// Synthetic checks (no ROS): ray-cast depth in a box room with pillars, PX4-like odometry with yaw / scale drift.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

#include "isam2_depth/depth_isam2.hpp"

using namespace isam2_depth;

static int failures = 0;
#define CHECK(cond, ...)                                    \
  do {                                                      \
    if (!(cond)) {                                          \
      std::printf("FAIL %s:%d: ", __FILE__, __LINE__);      \
      std::printf(__VA_ARGS__);                             \
      std::printf("\n");                                    \
      ++failures;                                           \
    }                                                       \
  } while (0)

namespace {
struct Box { Eigen::Vector3d lo, hi; };

// Room interior (NED: floor z = 0, ceiling z = -3) and solid pillars.
const Box kRoom{{-3.0, -3.0, -3.0}, {3.0, 3.0, 0.0}};
const std::vector<Box> kPillars = {{{1.8, -0.4, -3.0}, {2.2, 0.0, 0.0}},
                                   {{-0.5, 1.9, -3.0}, {0.1, 2.3, 0.0}},
                                   {{-2.4, -2.0, -1.2}, {-1.6, -1.2, 0.0}}};

double raycast(const Eigen::Vector3d& o, const Eigen::Vector3d& d) {
  double s = 1e9;
  for (int a = 0; a < 3; ++a) {   // exit from the room interior
    if (d[a] > 1e-12) s = std::min(s, (kRoom.hi[a] - o[a]) / d[a]);
    if (d[a] < -1e-12) s = std::min(s, (kRoom.lo[a] - o[a]) / d[a]);
  }
  for (const auto& b : kPillars) {   // slab entry
    double t0 = 0, t1 = 1e9;
    for (int a = 0; a < 3; ++a) {
      if (std::abs(d[a]) < 1e-12) {
        if (o[a] < b.lo[a] || o[a] > b.hi[a]) { t0 = 1e9; break; }
        continue;
      }
      double ta = (b.lo[a] - o[a]) / d[a], tb = (b.hi[a] - o[a]) / d[a];
      if (ta > tb) std::swap(ta, tb);
      t0 = std::max(t0, ta);
      t1 = std::min(t1, tb);
    }
    if (t0 <= t1 && t0 > 0) s = std::min(s, t0);
  }
  return s;
}

PinholeCamera camera() {   // OAK-D S2-like 640x400
  PinholeCamera c;
  c.width = 640; c.height = 400; c.fx = c.fy = 421.0; c.cx = 319.5; c.cy = 199.5;
  return c;
}

// camera optical (x right, y down, z fwd) -> body FRD, looking along body x, 10 cm ahead of the centre
gtsam::Pose3 bodyCam() {
  gtsam::Matrix3 R;
  R << 0, 0, 1, 1, 0, 0, 0, 1, 0;
  return gtsam::Pose3(gtsam::Rot3(R), gtsam::Point3(0.1, 0, 0));
}

DepthFrame render(const PinholeCamera& c, const gtsam::Pose3& T_world_cam, double stamp, std::mt19937& rng,
                  int stride = 2) {
  DepthFrame f;
  f.stamp = stamp; f.width = c.width; f.height = c.height;
  f.depth_m.assign(size_t(c.width) * c.height, std::numeric_limits<float>::quiet_NaN());
  std::normal_distribution<double> n01(0, 1);
  const Eigen::Matrix3d R = T_world_cam.rotation().matrix();
  const Eigen::Vector3d o = T_world_cam.translation();
  for (int v = 0; v < c.height; v += stride)
    for (int u = 0; u < c.width; u += stride) {
      const Eigen::Vector3d dc((u - c.cx) / c.fx, (v - c.cy) / c.fy, 1.0);
      const double z = raycast(o, R * dc);   // dc has unit z: s is z-depth
      f.depth_m[size_t(v) * c.width + u] = float(z + n01(rng) * 0.004 * z * z);
    }
  return f;
}

gtsam::Pose3 truthPose(double t) {   // circle r = 1.2 m at 1.4 m altitude, heading along the tangent, 0.4 m/s
  const double w = 0.4 / 1.2, a = w * t;
  return gtsam::Pose3(gtsam::Rot3::Ypr(a + M_PI / 2, 0.03 * std::sin(3 * a), 0.02 * std::cos(2 * a)),
                      gtsam::Point3(1.2 * std::cos(a) - 0.3, 1.2 * std::sin(a) - 0.3, -1.4));
}
gtsam::Vector3 truthVel(double t) {
  const double w = 0.4 / 1.2, a = w * t;
  return gtsam::Vector3(-0.4 * std::sin(a), 0.4 * std::cos(a), 0.0);
}

void testGrid() {
  GridParams gp;
  gp.min_hits = 1;
  std::vector<Eigen::Vector3f> pts = {Eigen::Vector3f(1.05f, 0.05f, 0.05f)};
  DistanceGrid g(pts, Eigen::Vector3d::Zero(), gp);
  CHECK(g.distance({1.05, 0.05, 0.05}) == 0.0f, "occupied voxel distance %f", g.distance({1.05, 0.05, 0.05}));
  CHECK(std::abs(g.distance({0.05, 0.05, 0.05}) - 1.0f) < 1e-4, "1 m away: %f", g.distance({0.05, 0.05, 0.05}));
  CHECK(std::abs(g.distance({1.05, 0.65, 0.45}) - std::sqrt(0.36f + 0.16f)) < 1e-4, "diagonal: %f",
        g.distance({1.05, 0.65, 0.45}));
  CHECK(g.distance({-3.0, 0.0, 0.0}) == gp.max_distance, "saturates");
  CHECK(!g.inside({0.0, 0.0, 10.0}), "outside");
}

void testIcp() {
  std::mt19937 rng(1);
  const PinholeCamera cam = camera();
  const gtsam::Pose3 Tbc = bodyCam();
  const gtsam::Pose3 Xa = truthPose(0.0), Xb = truthPose(1.5);
  const auto A = makeCloud(render(cam, Xa * Tbc, 0, rng), cam, {});
  const auto B = makeCloud(render(cam, Xb * Tbc, 0, rng), cam, {});
  const gtsam::Pose3 truth = (Xa * Tbc).between(Xb * Tbc);
  // camera frame: body yaw error is rotation about the optical y axis; PX4 roll / pitch are good
  const gtsam::Pose3 init = truth * gtsam::Pose3(gtsam::Rot3::Ypr(0.005, 0.04, 0.005), gtsam::Point3(0.06, 0.02, -0.05));
  const auto r = alignProjective(A, B, init, {});
  CHECK(r.has_value(), "ICP failed");
  if (!r) return;
  const gtsam::Pose3 e = truth.between(r->T_target_source);
  const gtsam::Point3 et = e.translation();
  const Eigen::Matrix<double, 6, 1> info = r->info.diagonal();
  std::printf("icp: %d inliers (%.0f%%), rmse %.4f m, %d it; error (camera x right, y down, z fwd) [%.3f %.3f %.3f] m "
              "%.3f deg; translation information [%.0f %.0f %.0f]\n", r->inliers, 100 * r->inlier_ratio, r->rmse,
              r->iterations, et.x(), et.y(), et.z(), gtsam::Rot3::Logmap(e.rotation()).norm() * 180 / M_PI, info(3),
              info(4), info(5));
  // Horizontal motion is observed (walls in two directions, pillars); vertical is not: the walls and posts run
  // floor to ceiling and the floor is out of view. ICP must say so (low information) rather than be accurate.
  CHECK(std::hypot(et.x(), et.z()) < 0.02, "ICP horizontal error %.3f", std::hypot(et.x(), et.z()));
  CHECK(gtsam::Rot3::Logmap(e.rotation()).norm() < 0.5 * M_PI / 180, "ICP rotation error");
  CHECK(info(4) < 0.1 * std::min(info(3), info(5)), "vertical must be reported as weakly observed");
}

void testGraph(double yaw_drift = 0.02, double scale = 1.04) {
  std::mt19937 rng(2);
  const PinholeCamera cam = camera();
  const gtsam::Pose3 Tbc = bodyCam();
  Isam2Params p;
  DepthIsam2 est(p, cam, Tbc);
  const double dt = 1.0 / 15.0, T = 40.0;          // > one full lap (2 pi r / v = 18.8 s): revisits
  // yaw_drift rad/s, scale: odometry distance scale error
  gtsam::Pose3 odom = truthPose(0);
  gtsam::Pose3 prev_truth = odom;
  double worst_ms = 0, t_last = 0;
  char worst[160] = "";
  gtsam::Vector3 v_last = gtsam::Vector3::Zero();
  for (double t = 0; t <= T; t += dt) {
    const gtsam::Pose3 X = truthPose(t);
    if (t > 0) {
      const gtsam::Pose3 rel = prev_truth.between(X);
      // drifting PX4 estimate: yaw bias and scale error on the relative motion (roll / pitch stay true)
      const gtsam::Pose3 noisy(gtsam::Rot3::Yaw(yaw_drift * dt) * rel.rotation(), scale * rel.translation());
      odom = odom * noisy;
      const auto ypr = odom.rotation().ypr(), ypr_true = X.rotation().ypr();
      odom = gtsam::Pose3(gtsam::Rot3::Ypr(ypr(0), ypr_true(1), ypr_true(2)), odom.translation());
    }
    prev_truth = X;
    OdomSample s;
    s.stamp = t;
    s.T_world_body = odom;
    s.v_world = odom.rotation() * X.rotation().inverse() * truthVel(t);
    if (est.addFrame(render(cam, X * Tbc, t, rng), s)) {
      const auto& st = est.stats();
      if (st.last_update_ms > worst_ms)
        std::snprintf(worst, sizeof worst, "%.1f ms (icp %.1f, isam2 %.1f, covariance + grid %.1f) at keyframe %zu",
                      st.last_update_ms, st.icp_ms, st.isam_ms, st.output_ms, st.keyframes);
      worst_ms = std::max(worst_ms, st.last_update_ms);
    }
    t_last = t;
    v_last = s.v_world;
  }
  // Compare at matching instants: latest() is the newest KEYFRAME (up to kf_max_interval old); predict() carries
  // it to the last odometry sample, which is what the MPPI would use.
  const MppiInput& out = est.latest();
  const auto& st = est.stats();
  const gtsam::Pose3 X_kf = truthPose(out.stamp), X_now = truthPose(t_last);
  OdomSample now;
  now.stamp = t_last;
  now.T_world_body = odom;
  now.v_world = v_last;
  const MppiInput pr = est.predict(now);
  auto yawErr = [](const gtsam::Pose3& a, const gtsam::Pose3& b) {
    return std::abs(gtsam::Rot3::Logmap(a.rotation().between(b.rotation())).z()) * 180 / M_PI;
  };
  const double e_odom = (odom.translation() - X_now.translation()).norm(), y_odom = yawErr(X_now, odom);
  const double e_kf = (out.T_world_body.translation() - X_kf.translation()).norm(), y_kf = yawErr(X_kf, out.T_world_body);
  const double e_now = (pr.T_world_body.translation() - X_now.translation()).norm(), y_now = yawErr(X_now, pr.T_world_body);
  std::printf("graph (yaw drift %.3f rad/s, scale %.2f): %zu keyframes, %zu factors, ICP ok %d / rejected %d, "
              "%d revisit links\n  worst update %s\n", yaw_drift, scale, st.keyframes, st.factors, st.icp_ok,
              st.icp_rejected, st.links, worst);
  std::printf("  error: odometry %.3f m %.2f deg | keyframe estimate %.3f m %.2f deg | predicted now %.3f m %.2f deg\n",
              e_odom, y_odom, e_kf, y_kf, e_now, y_now);
  CHECK(st.icp_rejected <= int(st.keyframes / 10), "too many ICP rejections");
  if (yaw_drift > 0) {
    CHECK(e_now < 0.3 * e_odom, "iSAM2 should beat the drifting odometry");
    CHECK(y_now < 0.3 * y_odom, "yaw drift should be corrected");
  } else {   // perfect odometry: the depth factors must not make it worse
    CHECK(e_now < 0.1, "position error with perfect odometry %.3f", e_now);
    CHECK(y_now < 2.0, "yaw error with perfect odometry %.2f", y_now);
  }
  CHECK(out.pose_cov.allFinite() && out.pose_cov.diagonal().minCoeff() > 0, "pose covariance");
  const double ev = (out.v_world - truthVel(out.stamp)).norm();
  CHECK(ev < 0.1, "velocity error %.3f", ev);

  // distance field (world NED): known structure is where the map says, the vehicle is in free space
  const auto& g = *out.distance;
  const double d_here = g.distance(out.T_world_body.translation());
  std::printf("  distance field %dx%dx%d, %zu occupied; at the vehicle %.2f m, pillar centre %.2f, wall %.2f, "
              "centre %.2f\n", g.size().x(), g.size().y(), g.size().z(), g.occupiedCentres().size(), d_here,
              g.distance({2.0, -0.2, -1.4}), g.distance({2.95, 0.0, -1.4}), g.distance({0.0, 0.0, -1.4}));
  CHECK(d_here > 0.5, "vehicle should be > 0.5 m from obstacles (%.2f)", d_here);
  CHECK(std::abs(g.distance({2.0, -0.2, -1.4}) - 0.2f) < 0.11f, "pillar centre is 0.2 m from its faces");
  CHECK(g.distance({2.95, 0.0, -1.4}) < 0.15f, "east wall (x = 3)");
  // nearest true structure is 1.4 m away (the floor, never in view near the centre: the observed map is sparser)
  CHECK(g.distance({0.0, 0.0, -1.4}) > 1.2f, "room centre is free: %.2f", g.distance({0.0, 0.0, -1.4}));

  // predict(): moves exactly by PX4's relative motion
  OdomSample b = now;
  b.stamp += 0.05;
  b.T_world_body = odom * gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(0.2, 0, 0));
  CHECK(std::abs((est.predict(b).T_world_body.translation() - pr.T_world_body.translation()).norm() - 0.2) < 1e-9,
        "predict moves by the odometry delta");
}
}  // namespace

int main() {
  testGrid();
  testIcp();
  testGraph(0.0, 1.0);    // perfect odometry
  testGraph(0.02, 1.04);  // PX4 drifting 1.1 deg/s in yaw, 4% scale error
  std::printf(failures ? "%d FAILURE(S)\n" : "all passed\n", failures);
  return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
