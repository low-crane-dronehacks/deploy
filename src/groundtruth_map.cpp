#include "maze_planning/groundtruth_map.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>

namespace maze_planning {

// ─────────────────────────────────────────────────────────────────────────────
// PlannedPath helpers
// ─────────────────────────────────────────────────────────────────────────────
PathPoint PlannedPath::sampleByTime(double t_s) const {
  if (points.empty()) return PathPoint{};
  if (points.size() == 1 || t_s <= points.front().t_s) return points.front();
  if (t_s >= points.back().t_s) return points.back();

  auto it = std::lower_bound(
      points.begin(), points.end(), t_s,
      [](const PathPoint& pt, double t) { return pt.t_s < t; });
  if (it == points.begin()) return points.front();
  const PathPoint& b = *it;
  const PathPoint& a = *(it - 1);
  const double dt = std::max(1e-6, b.t_s - a.t_s);
  const double u = std::clamp((t_s - a.t_s) / dt, 0.0, 1.0);

  PathPoint out;
  out.pos = (1.0 - u) * a.pos + u * b.pos;
  out.vel = (1.0 - u) * a.vel + u * b.vel;
  out.acc = (1.0 - u) * a.acc + u * b.acc;
  out.yaw_rad = wrapPi(a.yaw_rad + u * wrapPi(b.yaw_rad - a.yaw_rad));
  out.yawspeed_rad_s = (1.0 - u) * a.yawspeed_rad_s + u * b.yawspeed_rad_s;
  out.s_m = (1.0 - u) * a.s_m + u * b.s_m;
  out.t_s = t_s;
  return out;
}

PathPoint PlannedPath::sampleByArcLength(double s_m) const {
  if (points.empty()) return PathPoint{};
  if (points.size() == 1 || s_m <= points.front().s_m) return points.front();
  if (s_m >= points.back().s_m) return points.back();

  auto it = std::lower_bound(
      points.begin(), points.end(), s_m,
      [](const PathPoint& pt, double s) { return pt.s_m < s; });
  if (it == points.begin()) return points.front();
  const PathPoint& b = *it;
  const PathPoint& a = *(it - 1);
  const double ds = std::max(1e-6, b.s_m - a.s_m);
  const double u = std::clamp((s_m - a.s_m) / ds, 0.0, 1.0);

  PathPoint out;
  out.pos = (1.0 - u) * a.pos + u * b.pos;
  out.vel = (1.0 - u) * a.vel + u * b.vel;
  out.acc = (1.0 - u) * a.acc + u * b.acc;
  out.yaw_rad = wrapPi(a.yaw_rad + u * wrapPi(b.yaw_rad - a.yaw_rad));
  out.yawspeed_rad_s = (1.0 - u) * a.yawspeed_rad_s + u * b.yawspeed_rad_s;
  out.s_m = s_m;
  out.t_s = (1.0 - u) * a.t_s + u * b.t_s;
  return out;
}

int PlannedPath::findClosestIndex(const Eigen::Vector3d& pos, int min_idx) const {
  if (points.empty()) return 0;
  const int start = std::clamp(min_idx, 0, static_cast<int>(points.size()) - 1);
  // Search a bounded forward window to prevent jumping across self-intersecting maze corridors
  const int end = std::min(static_cast<int>(points.size()), start + 140);
  int best_i = start;
  double best_d2 = (points[start].pos - pos).squaredNorm();
  for (int i = start + 1; i < end; ++i) {
    const double d2 = (points[i].pos - pos).squaredNorm();
    if (d2 < best_d2) {
      best_d2 = d2;
      best_i = i;
    }
  }
  return best_i;
}

// ─────────────────────────────────────────────────────────────────────────────
// Analytical Primitive Signed Distance Functions
// ─────────────────────────────────────────────────────────────────────────────
double OBBSlabPrimitive::signedDistance(const Eigen::Vector3d& p_ned,
                                        Eigen::Vector3d* closest_pt_out) const {
  const double dx = p_ned.x() - center_xy.x();
  const double dy = p_ned.y() - center_xy.y();
  const double u = dx * axis_u_xy.x() + dy * axis_u_xy.y();
  const double v = dx * axis_v_xy.x() + dy * axis_v_xy.y();
  const double z_mid = 0.5 * (z_min_ned + z_max_ned);
  const double z_half = 0.5 * (z_max_ned - z_min_ned);
  const double w = p_ned.z() - z_mid;

  const double qx = std::abs(u) - half_len_m;
  const double qy = std::abs(v) - half_thick_m;
  const double qz = std::abs(w) - z_half;

  const double out_x = std::max(qx, 0.0);
  const double out_y = std::max(qy, 0.0);
  const double out_z = std::max(qz, 0.0);
  const double ext_dist = std::sqrt(out_x * out_x + out_y * out_y + out_z * out_z);
  const double int_dist = std::min(std::max({qx, qy, qz}), 0.0);

  if (closest_pt_out != nullptr) {
    const double cu = std::clamp(u, -half_len_m, half_len_m);
    const double cv = std::clamp(v, -half_thick_m, half_thick_m);
    const double cz = std::clamp(p_ned.z(), z_min_ned, z_max_ned);
    *closest_pt_out = Eigen::Vector3d(
        center_xy.x() + cu * axis_u_xy.x() + cv * axis_v_xy.x(),
        center_xy.y() + cu * axis_u_xy.y() + cv * axis_v_xy.y(),
        cz);
  }
  return ext_dist + int_dist;
}

double BoulderPrimitive::signedDistance(const Eigen::Vector3d& p_ned,
                                        Eigen::Vector3d* closest_pt_out) const {
  const Eigen::Vector3d d = p_ned - center_ned;
  const double nx = d.x() / std::max(1e-6, radii_m.x());
  const double ny = d.y() / std::max(1e-6, radii_m.y());
  const double nz = std::min(0.0, d.z()) / std::max(1e-6, radii_m.z());
  const double r_norm = std::sqrt(nx * nx + ny * ny + nz * nz);
  if (r_norm < 1e-8) {
    if (closest_pt_out) {
      *closest_pt_out = Eigen::Vector3d(center_ned.x(), center_ned.y(), center_ned.z() - radii_m.z());
    }
    return -radii_m.x();
  }
  const Eigen::Vector3d surf_pt(
      center_ned.x() + (nx / r_norm) * radii_m.x(),
      center_ned.y() + (ny / r_norm) * radii_m.y(),
      center_ned.z() + (nz / r_norm) * radii_m.z());
  if (closest_pt_out) *closest_pt_out = surf_pt;
  const double dist = (p_ned - surf_pt).norm();
  return (r_norm < 1.0 && p_ned.z() <= 0.0) ? -dist : dist;
}

double BVHNode::minDistanceSqToPoint(const Eigen::Vector3d& p) const {
  const double dx = std::max({bbox_min.x() - p.x(), 0.0, p.x() - bbox_max.x()});
  const double dy = std::max({bbox_min.y() - p.y(), 0.0, p.y() - bbox_max.y()});
  const double dz = std::max({bbox_min.z() - p.z(), 0.0, p.z() - bbox_max.z()});
  return dx * dx + dy * dy + dz * dz;
}

// ─────────────────────────────────────────────────────────────────────────────
// GroundTruthMap Implementation
// ─────────────────────────────────────────────────────────────────────────────
GroundTruthMap::GroundTruthMap() { initDefaultCourse(); }

OBBSlabPrimitive GroundTruthMap::makeOBB(double ax, double ay, double bx, double by,
                                         double half_thick, double z_min, double z_max) {
  OBBSlabPrimitive s{};
  s.center_xy = Eigen::Vector2d(0.5 * (ax + bx), 0.5 * (ay + by));
  const double dx = bx - ax;
  const double dy = by - ay;
  const double len = std::max(1e-6, std::hypot(dx, dy));
  s.axis_u_xy = Eigen::Vector2d(dx / len, dy / len);
  s.axis_v_xy = Eigen::Vector2d(-s.axis_u_xy.y(), s.axis_u_xy.x());
  s.half_len_m = 0.5 * len;
  s.half_thick_m = half_thick;
  s.z_min_ned = z_min;
  s.z_max_ned = z_max;
  return s;
}

void GroundTruthMap::initDefaultCourse() {
  arena_half_m_ = 2.99;
  wall_top_m_ = 1.7272;
  const double ht = 0.0508; // 4-inch wall half-thickness (0.0508 m)
  const double z_top = -wall_top_m_;

  obb_slabs_.clear();
  // wall0 decomposed into 4 slabs around the window aperture [-2.7015..-1.84225, -0.62735..-0.31455, z=-1.2192..-0.3048]
  obb_slabs_.push_back(makeOBB(-3.0297, -0.74675, -2.7015, -0.62735, ht, z_top, 0.0));
  obb_slabs_.push_back(makeOBB(-1.84225, -0.31455, -0.5472, 0.1568, ht, z_top, 0.0));
  obb_slabs_.push_back(makeOBB(-2.7015, -0.62735, -1.84225, -0.31455, ht, -0.3048, 0.0));
  obb_slabs_.push_back(makeOBB(-2.7015, -0.62735, -1.84225, -0.31455, ht, z_top, -1.2192));

  // wall1 .. wall6
  obb_slabs_.push_back(makeOBB(0.4572, -3.0480, 0.4572, 1.5748, ht, z_top, 0.0));
  obb_slabs_.push_back(makeOBB(-1.12855, 0.97335, 0.3871, 1.52505, ht, z_top, 0.0));
  obb_slabs_.push_back(makeOBB(0.5080, -1.9304, 2.1209, -1.9304, ht, z_top, 0.0));
  obb_slabs_.push_back(makeOBB(-0.6223, -0.9754, 0.4064, -0.9754, ht, z_top, 0.0));
  obb_slabs_.push_back(makeOBB(1.4351, -0.8128, 3.0480, -0.8128, ht, z_top, 0.0));
  obb_slabs_.push_back(makeOBB(0.5080, 0.3048, 2.1209, 0.3048, ht, z_top, 0.0));

  // Tunnel structural floor ribs & connector post
  obb_slabs_.push_back(makeOBB(0.5080, -2.4003, 0.5080, -1.6383, ht, -0.1016, 0.0));
  obb_slabs_.push_back(makeOBB(-0.0508, -2.4003, -0.0508, -1.6383, ht, -0.1016, 0.0));
  obb_slabs_.push_back(makeOBB(-0.6096, -2.4003, -0.6096, -1.6383, ht, -0.1016, 0.0));
  obb_slabs_.push_back(makeOBB(-0.04195, -1.6383, -0.04195, -1.3493, ht, -0.7620, 0.0));

  // 5 semi-ellipsoid boulders
  boulders_ = {
      BoulderPrimitive{Eigen::Vector3d(-1.09728, 2.19456, 0.0), Eigen::Vector3d(0.2286, 0.2286, 0.5334)},
      BoulderPrimitive{Eigen::Vector3d(0.12192, 1.58496, 0.0), Eigen::Vector3d(0.2286, 0.2286, 0.5334)},
      BoulderPrimitive{Eigen::Vector3d(1.64592, 1.88976, 0.0), Eigen::Vector3d(0.2286, 0.2286, 0.5334)},
      BoulderPrimitive{Eigen::Vector3d(0.12192, -0.24384, 0.0), Eigen::Vector3d(0.2286, 0.2286, 0.5334)},
      BoulderPrimitive{Eigen::Vector3d(-1.70688, -0.85344, 0.0), Eigen::Vector3d(0.2286, 0.2286, 0.5334)},
  };

  // Default course gates (Gate 0 -> Gate 1 -> Gate 2 -> Gate 3 -> Gate 4 Window -> Pad)
  default_gates_.clear();
  default_gates_.push_back(GateAperture{
      0, "gate0_launch_exit", Eigen::Vector3d(2.5540, -1.9304, -0.78),
      Eigen::Vector3d(0.0, 1.0, 0.0), Eigen::Vector2d(2.1209, -1.9304),
      Eigen::Vector2d(2.9870, -1.9304), 0.8661, -1.55, -0.25, false});
  default_gates_.push_back(GateAperture{
      1, "gate1_north_chute", Eigen::Vector3d(0.9462, -0.8128, -0.78),
      Eigen::Vector3d(0.0, 1.0, 0.0), Eigen::Vector2d(0.4572, -0.8128),
      Eigen::Vector2d(1.4351, -0.8128), 0.9779, -1.55, -0.25, false});
  default_gates_.push_back(GateAperture{
      2, "gate2_east_gap", Eigen::Vector3d(2.5540, 0.3048, -0.78),
      Eigen::Vector3d(0.0, 1.0, 0.0), Eigen::Vector2d(2.1209, 0.3048),
      Eigen::Vector2d(2.9870, 0.3048), 0.8661, -1.55, -0.25, false});
  default_gates_.push_back(GateAperture{
      3, "gate3_east_divider", Eigen::Vector3d(0.4572, 2.3800, -0.78),
      Eigen::Vector3d(-1.0, 0.0, 0.0), Eigen::Vector2d(0.4572, 1.5748),
      Eigen::Vector2d(0.4572, 2.9870), 1.4122, -1.55, -0.25, false});
  default_gates_.push_back(GateAperture{
      4, "gate4_window_bonus", Eigen::Vector3d(-2.2719, -0.4710, -0.7620),
      Eigen::Vector3d(-0.3455, -0.9384, 0.0), Eigen::Vector2d(-2.7015, -0.6274),
      Eigen::Vector2d(-1.8423, -0.3146), 0.9144, -1.2192, -0.3048, true});

  launch_mark_ned_ = Eigen::Vector3d(0.9144, -2.5146, -0.08);
  pad_center_ned_ = Eigen::Vector3d(-0.7925, -2.0422, -0.08);

  // Initialize bitset from analytical geometry so isStaticWallOccupied works immediately
  const std::size_t total_voxels =
      static_cast<std::size_t>(bit_nx_) * bit_ny_ * bit_nz_;
  const std::size_t n_words = (total_voxels + 63) / 64;
  occ_bitset_.assign(n_words, 0ULL);
  for (int ix = 0; ix < bit_nx_; ++ix) {
    const double x = bit_origin_.x() + (ix + 0.5) * bit_res_;
    for (int iy = 0; iy < bit_ny_; ++iy) {
      const double y = bit_origin_.y() + (iy + 0.5) * bit_res_;
      for (int iz = 0; iz < bit_nz_; ++iz) {
        const double z = bit_origin_.z() + (iz + 0.5) * bit_res_;
        Eigen::Vector3d cp;
        if (queryAnalytical(Eigen::Vector3d(x, y, z), cp) <= 0.03) {
          const std::size_t flat =
              (static_cast<std::size_t>(ix) * bit_ny_ + iy) * bit_nz_ + iz;
          occ_bitset_[flat >> 6] |= (1ULL << (flat & 63));
        }
      }
    }
  }
}

bool GroundTruthMap::loadFromBinary(const std::string& path) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs.is_open()) return false;

  char magic[4] = {0};
  uint32_t version = 0;
  int32_t bnx = 0, bny = 0, bnz = 0;
  float bres = 0.0f, ox = 0.0f, oy = 0.0f, oz = 0.0f;
  uint32_t n_obb = 0, n_boulders = 0, n_verts = 0, n_tris = 0, n_bit_words = 0;

  ifs.read(magic, 4);
  if (std::strncmp(magic, "CGTM", 4) != 0) return false;
  ifs.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));
  if (version != 1) return false;
  ifs.read(reinterpret_cast<char*>(&bnx), sizeof(int32_t));
  ifs.read(reinterpret_cast<char*>(&bny), sizeof(int32_t));
  ifs.read(reinterpret_cast<char*>(&bnz), sizeof(int32_t));
  ifs.read(reinterpret_cast<char*>(&bres), sizeof(float));
  ifs.read(reinterpret_cast<char*>(&ox), sizeof(float));
  ifs.read(reinterpret_cast<char*>(&oy), sizeof(float));
  ifs.read(reinterpret_cast<char*>(&oz), sizeof(float));
  ifs.read(reinterpret_cast<char*>(&n_obb), sizeof(uint32_t));
  ifs.read(reinterpret_cast<char*>(&n_boulders), sizeof(uint32_t));
  ifs.read(reinterpret_cast<char*>(&n_verts), sizeof(uint32_t));
  ifs.read(reinterpret_cast<char*>(&n_tris), sizeof(uint32_t));
  ifs.read(reinterpret_cast<char*>(&n_bit_words), sizeof(uint32_t));
  if (!ifs.good() || n_obb > 10000 || n_boulders > 10000 || n_verts > 1000000 || n_tris > 1000000) {
    return false;
  }

  bit_nx_ = static_cast<int>(bnx);
  bit_ny_ = static_cast<int>(bny);
  bit_nz_ = static_cast<int>(bnz);
  bit_res_ = static_cast<double>(bres);
  bit_origin_ = Eigen::Vector3d(ox, oy, oz);

  obb_slabs_.clear();
  obb_slabs_.reserve(n_obb);
  for (uint32_t i = 0; i < n_obb; ++i) {
    float buf[7];
    ifs.read(reinterpret_cast<char*>(buf), sizeof(buf));
    obb_slabs_.push_back(makeOBB(buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6]));
  }

  boulders_.resize(n_boulders);
  for (uint32_t i = 0; i < n_boulders; ++i) {
    float buf[6];
    ifs.read(reinterpret_cast<char*>(buf), sizeof(buf));
    BoulderPrimitive b{};
    b.center_ned = Eigen::Vector3d(buf[0], buf[1], buf[2]);
    b.radii_m = Eigen::Vector3d(buf[3], buf[4], buf[5]);
    boulders_[i] = b;
  }

  std::vector<Eigen::Vector3d> verts(n_verts);
  for (uint32_t i = 0; i < n_verts; ++i) {
    float vbuf[3];
    ifs.read(reinterpret_cast<char*>(vbuf), sizeof(vbuf));
    verts[i] = Eigen::Vector3d(vbuf[0], vbuf[1], vbuf[2]);
  }

  triangles_.resize(n_tris);
  for (uint32_t i = 0; i < n_tris; ++i) {
    int32_t fbuf[3];
    ifs.read(reinterpret_cast<char*>(fbuf), sizeof(fbuf));
    SurfaceTriangle t{};
    if (fbuf[0] >= 0 && static_cast<uint32_t>(fbuf[0]) < n_verts &&
        fbuf[1] >= 0 && static_cast<uint32_t>(fbuf[1]) < n_verts &&
        fbuf[2] >= 0 && static_cast<uint32_t>(fbuf[2]) < n_verts) {
      t.v0 = verts[fbuf[0]];
      t.v1 = verts[fbuf[1]];
      t.v2 = verts[fbuf[2]];
      Eigen::Vector3d n = (t.v1 - t.v0).cross(t.v2 - t.v0);
      const double nl = n.norm();
      t.normal = (nl > 1e-9) ? (n / nl) : Eigen::Vector3d(0.0, 0.0, -1.0);
      t.centroid = (t.v0 + t.v1 + t.v2) / 3.0;
    }
    triangles_[i] = t;
  }

  occ_bitset_.resize(n_bit_words, 0ULL);
  ifs.read(reinterpret_cast<char*>(occ_bitset_.data()),
           static_cast<std::streamsize>(n_bit_words * sizeof(uint64_t)));
  if (!ifs.good()) return false;

  buildBVH();
  binary_loaded_ = true;
  return true;
}

bool GroundTruthMap::loadFromObj(const std::string& obj_path) {
  std::ifstream ifs(obj_path);
  if (!ifs.is_open()) return false;

  std::vector<Eigen::Vector3d> verts_enu;
  std::vector<SurfaceTriangle> parsed_tris;
  std::string line;
  while (std::getline(ifs, line)) {
    if (line.size() < 2) continue;
    if (line[0] == 'v' && line[1] == ' ') {
      std::istringstream iss(line.substr(2));
      double x = 0.0, y = 0.0, z = 0.0;
      if (iss >> x >> y >> z) verts_enu.emplace_back(x, y, z);
    } else if (line[0] == 'f' && line[1] == ' ') {
      std::istringstream iss(line.substr(2));
      std::string tok;
      std::vector<int> idx;
      while (iss >> tok) {
        const auto slash = tok.find('/');
        const int vi = std::stoi(slash == std::string::npos ? tok : tok.substr(0, slash));
        if (vi > 0 && static_cast<std::size_t>(vi) <= verts_enu.size()) idx.push_back(vi - 1);
      }
      for (std::size_t k = 1; k + 1 < idx.size(); ++k) {
        const Eigen::Vector3d& a = verts_enu[idx[0]];
        const Eigen::Vector3d& b = verts_enu[idx[k]];
        const Eigen::Vector3d& c = verts_enu[idx[k + 1]];
        // Gazebo ENU (x=East, y=North, z=Up) -> NED (x=North, y=East, z=Down)
        SurfaceTriangle t{};
        t.v0 = Eigen::Vector3d(a.y(), a.x(), -a.z());
        t.v1 = Eigen::Vector3d(b.y(), b.x(), -b.z());
        t.v2 = Eigen::Vector3d(c.y(), c.x(), -c.z());
        Eigen::Vector3d n = (t.v1 - t.v0).cross(t.v2 - t.v0);
        const double nl = n.norm();
        t.normal = (nl > 1e-9) ? (n / nl) : Eigen::Vector3d(0.0, 0.0, -1.0);
        t.centroid = (t.v0 + t.v1 + t.v2) / 3.0;
        parsed_tris.push_back(t);
      }
    }
  }
  if (parsed_tris.empty()) return false;
  triangles_ = std::move(parsed_tris);
  buildBVH();
  obj_loaded_ = true;
  return true;
}

void GroundTruthMap::buildBVH() {
  bvh_nodes_.clear();
  if (triangles_.empty()) return;
  bvh_nodes_.reserve(2 * triangles_.size());
  std::vector<int> indices(triangles_.size());
  for (std::size_t i = 0; i < triangles_.size(); ++i) indices[i] = static_cast<int>(i);
  buildBVHRecursive(indices, 0, static_cast<int>(triangles_.size()));

  std::vector<SurfaceTriangle> reordered(triangles_.size());
  for (std::size_t i = 0; i < triangles_.size(); ++i) {
    reordered[i] = triangles_[indices[i]];
  }
  triangles_ = std::move(reordered);
}

int GroundTruthMap::buildBVHRecursive(std::vector<int>& tri_indices, int start, int end) {
  const int node_idx = static_cast<int>(bvh_nodes_.size());
  bvh_nodes_.push_back(BVHNode{});

  Eigen::Vector3d bmin(1e9, 1e9, 1e9);
  Eigen::Vector3d bmax(-1e9, -1e9, -1e9);
  for (int i = start; i < end; ++i) {
    const SurfaceTriangle& t = triangles_[tri_indices[i]];
    bmin = bmin.cwiseMin(t.v0).cwiseMin(t.v1).cwiseMin(t.v2);
    bmax = bmax.cwiseMax(t.v0).cwiseMax(t.v1).cwiseMax(t.v2);
  }
  bvh_nodes_[node_idx].bbox_min = bmin;
  bvh_nodes_[node_idx].bbox_max = bmax;

  const int count = end - start;
  if (count <= 8) {
    bvh_nodes_[node_idx].tri_start = start;
    bvh_nodes_[node_idx].tri_count = count;
    return node_idx;
  }

  const Eigen::Vector3d ext = bmax - bmin;
  int axis = 0;
  if (ext.y() > ext.x() && ext.y() >= ext.z()) axis = 1;
  else if (ext.z() > ext.x() && ext.z() > ext.y()) axis = 2;

  const int mid = start + count / 2;
  std::nth_element(
      tri_indices.begin() + start, tri_indices.begin() + mid, tri_indices.begin() + end,
      [this, axis](int ia, int ib) {
        return triangles_[ia].centroid[axis] < triangles_[ib].centroid[axis];
      });

  const int left = buildBVHRecursive(tri_indices, start, mid);
  const int right = buildBVHRecursive(tri_indices, mid, end);
  bvh_nodes_[node_idx].left_child = left;
  bvh_nodes_[node_idx].right_child = right;
  bvh_nodes_[node_idx].tri_count = 0;
  return node_idx;
}

Eigen::Vector3d GroundTruthMap::closestPointOnTriangle(const Eigen::Vector3d& p,
                                                       const SurfaceTriangle& tri) {
  const Eigen::Vector3d ab = tri.v1 - tri.v0;
  const Eigen::Vector3d ac = tri.v2 - tri.v0;
  const Eigen::Vector3d ap = p - tri.v0;

  const double d1 = ab.dot(ap);
  const double d2 = ac.dot(ap);
  if (d1 <= 0.0 && d2 <= 0.0) return tri.v0;

  const Eigen::Vector3d bp = p - tri.v1;
  const double d3 = ab.dot(bp);
  const double d4 = ac.dot(bp);
  if (d3 >= 0.0 && d4 <= d3) return tri.v1;

  const double vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
    const double v = d1 / (d1 - d3);
    return tri.v0 + v * ab;
  }

  const Eigen::Vector3d cp = p - tri.v2;
  const double d5 = ab.dot(cp);
  const double d6 = ac.dot(cp);
  if (d6 >= 0.0 && d5 <= d6) return tri.v2;

  const double vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
    const double w = d2 / (d2 - d6);
    return tri.v0 + w * ac;
  }

  const double va = d3 * d6 - d5 * d4;
  if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
    const double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    return tri.v1 + w * (tri.v2 - tri.v1);
  }

  const double denom = 1.0 / (va + vb + vc);
  const double v = vb * denom;
  const double w = vc * denom;
  return tri.v0 + v * ab + w * ac;
}

void GroundTruthMap::queryBVH(int node_idx, const Eigen::Vector3d& p_ned,
                              double& best_dist_sq, Eigen::Vector3d& best_closest_pt) const {
  if (node_idx < 0 || static_cast<std::size_t>(node_idx) >= bvh_nodes_.size()) return;
  const BVHNode& node = bvh_nodes_[node_idx];
  if (node.minDistanceSqToPoint(p_ned) >= best_dist_sq) return;

  if (node.isLeaf()) {
    for (int i = 0; i < node.tri_count; ++i) {
      const SurfaceTriangle& tri = triangles_[node.tri_start + i];
      const Eigen::Vector3d q = closestPointOnTriangle(p_ned, tri);
      const double d2 = (p_ned - q).squaredNorm();
      if (d2 < best_dist_sq) {
        best_dist_sq = d2;
        best_closest_pt = q;
      }
    }
    return;
  }

  const int c0 = node.left_child;
  const int c1 = node.right_child;
  const double d0 = (c0 >= 0) ? bvh_nodes_[c0].minDistanceSqToPoint(p_ned) : 1e18;
  const double d1 = (c1 >= 0) ? bvh_nodes_[c1].minDistanceSqToPoint(p_ned) : 1e18;
  if (d0 < d1) {
    if (d0 < best_dist_sq) queryBVH(c0, p_ned, best_dist_sq, best_closest_pt);
    if (d1 < best_dist_sq) queryBVH(c1, p_ned, best_dist_sq, best_closest_pt);
  } else {
    if (d1 < best_dist_sq) queryBVH(c1, p_ned, best_dist_sq, best_closest_pt);
    if (d0 < best_dist_sq) queryBVH(c0, p_ned, best_dist_sq, best_closest_pt);
  }
}

double GroundTruthMap::queryAnalytical(const Eigen::Vector3d& p_ned,
                                       Eigen::Vector3d& closest_pt_out) const {
  double best_d = 1e9;
  closest_pt_out = p_ned;

  // 1. Outer arena net perimeter walls (|x| <= arena_half_m_, |y| <= arena_half_m_)
  const double d_north = arena_half_m_ - p_ned.x();
  if (d_north < best_d) {
    best_d = d_north;
    closest_pt_out = Eigen::Vector3d(arena_half_m_, p_ned.y(), p_ned.z());
  }
  const double d_south = arena_half_m_ + p_ned.x();
  if (d_south < best_d) {
    best_d = d_south;
    closest_pt_out = Eigen::Vector3d(-arena_half_m_, p_ned.y(), p_ned.z());
  }
  const double d_east = arena_half_m_ - p_ned.y();
  if (d_east < best_d) {
    best_d = d_east;
    closest_pt_out = Eigen::Vector3d(p_ned.x(), arena_half_m_, p_ned.z());
  }
  const double d_west = arena_half_m_ + p_ned.y();
  if (d_west < best_d) {
    best_d = d_west;
    closest_pt_out = Eigen::Vector3d(p_ned.x(), -arena_half_m_, p_ned.z());
  }

  // 2. Analytical OBB wall & tunnel slabs
  for (const auto& slab : obb_slabs_) {
    Eigen::Vector3d cp;
    const double d = slab.signedDistance(p_ned, &cp);
    if (d < best_d) {
      best_d = d;
      closest_pt_out = cp;
    }
  }

  // 3. Analytical semi-ellipsoid boulders
  for (const auto& boulder : boulders_) {
    Eigen::Vector3d cp;
    const double d = boulder.signedDistance(p_ned, &cp);
    if (d < best_d) {
      best_d = d;
      closest_pt_out = cp;
    }
  }

  // 4. Analytical tunnel arch roof & side walls
  {
    const double dx = p_ned.x() - tunnel_.entry_xy.x();
    const double dy = p_ned.y() - tunnel_.entry_xy.y();
    const double s_along = dx * tunnel_.axis_u_xy.x() + dy * tunnel_.axis_u_xy.y();
    const double v_lat = dx * tunnel_.axis_v_xy.x() + dy * tunnel_.axis_v_xy.y();
    const double alt = -p_ned.z();
    if (s_along >= -0.35 && s_along <= tunnel_.length_m + 0.35 &&
        std::abs(v_lat) <= tunnel_.half_width_m + 0.45 && alt <= tunnel_.arch_top_m + 0.45) {
      const double s_clamped = std::clamp(s_along, 0.0, tunnel_.length_m);
      const double ds = std::max({0.0, -s_along, s_along - tunnel_.length_m});
      if (alt <= 0.45) {
        const double d_side = std::abs( std::abs(v_lat) - tunnel_.half_width_m ) - 0.5 * tunnel_.shell_thick_m;
        const double d3 = std::hypot(std::max(0.0, d_side), ds);
        if (d3 < best_d) {
          best_d = d3;
          const double sign_v = (v_lat >= 0.0) ? 1.0 : -1.0;
          const double wy = tunnel_.entry_xy.y() + s_clamped * tunnel_.axis_u_xy.y() +
                            sign_v * tunnel_.half_width_m * tunnel_.axis_v_xy.y();
          const double wx = tunnel_.entry_xy.x() + s_clamped * tunnel_.axis_u_xy.x() +
                            sign_v * tunnel_.half_width_m * tunnel_.axis_v_xy.x();
          closest_pt_out = Eigen::Vector3d(wx, wy, p_ned.z());
        }
      } else {
        const double dz_arch = alt - 0.3810;
        const double r_arch = std::hypot(v_lat, dz_arch);
        const double d_arch = std::abs(r_arch - tunnel_.half_width_m) - 0.5 * tunnel_.shell_thick_m;
        const double d3 = std::hypot(std::max(0.0, d_arch), ds);
        if (d3 < best_d) {
          best_d = d3;
          const double scale = (r_arch > 1e-6) ? (tunnel_.half_width_m / r_arch) : 1.0;
          const double cv = v_lat * scale;
          const double calt = 0.3810 + dz_arch * scale;
          const double wx = tunnel_.entry_xy.x() + s_clamped * tunnel_.axis_u_xy.x() + cv * tunnel_.axis_v_xy.x();
          const double wy = tunnel_.entry_xy.y() + s_clamped * tunnel_.axis_u_xy.y() + cv * tunnel_.axis_v_xy.y();
          closest_pt_out = Eigen::Vector3d(wx, wy, -calt);
        }
      }
    }
  }

  // 5. Blocked corridor spheres (FMEA Pillar Dead-Center global replan constraint)
  for (const auto& blk : blocked_corridors_) {
    const Eigen::Vector3d diff = p_ned - blk.center_ned;
    const double r = diff.norm();
    const double d = r - blk.radius_m;
    if (d < best_d) {
      best_d = d;
      closest_pt_out = (r > 1e-6) ? (blk.center_ned + (diff / r) * blk.radius_m) : blk.center_ned;
    }
  }

  return best_d;
}

double GroundTruthMap::clearance(const Eigen::Vector3d& p_ned) const {
  Eigen::Vector3d closest_pt;
  double best_d = queryAnalytical(p_ned, closest_pt);
  if (obb_slabs_.empty() && !bvh_nodes_.empty() && best_d > 0.0) {
    double best_d2 = best_d * best_d;
    queryBVH(0, p_ned, best_d2, closest_pt);
    best_d = std::min(best_d, std::sqrt(best_d2));
  }
  return best_d;
}

double GroundTruthMap::clearanceAndGradient(const Eigen::Vector3d& p_ned,
                                            Eigen::Vector3d& grad_out) const {
  Eigen::Vector3d closest_pt;
  double best_d = queryAnalytical(p_ned, closest_pt);
  if (obb_slabs_.empty() && !bvh_nodes_.empty() && best_d > 0.0) {
    double best_d2 = best_d * best_d;
    queryBVH(0, p_ned, best_d2, closest_pt);
    best_d = std::min(best_d, std::sqrt(best_d2));
  }
  const Eigen::Vector3d diff = p_ned - closest_pt;
  const double n = diff.norm();
  if (n > 1e-5) {
    grad_out = ((best_d >= 0.0) ? (1.0 / n) : (-1.0 / n)) * diff;
  } else {
    const double eps = 0.015;
    const double dx = clearance(p_ned + Eigen::Vector3d(eps, 0.0, 0.0)) -
                      clearance(p_ned - Eigen::Vector3d(eps, 0.0, 0.0));
    const double dy = clearance(p_ned + Eigen::Vector3d(0.0, eps, 0.0)) -
                      clearance(p_ned - Eigen::Vector3d(0.0, eps, 0.0));
    const double dz = clearance(p_ned + Eigen::Vector3d(0.0, 0.0, eps)) -
                      clearance(p_ned - Eigen::Vector3d(0.0, 0.0, eps));
    const Eigen::Vector3d g(dx, dy, dz);
    const double gn = g.norm();
    grad_out = (gn > 1e-9) ? (g / gn) : Eigen::Vector3d(1.0, 0.0, 0.0);
  }
  return best_d;
}

bool GroundTruthMap::isStaticWallOccupied(const Eigen::Vector3d& p_ned) const {
  if (!occ_bitset_.empty()) {
    const int ix = static_cast<int>(std::floor((p_ned.x() - bit_origin_.x()) / bit_res_));
    const int iy = static_cast<int>(std::floor((p_ned.y() - bit_origin_.y()) / bit_res_));
    const int iz = static_cast<int>(std::floor((p_ned.z() - bit_origin_.z()) / bit_res_));
    if (ix >= 0 && ix < bit_nx_ && iy >= 0 && iy < bit_ny_ && iz >= 0 && iz < bit_nz_) {
      const std::size_t flat =
          (static_cast<std::size_t>(ix) * bit_ny_ + iy) * bit_nz_ + iz;
      if ((occ_bitset_[flat >> 6] >> (flat & 63)) & 1ULL) return true;
    }
  }
  Eigen::Vector3d cp;
  return queryAnalytical(p_ned, cp) <= 0.02;
}

bool GroundTruthMap::isSegmentValid(const Eigen::Vector3d& a_ned, const Eigen::Vector3d& b_ned,
                                    double radius_m, double min_step_m,
                                    double alt_min_m, double alt_max_m) const {
  const double alt_a = -a_ned.z();
  const double alt_b = -b_ned.z();
  if (alt_a < alt_min_m || alt_a > alt_max_m || alt_b < alt_min_m || alt_b > alt_max_m) {
    return false;
  }
  const Eigen::Vector3d ab = b_ned - a_ned;
  const double total_len = ab.norm();
  if (total_len < 1e-6) {
    return clearance(a_ned) >= radius_m;
  }
  const Eigen::Vector3d dir = ab / total_len;
  double s = 0.0;
  while (s <= total_len) {
    const Eigen::Vector3d p = a_ned + dir * s;
    if (isStaticWallOccupied(p)) return false;
    const double c = clearance(p);
    if (c < radius_m) return false;
    if (s >= total_len) break;
    const double step = std::max(min_step_m, 0.85 * (c - radius_m));
    s = (s + step > total_len && s < total_len) ? total_len : (s + step);
  }
  return true;
}

Eigen::Vector3d GroundTruthMap::refineGateCenter(const GateAperture& gate,
                                                 double margin_from_posts_m) const {
  const Eigen::Vector2d ab = gate.post_b_xy - gate.post_a_xy;
  const double w = ab.norm();
  if (w < 2.0 * margin_from_posts_m + 0.05) return gate.center_map;

  const double u_min = std::clamp(margin_from_posts_m / w, 0.15, 0.48);
  const double u_max = 1.0 - u_min;
  const double target_z = std::clamp(gate.center_map.z(), gate.z_min_map + 0.18, gate.z_max_map - 0.18);

  double best_c = -1e9;
  Eigen::Vector3d best_pt = gate.center_map;
  best_pt.z() = target_z;
  for (int i = 0; i <= 14; ++i) {
    const double u = u_min + (u_max - u_min) * (static_cast<double>(i) / 14.0);
    const Eigen::Vector2d xy = gate.post_a_xy + u * ab;
    const Eigen::Vector3d cand(xy.x(), xy.y(), target_z);
    const double c = clearance(cand);
    if (c > best_c) {
      best_c = c;
      best_pt = cand;
    }
  }
  return best_pt;
}

void GroundTruthMap::markCorridorBlocked(const Eigen::Vector3d& center_ned, double radius_m) {
  for (auto& existing : blocked_corridors_) {
    if ((existing.center_ned - center_ned).norm() < 0.25) {
      existing.radius_m = std::max(existing.radius_m, radius_m);
      return;
    }
  }
  blocked_corridors_.push_back(BlockedCorridorSphere{center_ned, radius_m});
}

}  // namespace maze_planning
