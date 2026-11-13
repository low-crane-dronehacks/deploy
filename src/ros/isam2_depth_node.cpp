// isam2_depth_node: depth + PX4 odometry -> iSAM2 keyframe graph -> state + distance field for an MPPI.
//
//   in   depth          sensor_msgs/Image (16UC1 mm, 0 = invalid | 32FC1 m, +inf = no return) + CameraInfo
//        calibration    TF base_link (FLU) <- depth frame_id, looked up once (converted to FRD as policynav/obs.py)
//        PX4            /fmu/out/vehicle_local_position_v1 (NED position / velocity), /fmu/out/vehicle_attitude
//   out  ~/odometry     nav_msgs/Odometry at the PX4 rate: newest keyframe estimate + PX4 motion since
//                       (pose world NED <- body FRD; twist in the body frame, as nav_msgs defines it)
//        ~/obstacles    sensor_msgs/PointCloud2 of occupied distance-field voxels (world NED), per keyframe: the
//                       local window the MPPI sees (map_radius / grid extent around the vehicle)
//        ~/keyframes    nav_msgs/Path of ALL keyframe poses at their current smoothed values, per keyframe
//                       (iSAM2 keeps re-estimating past poses; watch them move when ICP / revisits correct them)
//        ~/map          sensor_msgs/PointCloud2: the whole corrected map, every keyframe cloud at its smoothed pose,
//                       map_voxel cells with >= grid_min_hits points (world NED), every map_publish_period s
//        ~/landmarks    visualization_msgs/MarkerArray: estimated landmarks (sphere + id), per keyframe
//        All of these are published only while something subscribes; ~/keyframes and ~/map are latched.
//   save ~/save_map     std_srvs/Trigger: write the corrected map now; also written at shutdown if map_save_path is
//                       set. <path>.ply = the ~/map voxels as binary PLY in ENU, z up (east, north, up; the same
//                       convention as a phone scan, so ego_planner/scripts/view_course_scan.py reads it), and
//                       <path>_keyframes.csv = the smoothed keyframe poses (NED, as ~/keyframes).
//        MppiStateSink  in-process: setSink() and the MPPI gets MppiInput (state + DistanceGrid) at the PX4 rate
// Publishes nothing to PX4 and never arms.
#include <cmath>
#include <cstring>
#include <ctime>
#include <fstream>
#include <deque>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <px4_msgs/msg/vehicle_attitude.hpp>
#include <px4_msgs/msg/vehicle_local_position.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <visualization_msgs/msg/marker_array.hpp>

#include "isam2_depth/depth_isam2.hpp"

namespace isam2_depth {

class Isam2DepthNode : public rclcpp::Node {
 public:
  Isam2DepthNode() : Node("isam2_depth") {
    depth_topic_ = declare_parameter<std::string>("depth_topic", "/camera/depth/image_raw");
    info_topic_ = declare_parameter<std::string>("depth_info_topic", "/camera/depth/camera_info");
    base_frame_ = declare_parameter<std::string>("base_frame", "base_link");
    world_frame_ = declare_parameter<std::string>("world_frame", "isam2_world_ned");
    body_frame_ = declare_parameter<std::string>("body_frame", "isam2_body_frd");
    publish_obstacles_ = declare_parameter("publish_obstacles", true);
    max_odom_gap_ = declare_parameter("max_odom_gap", 0.1);   // s: a depth frame needs PX4 data this close
    map_period_ = declare_parameter("map_publish_period", 1.0);   // s between ~/map rebuilds (cost grows with keyframes)
    map_save_path_ = declare_parameter<std::string>("map_save_path", "");   // "" = do not save at shutdown
    cad_prior_path_ = declare_parameter<std::string>("cad_prior_path", ""); // optional CADP v1 3D ESDF prior
    if (!cad_prior_path_.empty()) {
      std::string err;
      cad_prior_ = CadPriorEsdf::load(cad_prior_path_, &err);
      if (cad_prior_) {
        const auto shp = cad_prior_->shape();
        RCLCPP_INFO(get_logger(), "loaded CAD prior ESDF %s (%dx%dx%d @ %.2f m)",
                    cad_prior_path_.c_str(), shp.x(), shp.y(), shp.z(), cad_prior_->resolution());
      } else {
        RCLCPP_WARN(get_logger(), "CAD prior ESDF disabled (%s)", err.c_str());
      }
    }
    loadParams();

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
    const auto px4_qos = rclcpp::QoS(1).best_effort().transient_local();
    sub_lpos_ = create_subscription<px4_msgs::msg::VehicleLocalPosition>(
        "/fmu/out/vehicle_local_position_v1", px4_qos, [this](px4_msgs::msg::VehicleLocalPosition::ConstSharedPtr m) { onLpos(*m); });
    sub_att_ = create_subscription<px4_msgs::msg::VehicleAttitude>(
        "/fmu/out/vehicle_attitude", px4_qos, [this](px4_msgs::msg::VehicleAttitude::ConstSharedPtr m) {
          q_ = gtsam::Rot3::Quaternion(m->q[0], m->q[1], m->q[2], m->q[3]);   // FRD -> NED, w first
        });
    sub_info_ = create_subscription<sensor_msgs::msg::CameraInfo>(
        info_topic_, rclcpp::SensorDataQoS(), [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr m) { onInfo(*m); });
    sub_depth_ = create_subscription<sensor_msgs::msg::Image>(
        depth_topic_, rclcpp::SensorDataQoS(), [this](sensor_msgs::msg::Image::ConstSharedPtr m) { onDepth(*m); });
    pub_odom_ = create_publisher<nav_msgs::msg::Odometry>("~/odometry", 10);
    pub_vio_icp_odom_ = create_publisher<nav_msgs::msg::Odometry>("~/vio_icp_odometry", 10);
    pub_obst_ = create_publisher<sensor_msgs::msg::PointCloud2>("~/obstacles", 1);
    const auto latched = rclcpp::QoS(1).reliable().transient_local();   // RViz joining late still gets the last one
    pub_kf_ = create_publisher<nav_msgs::msg::Path>("~/keyframes", latched);
    pub_map_ = create_publisher<sensor_msgs::msg::PointCloud2>("~/map", latched);
    pub_adapted_ = create_publisher<sensor_msgs::msg::PointCloud2>("~/adapted_map", latched);
    pub_priors_ = create_publisher<sensor_msgs::msg::PointCloud2>("~/priors", latched);
    pub_new_obst_ = create_publisher<sensor_msgs::msg::PointCloud2>("~/new_updated_obstacles", latched);
    pub_cleared_ = create_publisher<sensor_msgs::msg::PointCloud2>("~/cleared_empty_space", latched);
    pub_lm_ = create_publisher<visualization_msgs::msg::MarkerArray>("~/landmarks", latched);
    pub_prior_updates_ = create_publisher<visualization_msgs::msg::MarkerArray>("~/prior_updates", latched);
    if (cad_prior_) {
      const PriorEsdfMap seed_map(p_.prior_map, cad_prior_.get());
      const builtin_interfaces::msg::Time t0_stamp{};
      publishCloud(*pub_priors_, seed_map.priors(), t0_stamp);
      publishCloud(*pub_new_obst_, {}, t0_stamp);
      publishCloud(*pub_cleared_, {}, t0_stamp);
    }
    srv_save_ = create_service<std_srvs::srv::Trigger>(
        "~/save_map", [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                             std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
          res->success = saveMap(map_save_path_, &res->message);
        });
    timer_ = create_wall_timer(std::chrono::seconds(5), [this] { logStats(); });
    RCLCPP_INFO(get_logger(), "depth %s, PX4 vehicle_local_position_v1 + vehicle_attitude; no commands are sent",
                depth_topic_.c_str());
  }

  // The MPPI registers here (same process, e.g. a component container or a main() that owns both).
  void setSink(std::shared_ptr<MppiStateSink> sink) { sink_ = std::move(sink); }

 private:
  void loadParams() {
    // Only the values worth tuning on the vehicle are exposed; the rest are Isam2Params defaults.
    p_.kf_min_translation = declare_parameter("kf_min_translation", p_.kf_min_translation);
    p_.kf_min_rotation_deg = declare_parameter("kf_min_rotation_deg", p_.kf_min_rotation_deg);
    p_.kf_max_interval = declare_parameter("kf_max_interval", p_.kf_max_interval);
    p_.relinearize_threshold = declare_parameter("relinearize_threshold", p_.relinearize_threshold);
    p_.wildfire_threshold = declare_parameter("wildfire_threshold", p_.wildfire_threshold);
    p_.extra_links = static_cast<int>(declare_parameter("extra_links", static_cast<int64_t>(p_.extra_links)));
    p_.cloud.stride = static_cast<int>(declare_parameter("cloud_stride", static_cast<int64_t>(p_.cloud.stride)));
    p_.cloud.max_range = declare_parameter("max_range", p_.cloud.max_range);
    p_.icp.info_deflation = declare_parameter("icp_info_deflation", p_.icp.info_deflation);
    p_.grid.resolution = declare_parameter("grid_resolution", p_.grid.resolution);
    p_.grid.min_hits = static_cast<int>(declare_parameter("grid_min_hits", static_cast<int64_t>(p_.grid.min_hits)));
    p_.prior_map.pixel_stride =
        static_cast<int>(declare_parameter("prior_pixel_stride", static_cast<int64_t>(p_.prior_map.pixel_stride)));
    p_.prior_map.l_free =
        static_cast<float>(declare_parameter("prior_l_free", static_cast<double>(p_.prior_map.l_free)));
  }

  double now_s() { return get_clock()->now().seconds(); }

  void onLpos(const px4_msgs::msg::VehicleLocalPosition& m) {
    if (!q_ || !(m.xy_valid && m.z_valid && m.v_xy_valid && m.v_z_valid)) return;
    if (!std::isfinite(m.x) || !std::isfinite(m.y) || !std::isfinite(m.z) || !std::isfinite(m.vx) ||
        !std::isfinite(m.vy) || !std::isfinite(m.vz))
      return;
    OdomSample s;
    s.stamp = now_s();   // receive time on the node clock (sim time in simulation), the same clock as depth stamps
    s.T_world_body = gtsam::Pose3(*q_, gtsam::Point3(m.x, m.y, m.z));
    s.v_world = Eigen::Vector3d(m.vx, m.vy, m.vz);
    odom_.push_back(s);
    while (!odom_.empty() && odom_.front().stamp < s.stamp - 2.0) odom_.pop_front();
    if (!est_ || !est_->initialised()) return;
    const MppiInput in = est_->predict(s);
    publishOdometry(*pub_odom_, in);
    if (vio_icp_ && vio_icp_->initialised() && pub_vio_icp_odom_->get_subscription_count() > 0) {
      publishOdometry(*pub_vio_icp_odom_, vio_icp_->predict(s));
    }
    if (sink_) sink_->onEstimate(in);
  }

  // PX4 estimate at time t: linear / slerp between the bracketing samples.
  std::optional<OdomSample> odomAt(double t) const {
    if (odom_.empty()) return std::nullopt;
    if (t >= odom_.back().stamp) {
      if (t - odom_.back().stamp > max_odom_gap_) return std::nullopt;
      OdomSample s = odom_.back();
      s.stamp = t;
      return s;
    }
    for (size_t i = odom_.size() - 1; i > 0; --i) {
      const OdomSample &a = odom_[i - 1], &b = odom_[i];
      if (a.stamp <= t && t <= b.stamp) {
        if (b.stamp - a.stamp > max_odom_gap_) return std::nullopt;
        const double w = b.stamp > a.stamp ? (t - a.stamp) / (b.stamp - a.stamp) : 0.0;
        OdomSample s;
        s.stamp = t;
        s.T_world_body = gtsam::Pose3(a.T_world_body.rotation().slerp(w, b.T_world_body.rotation()),
                                      (1 - w) * a.T_world_body.translation() + w * b.T_world_body.translation());
        s.v_world = (1 - w) * a.v_world + w * b.v_world;
        return s;
      }
    }
    return std::nullopt;   // older than the buffer
  }

  void onInfo(const sensor_msgs::msg::CameraInfo& m) {
    if (cam_ || m.k[0] <= 0) return;
    PinholeCamera c;
    c.fx = m.k[0]; c.fy = m.k[4]; c.cx = m.k[2]; c.cy = m.k[5];
    c.width = static_cast<int>(m.width); c.height = static_cast<int>(m.height);
    cam_ = c;
  }

  static std::optional<DepthFrame> decode(const sensor_msgs::msg::Image& m) {
    DepthFrame f;
    f.width = static_cast<int>(m.width);
    f.height = static_cast<int>(m.height);
    f.depth_m.resize(static_cast<size_t>(f.width) * f.height);
    const bool mm = m.encoding == "16UC1" || m.encoding == "mono16";
    if (!mm && m.encoding != "32FC1") return std::nullopt;
    if (m.is_bigendian) return std::nullopt;
    for (int v = 0; v < f.height; ++v) {
      const uint8_t* row = m.data.data() + static_cast<size_t>(v) * m.step;
      for (int u = 0; u < f.width; ++u) {
        float z;
        if (mm) {
          uint16_t d;
          std::memcpy(&d, row + 2 * u, 2);
          z = d == 0 ? std::numeric_limits<float>::quiet_NaN() : d * 1e-3f;   // 0 = no stereo match
        } else {
          std::memcpy(&z, row + 4 * u, 4);
          if (!(z > 0) && !std::isinf(z)) z = std::numeric_limits<float>::quiet_NaN();   // keep +inf = far
        }
        f.depth_m[static_cast<size_t>(v) * f.width + u] = z;
      }
    }
    return f;
  }

  void onDepth(const sensor_msgs::msg::Image& m) {
    if (!cam_) return;
    if (!est_) {
      geometry_msgs::msg::TransformStamped tf;
      try {
        tf = tf_buffer_->lookupTransform(base_frame_, m.header.frame_id, tf2::TimePointZero);
      } catch (const std::exception& e) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "no TF %s <- %s: %s", base_frame_.c_str(),
                             m.header.frame_id.c_str(), e.what());
        return;
      }
      const auto& r = tf.transform.rotation;
      const auto& t = tf.transform.translation;
      // TF is FLU; PX4 body is FRD: flip y and z (as policynav/obs.py DepthReprojector)
      const gtsam::Matrix3 F = gtsam::Vector3(1, -1, -1).asDiagonal();
      const gtsam::Rot3 R_flu = gtsam::Rot3::Quaternion(r.w, r.x, r.y, r.z);
      const gtsam::Pose3 T_bc(gtsam::Rot3(F * R_flu.matrix()), gtsam::Point3(F * gtsam::Vector3(t.x, t.y, t.z)));
      est_ = std::make_unique<DepthIsam2>(p_, *cam_, T_bc);
      if (cad_prior_) est_->setCadPrior(cad_prior_);
      vio_icp_ = std::make_unique<VioIcpTracker>(p_, *cam_, T_bc);
      const gtsam::Vector3 axis = T_bc.rotation().matrix().col(2);
      RCLCPP_INFO(get_logger(), "camera %dx%d fx %.1f; optical axis in body FRD (%.2f, %.2f, %.2f), t (%.3f, %.3f, %.3f)",
                  cam_->width, cam_->height, cam_->fx, axis.x(), axis.y(), axis.z(), T_bc.translation().x(),
                  T_bc.translation().y(), T_bc.translation().z());
    }
    if (static_cast<int>(m.width) != cam_->width || static_cast<int>(m.height) != cam_->height) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000, "depth %ux%u does not match CameraInfo %dx%d", m.width,
                            m.height, cam_->width, cam_->height);
      return;
    }
    auto frame = decode(m);
    if (!frame) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000, "unsupported depth encoding %s", m.encoding.c_str());
      return;
    }
    frame->stamp = rclcpp::Time(m.header.stamp).seconds();
    const auto odom = odomAt(frame->stamp);
    if (!odom) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "no PX4 estimate at the depth stamp (%zu samples)",
                           odom_.size());
      return;
    }
    if (vio_icp_) vio_icp_->addFrame(*frame, *odom);
    if (!est_->addFrame(*frame, *odom)) return;
    if (publish_obstacles_ && pub_obst_->get_subscription_count() > 0) publishObstacles(m.header.stamp);
    publishCloud(*pub_priors_, est_->priors(), m.header.stamp);
    publishCloud(*pub_new_obst_, est_->newUpdatedObstacles(), m.header.stamp);
    publishCloud(*pub_cleared_, est_->clearedEmptySpace(), m.header.stamp);
    if (pub_kf_->get_subscription_count() > 0) publishKeyframes(m.header.stamp);
    if (pub_lm_->get_subscription_count() > 0) publishLandmarks(m.header.stamp);
    if (pub_prior_updates_->get_subscription_count() > 0) publishPriorUpdates(m.header.stamp);
    const double t = rclcpp::Time(m.header.stamp).seconds();
    const bool want_map = pub_map_->get_subscription_count() > 0;
    const bool want_adapted = pub_adapted_->get_subscription_count() > 0;
    if ((want_map || want_adapted) && (t - last_map_ >= map_period_ || t < last_map_)) {
      last_map_ = t;
      if (want_adapted) {
        const auto cm = est_->globalMapClassified(p_.map_voxel, p_.grid.min_hits, p_.cad_prior.max_corr_dist);
        publishClassifiedMap(*pub_adapted_, cm, m.header.stamp);
        if (want_map) {
          std::vector<Eigen::Vector3f> all = cm.cad_matched;
          all.insert(all.end(), cm.adapted_new.begin(), cm.adapted_new.end());
          publishCloud(*pub_map_, all, m.header.stamp);
        }
      } else {
        publishCloud(*pub_map_, est_->globalMap(p_.map_voxel, p_.grid.min_hits), m.header.stamp);
      }
    }
  }

  void publishOdometry(rclcpp::Publisher<nav_msgs::msg::Odometry>& pub, const MppiInput& in) {
    nav_msgs::msg::Odometry o;
    o.header.stamp = rclcpp::Time(static_cast<int64_t>(in.stamp * 1e9), get_clock()->get_clock_type());
    o.header.frame_id = world_frame_;
    o.child_frame_id = body_frame_;
    setPose(o.pose.pose, in.T_world_body);
    const gtsam::Vector3 vb = in.T_world_body.rotation().unrotate(gtsam::Point3(in.v_world));
    o.twist.twist.linear.x = vb.x(); o.twist.twist.linear.y = vb.y(); o.twist.twist.linear.z = vb.z();
    // GTSAM orders the pose tangent [rot, trans]; ROS wants [trans, rot] (both in the body frame here)
    for (int i = 0; i < 6; ++i)
      for (int j = 0; j < 6; ++j) o.pose.covariance[i * 6 + j] = in.pose_cov((i + 3) % 6, (j + 3) % 6);
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) o.twist.covariance[i * 6 + j] = in.vel_cov(i, j);
    pub.publish(o);
  }

  void publishObstacles(const builtin_interfaces::msg::Time& stamp) {
    const auto& g = est_->latest().distance;
    if (g) publishCloud(*pub_obst_, g->occupiedCentres(), stamp);
  }

  void publishCloud(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>& pub, const std::vector<Eigen::Vector3f>& pts,
                    const builtin_interfaces::msg::Time& stamp) {
    sensor_msgs::msg::PointCloud2 c;
    c.header.stamp = stamp;
    c.header.frame_id = world_frame_;
    sensor_msgs::PointCloud2Modifier mod(c);
    mod.setPointCloud2FieldsByString(1, "xyz");
    mod.resize(pts.size());
    sensor_msgs::PointCloud2Iterator<float> x(c, "x"), y(c, "y"), z(c, "z");
    for (const auto& q : pts) {
      *x = q.x(); *y = q.y(); *z = q.z();
      ++x; ++y; ++z;
    }
    pub.publish(c);
  }

  builtin_interfaces::msg::Time toStamp(double t) const {
    return rclcpp::Time(static_cast<int64_t>(t * 1e9), get_clock()->get_clock_type());
  }

  static void setPose(geometry_msgs::msg::Pose& out, const gtsam::Pose3& T) {
    const gtsam::Point3 p = T.translation();
    const gtsam::Quaternion q = T.rotation().toQuaternion();
    out.position.x = p.x(); out.position.y = p.y(); out.position.z = p.z();
    out.orientation.w = q.w(); out.orientation.x = q.x(); out.orientation.y = q.y(); out.orientation.z = q.z();
  }

  void publishKeyframes(const builtin_interfaces::msg::Time& stamp) {
    nav_msgs::msg::Path path;
    path.header.stamp = stamp;
    path.header.frame_id = world_frame_;
    for (const auto& kf : est_->keyframePoses()) {
      geometry_msgs::msg::PoseStamped ps;
      ps.header.stamp = toStamp(kf.stamp);
      ps.header.frame_id = world_frame_;
      setPose(ps.pose, kf.T_world_body);
      path.poses.push_back(ps);
    }
    pub_kf_->publish(path);
  }

  void publishClassifiedMap(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>& pub,
                            const DepthIsam2::ClassifiedMap& cm,
                            const builtin_interfaces::msg::Time& stamp) {
    sensor_msgs::msg::PointCloud2 c;
    c.header.stamp = stamp;
    c.header.frame_id = world_frame_;
    sensor_msgs::PointCloud2Modifier mod(c);
    mod.setPointCloud2FieldsByString(2, "xyz", "rgb");
    const size_t total = cm.cad_matched.size() + cm.adapted_new.size();
    mod.resize(total);
    sensor_msgs::PointCloud2Iterator<float> x(c, "x"), y(c, "y"), z(c, "z");
    sensor_msgs::PointCloud2Iterator<uint8_t> r(c, "r"), g(c, "g"), b(c, "b");
    for (const auto& q : cm.cad_matched) {
      *x = q.x(); *y = q.y(); *z = q.z();
      *r = 35; *g = 195; *b = 185;   // teal-green: agrees with CAD prior ESDF
      ++x; ++y; ++z; ++r; ++g; ++b;
    }
    for (const auto& q : cm.adapted_new) {
      *x = q.x(); *y = q.y(); *z = q.z();
      *r = 255; *g = 140; *b = 25;   // warm amber-orange: newly discovered / shifted structure beyond CAD prior
      ++x; ++y; ++z; ++r; ++g; ++b;
    }
    pub.publish(c);
  }

  void publishPriorUpdates(const builtin_interfaces::msg::Time& stamp) {
    const auto gv = est_->graphVizData();
    visualization_msgs::msg::MarkerArray arr;
    auto makeLine = [&](int id, const std::string& ns, float width, float cr, float cg, float cb, float ca) {
      visualization_msgs::msg::Marker m;
      m.header.stamp = stamp;
      m.header.frame_id = world_frame_;
      m.ns = ns;
      m.id = id;
      m.type = visualization_msgs::msg::Marker::LINE_LIST;
      m.action = visualization_msgs::msg::Marker::ADD;
      m.pose.orientation.w = 1.0;
      m.scale.x = width;
      m.color.r = cr; m.color.g = cg; m.color.b = cb; m.color.a = ca;
      return m;
    };
    auto toPt = [](const gtsam::Point3& p) {
      geometry_msgs::msg::Point g;
      g.x = p.x(); g.y = p.y(); g.z = p.z();
      return g;
    };

    // 1. Keyframe smoothed-pose trajectory strip + initial-to-smoothed correction springs
    visualization_msgs::msg::Marker kf_strip;
    kf_strip.header.stamp = stamp;
    kf_strip.header.frame_id = world_frame_;
    kf_strip.ns = "smoothed_keyframes";
    kf_strip.id = 0;
    kf_strip.type = visualization_msgs::msg::Marker::LINE_STRIP;
    kf_strip.action = visualization_msgs::msg::Marker::ADD;
    kf_strip.pose.orientation.w = 1.0;
    kf_strip.scale.x = 0.012f;
    kf_strip.color.r = 0.05f; kf_strip.color.g = 0.68f; kf_strip.color.b = 0.96f; kf_strip.color.a = 0.90f;

    auto shifts = makeLine(1, "smoothing_shifts", 0.008f, 0.98f, 0.72f, 0.12f, 0.85f);
    for (size_t i = 0; i < gv.smoothed_poses.size(); ++i) {
      const auto ps = gv.smoothed_poses[i].translation();
      const auto pi = gv.initial_poses[i].translation();
      kf_strip.points.push_back(toPt(ps));
      if ((ps - pi).norm() > 0.004) {
        shifts.points.push_back(toPt(pi));
        shifts.points.push_back(toPt(ps));
      }
    }
    arr.markers.push_back(std::move(kf_strip));
    arr.markers.push_back(std::move(shifts));

    // 2. Accepted CAD-prior Pose3 anchors -> current smoothed keyframe pose X(k)
    auto cad_lines = makeLine(2, "cad_prior_anchors", 0.014f, 0.10f, 0.85f, 0.42f, 0.92f);
    visualization_msgs::msg::Marker cad_pts;
    cad_pts.header.stamp = stamp;
    cad_pts.header.frame_id = world_frame_;
    cad_pts.ns = "cad_anchor_points";
    cad_pts.id = 3;
    cad_pts.type = visualization_msgs::msg::Marker::POINTS;
    cad_pts.action = visualization_msgs::msg::Marker::ADD;
    cad_pts.pose.orientation.w = 1.0;
    cad_pts.scale.x = cad_pts.scale.y = 0.045f;
    cad_pts.color.r = 0.10f; cad_pts.color.g = 0.88f; cad_pts.color.b = 0.45f; cad_pts.color.a = 0.95f;
    for (const auto& a : gv.cad_priors) {
      if (a.keyframe >= gv.smoothed_poses.size()) continue;
      const auto pa = a.T_prior.translation();
      const auto ps = gv.smoothed_poses[a.keyframe].translation();
      cad_pts.points.push_back(toPt(pa));
      cad_lines.points.push_back(toPt(pa));
      cad_lines.points.push_back(toPt(ps));
    }
    arr.markers.push_back(std::move(cad_lines));
    arr.markers.push_back(std::move(cad_pts));

    // 3. Revisit loop-closure ICP edges X(j) <-> X(k)
    auto loops = makeLine(4, "revisit_closures", 0.009f, 0.12f, 0.78f, 0.92f, 0.70f);
    for (const auto& [j, k] : gv.revisit_links) {
      if (j < gv.smoothed_poses.size() && k < gv.smoothed_poses.size()) {
        loops.points.push_back(toPt(gv.smoothed_poses[j].translation()));
        loops.points.push_back(toPt(gv.smoothed_poses[k].translation()));
      }
    }
    arr.markers.push_back(std::move(loops));
    pub_prior_updates_->publish(arr);
  }

  void publishLandmarks(const builtin_interfaces::msg::Time& stamp) {
    visualization_msgs::msg::MarkerArray arr;
    visualization_msgs::msg::Marker clear;
    clear.header.frame_id = world_frame_;
    clear.action = visualization_msgs::msg::Marker::DELETEALL;   // ids can disappear if the graph is reset
    arr.markers.push_back(clear);
    for (const auto& [id, p] : est_->latest().landmarks) {
      visualization_msgs::msg::Marker m;
      m.header.stamp = stamp;
      m.header.frame_id = world_frame_;
      m.ns = "landmark";
      m.id = id;
      m.type = visualization_msgs::msg::Marker::SPHERE;
      m.action = visualization_msgs::msg::Marker::ADD;
      m.pose.position.x = p.x(); m.pose.position.y = p.y(); m.pose.position.z = p.z();
      m.pose.orientation.w = 1.0;
      m.scale.x = m.scale.y = m.scale.z = 0.15;
      m.color.r = 1.0f; m.color.g = 0.0f; m.color.b = 1.0f; m.color.a = 0.9f;
      arr.markers.push_back(m);
      visualization_msgs::msg::Marker label = m;
      label.ns = "landmark_id";
      label.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
      label.text = "L" + std::to_string(id);
      label.pose.position.z -= 0.2;   // NED: 0.2 m above the sphere
      label.scale.x = label.scale.y = 0.0;
      label.scale.z = 0.12;
      label.color.r = label.color.g = label.color.b = 1.0f;
      arr.markers.push_back(label);
    }
    pub_lm_->publish(arr);
  }

  // Writes <base>.ply + <base>_keyframes.csv; base "" -> ./isam2_map_<local time>. Returns false if nothing to save.
  bool saveMap(std::string base, std::string* msg = nullptr) {
    auto fail = [&](const std::string& why) {
      if (msg) *msg = why;
      RCLCPP_WARN(get_logger(), "map not saved: %s", why.c_str());
      return false;
    };
    if (!est_ || !est_->initialised()) return fail("no keyframes yet");
    if (base.empty()) {
      char buf[32];
      const std::time_t now = std::time(nullptr);
      std::strftime(buf, sizeof buf, "%Y%m%d_%H%M%S", std::localtime(&now));
      base = std::string("isam2_map_") + buf;
    }
    if (base.size() > 4 && base.compare(base.size() - 4, 4, ".ply") == 0) base.resize(base.size() - 4);
    const auto pts = est_->globalMap(p_.map_voxel, p_.grid.min_hits);
    std::ofstream ply(base + ".ply", std::ios::binary);
    if (!ply) return fail("cannot write " + base + ".ply");
    ply << "ply\nformat binary_little_endian 1.0\n"
        << "comment isam2_depth corrected map: voxel " << p_.map_voxel << " m, min_hits " << p_.grid.min_hits << "\n"
        << "comment frame ENU z up: x = east (NED y), y = north (NED x), z = up (-NED z), origin = PX4 local origin\n"
        << "element vertex " << pts.size() << "\nproperty float x\nproperty float y\nproperty float z\nend_header\n";
    for (const auto& q : pts) {
      const float v[3] = {q.y(), q.x(), -q.z()};
      ply.write(reinterpret_cast<const char*>(v), sizeof v);
    }
    std::ofstream csv(base + "_keyframes.csv");
    csv << "stamp,x_ned,y_ned,z_ned,qw,qx,qy,qz\n";
    for (const auto& kf : est_->keyframePoses()) {
      const auto p = kf.T_world_body.translation();
      const auto q = kf.T_world_body.rotation().toQuaternion();
      csv << std::fixed << kf.stamp << ',' << p.x() << ',' << p.y() << ',' << p.z() << ',' << q.w() << ','
          << q.x() << ',' << q.y() << ',' << q.z() << '\n';
    }
    const std::string ok = "saved " + std::to_string(pts.size()) + " voxels to " + base + ".ply and " +
                           std::to_string(est_->keyframePoses().size()) + " keyframes to " + base + "_keyframes.csv";
    RCLCPP_INFO(get_logger(), "%s", ok.c_str());
    if (msg) *msg = ok;
    return true;
  }

 public:
  void saveOnExit() {
    if (!map_save_path_.empty()) saveMap(map_save_path_);
  }

 private:
  void logStats() {
    if (!est_) return;
    const auto& s = est_->stats();
    RCLCPP_INFO(get_logger(), "%zu keyframes, %zu factors, ICP ok %d / rej %d, CAD ok %d / rej %d, %d revisit links, "
                "last update %.1f ms (icp %.1f, cad %.1f, isam2 %.1f, grid %.1f)",
                s.keyframes, s.factors, s.icp_ok, s.icp_rejected, s.cad_ok, s.cad_rejected, s.links,
                s.last_update_ms, s.icp_ms, s.cad_ms, s.isam_ms, s.output_ms);
  }

  Isam2Params p_;
  std::string depth_topic_, info_topic_, base_frame_, world_frame_, body_frame_;
  bool publish_obstacles_ = true;
  double max_odom_gap_ = 0.1;
  double map_period_ = 1.0, last_map_ = -1e18;
  std::string map_save_path_, cad_prior_path_;
  std::shared_ptr<const CadPriorEsdf> cad_prior_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_save_;
  std::optional<PinholeCamera> cam_;
  std::optional<gtsam::Rot3> q_;
  std::deque<OdomSample> odom_;
  std::unique_ptr<DepthIsam2> est_;
  std::unique_ptr<VioIcpTracker> vio_icp_;
  std::shared_ptr<MppiStateSink> sink_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp::Subscription<px4_msgs::msg::VehicleLocalPosition>::SharedPtr sub_lpos_;
  rclcpp::Subscription<px4_msgs::msg::VehicleAttitude>::SharedPtr sub_att_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr sub_info_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_depth_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom_, pub_vio_icp_odom_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_obst_, pub_map_, pub_adapted_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_priors_, pub_new_obst_, pub_cleared_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_kf_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_lm_, pub_prior_updates_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace isam2_depth

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<isam2_depth::Isam2DepthNode>();
  // MPPI: node->setSink(std::make_shared<YourMppi>(...));
  rclcpp::spin(node);
  node->saveOnExit();   // after Ctrl-C: the estimator is plain C++, still valid here
  rclcpp::shutdown();
  return 0;
}
