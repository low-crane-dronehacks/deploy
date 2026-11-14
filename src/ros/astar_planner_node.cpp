// astar_planner_node: grid A* global planner (src/astar_planner.cpp, ported from src/test/spoof_px4_depth.py).
//
// Same contract as rrt_star_planner_node (docs/AUTONOMY_ARCHITECTURE.md section 7), so either can run:
// in   /odometry/local                   nav_msgs/Odometry (odom frame): current pose + speed for replans
//      /plan/replan_blocked_corridor     geometry_msgs/PointStamped (map frame): block it, replan
//      TF map -> odom
// out  /plan/global_path                  nav_msgs/Path, map frame, latched, timed pose stamps, one per new plan
//      /viz/groundtruth_map_cloud        sensor_msgs/PointCloud2 (map frame, latched)
//
// Goals (launch file, from assets/arena_prior.json): `goals` = 9 doubles per goal
//   [cx, cy, cz, ax, ay, bx, by, nx, ny]   gate line A-B crossed along n; A = B = 0 for a plain goal (the pad)
// and `goal_names`.
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <geometry_msgs/msg/point_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include "maze_planning/astar_planner.hpp"
#include "maze_planning/course_progress.hpp"
#include "maze_planning/groundtruth_map.hpp"
#include "maze_planning/ros_conversions.hpp"

namespace maze_planning {

class AStarPlannerNode : public rclcpp::Node {
 public:
  AStarPlannerNode() : Node("astar_planner_node"), tf_buffer_(get_clock()), tf_listener_(tf_buffer_) {
    const std::string gt_path = declare_parameter<std::string>("groundtruth_map_path", "assets/course_groundtruth.bin");
    map_frame_ = declare_parameter<std::string>("map_frame", "map_ned");
    odom_frame_ = declare_parameter<std::string>("odom_frame", "odom_ned");
    replan_min_interval_s_ = declare_parameter<double>("replan_min_interval_s", 1.5);
    block_radius_m_ = declare_parameter<double>("block_radius_m", 0.20);
    const std::string isam2_priors_topic = declare_parameter<std::string>("isam2_priors_topic", "");
    const std::string isam2_new_obst_topic = declare_parameter<std::string>("isam2_new_obstacles_topic", "");
    isam2_voxel_res_m_ = declare_parameter<double>("isam2_voxel_resolution_m", 0.08);

    AStarConfig cfg{};
    cfg.resolution_m = declare_parameter<double>("resolution_m", cfg.resolution_m);
    cfg.arena_half_m = declare_parameter<double>("arena_half_m", cfg.arena_half_m);
    cfg.flight_z_ned = declare_parameter<double>("flight_z_ned", cfg.flight_z_ned);
    cfg.clearance_m = declare_parameter<double>("clearance_m", cfg.clearance_m);
    cfg.clearance_fallback_ratio = declare_parameter<double>("clearance_fallback_ratio", cfg.clearance_fallback_ratio);
    cfg.centre_dist_m = declare_parameter<double>("centre_dist_m", cfg.centre_dist_m);
    cfg.centre_weight = declare_parameter<double>("centre_weight", cfg.centre_weight);
    cfg.snap_radius_m = declare_parameter<double>("snap_radius_m", cfg.snap_radius_m);
    if (cfg.clearance_m <= 0.0 || cfg.clearance_fallback_ratio <= 0.0 || cfg.clearance_fallback_ratio > 1.0) {
      throw std::runtime_error("astar_planner_node: need clearance_m > 0 and 0 < clearance_fallback_ratio <= 1");
    }
    RCLCPP_INFO(get_logger(), "clearance %.3f m (%.3f m near goals/gates and as a last resort), centre %.2f m x %.2f, "
                "snap %.2f m", cfg.clearance_m, cfg.clearance_m * cfg.clearance_fallback_ratio, cfg.centre_dist_m,
                cfg.centre_weight, cfg.snap_radius_m);
    cfg.cruise_speed_m_s = declare_parameter<double>("cruise_speed_m_s", cfg.cruise_speed_m_s);
    cfg.lat_acc_m_s2 = declare_parameter<double>("lat_acc_m_s2", cfg.lat_acc_m_s2);
    cfg.yaw_rate_max_rad_s = declare_parameter<double>("yaw_rate_max_rad_s", cfg.yaw_rate_max_rad_s);
    cfg.acc_m_s2 = declare_parameter<double>("acc_m_s2", cfg.acc_m_s2);
    cfg.cp_margin_m = declare_parameter<double>("cp_margin_m", cfg.cp_margin_m);

    const std::vector<double> flat = declare_parameter<std::vector<double>>("goals", std::vector<double>{});
    const std::vector<std::string> names =
        declare_parameter<std::vector<std::string>>("goal_names", std::vector<std::string>{});
    if (flat.empty() || flat.size() % 9 != 0) {
      throw std::runtime_error("astar_planner_node: parameter 'goals' needs 9 doubles per goal (launch file sets it)");
    }
    for (std::size_t i = 0; i < flat.size() / 9; ++i) {
      const double* g = &flat[9 * i];
      PlanGoal q;
      q.name = i < names.size() ? names[i] : ("goal" + std::to_string(i));
      q.center = Eigen::Vector3d(g[0], g[1], g[2]);
      q.seg_a = Eigen::Vector2d(g[3], g[4]);
      q.seg_b = Eigen::Vector2d(g[5], g[6]);
      q.normal = Eigen::Vector2d(g[7], g[8]);
      q.has_segment = (q.seg_b - q.seg_a).norm() > 1e-6 && q.normal.norm() > 1e-6;
      goals_.push_back(q);
    }

    gt_map_ = std::make_shared<GroundTruthMap>();
    if (!gt_map_->loadFromBinary(gt_path)) {
      RCLCPP_WARN(get_logger(), "no ground-truth map at %s; using the built-in default course", gt_path.c_str());
      gt_map_->initDefaultCourse();
    }
    planner_ = std::make_unique<AStarPlanner>(cfg, gt_map_);
    // The elevated-opening altitude search is for the window: only near its gate line, elsewhere stay at flight_z.
    std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> window_lines;
    for (const auto& g : goals_) {
      if (g.has_segment && g.name == "window") window_lines.emplace_back(g.seg_a, g.seg_b);
    }
    planner_->setAltitudeSearchZones(std::move(window_lines), 0.6);

    const rclcpp::QoS latched = rclcpp::QoS(1).reliable().transient_local();
    pub_path_ = create_publisher<nav_msgs::msg::Path>("/plan/global_path", latched);
    pub_map_cloud_ = create_publisher<sensor_msgs::msg::PointCloud2>("/viz/groundtruth_map_cloud", latched);
    publishGroundTruthCloud();

    current_pos_map_ = Eigen::Vector3d(gt_map_->launchMarkNed().x(), gt_map_->launchMarkNed().y(), cfg.flight_z_ned);
    planAndPublish("initial plan from the launch mark", 0.0);

    sub_odom_ = create_subscription<nav_msgs::msg::Odometry>(
        "/odometry/local", rclcpp::SensorDataQoS(),
        [this](const nav_msgs::msg::Odometry::SharedPtr m) { onOdometry(*m); });
    sub_replan_ = create_subscription<geometry_msgs::msg::PointStamped>(
        "/plan/replan_blocked_corridor", 10,
        [this](const geometry_msgs::msg::PointStamped::SharedPtr m) { onBlocked(*m); });
    if (!isam2_priors_topic.empty()) {
      sub_isam2_priors_ = create_subscription<sensor_msgs::msg::PointCloud2>(
          isam2_priors_topic, latched,
          [this](const sensor_msgs::msg::PointCloud2::SharedPtr m) { onIsam2Cloud(*m, isam2_priors_map_); });
    }
    if (!isam2_new_obst_topic.empty()) {
      sub_isam2_new_obst_ = create_subscription<sensor_msgs::msg::PointCloud2>(
          isam2_new_obst_topic, latched,
          [this](const sensor_msgs::msg::PointCloud2::SharedPtr m) {
            onIsam2Cloud(*m, isam2_new_obst_map_);
            has_isam2_updates_ = true;
          });
    }
  }

 private:
  bool planAndPublish(const char* why, double start_speed) {
    const std::size_t first = progress_.nextAnchor();
    if (first >= goals_.size()) return false;
    const std::vector<PlanGoal> rest(goals_.begin() + static_cast<std::ptrdiff_t>(first), goals_.end());
    AStarPlanResult r = planner_->plan(current_pos_map_, rest, start_speed);
    if (!r.ok) {
      RCLCPP_ERROR(get_logger(), "%s: A* found no path (from goal #%zu); keeping the previous path", why, first);
      return false;
    }
    for (std::size_t i : r.skipped) RCLCPP_ERROR(get_logger(), "%s: unreachable, skipped: %s", why, rest[i].name.c_str());
    for (std::size_t i : r.not_crossed) {
      RCLCPP_WARN(get_logger(), "%s: path does not clear %s by the scoring rule", why, rest[i].name.c_str());
    }
    progress_.reset(r.path, r.goal_s_m, first);
    pub_path_->publish(ros_conv::toRosPath(r.path, map_frame_, now()));
    RCLCPP_INFO(get_logger(), "%s: %zu goals from #%zu, %.2f m, %.2f s, min clearance %.3f m", why, rest.size(), first,
                r.path.total_length_m, r.path.total_duration_s, r.min_clearance_m);
    return true;
  }

  void publishGroundTruthCloud() {
    std::vector<Eigen::Vector3f> pts;
    for (double x = -2.95; x <= 2.95; x += 0.06)
      for (double y = -2.95; y <= 2.95; y += 0.06)
        for (double z = -1.70; z <= -0.05; z += 0.08)
          if (gt_map_->isStaticWallOccupied(Eigen::Vector3d(x, y, z))) pts.emplace_back(x, y, z);
    sensor_msgs::msg::PointCloud2 cloud;
    cloud.header.frame_id = map_frame_;
    cloud.header.stamp = now();
    cloud.height = 1;
    sensor_msgs::PointCloud2Modifier mod(cloud);
    mod.setPointCloud2FieldsByString(1, "xyz");
    mod.resize(pts.size());
    sensor_msgs::PointCloud2Iterator<float> ix(cloud, "x"), iy(cloud, "y"), iz(cloud, "z");
    for (const auto& p : pts) {
      *ix = p.x(); *iy = p.y(); *iz = p.z();
      ++ix; ++iy; ++iz;
    }
    pub_map_cloud_->publish(cloud);
  }

  void onOdometry(const nav_msgs::msg::Odometry& m) {
    const VehicleKinematics k = ros_conv::fromOdometry(m);
    SE3Transform T_map_odom{};
    if (!ros_conv::lookupSE3(tf_buffer_, map_frame_, odom_frame_, T_map_odom)) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "no TF %s <- %s; treating them as identical",
                           map_frame_.c_str(), odom_frame_.c_str());
    }
    current_pos_map_ = T_map_odom.apply(k.pos_odom);
    has_odom_ = true;
    speed_ = k.vel_odom.head<2>().norm();
    progress_.update(current_pos_map_);
  }

  void onIsam2Cloud(const sensor_msgs::msg::PointCloud2& m, std::vector<Eigen::Vector3f>& dst) {
    SE3Transform T_map_cloud{};
    if (m.header.frame_id != map_frame_ &&
        !ros_conv::lookupSE3(tf_buffer_, map_frame_, m.header.frame_id, T_map_cloud)) {
      T_map_cloud = SE3Transform{};
    }
    dst.clear();
    dst.reserve(static_cast<std::size_t>(m.width) * m.height);
    sensor_msgs::PointCloud2ConstIterator<float> ix(m, "x"), iy(m, "y"), iz(m, "z");
    for (; ix != ix.end(); ++ix, ++iy, ++iz) {
      if (!std::isfinite(*ix) || !std::isfinite(*iy) || !std::isfinite(*iz)) continue;
      dst.push_back(T_map_cloud.apply(Eigen::Vector3d(*ix, *iy, *iz)).cast<float>());
    }
  }

  void onBlocked(const geometry_msgs::msg::PointStamped& m) {
    if (!has_odom_) return;
    const double now_s = now().seconds();
    if (now_s - last_replan_s_ < replan_min_interval_s_) return;
    last_replan_s_ = now_s;
    if (has_isam2_updates_ && (!isam2_priors_map_.empty() || !isam2_new_obst_map_.empty())) {
      // iSAM2's PriorEsdfMap already contains the exact physical obstacle voxels; only add a small local nudge
      // (<= 0.08 m) if needed rather than a huge blind 0.38 m sphere that seals adjacent narrow corridors!
      gt_map_->markCorridorBlocked(Eigen::Vector3d(m.point.x, m.point.y, m.point.z),
                                   std::min(block_radius_m_, 0.08));
      std::vector<Eigen::Vector3f> active = isam2_priors_map_;
      active.insert(active.end(), isam2_new_obst_map_.begin(), isam2_new_obst_map_.end());
      isam2_depth::GridParams gp{};
      gp.resolution = isam2_voxel_res_m_;
      gp.half_extent = Eigen::Vector3d(3.2, 3.2, 1.5);
      gp.min_hits = 1;
      gp.max_distance = 3.0f;
      const isam2_depth::DistanceGrid adapted_grid(
          active, Eigen::Vector3d(0.0, 0.0, -0.95), gp);
      planner_->rebuildClearance(&adapted_grid, 0.0);
    } else {
      gt_map_->markCorridorBlocked(Eigen::Vector3d(m.point.x, m.point.y, m.point.z), block_radius_m_);
      planner_->rebuildClearance();
    }
    planAndPublish("replan around blocked corridor", speed_);
  }

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  std::string map_frame_, odom_frame_;
  double replan_min_interval_s_{1.5};
  double block_radius_m_{0.20};
  double isam2_voxel_res_m_{0.08};
  std::vector<Eigen::Vector3f> isam2_priors_map_;
  std::vector<Eigen::Vector3f> isam2_new_obst_map_;
  bool has_isam2_updates_{false};
  bool has_odom_{false};
  std::vector<PlanGoal> goals_;
  std::shared_ptr<GroundTruthMap> gt_map_;
  std::unique_ptr<AStarPlanner> planner_;
  CourseProgress progress_;
  Eigen::Vector3d current_pos_map_{Eigen::Vector3d::Zero()};
  double speed_{0.0};
  double last_replan_s_{-1e9};

  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_map_cloud_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr sub_replan_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_isam2_priors_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_isam2_new_obst_;
};

}  // namespace maze_planning

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<maze_planning::AStarPlannerNode>());
  rclcpp::shutdown();
  return 0;
}
