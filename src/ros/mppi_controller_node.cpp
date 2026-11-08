// mppi_controller_node: 2048-rollout MPPI controller + STVL voxel costmap (Section 2.2, 2.3, WP-3 Tasks 3-4).
//
// in   /plan/local_reference          nav_msgs/Path (odom frame, timed pose stamps) from path_transformer_node
//      /odometry/local                nav_msgs/Odometry (odom frame)
//      obstacle_cloud_topic           sensor_msgs/PointCloud2, any frame with a TF to map (default
//                                     /oak/points/clamped): raw depth hits, ray-cleared from the sensor origin
//      isam2_obstacles_topic          sensor_msgs/PointCloud2 (optional, default /isam2_depth_node/obstacles):
//                                     occupied voxel centres of the estimator's distance field, any frame with a
//                                     TF to odom. Rebuilt here into a local DistanceGrid that MPPI samples.
//      TF map -> odom                 (identity with a warning if missing)
// out  /control/trajectory_setpoint   px4_msgs/TrajectorySetpoint (to control_mux_node), when px4_msgs exists
//      /control/cmd_vel               geometry_msgs/TwistStamped (NED velocity + yaw rate; debug / sim_loopback)
//      /costmap/local_voxels          nav_msgs/OccupancyGrid (map frame) slice at the vehicle height
//      /viz/mppi_candidates           visualization_msgs/MarkerArray (odom frame): sampled rollouts coloured
//                                     by cost (green = cheap, red = expensive) + the optimal rollout (blue)
//      /plan/replan_blocked_corridor  geometry_msgs/PointStamped (map frame) on FMEA Pillar Dead-Center
//
// This node only controls. It does not simulate the vehicle or broadcast TF: sim_loopback_node does that for
// the standalone demo, and the PX4 adapter publishes odometry + TF on SITL / the drone.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#ifdef HAVE_PX4_MSGS
#include <px4_msgs/msg/trajectory_setpoint.hpp>
#endif
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "isam2_depth/distance_grid.hpp"
#include "maze_planning/groundtruth_map.hpp"
#include "maze_planning/mppi_controller.hpp"
#include "maze_planning/ros_conversions.hpp"
#include "maze_planning/voxel_costmap.hpp"

namespace maze_planning {

namespace {

std::vector<Eigen::Vector3d> readXYZ(const sensor_msgs::msg::PointCloud2& m, const SE3Transform& T) {
  std::vector<Eigen::Vector3d> out;
  out.reserve(static_cast<std::size_t>(m.width) * m.height);
  sensor_msgs::PointCloud2ConstIterator<float> ix(m, "x"), iy(m, "y"), iz(m, "z");
  for (; ix != ix.end(); ++ix, ++iy, ++iz) {
    if (!std::isfinite(*ix) || !std::isfinite(*iy) || !std::isfinite(*iz)) continue;
    out.push_back(T.apply(Eigen::Vector3d(*ix, *iy, *iz)));
  }
  return out;
}

geometry_msgs::msg::Point toPoint(const Eigen::Vector3d& p) {
  geometry_msgs::msg::Point q;
  q.x = p.x();
  q.y = p.y();
  q.z = p.z();
  return q;
}

}  // namespace

class MPPIControllerNode : public rclcpp::Node {
 public:
  MPPIControllerNode()
      : Node("mppi_controller_node"),
        tf_buffer_(get_clock()),
        tf_listener_(tf_buffer_) {
    const std::string gt_path =
        declare_parameter<std::string>("groundtruth_map_path", "config/course_groundtruth.bin");
    const std::string obj_path =
        declare_parameter<std::string>("course_obj_path", "sitl/world/course.obj");
    cruise_speed_m_s_ = declare_parameter<double>("cruise_speed_m_s", 2.60);
    const bool inject_demo_pillar = declare_parameter<bool>("inject_demo_pillar", false);
    map_frame_ = declare_parameter<std::string>("map_frame", "map_ned");
    odom_frame_ = declare_parameter<std::string>("odom_frame", "odom_ned");
    const std::string cloud_topic = declare_parameter<std::string>("obstacle_cloud_topic", "/oak/points/clamped");
    const std::string isam2_topic =
        declare_parameter<std::string>("isam2_obstacles_topic", "/isam2_depth_node/obstacles");
    isam2_grid_timeout_s_ = declare_parameter<double>("isam2_grid_timeout_s", 1.0);
    isam2_grid_params_.resolution = declare_parameter<double>("isam2_grid_resolution", 0.1);
    viz_every_n_ = std::max<int64_t>(1, declare_parameter<int64_t>("viz_candidates_every_n", 5));

    gt_map_ = std::make_shared<GroundTruthMap>();
    if (!gt_map_->loadFromBinary(gt_path)) {
      if (!gt_map_->loadFromObj(obj_path)) {
        gt_map_->initDefaultCourse();
      }
    }

    VoxelCostmapConfig vcfg{};
    vcfg.max_sensor_range_m = declare_parameter<double>("max_sensor_range_m", 3.50);
    costmap_ = std::make_shared<VoxelCostmap>(vcfg, gt_map_);
    if (inject_demo_pillar) {
      costmap_->injectUnmodeledPillar(Eigen::Vector2d(1.75, -2.5146), 0.10, 1.65, 0.0);
    }

    MPPIConfig mcfg{};
    mcfg.num_samples = static_cast<int>(declare_parameter<int64_t>("mppi_num_samples", 2048));
    mcfg.horizon_steps = static_cast<int>(declare_parameter<int64_t>("mppi_horizon_steps", 16));
    mcfg.viz_num_candidates = static_cast<int>(declare_parameter<int64_t>("viz_num_candidates", 64));
    MultirotorPlantParams plant{};
    mppi_ = std::make_shared<MPPIController>(mcfg, plant, costmap_);

    pub_costmap_ = create_publisher<nav_msgs::msg::OccupancyGrid>("/costmap/local_voxels", 10);
    pub_candidates_ = create_publisher<visualization_msgs::msg::MarkerArray>("/viz/mppi_candidates", 10);
    pub_replan_ = create_publisher<geometry_msgs::msg::PointStamped>("/plan/replan_blocked_corridor", 10);
    pub_cmd_vel_ = create_publisher<geometry_msgs::msg::TwistStamped>("/control/cmd_vel", 10);
#ifdef HAVE_PX4_MSGS
    rclcpp::QoS px4_qos(1);
    px4_qos.best_effort();
    px4_qos.transient_local();
    pub_traj_sp_ =
        create_publisher<px4_msgs::msg::TrajectorySetpoint>("/control/trajectory_setpoint", px4_qos);
#endif

    sub_local_ref_ = create_subscription<nav_msgs::msg::Path>(
        "/plan/local_reference", 10,
        [this](const nav_msgs::msg::Path::SharedPtr m) {
          local_ref_odom_ = ros_conv::fromRosPath(*m, odom_frame_, cruise_speed_m_s_);
        });
    sub_odom_ = create_subscription<nav_msgs::msg::Odometry>(
        "/odometry/local", rclcpp::SensorDataQoS(),
        [this](const nav_msgs::msg::Odometry::SharedPtr m) {
          state_odom_ = ros_conv::fromOdometry(*m);
          has_odom_ = true;
        });
    if (!cloud_topic.empty()) {
      sub_points_ = create_subscription<sensor_msgs::msg::PointCloud2>(
          cloud_topic, rclcpp::SensorDataQoS(),
          [this](const sensor_msgs::msg::PointCloud2::SharedPtr m) { onObstacleCloud(*m); });
    }
    if (!isam2_topic.empty()) {
      sub_isam2_ = create_subscription<sensor_msgs::msg::PointCloud2>(
          isam2_topic, 1,
          [this](const sensor_msgs::msg::PointCloud2::SharedPtr m) { onIsam2Obstacles(*m); });
    }

    timer_costmap_ = create_wall_timer(std::chrono::milliseconds(33), [this]() { onCostmapTimer(); });
    timer_control_ = create_wall_timer(std::chrono::milliseconds(20), [this]() { onControlTimer(); });

    RCLCPP_INFO(get_logger(), "mppi_controller_node started (rollouts=%d, horizon=%d, cloud=%s, isam2=%s)",
                mcfg.num_samples, mcfg.horizon_steps, cloud_topic.c_str(), isam2_topic.c_str());
  }

 private:
  void updateMapToOdomTF() {
    if (!ros_conv::lookupSE3(tf_buffer_, map_frame_, odom_frame_, T_map_odom_)) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                           "no TF %s <- %s; treating them as identical", map_frame_.c_str(), odom_frame_.c_str());
      T_map_odom_ = SE3Transform{};
    }
    mppi_->setOdomToMapTransform(T_map_odom_);
  }

  // Raw depth hits: into the map frame through TF of the cloud's own frame (camera optical, body, ...).
  void onObstacleCloud(const sensor_msgs::msg::PointCloud2& m) {
    SE3Transform T_map_sensor{};
    if (!ros_conv::lookupSE3(tf_buffer_, map_frame_, m.header.frame_id, T_map_sensor)) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "no TF %s <- %s; dropping obstacle cloud",
                           map_frame_.c_str(), m.header.frame_id.c_str());
      return;
    }
    costmap_->updateFromClampedPointCloud(now().seconds(), T_map_sensor.t, readXYZ(m, T_map_sensor));
  }

  // Estimator distance field: occupied voxel centres -> DistanceGrid (odom frame) centred on the vehicle.
  void onIsam2Obstacles(const sensor_msgs::msg::PointCloud2& m) {
    if (!has_odom_) return;
    SE3Transform T_odom_cloud{};
    if (!ros_conv::lookupSE3(tf_buffer_, odom_frame_, m.header.frame_id, T_odom_cloud)) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "no TF %s <- %s; dropping iSAM2 obstacles",
                           odom_frame_.c_str(), m.header.frame_id.c_str());
      return;
    }
    const std::vector<Eigen::Vector3d> pts = readXYZ(m, T_odom_cloud);
    std::vector<Eigen::Vector3f> ptsf;
    ptsf.reserve(pts.size());
    for (const auto& p : pts) ptsf.push_back(p.cast<float>());
    isam2_depth::GridParams gp = isam2_grid_params_;
    gp.min_hits = 1;   // the input is already occupied voxel centres
    mppi_->setLocalDistanceGrid(
        std::make_shared<const isam2_depth::DistanceGrid>(ptsf, state_odom_.pos_odom, gp));
    last_isam2_grid_s_ = now().seconds();
    has_isam2_grid_ = true;
  }

  void onCostmapTimer() {
    int nx = 0, ny = 0;
    Eigen::Vector2d origin_xy = Eigen::Vector2d::Zero();
    std::vector<int8_t> grid_data;
    const double z_slice = has_odom_ ? T_map_odom_.apply(state_odom_.pos_odom).z() : -0.78;
    costmap_->exportOccupancyGrid2D(z_slice, 0.08, nx, ny, origin_xy, grid_data);

    nav_msgs::msg::OccupancyGrid msg;
    msg.header.stamp = now();
    msg.header.frame_id = map_frame_;
    msg.info.resolution = 0.08f;
    msg.info.width = static_cast<uint32_t>(nx);
    msg.info.height = static_cast<uint32_t>(ny);
    msg.info.origin.position.x = origin_xy.x();
    msg.info.origin.position.y = origin_xy.y();
    msg.info.origin.position.z = z_slice;
    msg.info.origin.orientation.w = 1.0;
    msg.data = std::move(grid_data);
    pub_costmap_->publish(msg);
  }

  void publishCandidates(const rclcpp::Time& stamp) {
    if (pub_candidates_->get_subscription_count() == 0) return;
    const MPPICandidateSet& c = mppi_->lastCandidates();
    visualization_msgs::msg::MarkerArray arr;
    arr.markers.reserve(static_cast<std::size_t>(c.num_candidates) + 1);

    // Costs span orders of magnitude (one lethal sample dominates a linear scale): colour by log cost.
    const double span = std::log1p(std::max(0.0, c.max_cost - c.min_cost));
    for (int k = 0; k < c.num_candidates; ++k) {
      visualization_msgs::msg::Marker mk;
      mk.header.stamp = stamp;
      mk.header.frame_id = odom_frame_;
      mk.ns = "mppi_candidates";
      mk.id = k;
      mk.type = visualization_msgs::msg::Marker::LINE_STRIP;
      mk.action = visualization_msgs::msg::Marker::ADD;
      mk.pose.orientation.w = 1.0;
      mk.scale.x = 0.008;
      mk.lifetime = rclcpp::Duration::from_seconds(0.5);
      const double u = span > 1e-9 ? std::log1p(c.costs[static_cast<std::size_t>(k)] - c.min_cost) / span : 0.0;
      mk.color.r = static_cast<float>(u);
      mk.color.g = static_cast<float>(1.0 - u);
      mk.color.b = 0.1f;
      mk.color.a = 0.45f;
      const Eigen::Vector3d* pts = &c.positions[static_cast<std::size_t>(k * c.points_per_candidate)];
      mk.points.reserve(static_cast<std::size_t>(c.points_per_candidate));
      for (int t = 0; t < c.points_per_candidate; ++t) mk.points.push_back(toPoint(pts[t]));
      arr.markers.push_back(std::move(mk));
    }

    visualization_msgs::msg::Marker opt;
    opt.header.stamp = stamp;
    opt.header.frame_id = odom_frame_;
    opt.ns = "mppi_optimal";
    opt.id = 0;
    opt.type = visualization_msgs::msg::Marker::LINE_STRIP;
    opt.action = visualization_msgs::msg::Marker::ADD;
    opt.pose.orientation.w = 1.0;
    opt.scale.x = 0.03;
    opt.lifetime = rclcpp::Duration::from_seconds(0.5);
    opt.color.r = 0.1f;
    opt.color.g = 0.55f;
    opt.color.b = 1.0f;
    opt.color.a = 1.0f;
    for (const auto& p : mppi_->lastOptimalRollout()) opt.points.push_back(toPoint(p));
    arr.markers.push_back(std::move(opt));
    pub_candidates_->publish(arr);
  }

  void onControlTimer() {
    if (!has_odom_) return;
    updateMapToOdomTF();
    if (has_isam2_grid_ && now().seconds() - last_isam2_grid_s_ > isam2_grid_timeout_s_) {
      RCLCPP_WARN(get_logger(), "iSAM2 obstacle grid is stale (> %.1f s); dropping it", isam2_grid_timeout_s_);
      mppi_->setLocalDistanceGrid(nullptr);
      has_isam2_grid_ = false;
    }
    if (local_ref_odom_.empty()) return;

    const TrajectorySetpointCmd cmd = mppi_->computeCommand(state_odom_, local_ref_odom_);
    const rclcpp::Time stamp = now();
    if (++tick_ % viz_every_n_ == 0) publishCandidates(stamp);

    if (cmd.request_global_replan) {
      geometry_msgs::msg::PointStamped blk;
      blk.header.stamp = stamp;
      blk.header.frame_id = map_frame_;
      blk.point.x = cmd.blocked_obstacle_pos.x();
      blk.point.y = cmd.blocked_obstacle_pos.y();
      blk.point.z = cmd.blocked_obstacle_pos.z();
      pub_replan_->publish(blk);
      mppi_->reset();
    }

    geometry_msgs::msg::TwistStamped tw{};
    tw.header.stamp = stamp;
    tw.header.frame_id = odom_frame_;
    tw.twist.linear.x = cmd.vel_ned.x();
    tw.twist.linear.y = cmd.vel_ned.y();
    tw.twist.linear.z = cmd.vel_ned.z();
    tw.twist.angular.z = cmd.yawspeed_rad_s;
    pub_cmd_vel_->publish(tw);

#ifdef HAVE_PX4_MSGS
    px4_msgs::msg::TrajectorySetpoint sp{};
    sp.timestamp = static_cast<uint64_t>(stamp.nanoseconds() / 1000);
    sp.position = {static_cast<float>(cmd.pos_ned.x()), static_cast<float>(cmd.pos_ned.y()),
                   static_cast<float>(cmd.pos_ned.z())};
    sp.velocity = {static_cast<float>(cmd.vel_ned.x()), static_cast<float>(cmd.vel_ned.y()),
                   static_cast<float>(cmd.vel_ned.z())};
    sp.acceleration = {static_cast<float>(cmd.acc_ned.x()), static_cast<float>(cmd.acc_ned.y()),
                       static_cast<float>(cmd.acc_ned.z())};
    sp.yaw = static_cast<float>(cmd.yaw_rad);
    sp.yawspeed = static_cast<float>(cmd.yawspeed_rad_s);
    pub_traj_sp_->publish(sp);
#endif
  }

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  std::string map_frame_;
  std::string odom_frame_;
  double cruise_speed_m_s_{2.60};
  int64_t viz_every_n_{5};
  int64_t tick_{0};

  std::shared_ptr<GroundTruthMap> gt_map_;
  std::shared_ptr<VoxelCostmap> costmap_;
  std::shared_ptr<MPPIController> mppi_;
  SE3Transform T_map_odom_{};
  PlannedPath local_ref_odom_;
  VehicleKinematics state_odom_{};
  bool has_odom_{false};

  isam2_depth::GridParams isam2_grid_params_{};
  double isam2_grid_timeout_s_{1.0};
  double last_isam2_grid_s_{0.0};
  bool has_isam2_grid_{false};

  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr pub_costmap_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_candidates_;
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr pub_replan_;
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr pub_cmd_vel_;
#ifdef HAVE_PX4_MSGS
  rclcpp::Publisher<px4_msgs::msg::TrajectorySetpoint>::SharedPtr pub_traj_sp_;
#endif

  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr sub_local_ref_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_points_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_isam2_;
  rclcpp::TimerBase::SharedPtr timer_costmap_;
  rclcpp::TimerBase::SharedPtr timer_control_;
};

}  // namespace maze_planning

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<maze_planning::MPPIControllerNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
