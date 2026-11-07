// maze_planning_node: RRT* Global Planner + Path Transformer + STVL Costmap + 2048-Rollout MPPI Controller
// Implements the full ROS 2 topic & TF2 pipeline specified in system_architecture_plan.md (Section 2.2, 2.3, WP-3).
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
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
#include <std_msgs/msg/float32_multi_array.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

#include "maze_planning/groundtruth_map.hpp"
#include "maze_planning/mppi_controller.hpp"
#include "maze_planning/path_transformer.hpp"
#include "maze_planning/rrt_star_planner.hpp"
#include "maze_planning/voxel_costmap.hpp"

namespace maze_planning {

class MazePlanningNode : public rclcpp::Node {
 public:
  MazePlanningNode()
      : Node("maze_planning_node"),
        tf_buffer_(get_clock()),
        tf_listener_(tf_buffer_),
        tf_broadcaster_(this) {
    const std::string gt_path =
        declare_parameter<std::string>("groundtruth_map_path", "config/course_groundtruth.bin");
    const std::string obj_path =
        declare_parameter<std::string>("course_obj_path", "sitl/world/course.obj");
    prefer_window_ = declare_parameter<bool>("prefer_window", true);
    include_return_leg_ = declare_parameter<bool>("include_return_leg", false);
    sim_loopback_ = declare_parameter<bool>("sim_loopback", false);
    inject_demo_pillar_ = declare_parameter<bool>("inject_demo_pillar", false);
    map_frame_ = declare_parameter<std::string>("map_frame", "map");
    odom_frame_ = declare_parameter<std::string>("odom_frame", "odom");
    base_frame_ = declare_parameter<std::string>("base_frame", "base_link");

    gt_map_ = std::make_shared<GroundTruthMap>();
    if (!gt_map_->loadFromBinary(gt_path)) {
      if (!gt_map_->loadFromObj(obj_path)) {
        gt_map_->initDefaultCourse();
      }
    }

    VoxelCostmapConfig vcfg{};
    vcfg.max_sensor_range_m = declare_parameter<double>("max_sensor_range_m", 3.50);
    costmap_ = std::make_shared<VoxelCostmap>(vcfg, gt_map_);
    if (inject_demo_pillar_) {
      costmap_->injectUnmodeledPillar(Eigen::Vector2d(1.75, -2.5146), 0.10, 1.65, 0.0);
    }

    RRTStarConfig rcfg{};
    rcfg.cruise_speed_m_s = declare_parameter<double>("cruise_speed_m_s", 2.60);
    rcfg.max_iterations = static_cast<int>(declare_parameter<int64_t>("rrt_max_iterations", 1200));
    rrt_planner_ = std::make_unique<RRTStarPlanner>(rcfg, gt_map_, costmap_);

    PathTransformerConfig ptcfg{};
    path_transformer_ = std::make_unique<PathTransformer>(ptcfg);

    MPPIConfig mcfg{};
    mcfg.num_samples = static_cast<int>(declare_parameter<int64_t>("mppi_num_samples", 2048));
    mcfg.horizon_steps = static_cast<int>(declare_parameter<int64_t>("mppi_horizon_steps", 16));
    MultirotorPlantParams plant{};
    mppi_ = std::make_shared<MPPIController>(mcfg, plant, costmap_);

    // Initial global RRT* plan on GroundTruthMap
    const Eigen::Vector3d launch_pos(
        gt_map_->launchMarkNed().x(), gt_map_->launchMarkNed().y(), -0.78);
    rrt_planner_->planFullCourse(launch_pos, prefer_window_, include_return_leg_, global_path_map_);
    path_transformer_->setGlobalPathMap(global_path_map_);

    if (sim_loopback_) {
      state_odom_.timestamp_s = get_clock()->now().seconds();
      state_odom_.pos_odom = launch_pos;
      state_odom_.vel_odom = Eigen::Vector3d::Zero();
      state_odom_.q_odom_base = Eigen::Quaterniond::Identity();
      state_odom_.tracking_valid = true;
      has_odom_ = true;
    }

    // Publishers matching Section 2.3 Detailed Topic Specification + RViz visualization topics
    pub_global_rrt_ = create_publisher<nav_msgs::msg::Path>("/plan/global_rrt", 10);
    pub_local_ref_ = create_publisher<nav_msgs::msg::Path>("/plan/local_reference", 10);
    pub_mppi_rollout_ = create_publisher<nav_msgs::msg::Path>("/plan/mppi_rollout", 10);
    pub_drone_pose_ = create_publisher<geometry_msgs::msg::PoseStamped>("/viz/drone_pose", 10);
    pub_map_cloud_ = create_publisher<sensor_msgs::msg::PointCloud2>("/viz/groundtruth_map_cloud", 1);
    pub_rl_lookaheads_ =
        create_publisher<std_msgs::msg::Float32MultiArray>("/plan/rl_lookaheads", 10);
    pub_costmap_ = create_publisher<nav_msgs::msg::OccupancyGrid>("/costmap/local_voxels", 10);
    buildGroundTruthPointCloudMsg();

    rclcpp::QoS px4_qos(1);
    px4_qos.best_effort();
    px4_qos.transient_local();
#ifdef HAVE_PX4_MSGS
    pub_traj_sp_ =
        create_publisher<px4_msgs::msg::TrajectorySetpoint>("/control/trajectory_setpoint", px4_qos);
#endif
    pub_cmd_vel_ =
        create_publisher<geometry_msgs::msg::TwistStamped>("/control/cmd_vel", 10);

    // Subscribers matching Section 2.3 Detailed Topic Specification
    sub_odom_ = create_subscription<nav_msgs::msg::Odometry>(
        "/odometry/local", rclcpp::SensorDataQoS(),
        [this](const nav_msgs::msg::Odometry::SharedPtr m) { onOdometry(*m); });
    sub_points_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        "/oak/points/clamped", rclcpp::SensorDataQoS(),
        [this](const sensor_msgs::msg::PointCloud2::SharedPtr m) { onClampedPointCloud(*m); });

    // Multi-rate timers matching Section 2.3:
    // - /plan/global_rrt @ 1 Hz (or event-driven on replan)
    // - /plan/local_reference @ 20 Hz (50 ms)
    // - /costmap/local_voxels @ 30 Hz (33 ms)
    // - /plan/rl_lookaheads & /control/trajectory_setpoint @ 50 Hz (20 ms)
    timer_global_ = create_wall_timer(std::chrono::milliseconds(1000), [this]() { onGlobalTimer(); });
    timer_local_ref_ = create_wall_timer(std::chrono::milliseconds(50), [this]() { onLocalRefTimer(); });
    timer_costmap_ = create_wall_timer(std::chrono::milliseconds(33), [this]() { onCostmapTimer(); });
    timer_control_ = create_wall_timer(std::chrono::milliseconds(20), [this]() { onControlTimer(); });

    RCLCPP_INFO(get_logger(),
                "maze_planning_node started: GroundTruthMap slabs=%zu boulders=%zu tris=%zu, MPPI rollouts=%d",
                gt_map_->numOBBSlabs(), gt_map_->numBoulders(), gt_map_->numTriangles(),
                mcfg.num_samples);
  }

  std::shared_ptr<MPPIController> mppiController() const { return mppi_; }

 private:
  void updateMapToOdomTF() {
    try {
      const auto tf = tf_buffer_.lookupTransform(map_frame_, odom_frame_, tf2::TimePointZero);
      SE3Transform T_map_odom{};
      T_map_odom.t = Eigen::Vector3d(
          tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z);
      T_map_odom.q = Eigen::Quaterniond(
          tf.transform.rotation.w, tf.transform.rotation.x,
          tf.transform.rotation.y, tf.transform.rotation.z).normalized();
      path_transformer_->setMapToOdomTransform(T_map_odom);
      mppi_->setOdomToMapTransform(T_map_odom);
    } catch (const std::exception&) {
      // Keep identity / last valid T_map_odom when TF is not yet broadcast
    }
  }

  void onOdometry(const nav_msgs::msg::Odometry& m) {
    state_odom_.timestamp_s = rclcpp::Time(m.header.stamp).seconds();
    state_odom_.pos_odom = Eigen::Vector3d(
        m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z);
    state_odom_.q_odom_base = Eigen::Quaterniond(
        m.pose.pose.orientation.w, m.pose.pose.orientation.x,
        m.pose.pose.orientation.y, m.pose.pose.orientation.z).normalized();
    const Eigen::Vector3d v_body(
        m.twist.twist.linear.x, m.twist.twist.linear.y, m.twist.twist.linear.z);
    state_odom_.vel_odom = state_odom_.q_odom_base * v_body;
    state_odom_.omega_body = Eigen::Vector3d(
        m.twist.twist.angular.x, m.twist.twist.angular.y, m.twist.twist.angular.z);
    state_odom_.pos_variance =
        std::max({m.pose.covariance[0], m.pose.covariance[7], m.pose.covariance[14]});
    state_odom_.tracking_valid = state_odom_.pos_odom.allFinite();
    has_odom_ = true;
  }

  void onClampedPointCloud(const sensor_msgs::msg::PointCloud2& m) {
    if (!has_odom_) return;
    updateMapToOdomTF();
    const SE3Transform& T_map_odom = path_transformer_->mapToOdomTransform();
    const Eigen::Vector3d sensor_map = T_map_odom.apply(state_odom_.pos_odom);

    std::vector<Eigen::Vector3d> pts_map;
    pts_map.reserve(static_cast<std::size_t>(m.width * m.height));
    sensor_msgs::PointCloud2ConstIterator<float> iter_x(m, "x"), iter_y(m, "y"), iter_z(m, "z");
    for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
      if (!std::isfinite(*iter_x) || !std::isfinite(*iter_y) || !std::isfinite(*iter_z)) continue;
      const Eigen::Vector3d p_body(*iter_x, *iter_y, *iter_z);
      const Eigen::Vector3d p_odom =
          state_odom_.pos_odom + state_odom_.q_odom_base * p_body;
      pts_map.push_back(T_map_odom.apply(p_odom));
    }
    costmap_->updateFromClampedPointCloud(get_clock()->now().seconds(), sensor_map, pts_map);
  }

  nav_msgs::msg::Path toRosPath(const PlannedPath& p, const std::string& frame_id) const {
    nav_msgs::msg::Path msg;
    msg.header.stamp = get_clock()->now();
    msg.header.frame_id = frame_id;
    msg.poses.reserve(p.points.size());
    for (const auto& pt : p.points) {
      geometry_msgs::msg::PoseStamped ps;
      ps.header = msg.header;
      ps.pose.position.x = pt.pos.x();
      ps.pose.position.y = pt.pos.y();
      ps.pose.position.z = pt.pos.z();
      const double half_yaw = 0.5 * pt.yaw_rad;
      ps.pose.orientation.w = std::cos(half_yaw);
      ps.pose.orientation.x = 0.0;
      ps.pose.orientation.y = 0.0;
      ps.pose.orientation.z = std::sin(half_yaw);
      msg.poses.push_back(ps);
    }
    return msg;
  }

  void buildGroundTruthPointCloudMsg() {
    std::vector<Eigen::Vector3f> pts;
    pts.reserve(18000);
    for (double x = -2.95; x <= 2.95; x += 0.06) {
      for (double y = -2.95; y <= 2.95; y += 0.06) {
        for (double z = -1.70; z <= -0.05; z += 0.08) {
          if (gt_map_->isStaticWallOccupied(Eigen::Vector3d(x, y, z))) {
            pts.emplace_back(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
          }
        }
      }
    }
    gt_cloud_msg_.header.frame_id = map_frame_;
    gt_cloud_msg_.height = 1;
    gt_cloud_msg_.width = static_cast<uint32_t>(pts.size());
    sensor_msgs::PointCloud2Modifier mod(gt_cloud_msg_);
    mod.setPointCloud2FieldsByString(1, "xyz");
    mod.resize(pts.size());
    sensor_msgs::PointCloud2Iterator<float> ix(gt_cloud_msg_, "x");
    sensor_msgs::PointCloud2Iterator<float> iy(gt_cloud_msg_, "y");
    sensor_msgs::PointCloud2Iterator<float> iz(gt_cloud_msg_, "z");
    for (const auto& p : pts) {
      *ix = p.x();
      *iy = p.y();
      *iz = p.z();
      ++ix;
      ++iy;
      ++iz;
    }
  }

  void onGlobalTimer() {
    pub_global_rrt_->publish(toRosPath(global_path_map_, map_frame_));
    gt_cloud_msg_.header.stamp = get_clock()->now();
    pub_map_cloud_->publish(gt_cloud_msg_);
  }

  void onLocalRefTimer() {
    if (!has_odom_) return;
    updateMapToOdomTF();
    latest_local_ref_odom_ = path_transformer_->extractLocalReferenceOdom(state_odom_);
    pub_local_ref_->publish(toRosPath(latest_local_ref_odom_, odom_frame_));
  }

  void onCostmapTimer() {
    int nx = 0, ny = 0;
    Eigen::Vector2d origin_xy = Eigen::Vector2d::Zero();
    std::vector<int8_t> grid_data;
    const double z_slice = has_odom_ ? state_odom_.pos_odom.z() : -0.78;
    costmap_->exportOccupancyGrid2D(z_slice, 0.08, nx, ny, origin_xy, grid_data);

    nav_msgs::msg::OccupancyGrid msg;
    msg.header.stamp = get_clock()->now();
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

  void onControlTimer() {
    if (!has_odom_) return;
    updateMapToOdomTF();

    // 1. Publish /plan/rl_lookaheads @ 50 Hz (Section 4.1: 4 relative waypoints in body frame B)
    std::array<float, 12> rl_flat{};
    path_transformer_->computeRLLookaheadsBody(state_odom_, &rl_flat);
    std_msgs::msg::Float32MultiArray rl_msg;
    rl_msg.data.assign(rl_flat.begin(), rl_flat.end());
    pub_rl_lookaheads_->publish(rl_msg);

    // 2. Compute 50 Hz MPPI command over 2048 rollouts
    if (latest_local_ref_odom_.empty()) {
      latest_local_ref_odom_ = path_transformer_->extractLocalReferenceOdom(state_odom_);
    }
    const TrajectorySetpointCmd cmd = mppi_->computeCommand(state_odom_, latest_local_ref_odom_);

    // Publish MPPI optimal 16-step horizon rollout for RViz (/plan/mppi_rollout)
    nav_msgs::msg::Path rollout_msg;
    rollout_msg.header.stamp = get_clock()->now();
    rollout_msg.header.frame_id = odom_frame_;
    for (const auto& pt : mppi_->lastOptimalRollout()) {
      geometry_msgs::msg::PoseStamped ps;
      ps.header = rollout_msg.header;
      ps.pose.position.x = pt.x();
      ps.pose.position.y = pt.y();
      ps.pose.position.z = pt.z();
      ps.pose.orientation.w = 1.0;
      rollout_msg.poses.push_back(ps);
    }
    pub_mppi_rollout_->publish(rollout_msg);

    // Optional standalone closed-loop loopback integrator for RViz visualization without Gazebo
    if (sim_loopback_) {
      const double dt = 0.02;
      state_odom_.timestamp_s += dt;
      state_odom_.vel_odom = 0.80 * state_odom_.vel_odom + 0.20 * cmd.vel_ned;
      state_odom_.pos_odom += state_odom_.vel_odom * dt;
      const double next_yaw = state_odom_.yaw() + cmd.yawspeed_rad_s * dt;
      state_odom_.q_odom_base = Eigen::Quaterniond(
          std::cos(0.5 * next_yaw), 0.0, 0.0, std::sin(0.5 * next_yaw));

      geometry_msgs::msg::TransformStamped tf_map_odom;
      tf_map_odom.header.stamp = get_clock()->now();
      tf_map_odom.header.frame_id = map_frame_;
      tf_map_odom.child_frame_id = odom_frame_;
      tf_map_odom.transform.rotation.w = 1.0;
      tf_broadcaster_.sendTransform(tf_map_odom);
    }

    // Publish current drone pose + TF odom -> base_link for RViz
    geometry_msgs::msg::PoseStamped pose_msg;
    pose_msg.header.stamp = get_clock()->now();
    pose_msg.header.frame_id = odom_frame_;
    pose_msg.pose.position.x = state_odom_.pos_odom.x();
    pose_msg.pose.position.y = state_odom_.pos_odom.y();
    pose_msg.pose.position.z = state_odom_.pos_odom.z();
    pose_msg.pose.orientation.w = state_odom_.q_odom_base.w();
    pose_msg.pose.orientation.x = state_odom_.q_odom_base.x();
    pose_msg.pose.orientation.y = state_odom_.q_odom_base.y();
    pose_msg.pose.orientation.z = state_odom_.q_odom_base.z();
    pub_drone_pose_->publish(pose_msg);

    geometry_msgs::msg::TransformStamped tf_odom_base;
    tf_odom_base.header.stamp = pose_msg.header.stamp;
    tf_odom_base.header.frame_id = odom_frame_;
    tf_odom_base.child_frame_id = base_frame_;
    tf_odom_base.transform.translation.x = state_odom_.pos_odom.x();
    tf_odom_base.transform.translation.y = state_odom_.pos_odom.y();
    tf_odom_base.transform.translation.z = state_odom_.pos_odom.z();
    tf_odom_base.transform.rotation = pose_msg.pose.orientation;
    tf_broadcaster_.sendTransform(tf_odom_base);

    // 3. FMEA Pillar Dead-Center: if MPPI detects a blocked narrow corridor, mark edge in RRT* & replan
    if (cmd.request_global_replan) {
      const Eigen::Vector3d cur_map =
          path_transformer_->mapToOdomTransform().apply(state_odom_.pos_odom);
      gt_map_->markCorridorBlocked(cmd.blocked_obstacle_pos, 0.36);
      if (rrt_planner_->planFullCourse(cur_map, prefer_window_, include_return_leg_,
                                       global_path_map_)) {
        path_transformer_->setGlobalPathMap(global_path_map_);
        latest_local_ref_odom_ = path_transformer_->extractLocalReferenceOdom(state_odom_);
        mppi_->reset();
        pub_global_rrt_->publish(toRosPath(global_path_map_, map_frame_));
      }
    }

    // 4. Publish /control/cmd_vel (always) and /control/trajectory_setpoint (when px4_msgs is present)
    geometry_msgs::msg::TwistStamped tw{};
    tw.header.stamp = get_clock()->now();
    tw.header.frame_id = odom_frame_;
    tw.twist.linear.x = cmd.vel_ned.x();
    tw.twist.linear.y = cmd.vel_ned.y();
    tw.twist.linear.z = cmd.vel_ned.z();
    tw.twist.angular.z = cmd.yawspeed_rad_s;
    pub_cmd_vel_->publish(tw);

#ifdef HAVE_PX4_MSGS
    px4_msgs::msg::TrajectorySetpoint sp{};
    sp.timestamp = static_cast<uint64_t>(get_clock()->now().nanoseconds() / 1000);
    sp.position = {
        static_cast<float>(cmd.pos_ned.x()),
        static_cast<float>(cmd.pos_ned.y()),
        static_cast<float>(cmd.pos_ned.z()),
    };
    sp.velocity = {
        static_cast<float>(cmd.vel_ned.x()),
        static_cast<float>(cmd.vel_ned.y()),
        static_cast<float>(cmd.vel_ned.z()),
    };
    sp.acceleration = {
        static_cast<float>(cmd.acc_ned.x()),
        static_cast<float>(cmd.acc_ned.y()),
        static_cast<float>(cmd.acc_ned.z()),
    };
    sp.yaw = static_cast<float>(cmd.yaw_rad);
    sp.yawspeed = static_cast<float>(cmd.yawspeed_rad_s);
    pub_traj_sp_->publish(sp);
#endif
  }

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  tf2_ros::TransformBroadcaster tf_broadcaster_;
  std::string map_frame_{"map"};
  std::string odom_frame_{"odom"};
  std::string base_frame_{"base_link"};
  bool prefer_window_{true};
  bool include_return_leg_{false};
  bool sim_loopback_{false};
  bool inject_demo_pillar_{false};

  std::shared_ptr<GroundTruthMap> gt_map_;
  std::shared_ptr<VoxelCostmap> costmap_;
  std::unique_ptr<RRTStarPlanner> rrt_planner_;
  std::unique_ptr<PathTransformer> path_transformer_;
  std::shared_ptr<MPPIController> mppi_;

  PlannedPath global_path_map_;
  PlannedPath latest_local_ref_odom_;
  VehicleKinematics state_odom_{};
  bool has_odom_{false};
  sensor_msgs::msg::PointCloud2 gt_cloud_msg_;

  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_global_rrt_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_local_ref_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_mppi_rollout_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub_drone_pose_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_map_cloud_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr pub_rl_lookaheads_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr pub_costmap_;
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr pub_cmd_vel_;
#ifdef HAVE_PX4_MSGS
  rclcpp::Publisher<px4_msgs::msg::TrajectorySetpoint>::SharedPtr pub_traj_sp_;
#endif

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_points_;

  rclcpp::TimerBase::SharedPtr timer_global_;
  rclcpp::TimerBase::SharedPtr timer_local_ref_;
  rclcpp::TimerBase::SharedPtr timer_costmap_;
  rclcpp::TimerBase::SharedPtr timer_control_;
};

}  // namespace maze_planning

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<maze_planning::MazePlanningNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
