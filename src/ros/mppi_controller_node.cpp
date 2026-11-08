// mppi_controller_node: Dedicated 2048-Rollout 6-DoF MPPI Controller & STVL Costmap Node
// (Section 2.2, 2.3, WP-3 Task 3 & Task 4).
//
// Responsibilities:
//   - Subscribes to /plan/local_reference (20 Hz nav_msgs/Path in odom frame from path_transformer_node)
//   - Subscribes to /odometry/local (100 Hz nav_msgs/Odometry in odom frame)
//   - Subscribes to /oak/points/clamped (30 Hz sensor_msgs/PointCloud2) to update STVL VoxelCostmap
//     with hard-locked static maze walls (FMEA Disparity Punch-Through protection)
//   - Publishes /costmap/local_voxels (30 Hz nav_msgs/OccupancyGrid)
//   - Evaluates 2048 rollouts at 50 Hz and publishes:
//       * /control/trajectory_setpoint (px4_msgs/TrajectorySetpoint when px4_msgs is present)
//       * /control/cmd_vel (geometry_msgs/TwistStamped)
//       * /plan/mppi_rollout (nav_msgs/Path in odom frame for RViz)
//       * /plan/replan_blocked_corridor (geometry_msgs/PointStamped when FMEA Pillar Dead-Center triggers)
#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <geometry_msgs/msg/point_stamped.hpp>
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
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

#include "maze_planning/groundtruth_map.hpp"
#include "maze_planning/mppi_controller.hpp"
#include "maze_planning/voxel_costmap.hpp"

namespace maze_planning {

class MPPIControllerNode : public rclcpp::Node {
 public:
  MPPIControllerNode()
      : Node("mppi_controller_node"),
        tf_buffer_(get_clock()),
        tf_listener_(tf_buffer_),
        tf_broadcaster_(this) {
    const std::string gt_path =
        declare_parameter<std::string>("groundtruth_map_path", "config/course_groundtruth.bin");
    const std::string obj_path =
        declare_parameter<std::string>("course_obj_path", "sitl/world/course.obj");
    cruise_speed_m_s_ = declare_parameter<double>("cruise_speed_m_s", 2.60);
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

    MPPIConfig mcfg{};
    mcfg.num_samples = static_cast<int>(declare_parameter<int64_t>("mppi_num_samples", 2048));
    mcfg.horizon_steps = static_cast<int>(declare_parameter<int64_t>("mppi_horizon_steps", 16));
    MultirotorPlantParams plant{};
    mppi_ = std::make_shared<MPPIController>(mcfg, plant, costmap_);

    if (sim_loopback_) {
      state_odom_.timestamp_s = this->now().seconds();
      state_odom_.pos_odom = Eigen::Vector3d(
          gt_map_->launchMarkNed().x(), gt_map_->launchMarkNed().y(), -0.78);
      state_odom_.vel_odom = Eigen::Vector3d::Zero();
      state_odom_.q_odom_base = Eigen::Quaterniond::Identity();
      state_odom_.tracking_valid = true;
      has_odom_ = true;
    }

    pub_costmap_ = create_publisher<nav_msgs::msg::OccupancyGrid>("/costmap/local_voxels", 10);
    pub_mppi_rollout_ = create_publisher<nav_msgs::msg::Path>("/plan/mppi_rollout", 10);
    pub_drone_pose_ = create_publisher<geometry_msgs::msg::PoseStamped>("/viz/drone_pose", 10);
    pub_replan_ = create_publisher<geometry_msgs::msg::PointStamped>(
        "/plan/replan_blocked_corridor", 10);
    pub_cmd_vel_ = create_publisher<geometry_msgs::msg::TwistStamped>("/control/cmd_vel", 10);
    if (sim_loopback_) {
      pub_sim_odom_ = create_publisher<nav_msgs::msg::Odometry>(
          "/odometry/local", rclcpp::SensorDataQoS());
    }

#ifdef HAVE_PX4_MSGS
    rclcpp::QoS px4_qos(1);
    px4_qos.best_effort();
    px4_qos.transient_local();
    pub_traj_sp_ =
        create_publisher<px4_msgs::msg::TrajectorySetpoint>("/control/trajectory_setpoint", px4_qos);
#endif

    sub_local_ref_ = create_subscription<nav_msgs::msg::Path>(
        "/plan/local_reference", 10,
        [this](const nav_msgs::msg::Path::SharedPtr m) { onLocalReference(*m); });
    sub_odom_ = create_subscription<nav_msgs::msg::Odometry>(
        "/odometry/local", rclcpp::SensorDataQoS(),
        [this](const nav_msgs::msg::Odometry::SharedPtr m) { onOdometry(*m); });
    sub_points_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        "/oak/points/clamped", rclcpp::SensorDataQoS(),
        [this](const sensor_msgs::msg::PointCloud2::SharedPtr m) { onClampedPointCloud(*m); });

    timer_costmap_ = create_wall_timer(
        std::chrono::milliseconds(33), [this]() { onCostmapTimer(); });
    timer_control_ = create_wall_timer(
        std::chrono::milliseconds(20), [this]() { onControlTimer(); });

    RCLCPP_INFO(get_logger(), "mppi_controller_node started (rollouts=%d, horizon=%d, sim_loopback=%s)",
                mcfg.num_samples, mcfg.horizon_steps, sim_loopback_ ? "true" : "false");
  }

 private:
  void updateMapToOdomTF() {
    try {
      const auto tf = tf_buffer_.lookupTransform(map_frame_, odom_frame_, tf2::TimePointZero);
      T_map_odom_.t = Eigen::Vector3d(
          tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z);
      T_map_odom_.q = Eigen::Quaterniond(
          tf.transform.rotation.w, tf.transform.rotation.x,
          tf.transform.rotation.y, tf.transform.rotation.z).normalized();
      mppi_->setOdomToMapTransform(T_map_odom_);
    } catch (const std::exception&) {
    }
  }

  void onLocalReference(const nav_msgs::msg::Path& m) {
    PlannedPath p;
    p.frame_id = m.header.frame_id.empty() ? odom_frame_ : m.header.frame_id;
    p.points.reserve(m.poses.size());
    double s = 0.0;
    for (std::size_t i = 0; i < m.poses.size(); ++i) {
      const auto& pos = m.poses[i].pose.position;
      const auto& q = m.poses[i].pose.orientation;
      PathPoint pt{};
      pt.pos = Eigen::Vector3d(pos.x, pos.y, pos.z);
      pt.yaw_rad = std::atan2(2.0 * (q.w * q.z + q.x * q.y),
                              1.0 - 2.0 * (q.y * q.y + q.z * q.z));
      if (i > 0) {
        const Eigen::Vector3d d = pt.pos - p.points.back().pos;
        const double ds = d.norm();
        s += ds;
        if (ds > 1e-5) {
          p.points.back().vel = (d / ds) * cruise_speed_m_s_;
        }
      }
      pt.s_m = s;
      pt.t_s = s / std::max(0.2, cruise_speed_m_s_);
      p.points.push_back(pt);
    }
    if (p.points.size() >= 2) {
      p.points.back().vel = p.points[p.points.size() - 2].vel;
    }
    p.total_length_m = s;
    p.total_duration_s = s / std::max(0.2, cruise_speed_m_s_);
    local_ref_odom_ = std::move(p);
  }

  void onOdometry(const nav_msgs::msg::Odometry& m) {
    if (sim_loopback_) return;
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
    const Eigen::Vector3d sensor_map = T_map_odom_.apply(state_odom_.pos_odom);

    std::vector<Eigen::Vector3d> pts_map;
    pts_map.reserve(static_cast<std::size_t>(m.width * m.height));
    sensor_msgs::PointCloud2ConstIterator<float> iter_x(m, "x"), iter_y(m, "y"), iter_z(m, "z");
    for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
      if (!std::isfinite(*iter_x) || !std::isfinite(*iter_y) || !std::isfinite(*iter_z)) continue;
      const Eigen::Vector3d p_body(*iter_x, *iter_y, *iter_z);
      const Eigen::Vector3d p_odom =
          state_odom_.pos_odom + state_odom_.q_odom_base * p_body;
      pts_map.push_back(T_map_odom_.apply(p_odom));
    }
    costmap_->updateFromClampedPointCloud(this->now().seconds(), sensor_map, pts_map);
  }

  void onCostmapTimer() {
    int nx = 0, ny = 0;
    Eigen::Vector2d origin_xy = Eigen::Vector2d::Zero();
    std::vector<int8_t> grid_data;
    const double z_slice = has_odom_ ? state_odom_.pos_odom.z() : -0.78;
    costmap_->exportOccupancyGrid2D(z_slice, 0.08, nx, ny, origin_xy, grid_data);

    nav_msgs::msg::OccupancyGrid msg;
    msg.header.stamp = this->now();
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

    // Broadcast simulated odometry when running standalone RViz loopback demo
    if (sim_loopback_ && pub_sim_odom_) {
      nav_msgs::msg::Odometry odom_msg;
      odom_msg.header.stamp = this->now();
      odom_msg.header.frame_id = odom_frame_;
      odom_msg.child_frame_id = base_frame_;
      odom_msg.pose.pose.position.x = state_odom_.pos_odom.x();
      odom_msg.pose.pose.position.y = state_odom_.pos_odom.y();
      odom_msg.pose.pose.position.z = state_odom_.pos_odom.z();
      odom_msg.pose.pose.orientation.w = state_odom_.q_odom_base.w();
      odom_msg.pose.pose.orientation.x = state_odom_.q_odom_base.x();
      odom_msg.pose.pose.orientation.y = state_odom_.q_odom_base.y();
      odom_msg.pose.pose.orientation.z = state_odom_.q_odom_base.z();
      const Eigen::Vector3d v_body = state_odom_.q_odom_base.conjugate() * state_odom_.vel_odom;
      odom_msg.twist.twist.linear.x = v_body.x();
      odom_msg.twist.twist.linear.y = v_body.y();
      odom_msg.twist.twist.linear.z = v_body.z();
      pub_sim_odom_->publish(odom_msg);

      geometry_msgs::msg::TransformStamped tf_map_odom;
      tf_map_odom.header.stamp = odom_msg.header.stamp;
      tf_map_odom.header.frame_id = map_frame_;
      tf_map_odom.child_frame_id = odom_frame_;
      tf_map_odom.transform.rotation.w = 1.0;
      tf_broadcaster_.sendTransform(tf_map_odom);
    }

    if (local_ref_odom_.empty()) return;

    const TrajectorySetpointCmd cmd = mppi_->computeCommand(state_odom_, local_ref_odom_);

    // Publish MPPI optimal 16-step horizon rollout for RViz (/plan/mppi_rollout)
    nav_msgs::msg::Path rollout_msg;
    rollout_msg.header.stamp = this->now();
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

    if (sim_loopback_) {
      const double dt = 0.02;
      state_odom_.timestamp_s += dt;
      state_odom_.vel_odom = 0.80 * state_odom_.vel_odom + 0.20 * cmd.vel_ned;
      state_odom_.pos_odom += state_odom_.vel_odom * dt;
      const double next_yaw = state_odom_.yaw() + cmd.yawspeed_rad_s * dt;
      state_odom_.q_odom_base = Eigen::Quaterniond(
          std::cos(0.5 * next_yaw), 0.0, 0.0, std::sin(0.5 * next_yaw));
    }

    // Publish current drone pose + TF odom -> base_link for RViz
    geometry_msgs::msg::PoseStamped pose_msg;
    pose_msg.header.stamp = this->now();
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

    if (cmd.request_global_replan) {
      geometry_msgs::msg::PointStamped blk;
      blk.header.stamp = this->now();
      blk.header.frame_id = map_frame_;
      blk.point.x = cmd.blocked_obstacle_pos.x();
      blk.point.y = cmd.blocked_obstacle_pos.y();
      blk.point.z = cmd.blocked_obstacle_pos.z();
      pub_replan_->publish(blk);
      mppi_->reset();
    }

    geometry_msgs::msg::TwistStamped tw{};
    tw.header.stamp = this->now();
    tw.header.frame_id = odom_frame_;
    tw.twist.linear.x = cmd.vel_ned.x();
    tw.twist.linear.y = cmd.vel_ned.y();
    tw.twist.linear.z = cmd.vel_ned.z();
    tw.twist.angular.z = cmd.yawspeed_rad_s;
    pub_cmd_vel_->publish(tw);

#ifdef HAVE_PX4_MSGS
    px4_msgs::msg::TrajectorySetpoint sp{};
    sp.timestamp = static_cast<uint64_t>(this->now().nanoseconds() / 1000);
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
  double cruise_speed_m_s_{2.60};
  bool sim_loopback_{false};
  bool inject_demo_pillar_{false};

  std::shared_ptr<GroundTruthMap> gt_map_;
  std::shared_ptr<VoxelCostmap> costmap_;
  std::shared_ptr<MPPIController> mppi_;
  SE3Transform T_map_odom_{};
  PlannedPath local_ref_odom_;
  VehicleKinematics state_odom_{};
  bool has_odom_{false};

  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr pub_costmap_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_mppi_rollout_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub_drone_pose_;
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr pub_replan_;
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr pub_cmd_vel_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_sim_odom_;
#ifdef HAVE_PX4_MSGS
  rclcpp::Publisher<px4_msgs::msg::TrajectorySetpoint>::SharedPtr pub_traj_sp_;
#endif

  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr sub_local_ref_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_points_;
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
