// rrt_star_planner_node: Dedicated Global Informed RRT* Planner ROS 2 Node (Section 2.2, 2.3, WP-3 Task 1).
//
// Responsibilities:
//   - Loads the non-ESDF GroundTruthMap (config/course_groundtruth.bin, "CGTM" v1)
//   - Computes clearance-constrained Informed RRT* + clamped cubic B-spline global path in `map` frame
//   - Publishes /plan/global_rrt (nav_msgs/Path @ 1 Hz or event-driven on replan)
//   - Publishes /viz/groundtruth_map_cloud (sensor_msgs/PointCloud2) for 3D RViz visualization
//   - Subscribes to /plan/replan_blocked_corridor (geometry_msgs/PointStamped) from MPPI/Supervisor
//     for FMEA Pillar Dead-Center recovery (marks corridor sphere blocked and triggers immediate replan)
#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include "maze_planning/groundtruth_map.hpp"
#include "maze_planning/rrt_star_planner.hpp"

namespace maze_planning {

class RRTStarPlannerNode : public rclcpp::Node {
 public:
  RRTStarPlannerNode()
      : Node("rrt_star_planner_node"),
        tf_buffer_(get_clock()),
        tf_listener_(tf_buffer_) {
    const std::string gt_path =
        declare_parameter<std::string>("groundtruth_map_path", "config/course_groundtruth.bin");
    const std::string obj_path =
        declare_parameter<std::string>("course_obj_path", "sitl/world/course.obj");
    prefer_window_ = declare_parameter<bool>("prefer_window", true);
    include_return_leg_ = declare_parameter<bool>("include_return_leg", false);
    map_frame_ = declare_parameter<std::string>("map_frame", "map");
    odom_frame_ = declare_parameter<std::string>("odom_frame", "odom");

    gt_map_ = std::make_shared<GroundTruthMap>();
    if (!gt_map_->loadFromBinary(gt_path)) {
      if (!gt_map_->loadFromObj(obj_path)) {
        gt_map_->initDefaultCourse();
      }
    }

    RRTStarConfig rcfg{};
    rcfg.cruise_speed_m_s = declare_parameter<double>("cruise_speed_m_s", 2.60);
    rcfg.max_iterations = static_cast<int>(declare_parameter<int64_t>("rrt_max_iterations", 1200));
    rrt_planner_ = std::make_unique<RRTStarPlanner>(rcfg, gt_map_);

    const Eigen::Vector3d launch_pos(
        gt_map_->launchMarkNed().x(), gt_map_->launchMarkNed().y(), -0.78);
    current_pos_map_ = launch_pos;
    rrt_planner_->planFullCourse(launch_pos, prefer_window_, include_return_leg_, global_path_map_);

    pub_global_rrt_ = create_publisher<nav_msgs::msg::Path>("/plan/global_rrt", 10);
    pub_map_cloud_ =
        create_publisher<sensor_msgs::msg::PointCloud2>("/viz/groundtruth_map_cloud", 1);
    buildGroundTruthPointCloudMsg();

    sub_odom_ = create_subscription<nav_msgs::msg::Odometry>(
        "/odometry/local", rclcpp::SensorDataQoS(),
        [this](const nav_msgs::msg::Odometry::SharedPtr m) { onOdometry(*m); });
    sub_replan_ = create_subscription<geometry_msgs::msg::PointStamped>(
        "/plan/replan_blocked_corridor", 10,
        [this](const geometry_msgs::msg::PointStamped::SharedPtr m) { onBlockedCorridorReplan(*m); });

    timer_ = create_wall_timer(std::chrono::milliseconds(1000), [this]() { onPublishTimer(); });
    onPublishTimer();

    RCLCPP_INFO(get_logger(),
                "rrt_star_planner_node started: GroundTruthMap slabs=%zu boulders=%zu tris=%zu, path_len=%.2fm",
                gt_map_->numOBBSlabs(), gt_map_->numBoulders(), gt_map_->numTriangles(),
                global_path_map_.total_length_m);
  }

 private:
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

  void onOdometry(const nav_msgs::msg::Odometry& m) {
    Eigen::Vector3d pos_odom(
        m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z);
    try {
      const auto tf = tf_buffer_.lookupTransform(map_frame_, odom_frame_, tf2::TimePointZero);
      SE3Transform T_map_odom{};
      T_map_odom.t = Eigen::Vector3d(
          tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z);
      T_map_odom.q = Eigen::Quaterniond(
          tf.transform.rotation.w, tf.transform.rotation.x,
          tf.transform.rotation.y, tf.transform.rotation.z).normalized();
      current_pos_map_ = T_map_odom.apply(pos_odom);
    } catch (const std::exception&) {
      current_pos_map_ = pos_odom;
    }
  }

  void onBlockedCorridorReplan(const geometry_msgs::msg::PointStamped& m) {
    const Eigen::Vector3d blocked_map(m.point.x, m.point.y, m.point.z);
    gt_map_->markCorridorBlocked(blocked_map, 0.36);
    if (rrt_planner_->planFullCourse(current_pos_map_, prefer_window_, include_return_leg_,
                                     global_path_map_)) {
      RCLCPP_WARN(get_logger(),
                  "Replanned global RRT* around blocked corridor [%.2f, %.2f, %.2f] -> new len=%.2fm",
                  blocked_map.x(), blocked_map.y(), blocked_map.z(),
                  global_path_map_.total_length_m);
      onPublishTimer();
    }
  }

  nav_msgs::msg::Path toRosPath(const PlannedPath& p, const std::string& frame_id) {
    nav_msgs::msg::Path msg;
    msg.header.stamp = this->now();
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

  void onPublishTimer() {
    pub_global_rrt_->publish(toRosPath(global_path_map_, map_frame_));
    gt_cloud_msg_.header.stamp = this->now();
    pub_map_cloud_->publish(gt_cloud_msg_);
  }

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  std::string map_frame_{"map"};
  std::string odom_frame_{"odom"};
  bool prefer_window_{true};
  bool include_return_leg_{false};

  std::shared_ptr<GroundTruthMap> gt_map_;
  std::unique_ptr<RRTStarPlanner> rrt_planner_;
  PlannedPath global_path_map_;
  Eigen::Vector3d current_pos_map_{Eigen::Vector3d::Zero()};
  sensor_msgs::msg::PointCloud2 gt_cloud_msg_;

  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_global_rrt_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_map_cloud_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr sub_replan_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace maze_planning

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<maze_planning::RRTStarPlannerNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
