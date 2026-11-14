// rrt_star_planner_node: Global Informed RRT* planner (Section 2.2, 2.3, WP-3 Task 1).
//
// in   /odometry/local                   nav_msgs/Odometry (odom frame); current pose for replans + progress
//      /plan/replan_blocked_corridor     geometry_msgs/PointStamped (map frame) from MPPI / supervisor:
//                                        mark the corridor blocked and replan (FMEA Pillar Dead-Center)
//      TF map -> odom                    (identity with a warning if missing)
// out  /plan/global_path                  nav_msgs/Path, map frame, transient_local (latched). Published once
//                                        per new plan, never re-sent on a timer: a consumer that sees a new
//                                        message knows it is a new plan starting at the vehicle.
//                                        Pose stamps carry the timing (ros_conversions.hpp).
//      /viz/groundtruth_map_cloud        sensor_msgs/PointCloud2 (map frame, latched) for RViz
//
// The path goes start -> the course anchors still ahead -> pad. CourseProgress records which anchors have
// been passed, so a replan from mid-course never loops back through the start of the course.
// The initial plan starts at the launch mark at cruise height (the vehicle is still on the ground).
//
// Swapping in another global planner: replace this node with anything that publishes the same
// /plan/global_path contract (docs/AUTONOMY_ARCHITECTURE.md, "Global planner contract").
#include <chrono>
#include <cmath>
#include <memory>
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

#include "maze_planning/course_progress.hpp"
#include "maze_planning/groundtruth_map.hpp"
#include "maze_planning/ros_conversions.hpp"
#include "maze_planning/rrt_star_planner.hpp"

namespace maze_planning {

class RRTStarPlannerNode : public rclcpp::Node {
 public:
  RRTStarPlannerNode()
      : Node("rrt_star_planner_node"),
        tf_buffer_(get_clock()),
        tf_listener_(tf_buffer_) {
    const std::string gt_path =
        declare_parameter<std::string>("groundtruth_map_path", "assets/course_groundtruth.bin");
    const std::string obj_path =
        declare_parameter<std::string>("course_obj_path", "sitl/world/course.obj");
    prefer_window_ = declare_parameter<bool>("prefer_window", true);
    include_return_leg_ = declare_parameter<bool>("include_return_leg", false);
    map_frame_ = declare_parameter<std::string>("map_frame", "map_ned");
    odom_frame_ = declare_parameter<std::string>("odom_frame", "odom_ned");
    replan_min_interval_s_ = declare_parameter<double>("replan_min_interval_s", 1.5);
    block_radius_m_ = declare_parameter<double>("block_radius_m", 0.20);
    const double cruise_z = declare_parameter<double>("cruise_z_ned", -0.78);

    gt_map_ = std::make_shared<GroundTruthMap>();
    if (!gt_map_->loadFromBinary(gt_path)) {
      RCLCPP_WARN(get_logger(), "no ground-truth map at %s; trying %s", gt_path.c_str(), obj_path.c_str());
      if (!gt_map_->loadFromObj(obj_path)) {
        RCLCPP_WARN(get_logger(), "no course OBJ either; using the built-in default course");
        gt_map_->initDefaultCourse();
      }
    }

    RRTStarConfig rcfg{};
    rcfg.cruise_speed_m_s = declare_parameter<double>("cruise_speed_m_s", 1.50);
    rcfg.max_iterations = static_cast<int>(declare_parameter<int64_t>("rrt_max_iterations", 1200));
    rrt_planner_ = std::make_unique<RRTStarPlanner>(rcfg, gt_map_);

    const rclcpp::QoS latched = rclcpp::QoS(1).reliable().transient_local();
    pub_global_path_ = create_publisher<nav_msgs::msg::Path>("/plan/global_path", latched);
    pub_map_cloud_ = create_publisher<sensor_msgs::msg::PointCloud2>("/viz/groundtruth_map_cloud", latched);
    publishGroundTruthCloud();

    sub_odom_ = create_subscription<nav_msgs::msg::Odometry>(
        "/odometry/local", rclcpp::SensorDataQoS(),
        [this](const nav_msgs::msg::Odometry::SharedPtr m) { onOdometry(*m); });
    sub_replan_ = create_subscription<geometry_msgs::msg::PointStamped>(
        "/plan/replan_blocked_corridor", 10,
        [this](const geometry_msgs::msg::PointStamped::SharedPtr m) { onBlockedCorridorReplan(*m); });

    current_pos_map_ = Eigen::Vector3d(gt_map_->launchMarkNed().x(), gt_map_->launchMarkNed().y(), cruise_z);
    planAndPublish("initial plan from the launch mark");

    RCLCPP_INFO(get_logger(),
                "rrt_star_planner_node started: GroundTruthMap slabs=%zu boulders=%zu tris=%zu, %zu anchors",
                gt_map_->numOBBSlabs(), gt_map_->numBoulders(), gt_map_->numTriangles(),
                rrt_planner_->courseAnchors(prefer_window_, include_return_leg_).size());
  }

 private:
  // Plan current pose -> anchors not yet passed -> pad. Keeps (and does not re-send) the old path on failure.
  bool planAndPublish(const char* why) {
    const std::size_t first = progress_.nextAnchor();
    PlannedPath candidate;
    std::vector<double> anchor_s;
    if (!rrt_planner_->planFullCourse(current_pos_map_, prefer_window_, include_return_leg_, candidate, first,
                                      &anchor_s)) {
      RCLCPP_ERROR(get_logger(), "%s: RRT* could not reach every remaining anchor (from #%zu); keeping the "
                   "previous path", why, first);
      return false;
    }
    global_path_map_ = std::move(candidate);
    progress_.reset(global_path_map_, std::move(anchor_s), first);
    pub_global_path_->publish(ros_conv::toRosPath(global_path_map_, map_frame_, now()));
    RCLCPP_INFO(get_logger(), "%s: from [%.2f, %.2f, %.2f] via anchor #%zu.. -> pad, len=%.2fm", why,
                current_pos_map_.x(), current_pos_map_.y(), current_pos_map_.z(), first,
                global_path_map_.total_length_m);
    return true;
  }

  void publishGroundTruthCloud() {
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
    sensor_msgs::msg::PointCloud2 cloud;
    cloud.header.frame_id = map_frame_;
    cloud.header.stamp = now();
    cloud.height = 1;
    sensor_msgs::PointCloud2Modifier mod(cloud);
    mod.setPointCloud2FieldsByString(1, "xyz");
    mod.resize(pts.size());
    sensor_msgs::PointCloud2Iterator<float> ix(cloud, "x"), iy(cloud, "y"), iz(cloud, "z");
    for (const auto& p : pts) {
      *ix = p.x();
      *iy = p.y();
      *iz = p.z();
      ++ix;
      ++iy;
      ++iz;
    }
    pub_map_cloud_->publish(cloud);
  }

  void onOdometry(const nav_msgs::msg::Odometry& m) {
    const Eigen::Vector3d pos_odom(m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z);
    SE3Transform T_map_odom{};
    if (!ros_conv::lookupSE3(tf_buffer_, map_frame_, odom_frame_, T_map_odom)) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                           "no TF %s <- %s; treating them as identical", map_frame_.c_str(), odom_frame_.c_str());
    }
    current_pos_map_ = T_map_odom.apply(pos_odom);
    progress_.update(current_pos_map_);
  }

  void onBlockedCorridorReplan(const geometry_msgs::msg::PointStamped& m) {
    const double now_s = now().seconds();
    if (now_s - last_replan_s_ < replan_min_interval_s_) return;
    last_replan_s_ = now_s;

    const Eigen::Vector3d blocked_map(m.point.x, m.point.y, m.point.z);
    gt_map_->markCorridorBlocked(blocked_map, block_radius_m_);
    RCLCPP_WARN(get_logger(), "corridor blocked at [%.2f, %.2f, %.2f]; replanning", blocked_map.x(),
                blocked_map.y(), blocked_map.z());
    planAndPublish("replan around blocked corridor");
  }

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  std::string map_frame_;
  std::string odom_frame_;
  bool prefer_window_{true};
  bool include_return_leg_{false};
  double replan_min_interval_s_{1.5};
  double block_radius_m_{0.20};

  std::shared_ptr<GroundTruthMap> gt_map_;
  std::unique_ptr<RRTStarPlanner> rrt_planner_;
  PlannedPath global_path_map_;
  CourseProgress progress_;
  Eigen::Vector3d current_pos_map_{Eigen::Vector3d::Zero()};
  double last_replan_s_{-1e9};

  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_global_path_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_map_cloud_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr sub_replan_;
};

}  // namespace maze_planning

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<maze_planning::RRTStarPlannerNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
