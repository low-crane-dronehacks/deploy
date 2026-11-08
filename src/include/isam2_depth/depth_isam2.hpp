// Keyframe factor graph over depth + PX4 odometry, solved incrementally with iSAM2 (Kaess et al., ICRA 2011),
// producing the state and obstacle distance field an MPPI controller rolls out from.
//
// Variables per keyframe k:  X(k) Pose3 world NED <- body FRD,  V(k) Vector3 velocity NED;  L(id) Point3 landmark.
// Factors:
//   prior        X(0), V(0) at PX4's estimate (anchors the graph in PX4's local frame)
//   odometry     BetweenFactor X(k-1)->X(k) from PX4 relative motion (sigma grows with distance / rotation)
//   depth ICP    BetweenFactor X(j)->X(k) from projective point-to-plane ICP of keyframe depth clouds, with the
//                ICP information matrix (robust Huber); j = k-1 plus up to `extra_links` older nearby keyframes
//   attitude     Pose3AttitudeFactor X(k): PX4 roll/pitch are gravity-referenced, ICP / odometry yaw is not
//   velocity     BodyVelocityFactor (X(k), V(k)) from PX4 (body frame: immune to its yaw drift), and
//                KinematicFactor (X(k-1), V(k-1), X(k), V(k))
//   landmark     BodyPointFactor X(k)->L(id) for a 3D point measured in depth (e.g. the ring-marker centre)
// iSAM2 settings follow the paper: Gauss-Newton with wildfire threshold alpha (partial state updates, Alg. 4),
// fluid relinearization threshold beta (Alg. 5); GTSAM orders the affected variables with constrained COLAMD,
// newest keys last (Sec. IV-A).
#pragma once

#include <map>
#include <optional>
#include <vector>

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/Values.h>

#include "isam2_depth/depth_icp.hpp"
#include "isam2_depth/distance_grid.hpp"
#include "isam2_depth/mppi_interface.hpp"
#include "isam2_depth/types.hpp"

namespace isam2_depth {

struct Isam2Params {
  // iSAM2 (paper Sec. IV-B, IV-C; their experiments: alpha 0.001, beta 0.1)
  double wildfire_threshold = 0.001;     // alpha
  double relinearize_threshold = 0.1;    // beta
  int relinearize_skip = 1;              // check every update (the paper: every 10 steps on 10k-pose datasets)

  // keyframe selection (from PX4 motion since the last keyframe)
  double kf_min_translation = 0.25;      // m
  double kf_min_rotation_deg = 10.0;
  double kf_max_interval = 0.5;          // s

  // PX4 relative-motion factor
  double odom_sigma_trans = 0.02, odom_sigma_trans_per_m = 0.05;     // m, m/m
  double odom_sigma_rot = 0.005, odom_sigma_rot_per_rad = 0.05;      // rad, rad/rad
  double attitude_sigma = 0.03;          // rad, PX4 roll / pitch
  double vel_sigma = 0.15;               // m/s, PX4 velocity
  double kin_sigma = 0.05;               // m, trapezoidal position/velocity consistency
  double prior_sigma_rot = 0.01, prior_sigma_trans = 0.01, prior_sigma_vel = 0.1;

  // depth
  CloudParams cloud;
  IcpParams icp;
  double icp_gate_trans = 0.3;           // m: ICP disagreeing with the prediction by more is dropped
  double icp_gate_rot_deg = 15.0;
  double icp_sigma_max_trans = 1.0, icp_sigma_max_rot = 0.5;   // regularises unobservable ICP directions
  int extra_links = 2;                   // ICP to older keyframes near the new one (revisits / loop closures)
  double link_max_dist = 1.5;            // m
  double link_max_angle_deg = 30.0;
  int link_min_age = 5;                  // keyframes; nearer ones are already chained by odometry + ICP
  int keep_dense = 150;                  // newest keyframes keeping their organised cloud for revisit ICP (~0.4 MB each)

  // map for the MPPI
  double map_voxel = 0.05;               // m, per-keyframe cloud downsampling
  double map_radius = 8.0;               // m, keyframes farther than this from the vehicle are left out
  GridParams grid;
};

class DepthIsam2 {
 public:
  DepthIsam2(const Isam2Params& p, const PinholeCamera& cam, const gtsam::Pose3& T_body_cam);

  // One depth frame with PX4's estimate at the frame's stamp. Returns true if it became a keyframe (the graph
  // was updated and latest() changed). Landmark observations force a keyframe.
  bool addFrame(const DepthFrame& depth, const OdomSample& odom, const std::vector<LandmarkObs>& obs = {});

  bool initialised() const { return !kfs_.empty(); }
  // Estimate at the newest keyframe.
  const MppiInput& latest() const { return latest_; }
  // Newest keyframe estimate carried forward by PX4's relative motion since then (call at the odometry rate).
  MppiInput predict(const OdomSample& now) const;

  struct Stats {
    size_t keyframes = 0, factors = 0;
    int icp_ok = 0, icp_rejected = 0, links = 0;
    double last_update_ms = 0;                       // whole keyframe, of which:
    double icp_ms = 0, isam_ms = 0, output_ms = 0;   // depth ICP, iSAM2 update + estimate, covariance + grid
  };
  const Stats& stats() const { return stats_; }
  const gtsam::Values& estimate() const { return estimate_; }

  // The full smoothed estimate (paper: every past pose, not only the newest), for visualisation / logging.
  struct KeyframePose {
    double stamp;
    gtsam::Pose3 T_world_body;   // world NED <- body FRD, current smoothed value
  };
  std::vector<KeyframePose> keyframePoses() const;
  // The whole corrected map: every keyframe's downsampled cloud placed at its CURRENT smoothed pose, merged into
  // `voxel` cells; a cell is kept if at least `min_hits` keyframe points fall in it. Voxel centres, world NED.
  // Unlike latest().distance (map_radius / grid window around the vehicle), this covers everything seen.
  // Cost grows with the number of keyframes (~1M points at a few hundred): call it at ~1 Hz, not per frame.
  std::vector<Eigen::Vector3f> globalMap(double voxel, int min_hits) const;

 private:
  struct Keyframe {
    double stamp;
    OdomSample odom;
    std::optional<OrganisedCloud> cloud;      // dropped after keep_dense newer keyframes
    std::vector<Eigen::Vector3f> map_points;  // body FRD, voxel-downsampled
  };

  // ICP between keyframe clouds as a body-frame BetweenFactor X(j)->X(k); false if rejected.
  bool addIcpFactor(size_t j, size_t k, const OrganisedCloud& cloud_k, const gtsam::Pose3& predicted_rel,
                    gtsam::NonlinearFactorGraph& g);
  void rebuildOutput(size_t k);

  Isam2Params p_;
  PinholeCamera cam_;
  gtsam::Pose3 T_bc_, T_cb_;
  gtsam::ISAM2 isam_;
  gtsam::Values estimate_;
  std::vector<Keyframe> kfs_;
  MppiInput latest_;
  Stats stats_;
};

}  // namespace isam2_depth
