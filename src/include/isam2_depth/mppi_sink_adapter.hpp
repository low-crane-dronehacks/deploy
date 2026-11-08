// In-process bridge from the iSAM2 estimator to maze_planning::MPPIController.
//
// The MPPI core has no GTSAM dependency; this header is the only place that knows both sides. Use it when the
// estimator and the MPPI run in one process (a composed node):
//
//   auto sink = std::make_shared<isam2_depth::MppiControllerSink>(mppi);
//   isam2_node->setSink(sink);
//
// Across processes, mppi_controller_node instead rebuilds the same DistanceGrid from the estimator's
// ~/obstacles cloud (see docs/AUTONOMY_ARCHITECTURE.md, "Map inputs").
#pragma once

#include <memory>

#include "isam2_depth/mppi_interface.hpp"
#include "maze_planning/mppi_controller.hpp"

namespace isam2_depth {

class MppiControllerSink : public MppiStateSink {
 public:
  explicit MppiControllerSink(std::shared_ptr<maze_planning::MPPIController> mppi) : mppi_(std::move(mppi)) {}

  void onEstimate(const MppiInput& in) override {
    const gtsam::Point3 p = in.T_world_body.translation();
    const gtsam::Quaternion q = in.T_world_body.rotation().toQuaternion();
    mppi_->onEstimatorUpdate(in.stamp, Eigen::Vector3d(p.x(), p.y(), p.z()),
                             Eigen::Quaterniond(q.w(), q.x(), q.y(), q.z()), in.v_world, in.distance);
  }

 private:
  std::shared_ptr<maze_planning::MPPIController> mppi_;
};

}  // namespace isam2_depth
