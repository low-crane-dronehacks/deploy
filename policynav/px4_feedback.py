"""Validated PX4 policy inputs, independent of ROS. NED velocity and FRD->NED attitude."""
import math

import numpy as np


class Px4Feedback:
    def __init__(self, timeout_s=0.5, rf_max=4.0, estimator_timeout_s=2.5):
        self.timeout_s = timeout_s
        self.estimator_timeout_s = estimator_timeout_s
        self.vel = self.q = None
        self.range_m = float(rf_max)  # until the first valid range; later gaps retain the last valid value
        self.range_t = None
        self.lpos_t = self.att_t = self.estimator_t = None
        self.heading_good = self.vision_heading_good = False
        self.range_fusion_good = False
        self.range_source = None
        self.fault = None
        self._samples = {}

    def _sample(self, key, stamp, resets, now, active, timeout_s=None):
        timeout_s = self.timeout_s if timeout_s is None else timeout_s
        previous = self._samples.get(key)
        if stamp <= 0:
            if active:
                self.fault = f'{key}: invalid sample timestamp'
            return False
        if previous is not None:
            old_stamp, old_resets, old_time = previous
            if active and (resets != old_resets or stamp < old_stamp or
                           (stamp - old_stamp) * 1e-6 > timeout_s or
                           now - old_time > timeout_s):
                self.fault = f'{key}: estimate reset or feedback gap'
            if stamp == old_stamp:
                return False  # repeated samples cannot renew freshness
        self._samples[key] = stamp, resets, now
        return True

    def on_lpos(self, m, now, active=False):
        velocity = (float(m.vx), float(m.vy), float(m.vz))
        valid = (m.v_xy_valid and m.v_z_valid and
                 all(math.isfinite(v) for v in velocity))
        # With RANGE as height reference, PX4 can fuse range into height while
        # terrain validity remains false. Require fresh, healthy fusion evidence.
        fusion_ok = (self.estimator_t is not None and
                     0 <= now - self.estimator_t <= self.estimator_timeout_s and self.range_fusion_good)
        range_ok = ((m.dist_bottom_valid or fusion_ok) and m.v_z_valid and
                    math.isfinite(m.dist_bottom) and m.dist_bottom > 0)
        if not valid:
            self.vel = None
        if active and not valid:
            self.fault = 'PX4 velocity invalid'
        if not self._sample('local position', m.timestamp_sample,
                            (m.vxy_reset_counter, m.vz_reset_counter, m.heading_reset_counter), now, active):
            return
        self.lpos_t = now
        self.heading_good = bool(m.heading_good_for_control)
        if active and self.heading_error(now):
            self.fault = self.heading_error(now)
        self.vel = np.asarray(velocity) if valid else None
        if range_ok:
            self.range_m, self.range_t = float(m.dist_bottom), now
            self.range_source = 'terrain' if m.dist_bottom_valid else 'range_height_fusion'

    def on_estimator(self, m, now, active=False):
        # PX4 1.16 deliberately leaves yaw_align false for LOCAL_FRAME_FRD
        # vision yaw. A north-referenced heading is unnecessary for this policy;
        # velocity and attitude must share PX4's local frame instead.
        valid = (m.cs_tilt_align and m.cs_ev_yaw and not m.cs_ev_yaw_fault
                 and not m.fs_bad_hdg and not m.reject_yaw)
        if active and not self.heading_good and not valid:
            self.fault = 'PX4 vision heading invalid'
        if not self._sample('estimator flags', m.timestamp_sample, (), now,
                            active and not self.heading_good, self.estimator_timeout_s):
            return
        self.estimator_t = now
        self.vision_heading_good = bool(valid)
        self.range_fusion_good = bool(m.cs_tilt_align and m.cs_rng_hgt and
                                      not m.cs_rng_fault and not m.cs_rng_stuck and not m.reject_hagl)

    def heading_error(self, now):
        if self.heading_good:
            return None
        if self.estimator_t is None or not 0 <= now - self.estimator_t <= self.estimator_timeout_s:
            return 'PX4 heading unavailable: estimator flags missing or stale'
        if not self.vision_heading_good:
            return 'PX4 heading invalid: no healthy external-vision yaw fusion'
        return None

    def on_att(self, m, now, active=False):
        q = np.asarray(m.q, dtype=float)
        norm = float(np.linalg.norm(q))
        valid = q.shape == (4,) and np.isfinite(q).all() and abs(norm - 1.) < .01
        if not valid:
            self.q = None
            if active:
                self.fault = 'PX4 attitude invalid'
        if not self._sample('attitude', m.timestamp_sample, (m.quat_reset_counter,), now, active):
            return
        self.att_t = now
        self.q = q / norm if valid else None

    def error(self, now, timeout_s):
        if self.fault:
            return self.fault
        for name, stamp in (('PX4 velocity', self.lpos_t), ('PX4 attitude', self.att_t)):
            if stamp is None or not 0 <= now - stamp <= timeout_s:
                return name + ' missing or stale'
        if self.vel is None:
            return 'PX4 velocity invalid'
        if self.q is None:
            return 'PX4 attitude invalid'
        if self.heading_error(now):
            return self.heading_error(now)
        if self.range_t is None:
            return ('PX4 downward range unavailable: no accepted dist_bottom; '
                    'need valid terrain or fresh healthy range-height fusion')
        return None
