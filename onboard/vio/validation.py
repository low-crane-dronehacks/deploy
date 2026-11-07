"""Measurement validation shared by onboard consumers; no simulator dependencies."""
import math


class CaptureGate:
    """Validate captures on the caller's ROS clock, preserving original sample time."""
    def __init__(self, max_age_s=0.3, future_tol_s=0.005):
        if not all(math.isfinite(v) and v >= 0 for v in (max_age_s, future_tol_s)):
            raise ValueError("capture age/skew limits must be finite and nonnegative")
        self.max_age_us=max_age_s*1e6;self.future_tol_us=future_tol_s*1e6
        self.counts={k:0 for k in ('ok','duplicate','reordered','stale','future','invalid')}
        self.last_us=None

    def reset(self):
        self.last_us=None

    def check(self, capture_us, now_us):
        if (capture_us is None or now_us is None or
                not all(math.isfinite(v) for v in (capture_us,now_us)) or capture_us <= 0):
            kind='invalid'
        elif self.last_us is not None and capture_us==self.last_us:kind='duplicate'
        elif self.last_us is not None and capture_us<self.last_us:kind='reordered'
        elif now_us-capture_us>self.max_age_us:kind='stale'
        elif capture_us-now_us>self.future_tol_us:kind='future'
        else:kind='ok'
        self.counts[kind]+=1
        age=None if kind=='invalid' else (now_us-capture_us)*1e-6
        if kind=='ok':self.last_us=capture_us
        return kind=='ok',kind,age


def sample_problem(required, position, quaternion, velocity):
    """Return a reason when any field the target estimator consumes is invalid."""
    if 'position' in required and not all(math.isfinite(float(v)) for v in position):
        return 'non-finite position'
    if 'orientation' in required:
        if not all(math.isfinite(float(v)) for v in quaternion):return 'non-finite orientation'
        norm=math.sqrt(sum(float(v)**2 for v in quaternion))
        if not 0.9<=norm<=1.1:return 'invalid quaternion norm'
    if 'velocity' in required and not all(math.isfinite(float(v)) for v in velocity):
        return 'non-finite velocity'
    return None
