"""SDK xyz/Euler-rpy covariance through O = A W inverse(A), NumPy only.

SDK17 PoseCovToXYZRollPitchYawCov exports Cartesian position and Euler roll,
pitch and yaw, not a rotation-vector covariance. E(R) maps Euler coordinate
increments to a small WORLD rotation. With c = R_W R_A.T p_A, the Jacobian is
[[R_A, R_A [c]x E(W)], [0, inverse(E(O)) R_A E(W)]]. Output is xyz/Euler-rpy.
The native SDK's near-gimbal branch discards rotation coordinates: refuse that
covariance rather than manufacturing certainty. This corrects the previous
rotation-vector assumption; see the October27 IMU investigation for evidence.
"""
from __future__ import annotations

import numpy as np


def skew(v):
    x, y, z = (float(c) for c in v)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _euler_axes(rotation, *, sdk_input=False):
    """World rotation-vector differential per xyz Euler coordinate increment."""
    R = np.asarray(rotation, float)
    if (R.shape != (3, 3) or not np.all(np.isfinite(R)) or
            not np.allclose(R.T @ R, np.eye(3), rtol=0, atol=1e-5) or
            not np.isclose(np.linalg.det(R), 1., rtol=0, atol=1e-5)):
        raise ValueError('invalid pose rotation')
    # SDK determinant = sin(pitch)/2; its branch tolerance is1e-5 in float32.
    # Allow float32 rounding when deciding whether the native branch lost data.
    if sdk_input and 1. - abs(float(R[2, 0])) <= 2. * (1e-5 + 2. * np.finfo(np.float32).eps):
        raise ValueError('SDK Euler covariance unavailable near pitch singularity')
    cp = float(np.hypot(R[0, 0], R[1, 0]))
    if cp < 1e-7:
        raise ValueError('output Euler covariance singularity')
    cy, sy, sp = R[0, 0] / cp, R[1, 0] / cp, -R[2, 0]
    return np.array([[cy * cp, -sy, 0.], [sy * cp, cy, 0.], [-sp, 0., 1.]])


def pose_jacobian(A, W):
    """Derivative of output xyz/rpy w.r.t. SDK xyz/rpy, not rotation vectors."""
    A, W = np.asarray(A, float), np.asarray(W, float)
    if A.shape != (4, 4) or W.shape != (4, 4) or not np.isfinite(A).all() or not np.isfinite(W).all():
        raise ValueError('pose transforms must be finite4x4 matrices')
    Ra, pa, Rw = A[:3, :3], A[:3, 3], W[:3, :3]
    if not np.allclose(Ra.T @ Ra, np.eye(3), rtol=0, atol=1e-5) or not np.isclose(np.linalg.det(Ra), 1., rtol=0, atol=1e-5):
        raise ValueError('invalid mount rotation')
    Ei = _euler_axes(Rw, sdk_input=True)
    Eo = _euler_axes(Ra @ Rw @ Ra.T)
    c = Rw @ Ra.T @ pa
    J = np.zeros((6, 6))
    J[:3, :3] = Ra
    J[:3, 3:] = Ra @ skew(c) @ Ei
    J[3:, 3:] = np.linalg.solve(Eo, Ra @ Ei)
    return J


def transform_pose_covariance(C, A, W, rel_tol=1e-9):
    """Sigma_W (6x6, xyz-rpy) -> Sigma_O. Raises ValueError for a non-finite or
    clearly indefinite input (that is not a covariance; nothing should be published)."""
    C = np.asarray(C, float).reshape(6, 6)
    if not np.all(np.isfinite(C)):
        raise ValueError("non-finite covariance")
    Cs = 0.5 * (C + C.T)
    scale = max(float(np.max(np.abs(np.diag(Cs)))), 1e-30)
    if float(np.linalg.eigvalsh(Cs).min()) < -1e-6 * scale:
        raise ValueError("input covariance is not positive semidefinite")
    J = pose_jacobian(A, W)
    out = J @ Cs @ J.T
    out = 0.5 * (out + out.T)
    w, V = np.linalg.eigh(out)
    if w.min() < 0.0:                                   # round-off only (a congruence preserves PSD)
        w = np.maximum(w, 0.0)
        out = (V * w) @ V.T
        out = 0.5 * (out + out.T)
    return out


def is_covariance(C, tol=1e-9):
    C = np.asarray(C, float)
    if C.shape != (6, 6) or not np.all(np.isfinite(C)) or not np.allclose(C, C.T, atol=tol):
        return False
    scale = max(float(np.max(np.abs(np.diag(C)))), 1e-30)
    return float(np.linalg.eigvalsh(C).min()) >= -tol * scale
