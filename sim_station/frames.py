"""SE(3) checks. A pose is a 4x4 rigid transform: rotation orthonormal, determinant +1."""

from __future__ import annotations

import numpy as np


def rotation_ok(R: np.ndarray, tol: float = 1e-6) -> bool:
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3) or not np.isfinite(R).all():
        return False
    if abs(float(np.linalg.det(R)) - 1.0) > tol:
        return False
    return float(np.linalg.norm(R.T @ R - np.eye(3))) <= tol


def pose_ok(T: np.ndarray, tol: float = 1e-6) -> bool:
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        return False
    if not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=tol):
        return False
    return rotation_ok(T[:3, :3], tol)


def invert_pose(T: np.ndarray) -> np.ndarray:
    R = np.asarray(T, dtype=float)[:3, :3]
    p = np.asarray(T, dtype=float)[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ p
    return out


def compose_pose(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.asarray(A, dtype=float) @ np.asarray(B, dtype=float)
