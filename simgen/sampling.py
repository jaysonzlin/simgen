"""Persistent object-point selection and Gaussian DC color conversion."""

from __future__ import annotations

import numpy as np


SH_C0 = np.float32(0.28209479177387814)


def farthest_point_indices(points: np.ndarray, count: int) -> np.ndarray:
    """Return deterministic greedy FPS indices for a single `(points, 3)` frame."""

    coordinates = np.asarray(points, dtype=np.float32)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {coordinates.shape}")
    if not 1 <= count <= len(coordinates):
        raise ValueError(f"count must be between 1 and {len(coordinates)}, got {count}")
    if not np.isfinite(coordinates).all():
        raise ValueError("points must be finite")

    selected = np.empty(count, dtype=np.int64)
    selected[0] = 0
    distances = np.sum((coordinates - coordinates[0]) ** 2, axis=1)
    for index in range(1, count):
        selected[index] = int(np.argmax(distances))
        distances = np.minimum(
            distances,
            np.sum((coordinates - coordinates[selected[index]]) ** 2, axis=1),
        )
    return selected


def build_object_trajectory(
    raw_positions: np.ndarray, point_range: tuple[int, int], local_indices: np.ndarray
) -> np.ndarray:
    """Apply frame-zero local point indices to a global raw simulation trajectory."""

    frames = np.asarray(raw_positions, dtype=np.float32)
    start, end = point_range
    indices = np.asarray(local_indices, dtype=np.int64)
    if frames.ndim != 3 or frames.shape[2] != 3:
        raise ValueError(f"raw_positions must have shape (T, N, 3), got {frames.shape}")
    if start < 0 or end <= start or end > frames.shape[1]:
        raise ValueError(f"invalid point range {point_range} for {frames.shape[1]} points")
    if indices.ndim != 1 or not len(indices):
        raise ValueError("local_indices must be a non-empty one-dimensional array")
    if np.any(indices < 0) or np.any(indices >= end - start):
        raise ValueError("local_indices fall outside point_range")

    return frames[:, start:end][:, indices, :][:, np.newaxis, :, :]


def sh_dc_to_rgb(dc_coefficients: np.ndarray) -> np.ndarray:
    """Convert selected Gaussian DC SH coefficients to static RGB in `[0, 1]`."""

    dc = np.asarray(dc_coefficients, dtype=np.float32)
    if dc.ndim != 2 or dc.shape[1] != 3:
        raise ValueError(f"DC coefficients must have shape (N, 3), got {dc.shape}")
    return np.clip(dc * SH_C0 + 0.5, 0.0, 1.0).astype(np.float32, copy=False)
