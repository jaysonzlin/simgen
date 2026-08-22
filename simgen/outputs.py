"""Canonical sample artifact writers and structural validation."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def _array(value: np.ndarray, shape: tuple[int | None, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(array.shape, shape, strict=True)
    ):
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def write_object_hdf5(
    path: Path,
    point_cloud: np.ndarray,
    rgb: np.ndarray,
    initial_linear_velocity: np.ndarray,
    initial_angular_velocity: np.ndarray,
) -> None:
    """Write the per-object trajectory contract consumed by downstream datasets."""

    trajectory = _array(point_cloud, (None, 1, 2048, 3), "point_cloud")
    colors = _array(rgb, (2048, 3), "rgb")
    if np.any(colors < 0.0) or np.any(colors > 1.0):
        raise ValueError("rgb must be within [0, 1]")
    linear = _array(initial_linear_velocity, (1, 3), "initial_linear_velocity")
    angular = _array(initial_angular_velocity, (1, 3), "initial_angular_velocity")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(destination, "w") as output:
        output.create_dataset("point_cloud", data=trajectory, compression="gzip")
        output.create_dataset("rgb", data=colors, compression="gzip")
        output.create_dataset("initial_linear_velocity", data=linear)
        output.create_dataset("initial_angular_velocity", data=angular)
