"""Optional RGB-faithful visual inspection of final object trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np


@dataclass(frozen=True)
class VideoTrajectories:
    points: np.ndarray
    colors: np.ndarray


def load_trajectories_for_video(paths: Sequence[Path]) -> VideoTrajectories:
    """Read final HDF5s in lexical order and preserve their stored RGB colors."""

    points, colors = [], []
    expected_frames: int | None = None
    for path in sorted(Path(item) for item in paths):
        with h5py.File(path) as source:
            trajectory = np.asarray(source["point_cloud"], dtype=np.float32)
            rgb = np.asarray(source["rgb"], dtype=np.float32)
        if trajectory.ndim != 4 or trajectory.shape[1:] != (1, 2048, 3):
            raise ValueError(f"{path}: invalid point_cloud shape {trajectory.shape}")
        if rgb.shape != (2048, 3):
            raise ValueError(f"{path}: invalid rgb shape {rgb.shape}")
        if expected_frames is None:
            expected_frames = trajectory.shape[0]
        elif trajectory.shape[0] != expected_frames:
            raise ValueError("all object trajectories must have the same frame count")
        points.append(trajectory[:, 0])
        colors.append(rgb)
    if not points:
        raise ValueError("at least one object trajectory is required")
    return VideoTrajectories(points=np.stack(points, axis=1), colors=np.stack(colors, axis=0))


def render_trajectory_video(object_paths: Sequence[Path], destination: Path, fps: int) -> Path:
    """Render final HDF5 trajectories with their stored Gaussian RGB colors."""

    import imageio.v2 as imageio
    import matplotlib.pyplot as plt

    trajectories = load_trajectories_for_video(object_paths)
    figure = plt.figure(figsize=(8, 8))
    axes = figure.add_subplot(111, projection="3d")
    flattened = trajectories.points.reshape(-1, 3)
    lower, upper = flattened.min(axis=0), flattened.max(axis=0)
    center = (lower + upper) / 2
    span = max(float(np.max(upper - lower)), 1e-3)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(destination, fps=fps) as writer:
        for frame in range(trajectories.points.shape[0]):
            axes.clear()
            axes.set_xlim(center[0] - span / 2, center[0] + span / 2)
            axes.set_ylim(center[1] - span / 2, center[1] + span / 2)
            axes.set_zlim(center[2] - span / 2, center[2] + span / 2)
            for object_index in range(trajectories.points.shape[1]):
                axes.scatter(
                    *trajectories.points[frame, object_index].T,
                    c=trajectories.colors[object_index],
                    s=2,
                    edgecolors="none",
                )
            figure.canvas.draw()
            writer.append_data(np.asarray(figure.canvas.buffer_rgba())[:, :, :3])
    plt.close(figure)
    return destination
