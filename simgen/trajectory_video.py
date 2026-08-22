"""Optional Kubric-style visual inspection of final object trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np


@dataclass(frozen=True)
class VideoTrajectories:
    points: np.ndarray


def load_trajectories_for_video(paths: Sequence[Path]) -> VideoTrajectories:
    """Read final HDF5 trajectories in lexical object order."""

    points = []
    expected_frames: int | None = None
    for path in sorted(Path(item) for item in paths):
        with h5py.File(path) as source:
            trajectory = np.asarray(source["point_cloud"], dtype=np.float32)
        if trajectory.ndim != 4 or trajectory.shape[1:] != (1, 2048, 3):
            raise ValueError(f"{path}: invalid point_cloud shape {trajectory.shape}")
        if expected_frames is None:
            expected_frames = trajectory.shape[0]
        elif trajectory.shape[0] != expected_frames:
            raise ValueError("all object trajectories must have the same frame count")
        points.append(trajectory[:, 0])
    if not points:
        raise ValueError("at least one object trajectory is required")
    return VideoTrajectories(points=np.stack(points, axis=1))


def compute_point_colors(pc_data: np.ndarray) -> np.ndarray:
    """Color each object by its points' relative initial heights, as Kubric does."""

    import matplotlib.pyplot as plt

    initial_heights = pc_data[0, :, :, 2]
    min_heights = initial_heights.min(axis=1, keepdims=True)
    height_ranges = np.ptp(initial_heights, axis=1, keepdims=True)
    normalized_heights = np.full(initial_heights.shape, 0.5, dtype=np.float64)
    np.divide(
        initial_heights - min_heights,
        height_ranges,
        out=normalized_heights,
        where=height_ranges > 0,
    )
    return plt.get_cmap("viridis")(normalized_heights)


def render_trajectory_video(
    object_paths: Sequence[Path], destination: Path, fps: int = 12
) -> Path:
    """Render final HDF5 trajectories with Kubric's 10-inch height-colored style."""

    import imageio.v2 as imageio
    import matplotlib.pyplot as plt

    trajectories = load_trajectories_for_video(object_paths)
    figure = plt.figure(figsize=(10, 10))
    axes = figure.add_subplot(111, projection="3d")
    flattened = trajectories.points.reshape(-1, 3)
    minimum, maximum = flattened.min(axis=0), flattened.max(axis=0)
    midpoint = (minimum + maximum) / 2
    span = max(maximum[0] - minimum[0], maximum[1] - minimum[1], maximum[2]) + 1.0
    x_limits = (midpoint[0] - span / 2, midpoint[0] + span / 2)
    y_limits = (midpoint[1] - span / 2, midpoint[1] + span / 2)
    z_limits = (0.0, span)
    point_colors = compute_point_colors(trajectories.points)
    colorbar_mappable = plt.cm.ScalarMappable(
        norm=plt.Normalize(vmin=0.0, vmax=1.0), cmap="viridis"
    )
    colorbar_mappable.set_array([])
    colorbar = figure.colorbar(colorbar_mappable, ax=axes, pad=0.1, shrink=0.7)
    colorbar.set_label("Relative initial height")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(destination, fps=fps) as writer:
        for frame in range(trajectories.points.shape[0]):
            axes.clear()
            axes.set_xlim(x_limits)
            axes.set_ylim(y_limits)
            axes.set_zlim(z_limits)
            axes.set_box_aspect((1, 1, 1))
            axes.set_xlabel("X")
            axes.set_ylabel("Y")
            axes.set_zlabel("Z")
            axes.set_title(
                f"Point Cloud Trajectories - Frame {frame:03d} / "
                f"{trajectories.points.shape[0] - 1:03d}",
                fontsize=14,
            )
            axes.grid(True)
            for object_index in range(trajectories.points.shape[1]):
                points = trajectories.points[frame, object_index]
                axes.scatter(
                    points[:, 0],
                    points[:, 1],
                    points[:, 2],
                    c=point_colors[object_index],
                    s=4,
                    alpha=0.8,
                    edgecolors="none",
                    label=f"Object {object_index} (Instance {object_index})",
                )
            axes.legend(loc="upper right")
            figure.canvas.draw()
            writer.append_data(np.asarray(figure.canvas.buffer_rgba())[:, :, :3])
    plt.close(figure)
    return destination
