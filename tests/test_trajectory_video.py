from __future__ import annotations

import numpy as np

from simgen.outputs import write_object_hdf5
from simgen.trajectory_video import compute_point_colors, load_trajectories_for_video


def test_video_loader_preserves_object_trajectory_order(tmp_path) -> None:
    paths = []
    for index, color in enumerate(([1, 0, 0], [0, 1, 0])):
        path = tmp_path / f"{index:03d}" / "pc.hdf5"
        write_object_hdf5(
            path,
            np.zeros((1, 1, 2048, 3), dtype=np.float32),
            np.tile(np.asarray(color, dtype=np.float32), (2048, 1)),
            np.zeros((1, 3), dtype=np.float32),
            np.zeros((1, 3), dtype=np.float32),
        )
        paths.append(path)

    trajectories = load_trajectories_for_video(paths)

    assert trajectories.points.shape == (1, 2, 2048, 3)


def test_video_colors_encode_relative_initial_height_with_kubric_viridis_scheme() -> None:
    points = np.zeros((1, 1, 2, 3), dtype=np.float32)
    points[0, 0, :, 2] = [2.0, 4.0]

    colors = compute_point_colors(points)

    np.testing.assert_allclose(colors[0, 0], [0.267004, 0.004874, 0.329415, 1.0], rtol=1e-5)
    np.testing.assert_allclose(colors[0, 1], [0.993248, 0.906157, 0.143936, 1.0], rtol=1e-5)
