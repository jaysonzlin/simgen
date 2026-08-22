from __future__ import annotations

import numpy as np

from simgen.outputs import write_object_hdf5
from simgen.trajectory_video import load_trajectories_for_video


def test_video_loader_uses_stored_object_rgb(tmp_path) -> None:
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

    np.testing.assert_allclose(trajectories.colors[0, 0], [1, 0, 0])
    np.testing.assert_allclose(trajectories.colors[1, 0], [0, 1, 0])
