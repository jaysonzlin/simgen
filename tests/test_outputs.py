from __future__ import annotations

import h5py
import numpy as np

from simgen.outputs import write_object_hdf5


def test_object_hdf5_has_required_shapes_and_rgb(tmp_path) -> None:
    path = tmp_path / "pc.hdf5"
    write_object_hdf5(
        path,
        np.zeros((49, 1, 2048, 3), dtype=np.float32),
        np.ones((2048, 3), dtype=np.float32),
        np.zeros((1, 3), dtype=np.float32),
        np.zeros((1, 3), dtype=np.float32),
    )

    with h5py.File(path) as source:
        assert source["point_cloud"].shape == (49, 1, 2048, 3)
        assert source["point_cloud"].dtype == np.float32
        assert source["rgb"].shape == (2048, 3)
        assert source["initial_linear_velocity"].shape == (1, 3)
        assert source["initial_angular_velocity"].shape == (1, 3)
