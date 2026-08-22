from __future__ import annotations

import h5py
import numpy as np

from simgen.point_views import PointViewOptions, export_point_views


def test_point_views_use_detector_tracker_masks_and_depth(tmp_path) -> None:
    view = tmp_path / "view_0"
    view.mkdir()
    rgb = np.array([[[255, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]], dtype=np.uint8)
    np.save(view / "00000000.npy", rgb)
    (view / "cameras.json").write_text(
        '[{"rotation": [[1,0,0],[0,1,0],[0,0,1]], "position": [0,0,0], "fx": 1, "fy": 1}]'
    )
    with h5py.File(view / "depth.h5", "w") as source:
        source.create_dataset("depth", data=np.ones((1, 1, 2, 2), dtype=np.float32))

    outputs = export_point_views(
        view,
        detector=lambda _: ["ball"],
        tracker=lambda _paths, _labels: [np.array([[True, False], [False, False]])],
        options=PointViewOptions(view=0, downsample_factor=1),
        image_loader=lambda path: np.load(path),
    )

    with h5py.File(outputs[0]) as result:
        assert result["xyz"].shape == (1, 3)
        assert result["rgb"][:].tolist() == [[255, 0, 0]]
        assert result.attrs["detected_labels"] == "ball"
