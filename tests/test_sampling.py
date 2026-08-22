from __future__ import annotations

import numpy as np

from simgen.sampling import build_object_trajectory, farthest_point_indices, sh_dc_to_rgb


def test_fps_indices_selected_at_frame_zero_are_reused_for_every_frame() -> None:
    frames = np.array(
        [
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            [[10, 0, 0], [11, 0, 0], [10, 1, 0]],
        ],
        dtype=np.float32,
    )

    indices = farthest_point_indices(frames[0], 2)
    trajectory = build_object_trajectory(frames, (0, 3), indices)

    np.testing.assert_allclose(trajectory[:, 0], frames[:, indices])


def test_sh_dc_to_rgb_uses_ngff_conversion_and_clipping() -> None:
    dc = np.array([[-2.0, 0.0, 2.0]], dtype=np.float32)

    rgb = sh_dc_to_rgb(dc)

    np.testing.assert_allclose(rgb, [[0.0, 0.5, 1.0]], atol=1e-6)
