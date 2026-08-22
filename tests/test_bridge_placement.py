from __future__ import annotations

import numpy as np

from simgen.ngff_runtime.bridge import resolve_translations
from simgen.placement import AssetBounds


class Object:
    def __init__(self, instance_id: str, asset: str, scale: float | None = None, pose=None):
        self.instance_id = instance_id
        self.asset = asset
        self.scale = scale
        self.pose = pose


class Scene:
    seed = 23
    objects = (Object("ball_a", "ball"), Object("ball_b", "ball"))


def test_bridge_uses_seeded_non_overlapping_positions_when_pose_is_omitted() -> None:
    bounds = {"ball": AssetBounds(center=np.zeros(3), radius=0.2)}

    first = resolve_translations(Scene(), bounds)
    second = resolve_translations(Scene(), bounds)

    assert first == second
    assert np.linalg.norm(np.asarray(first[0]) - np.asarray(first[1])) >= 0.4
