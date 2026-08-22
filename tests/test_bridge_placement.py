from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from simgen.ngff_runtime.bridge import build_dynamic_config, resolve_translations
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


def test_dynamic_config_uses_declared_panda_ball_can_material_values() -> None:
    scene = SimpleNamespace(
        timeline=SimpleNamespace(substep_dt=0.0001, frame_dt=1 / 24, frames=49),
        objects=(
            SimpleNamespace(asset="panda", physics={"E": 1e5, "density": 700}),
            SimpleNamespace(asset="ball", physics={"E": 8e5, "density": 600}),
            SimpleNamespace(asset="can", physics={"E": 1e6, "density": 800}),
        ),
        physics=SimpleNamespace(
            overrides={
                "nu": 0.3,
                "rpic_damping": 0.9,
                "n_grid": 200,
                "grid_lim": 4,
                "material": "jelly",
                "g": [0, 0, -5],
            }
        ),
    )

    config = build_dynamic_config(scene)

    assert config["E"] == {"panda": 1e5, "ball": 8e5, "can": 1e6}
    assert config["density"] == {"panda": 700, "ball": 600, "can": 800}
    assert config["frame_num"] == 49
    assert config["frame_dt"] == 1 / 24
    assert config["n_grid"] == 200
