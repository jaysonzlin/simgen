from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from simgen.ngff_runtime.bridge import (
    build_dynamic_config,
    count_opacity_filtered_points,
    resolve_translations,
)


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
    cube = np.array(
        [[x, y, z] for x in (-0.01, 0.01) for y in (-0.01, 0.01) for z in (-0.01, 0.01)],
        dtype=np.float32,
    )
    points = {"ball_a": cube, "ball_b": cube}

    first = resolve_translations(Scene(), points)
    second = resolve_translations(Scene(), points)

    assert first == second
    assert np.linalg.norm(np.asarray(first[0]) - np.asarray(first[1])) >= 0.05


def test_bridge_uses_ngff_randomized_placement_order_from_gaussian_points() -> None:
    scene = SimpleNamespace(
        seed=1,
        objects=(
            Object("panda_0", "panda", scale=0.6),
            Object("ball_0", "ball", scale=0.4),
            Object("can_0", "can", scale=1.0),
        ),
    )
    cube = np.array(
        [[x, y, z] for x in (-0.01, 0.01) for y in (-0.01, 0.01) for z in (-0.01, 0.01)],
        dtype=np.float32,
    )

    translations = resolve_translations(
        scene,
        {"panda_0": cube, "ball_0": cube, "can_0": cube},
    )

    assert len(translations) == 3
    assert any(translation == (0.0, 0.0, 0.0) for translation in translations)
    assert translations[0] != (0.0, 0.0, 0.0)


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


def test_opacity_filtered_point_count_matches_ngff_strict_threshold() -> None:
    retained_opacities = np.array([0.01, 0.02, 0.5], dtype=np.float32)
    logits = np.log(retained_opacities / (1.0 - retained_opacities))

    assert count_opacity_filtered_points(logits, threshold=0.02) == 1
