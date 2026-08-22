from __future__ import annotations

import numpy as np

from simgen.placement import (
    AssetBounds,
    NgffPlacementObject,
    PlacementObject,
    resolve_ngff_placement,
    resolve_placement,
)


def test_seeded_placement_is_reproducible_and_non_overlapping() -> None:
    bounds = {"ball": AssetBounds(center=np.zeros(3), radius=0.2)}
    objects = [PlacementObject("a", "ball"), PlacementObject("b", "ball")]

    first = resolve_placement(seed=17, objects=objects, bounds=bounds)
    second = resolve_placement(seed=17, objects=objects, bounds=bounds)

    assert first == second
    assert np.linalg.norm(np.asarray(first[0].position) - np.asarray(first[1].position)) >= 0.4


def test_ngff_placement_randomizes_placement_order_but_returns_yaml_order() -> None:
    cube = np.array(
        [[x, y, z] for x in (-0.01, 0.01) for y in (-0.01, 0.01) for z in (-0.01, 0.01)],
        dtype=np.float32,
    )
    objects = [
        NgffPlacementObject("panda_0", "panda", cube, 0.6),
        NgffPlacementObject("ball_0", "ball", cube, 0.4),
        NgffPlacementObject("can_0", "can", cube, 1.0),
    ]

    result = resolve_ngff_placement(seed=1, objects=objects)

    assert [item.instance_id for item in result.placements] == [item.instance_id for item in objects]
    assert result.placement_order != tuple(item.instance_id for item in objects)
    origin_instance = result.placement_order[0]
    position_by_instance = {item.instance_id: item.position for item in result.placements}
    assert position_by_instance[origin_instance] == (0.0, 0.0, 0.0)


def test_ngff_three_object_proposals_stay_near_a_previously_placed_object() -> None:
    cube = np.array(
        [[x, y, z] for x in (-0.01, 0.01) for y in (-0.01, 0.01) for z in (-0.01, 0.01)],
        dtype=np.float32,
    )
    result = resolve_ngff_placement(
        seed=9,
        objects=[
            NgffPlacementObject("a", "a", cube),
            NgffPlacementObject("b", "b", cube),
            NgffPlacementObject("c", "c", cube),
        ],
    )

    positions = {item.instance_id: np.asarray(item.position) for item in result.placements}
    for index, instance_id in enumerate(result.placement_order[1:], start=1):
        position = positions[instance_id]
        previous_positions = [positions[previous] for previous in result.placement_order[:index]]
        assert any(
            abs(position[0] - previous[0]) <= 0.1
            and abs(position[1] - previous[1]) <= 0.1
            for previous in previous_positions
        )
