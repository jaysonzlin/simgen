from __future__ import annotations

import numpy as np

from simgen.placement import AssetBounds, PlacementObject, resolve_placement


def test_seeded_placement_is_reproducible_and_non_overlapping() -> None:
    bounds = {"ball": AssetBounds(center=np.zeros(3), radius=0.2)}
    objects = [PlacementObject("a", "ball"), PlacementObject("b", "ball")]

    first = resolve_placement(seed=17, objects=objects, bounds=bounds)
    second = resolve_placement(seed=17, objects=objects, bounds=bounds)

    assert first == second
    assert np.linalg.norm(np.asarray(first[0].position) - np.asarray(first[1].position)) >= 0.4
