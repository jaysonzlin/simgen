"""Seeded non-overlapping placement for resolved Gaussian asset instances."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class AssetBounds:
    center: np.ndarray
    radius: float


@dataclass(frozen=True)
class PlacementObject:
    instance_id: str
    asset: str
    scale: float = 1.0
    explicit_position: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class PlacedObject:
    instance_id: str
    asset: str
    scale: float
    position: tuple[float, float, float]


def resolve_placement(
    *,
    seed: int,
    objects: Sequence[PlacementObject],
    bounds: Mapping[str, AssetBounds],
    space_min: tuple[float, float, float] = (-0.9, -0.9, -0.9),
    space_max: tuple[float, float, float] = (0.9, 0.9, 0.9),
    max_attempts_per_object: int = 500,
) -> tuple[PlacedObject, ...]:
    """Resolve explicit or seeded positions, rejecting overlapping bounding spheres."""

    generator = np.random.default_rng(seed)
    placed: list[PlacedObject] = []

    for obj in objects:
        if obj.asset not in bounds:
            raise KeyError(f"no bounds were provided for asset {obj.asset!r}")
        asset_bounds = bounds[obj.asset]
        radius = float(asset_bounds.radius) * obj.scale
        if radius <= 0:
            raise ValueError(f"asset {obj.asset!r} has a non-positive scaled radius")

        candidates = (
            [obj.explicit_position]
            if obj.explicit_position is not None
            else [
                tuple(
                    float(value)
                    for value in generator.uniform(
                        np.asarray(space_min, dtype=np.float64) + radius,
                        np.asarray(space_max, dtype=np.float64) - radius,
                    )
                )
                for _ in range(max_attempts_per_object)
            ]
        )
        for candidate in candidates:
            assert candidate is not None
            position = np.asarray(candidate, dtype=np.float64)
            if np.any(position - radius < np.asarray(space_min)) or np.any(
                position + radius > np.asarray(space_max)
            ):
                continue
            if all(
                np.linalg.norm(position - np.asarray(other.position))
                >= radius + float(bounds[other.asset].radius) * other.scale
                for other in placed
            ):
                placed.append(
                    PlacedObject(
                        instance_id=obj.instance_id,
                        asset=obj.asset,
                        scale=obj.scale,
                        position=tuple(float(value) for value in position),
                    )
                )
                break
        else:
            raise ValueError(
                f"could not place object {obj.instance_id!r} without overlap after "
                f"{len(candidates)} attempt(s)"
            )

    return tuple(placed)
