"""Seeded non-overlapping placement for resolved Gaussian asset instances."""

from __future__ import annotations

from dataclasses import dataclass
import random
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


@dataclass(frozen=True)
class NgffPlacementObject:
    """An explicit instance with the Gaussian points used by NGFF placement."""

    instance_id: str
    asset: str
    points: np.ndarray
    scale: float = 1.0
    explicit_position: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class NgffPlacementResult:
    """Placements in YAML order plus the independently shuffled placement order."""

    placements: tuple[PlacedObject, ...]
    placement_order: tuple[str, ...]


def _ngff_geometry(object_spec: NgffPlacementObject) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    points = np.asarray(object_spec.points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError(f"{object_spec.instance_id}: points must have shape (N, 3)")
    if object_spec.scale <= 0:
        raise ValueError(f"{object_spec.instance_id}: scale must be positive")
    points = points * object_spec.scale
    lower, upper = points.min(axis=0), points.max(axis=0)
    radius = float(np.linalg.norm((upper - lower) / 2.0))
    return points, lower, upper, radius


def _aabbs_overlap(
    candidate_lower: np.ndarray,
    candidate_upper: np.ndarray,
    other_lower: np.ndarray,
    other_upper: np.ndarray,
) -> bool:
    return bool(np.all(candidate_lower <= other_upper) and np.all(candidate_upper >= other_lower))


def _ngff_point_collision(candidate: np.ndarray, other: np.ndarray, *, threshold: float = 0.05) -> bool:
    """Match NGFF's batched `torch.cdist` collision check on the available device."""

    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        candidate_tensor = torch.as_tensor(candidate, dtype=torch.float32, device=device)
        other_tensor = torch.as_tensor(other, dtype=torch.float32, device=device)
        with torch.no_grad():
            for start in range(0, len(candidate_tensor), 5000):
                if torch.any(torch.cdist(candidate_tensor[start : start + 5000], other_tensor) < threshold):
                    return True
        return False
    except ImportError:
        for start in range(0, len(candidate), 5000):
            distances = np.linalg.norm(candidate[start : start + 5000, None] - other[None], axis=2)
            if np.any(distances < threshold):
                return True
        return False


def resolve_ngff_placement(
    *,
    seed: int,
    objects: Sequence[NgffPlacementObject],
    max_attempts_per_object: int = 500,
    max_scene_restarts: int = 100,
) -> NgffPlacementResult:
    """Place explicit YAML objects using edited-NGFF's seeded scene policy.

    The objects are shuffled only for placement. Returned placements retain YAML order so
    Gaussian concatenation and per-object trajectory partitioning remain stable.
    """

    if not objects:
        raise ValueError("at least one object is required")
    if len({object_spec.instance_id for object_spec in objects}) != len(objects):
        raise ValueError("instance ids must be unique")

    py_rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    geometries = {object_spec.instance_id: _ngff_geometry(object_spec) for object_spec in objects}
    objects_by_id = {object_spec.instance_id: object_spec for object_spec in objects}
    automatic = [object_spec.instance_id for object_spec in objects if object_spec.explicit_position is None]
    explicit = [object_spec.instance_id for object_spec in objects if object_spec.explicit_position is not None]

    for _ in range(max_scene_restarts):
        placement_order = automatic.copy()
        py_rng.shuffle(placement_order)
        positions = {
            instance_id: np.asarray(objects_by_id[instance_id].explicit_position, dtype=np.float32)
            for instance_id in explicit
        }
        placed_order = explicit.copy()

        if placement_order:
            first_instance = placement_order[0]
            positions[first_instance] = np.zeros(3, dtype=np.float32)
            placed_order.append(first_instance)
            remaining = placement_order[1:]
        else:
            remaining = []

        for instance_id in remaining:
            _, lower, upper, radius = geometries[instance_id]
            placed = False
            for _ in range(max_attempts_per_object):
                # The original generator always enters this branch (`random.random() < 1`).
                py_rng.random()
                base_instance = py_rng.choice(placed_order)
                base_position = positions[base_instance]
                _, base_lower, base_upper, _ = geometries[base_instance]

                if len(objects) in (2, 3, 4):
                    candidate_position = np.array(
                        [
                            np_rng.uniform(base_position[0] - 0.1, base_position[0] + 0.1),
                            np_rng.uniform(base_position[1] - 0.1, base_position[1] + 0.1),
                            np_rng.uniform(-0.9, 0.9),
                        ],
                        dtype=np.float32,
                    )
                else:
                    base_side = base_upper - base_lower
                    candidate_side = upper - lower
                    candidate_position = np.array(
                        [
                            np_rng.uniform(
                                base_position[0] - base_side[0] / 2 - candidate_side[0] / 2,
                                base_position[0] + base_side[0] / 2 + candidate_side[0] / 2,
                            ),
                            np_rng.uniform(
                                base_position[1] - base_side[1] / 2 - candidate_side[1] / 2,
                                base_position[1] + base_side[1] / 2 + candidate_side[1] / 2,
                            ),
                            np_rng.uniform(-0.9, 0.9),
                        ],
                        dtype=np.float32,
                    )

                if np.any(candidate_position - radius < -0.9) or np.any(candidate_position + radius > 0.9):
                    continue
                candidate_points = geometries[instance_id][0] + candidate_position
                candidate_lower, candidate_upper = lower + candidate_position, upper + candidate_position
                collision = False
                for other_instance in placed_order:
                    other_points, other_lower, other_upper, _ = geometries[other_instance]
                    other_position = positions[other_instance]
                    if _aabbs_overlap(
                        candidate_lower,
                        candidate_upper,
                        other_lower + other_position,
                        other_upper + other_position,
                    ) and _ngff_point_collision(candidate_points, other_points + other_position):
                        collision = True
                        break
                if not collision:
                    positions[instance_id] = candidate_position
                    placed_order.append(instance_id)
                    placed = True
                    break
            if not placed:
                break
        else:
            final_positions = [positions[instance_id] for instance_id in placed_order]
            if all(
                np.linalg.norm(first - second) >= 0.05
                for index, first in enumerate(final_positions)
                for second in final_positions[index + 1 :]
            ):
                returned = tuple(
                    PlacedObject(
                        instance_id=object_spec.instance_id,
                        asset=object_spec.asset,
                        scale=object_spec.scale,
                        position=tuple(float(value) for value in positions[object_spec.instance_id]),
                    )
                    for object_spec in objects
                )
                return NgffPlacementResult(returned, tuple(placed_order))

    raise ValueError("NGFF placement could not find a collision-free scene after restarts")


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
