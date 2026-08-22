"""Asset point-range manifests built before MPM can reorder particles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ManifestInstance:
    instance_id: str
    asset: str
    ordinal: str
    point_range: tuple[int, int]


@dataclass(frozen=True)
class CombinedAssetManifest:
    total_points: int
    instances: tuple[ManifestInstance, ...]


def build_manifest(
    entries: Iterable[tuple[str, str, int]]
) -> CombinedAssetManifest:
    """Assign contiguous half-open point ranges in declaration order."""

    cursor = 0
    instances: list[ManifestInstance] = []
    for index, (instance_id, asset, point_count) in enumerate(entries):
        if point_count < 1:
            raise ValueError(f"{instance_id}: point_count must be positive")
        instances.append(
            ManifestInstance(
                instance_id=instance_id,
                asset=asset,
                ordinal=f"{index:03d}",
                point_range=(cursor, cursor + point_count),
            )
        )
        cursor += point_count
    return CombinedAssetManifest(total_points=cursor, instances=tuple(instances))


def resolve_asset_paths(assets_root: Path, objects: Sequence[object]) -> tuple[Path, ...]:
    """Resolve standard NGFF asset PLY paths with instance-aware errors."""

    paths = []
    for item in objects:
        instance_id = getattr(item, "instance_id")
        asset = getattr(item, "asset")
        path = Path(assets_root) / asset / "point_cloud" / "iteration_30000" / "point_cloud.ply"
        if not path.is_file():
            raise FileNotFoundError(f"{instance_id} ({asset}): asset PLY not found: {path}")
        paths.append(path)
    return tuple(paths)
