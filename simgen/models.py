"""Immutable configuration values shared by every SimGen stage."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class Pose:
    """An optional user-authored object transform in world coordinates."""

    position: tuple[float, float, float]
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class ObjectSpec:
    """One declared object instance, in YAML declaration order."""

    instance_id: str
    asset: str
    pose: Pose | None = None
    scale: float | None = None
    physics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Timeline:
    frames: int = 49
    fps: int = 24
    substep_dt: float = 0.0001

    @property
    def frame_dt(self) -> float:
        return 1.0 / self.fps


@dataclass(frozen=True)
class RenderSpec:
    width: int = 480
    height: int = 480
    camera_preset: str = "simgen_camera_v1"
    background_path: Path | None = None


@dataclass(frozen=True)
class OutputSpec:
    keep_simulation: bool = False
    point_views: bool = False
    rgb_video: bool = False
    trajectory_video: bool = False
    trajectory_video_fps: int = 12


@dataclass(frozen=True)
class ModelPaths:
    grounding_dino_model_dir: Path | None = None
    sam2_config: Path | None = None
    sam2_checkpoint: Path | None = None


@dataclass(frozen=True)
class PhysicsSpec:
    profile: str = "ngff_dynamic"
    overrides: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedScene:
    """All author intent and defaults after path/configuration resolution."""

    source_path: Path
    seed: int
    assets_root: Path
    objects: tuple[ObjectSpec, ...]
    timeline: Timeline
    render: RenderSpec
    outputs: OutputSpec
    models: ModelPaths
    physics: PhysicsSpec
    source_data: Mapping[str, Any]
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable resolved configuration."""

        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if hasattr(value, "__dataclass_fields__"):
                return {key: convert(item) for key, item in asdict(value).items()}
            if isinstance(value, Mapping):
                return {str(key): convert(item) for key, item in value.items()}
            if isinstance(value, tuple | list):
                return [convert(item) for item in value]
            return value

        return convert(self)
