"""YAML scene parsing, defaults, precedence, and user-facing validation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import yaml

from .models import (
    ModelPaths,
    ObjectSpec,
    OutputSpec,
    PhysicsSpec,
    Pose,
    RenderSpec,
    ResolvedScene,
    Timeline,
)


DEFAULT_ASSETS_ROOT = Path(__file__).resolve().parents[1] / "data" / "GSCollision" / "objects"


def _mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def _positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _positive_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{field_name} must be a positive number")
    return float(value)


def _vector(value: object, field_name: str) -> tuple[float, float, float]:
    if not isinstance(value, list | tuple) or len(value) != 3:
        raise ValueError(f"{field_name} must contain exactly three numbers")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ValueError(f"{field_name} must contain only numbers")
    return tuple(float(item) for item in value)


def _optional_path(value: object) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("model paths must be non-empty strings")
    return Path(value)


def _model_path(
    key: str, models: Mapping[str, Any], cli_overrides: Mapping[str, object], env_name: str
) -> Path | None:
    cli_value = cli_overrides.get(key)
    if cli_value is not None:
        return _optional_path(cli_value)
    if models.get(key) is not None:
        return _optional_path(models[key])
    return _optional_path(os.environ.get(env_name))


def _parse_pose(value: object, instance_id: str) -> Pose | None:
    if value is None:
        return None
    pose = _mapping(value, f"objects[{instance_id}].pose")
    return Pose(
        position=_vector(pose.get("position"), f"objects[{instance_id}].pose.position"),
        rotation=_vector(
            pose.get("rotation", [0.0, 0.0, 0.0]),
            f"objects[{instance_id}].pose.rotation",
        ),
    )


def _parse_objects(data: Mapping[str, Any]) -> tuple[ObjectSpec, ...]:
    raw_objects = data.get("objects")
    if not isinstance(raw_objects, list) or not raw_objects:
        raise ValueError("objects must be a non-empty list")

    result = []
    seen_ids: set[str] = set()
    for index, raw_object in enumerate(raw_objects):
        object_data = _mapping(raw_object, f"objects[{index}]")
        instance_id = object_data.get("id")
        asset = object_data.get("asset")
        if not isinstance(instance_id, str) or not instance_id:
            raise ValueError(f"objects[{index}].id must be a non-empty string")
        if instance_id in seen_ids:
            raise ValueError(f"duplicate object id: {instance_id}")
        if not isinstance(asset, str) or not asset:
            raise ValueError(f"objects[{index}].asset must be a non-empty string")
        scale_value = object_data.get("scale")
        result.append(
            ObjectSpec(
                instance_id=instance_id,
                asset=asset,
                pose=_parse_pose(object_data.get("pose"), instance_id),
                scale=(
                    _positive_float(scale_value, f"objects[{index}].scale")
                    if scale_value is not None
                    else None
                ),
                physics=_mapping(object_data.get("physics"), f"objects[{index}].physics"),
            )
        )
        seen_ids.add(instance_id)
    return tuple(result)


def load_scene(path: Path, *, cli_overrides: Mapping[str, object]) -> ResolvedScene:
    """Load a YAML scene and resolve portable defaults without GPU imports."""

    source_path = Path(path).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"scene YAML does not exist: {source_path}")
    loaded = yaml.safe_load(source_path.read_text())
    data = _mapping(loaded, "scene")

    seed = data.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    assets_root = Path(data.get("assets_root", DEFAULT_ASSETS_ROOT)).expanduser()
    if not assets_root.is_absolute():
        assets_root = (source_path.parent / assets_root).resolve()
    if not assets_root.is_dir():
        raise FileNotFoundError(f"assets_root does not exist: {assets_root}")

    timeline_data = _mapping(data.get("timeline"), "timeline")
    timeline = Timeline(
        frames=_positive_int(timeline_data.get("frames", 49), "timeline.frames"),
        fps=_positive_int(timeline_data.get("fps", 24), "timeline.fps"),
        substep_dt=_positive_float(
            timeline_data.get("substep_dt", 0.0001), "timeline.substep_dt"
        ),
    )
    render_data = _mapping(data.get("render"), "render")
    background_name = render_data.get("background")
    if background_name is not None and (not isinstance(background_name, str) or not background_name):
        raise ValueError("render.background must be a non-empty background name")
    background_path = (
        assets_root.parent / "backgrounds" / background_name if background_name is not None else None
    )
    if background_path is not None and not background_path.is_dir():
        raise FileNotFoundError(f"render.background does not exist: {background_path}")
    render = RenderSpec(
        width=_positive_int(render_data.get("width", 480), "render.width"),
        height=_positive_int(render_data.get("height", 480), "render.height"),
        camera_preset=str(render_data.get("camera_preset", "simgen_camera_v1")),
        background_path=background_path,
    )
    outputs_data = _mapping(data.get("outputs"), "outputs")
    outputs = OutputSpec(
        keep_simulation=bool(outputs_data.get("keep_simulation", False)),
        point_views=bool(outputs_data.get("point_views", False)),
        trajectory_video=bool(outputs_data.get("trajectory_video", False)),
        trajectory_video_fps=_positive_int(
            outputs_data.get("trajectory_video_fps", 12), "outputs.trajectory_video_fps"
        ),
    )
    models_data = _mapping(data.get("models"), "models")
    models = ModelPaths(
        grounding_dino_model_dir=_model_path(
            "grounding_dino_model_dir",
            models_data,
            cli_overrides,
            "SIMGEN_GROUNDING_DINO_MODEL_DIR",
        ),
        sam2_config=_model_path("sam2_config", models_data, cli_overrides, "SIMGEN_SAM2_CONFIG"),
        sam2_checkpoint=_model_path(
            "sam2_checkpoint", models_data, cli_overrides, "SIMGEN_SAM2_CHECKPOINT"
        ),
    )
    physics_data = _mapping(data.get("physics"), "physics")
    profile = physics_data.get("profile", "ngff_dynamic")
    if not isinstance(profile, str) or not profile:
        raise ValueError("physics.profile must be a non-empty string")

    return ResolvedScene(
        source_path=source_path,
        seed=seed,
        assets_root=assets_root,
        objects=_parse_objects(data),
        timeline=timeline,
        render=render,
        outputs=outputs,
        models=models,
        physics=PhysicsSpec(profile=profile, overrides=physics_data.get("overrides", {})),
        source_data=data,
    )
