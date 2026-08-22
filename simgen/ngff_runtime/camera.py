"""Stationary view-0 camera records without importing CUDA dependencies."""

from __future__ import annotations

from dataclasses import asdict, dataclass


CAMERA_ZERO_NAME = "view_0"
SIMGEN_CAMERA_V1 = {
    "name": "simgen_camera_v1",
    "default_camera_index": -1,
    "init_azimuthm": 120.0,
    "init_elevation": 20.0,
    "init_radius": 4.0,
    "move_camera": False,
    "delta_a": 0.0,
    "delta_e": 0.0,
    "delta_r": 0.0,
    "fov_scale": 1.5,
}


@dataclass(frozen=True)
class CameraRecord:
    rotation: tuple[tuple[float, float, float], ...]
    position: tuple[float, float, float]
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def stationary_camera_records(record: dict[str, object], frames: int) -> list[dict[str, object]]:
    """Repeat the rendered camera-0 pose once for every output frame."""

    if frames < 1:
        raise ValueError("frames must be positive")
    return [dict(record) for _ in range(frames)]
