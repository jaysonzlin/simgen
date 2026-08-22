"""Ordered single-sample orchestration with an injectable GPU runtime."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import tempfile
from typing import Protocol

import h5py
import numpy as np

from .config import load_scene
from .metadata import write_metadata
from .outputs import write_object_hdf5
from .sampling import build_object_trajectory, farthest_point_indices, sh_dc_to_rgb


COMPLETION_MARKER = ".simgen_complete"


@dataclass(frozen=True)
class RawSimulation:
    positions: np.ndarray
    sh_dc: np.ndarray
    point_counts: list[int]


@dataclass(frozen=True)
class RenderedView:
    images: list[np.ndarray]
    depth: np.ndarray
    alpha: np.ndarray
    camera: dict[str, object]


class Runtime(Protocol):
    def simulate(self, scene, workdir: Path) -> RawSimulation: ...

    def render(self, scene, raw: RawSimulation) -> RenderedView: ...


def _write_view(output: Path, view: RenderedView, frames: int) -> None:
    from PIL import Image

    if len(view.images) != frames:
        raise ValueError(f"renderer returned {len(view.images)} images for {frames} frames")
    if view.depth.shape[0] != frames or view.alpha.shape != view.depth.shape:
        raise ValueError("renderer depth and alpha must be aligned to the configured frame count")
    root = output / "view_0"
    root.mkdir(parents=True)
    for frame, image in enumerate(view.images):
        Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(
            root / f"{frame:08d}.png"
        )
    with h5py.File(root / "depth.h5", "w") as depth_file:
        depth_file.create_dataset("depth", data=np.asarray(view.depth, dtype=np.float32), compression="gzip")
        depth_file.create_dataset("alpha", data=np.asarray(view.alpha, dtype=np.float32), compression="gzip")
    (root / "cameras.json").write_text(json.dumps([view.camera] * frames, indent=2) + "\n")


def _write_trajectories(output: Path, scene, raw: RawSimulation) -> list[dict[str, object]]:
    if raw.positions.ndim != 3 or raw.positions.shape[2] != 3:
        raise ValueError("raw positions must have shape (frames, points, 3)")
    if len(raw.point_counts) != len(scene.objects) or sum(raw.point_counts) != raw.positions.shape[1]:
        raise ValueError("runtime point counts must exactly partition the raw points by scene object")
    if raw.sh_dc.shape != (raw.positions.shape[1], 3):
        raise ValueError("raw sh_dc must align one-to-one with raw simulation points")

    metadata_instances = []
    start = 0
    for index, (object_spec, count) in enumerate(zip(scene.objects, raw.point_counts, strict=True)):
        end = start + count
        local_indices = farthest_point_indices(raw.positions[0, start:end], 2048)
        trajectory = build_object_trajectory(raw.positions, (start, end), local_indices)
        rgb = sh_dc_to_rgb(raw.sh_dc[start:end][local_indices])
        write_object_hdf5(
            output / "objects" / f"{index:03d}" / "pc.hdf5",
            trajectory,
            rgb,
            np.zeros((1, 3), dtype=np.float32),
            np.zeros((1, 3), dtype=np.float32),
        )
        metadata_instances.append(
            {
                "id": f"{index:03d}",
                "instance_id": object_spec.instance_id,
                "name": object_spec.asset,
                "source_point_range": [start, end],
                "fps_indices": local_indices.tolist(),
            }
        )
        start = end
    return metadata_instances


def run(
    scene_path: Path,
    output: Path,
    *,
    resume: bool,
    force: set[str],
    runtime: Runtime,
    cli_overrides: dict[str, object] | None = None,
) -> Path:
    """Generate one validated compact sample using a concrete runtime implementation."""

    if force:
        raise NotImplementedError("selective --force execution will be enabled with remote stage adapters")
    destination = Path(output)
    if destination.exists():
        if resume and (destination / COMPLETION_MARKER).is_file():
            return destination
        if (destination / COMPLETION_MARKER).is_file():
            raise FileExistsError(f"output already exists: {destination}")
        shutil.rmtree(destination)

    scene = load_scene(Path(scene_path), cli_overrides=cli_overrides or {})
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.simgen-", dir=destination.parent))
    try:
        shutil.copy2(scene.source_path, staging / "scene.yaml")
        raw = runtime.simulate(scene, staging)
        instances = _write_trajectories(staging, scene, raw)
        view = runtime.render(scene, raw)
        _write_view(staging, view, scene.timeline.frames)
        if scene.outputs.rgb_video:
            from .rgb_video import write_rgb_video

            write_rgb_video(view.images, staging / "view_0" / "rgb.mp4", scene.timeline.fps)
        if scene.outputs.trajectory_video:
            from .trajectory_video import render_trajectory_video

            render_trajectory_video(
                [staging / "objects" / f"{index:03d}" / "pc.hdf5" for index in range(len(scene.objects))],
                staging / "pc_trajectory.mp4",
                scene.outputs.trajectory_video_fps,
            )
        write_metadata(staging / "metadata.json", scene)
        metadata_path = staging / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["instances"] = instances
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        if not scene.outputs.keep_simulation:
            raw_ngff = staging / "raw_ngff"
            if raw_ngff.exists():
                shutil.rmtree(raw_ngff)
        (staging / COMPLETION_MARKER).touch()
        staging.replace(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination
