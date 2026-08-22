"""Observation-only foreground point views from RGB, depth, and tracked masks."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Sequence

import h5py
import numpy as np


@dataclass(frozen=True)
class PointViewOptions:
    view: int = 0
    downsample_factor: int = 10


def _default_image_loader(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _unproject(depth: np.ndarray, mask: np.ndarray, camera: dict[str, object]) -> np.ndarray:
    rows, columns = np.nonzero(mask)
    values = depth[rows, columns]
    fx, fy = float(camera["fx"]), float(camera["fy"])
    cx, cy = float(camera.get("cx", 0.0)), float(camera.get("cy", 0.0))
    camera_points = np.stack(
        [(columns - cx) * values / fx, (rows - cy) * values / fy, values], axis=1
    ).astype(np.float32)
    rotation = np.asarray(camera["rotation"], dtype=np.float32)
    position = np.asarray(camera["position"], dtype=np.float32)
    return camera_points @ rotation.T + position


def export_point_views(
    render_dir: Path,
    *,
    detector: Callable[[np.ndarray], Sequence[str]],
    tracker: Callable[[Sequence[Path], Sequence[str]], Sequence[np.ndarray]],
    options: PointViewOptions,
    image_loader: Callable[[Path], np.ndarray] = _default_image_loader,
) -> list[Path]:
    """Write foreground-only HDF5 views from detector/tracker RGB masks and depth."""

    if options.downsample_factor < 1:
        raise ValueError("downsample_factor must be at least one")
    root = Path(render_dir)
    frame_paths = sorted(root.glob("*.png")) or sorted(root.glob("*.npy"))
    if not frame_paths:
        raise FileNotFoundError(f"no RGB frames found in {root}")
    cameras = json.loads((root / "cameras.json").read_text())
    if len(cameras) < len(frame_paths):
        raise ValueError("cameras.json must provide one camera per RGB frame")
    labels = list(detector(image_loader(frame_paths[0])))
    masks = list(tracker(frame_paths, labels))
    if len(masks) != len(frame_paths):
        raise ValueError("tracker must provide one mask per RGB frame")

    destination = root / "point_views"
    destination.mkdir(exist_ok=True)
    outputs: list[Path] = []
    with h5py.File(root / "depth.h5") as depth_file:
        depth_data = depth_file["depth"]
        for frame, frame_path in enumerate(frame_paths):
            rgb = np.asarray(image_loader(frame_path), dtype=np.uint8)
            depth = np.asarray(depth_data[frame, options.view], dtype=np.float32)
            mask = np.asarray(masks[frame], dtype=bool) & np.isfinite(depth) & (depth > 0)
            if rgb.shape != (*depth.shape, 3) or mask.shape != depth.shape:
                raise ValueError("RGB, depth, and tracked mask dimensions must match")
            xyz = _unproject(depth, mask, cameras[frame])[:: options.downsample_factor]
            colors = rgb[mask][:: options.downsample_factor]
            output = destination / f"{frame_path.stem}.h5"
            with h5py.File(output, "w") as result:
                result.create_dataset("xyz", data=xyz.astype(np.float32))
                result.create_dataset("rgb", data=colors.astype(np.uint8))
                result.attrs["frame"] = frame
                result.attrs["view"] = options.view
                result.attrs["detected_labels"] = ",".join(labels)
            outputs.append(output)
    return outputs
