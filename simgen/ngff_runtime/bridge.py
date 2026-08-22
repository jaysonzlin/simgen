"""Bridge a SimGen YAML scene to the vendored NGFF command-line programs."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from ..assets import resolve_asset_paths
from .camera import SIMGEN_CAMERA_V1
from .simulation import run_mpm, run_render
from ..placement import AssetBounds, PlacementObject, resolve_placement

ASSET_IDS = {n: i for i, n in enumerate(("ball", "bear", "bowl", "can", "cloth", "cloth2", "cloth3", "duck", "duck2", "miku", "panda", "phone", "pillow", "pillow2", "rope", "rope2", "soccer", "toy"))}

NGFF_DYNAMIC_ELASTIC_MODULI = {"ball": 8e5, "bear": 5e4, "bowl": 1e7, "can": 1e6, "cloth": 3e3, "cloth2": 3e3, "cloth3": 3e3, "duck": 3e5, "duck2": 5e5, "miku": 1e5, "panda": 1e5, "phone": 2e7, "pillow": 1e4, "pillow2": 1e4, "rope": 5e3, "rope2": 5e3, "soccer": 3e6, "toy": 1e5}
NGFF_DYNAMIC_DENSITIES = {"ball": 600, "bear": 500, "bowl": 800, "can": 800, "cloth": 100, "cloth2": 100, "cloth3": 100, "duck": 300, "duck2": 1200, "miku": 400, "panda": 700, "phone": 2600, "pillow": 500, "pillow2": 500, "rope": 500, "rope2": 500, "soccer": 700, "toy": 500}


def count_opacity_filtered_points(opacity_logits: np.ndarray, *, threshold: float) -> int:
    """Count points retained by NGFF's opacity preprocessing."""

    opacities = 1.0 / (1.0 + np.exp(-np.asarray(opacity_logits, dtype=np.float32)))
    return int(np.count_nonzero(opacities > threshold))


def build_dynamic_config(scene) -> dict[str, object]:
    """Build the NGFF dynamic config while preserving scene-declared material values."""
    config = {
        "opacity_threshold": 0.02,
        "rotation_degree": [],
        "rotation_axis": [],
        "substep_dt": scene.timeline.substep_dt,
        "frame_dt": scene.timeline.frame_dt,
        "frame_num": scene.timeline.frames,
        "nu": 0.3,
        "rpic_damping": 0.9,
        "n_grid": 200,
        "grid_lim": 4,
        "material": "jelly",
        "g": [0, 0, -5],
        "sim_area": None,
        "boundary_conditions": [{"type": "bounding_box"}],
        "mpm_space_vertical_upward_axis": [0, 0, 1],
        "mpm_space_viewpoint_center": [2, 2, 0.7],
    }
    config.update(scene.physics.overrides)
    config["E"] = {
        item.asset: item.physics.get("E", NGFF_DYNAMIC_ELASTIC_MODULI.get(item.asset, 1e5))
        for item in scene.objects
    }
    config["density"] = {
        item.asset: item.physics.get("density", NGFF_DYNAMIC_DENSITIES.get(item.asset, 700))
        for item in scene.objects
    }
    config.update(SIMGEN_CAMERA_V1)
    return config

def resolve_translations(scene, bounds: dict[str, AssetBounds]) -> tuple[tuple[float, float, float], ...]:
    """Honor explicit poses and otherwise reproduce NGFF-style seeded placement."""
    placed = resolve_placement(
        seed=scene.seed,
        objects=[PlacementObject(item.instance_id, item.asset, item.scale or 1.0,
                                 item.pose.position if item.pose else None) for item in scene.objects],
        bounds=bounds,
    )
    return tuple(item.position for item in placed)

class RemoteNgffRuntime:
    def __init__(self): self.raw_dir = self.config = None; self.counts = []
    def simulate(self, scene, workdir: Path):
        from plyfile import PlyData
        from .vendor.dataset.generate_scene import combine_ply_files
        from ..pipeline import RawSimulation
        paths = resolve_asset_paths(scene.assets_root, scene.objects)
        if any(item.asset not in ASSET_IDS for item in scene.objects): raise ValueError("scene contains an unsupported NGFF asset")
        dynamic_config = build_dynamic_config(scene)
        opacity_threshold = float(dynamic_config["opacity_threshold"])
        ply_data = [PlyData.read(path).elements[0] for path in paths]
        self.counts = [
            count_opacity_filtered_points(data["opacity"], threshold=opacity_threshold)
            for data in ply_data
        ]
        bounds = {}
        for item, data in zip(scene.objects, ply_data, strict=True):
            points = np.column_stack((data["x"], data["y"], data["z"])).astype(np.float32)
            radius = float(np.linalg.norm((points.max(axis=0) - points.min(axis=0)) / 2.0))
            bounds[item.asset] = AssetBounds(center=points.mean(axis=0), radius=radius)
        root = workdir / "GSCollision"; model = root / "scenes" / str(len(paths)) / "sample"; cloud = model / "point_cloud" / "iteration_30000"; cloud.mkdir(parents=True)
        translations = resolve_translations(scene, bounds)
        combine_ply_files([str(p) for p in paths], str(cloud / "point_cloud.ply"), translations, [item.scale or 1.0 for item in scene.objects])
        (root / "scene_configs").mkdir(); (root / "scene_configs" / f"{len(paths)}.json").write_text(json.dumps({"sample": {"scene_object_idxs": [ASSET_IDS[x.asset] for x in scene.objects]}}))
        self.config = root / "dynamic_config.json"; self.config.write_text(json.dumps(dynamic_config))
        self.raw_dir = workdir / "raw_ngff"; run_mpm(model_path=model, config_path=self.config, output_path=self.raw_dir)
        import h5py
        frames = sorted(self.raw_dir.glob("[0-9][0-9][0-9][0-9].h5"))
        with h5py.File(self.raw_dir / "shs.h5") as source: sh = source["shs"][:,0,:].astype(np.float32)
        positions = []
        for frame in frames:
            with h5py.File(frame) as source: positions.append(source["pos"][:])
        return RawSimulation(np.asarray(positions, dtype=np.float32), sh, self.counts)
    def render(self, scene, raw):
        from PIL import Image
        import h5py
        from ..pipeline import RenderedView
        target = self.raw_dir.parent / "render"; run_render(model_path=self.raw_dir, config_path=self.config, output_path=target, background_path=scene.render.background_path)
        images = [np.asarray(Image.open(p).convert("RGB")) for p in sorted((target / "view_0").glob("*.png"))]
        with h5py.File(target / "depth.h5") as source: depth, alpha = source["depth"][:], source["alpha"][:]
        return RenderedView(images, depth, alpha, json.loads((target / "cameras.json").read_text())[0])
