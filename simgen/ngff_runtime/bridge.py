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
        ply_data = [PlyData.read(path).elements[0] for path in paths]
        self.counts = [len(data) for data in ply_data]
        bounds = {}
        for item, data in zip(scene.objects, ply_data, strict=True):
            points = np.column_stack((data["x"], data["y"], data["z"])).astype(np.float32)
            radius = float(np.linalg.norm((points.max(axis=0) - points.min(axis=0)) / 2.0))
            bounds[item.asset] = AssetBounds(center=points.mean(axis=0), radius=radius)
        root = workdir / "GSCollision"; model = root / "scenes" / str(len(paths)) / "sample"; cloud = model / "point_cloud" / "iteration_30000"; cloud.mkdir(parents=True)
        translations = resolve_translations(scene, bounds)
        combine_ply_files([str(p) for p in paths], str(cloud / "point_cloud.ply"), translations, [item.scale or 1.0 for item in scene.objects])
        (root / "scene_configs").mkdir(); (root / "scene_configs" / f"{len(paths)}.json").write_text(json.dumps({"sample": {"scene_object_idxs": [ASSET_IDS[x.asset] for x in scene.objects]}}))
        self.config = root / "dynamic_config.json"; self.config.write_text(json.dumps({"opacity_threshold": 0.0, "rotation_degree": [], "rotation_axis": [], "substep_dt": scene.timeline.substep_dt, "frame_dt": scene.timeline.frame_dt, "frame_num": scene.timeline.frames, "E": {x.asset: 1e5 for x in scene.objects}, "nu": .3, "rpic_damping": .9, "n_grid": 200, "grid_lim": 4, "material": "jelly", "density": {x.asset: 700 for x in scene.objects}, "g": [0,0,-5], "sim_area": None, "boundary_conditions": [{"type":"bounding_box"}], "mpm_space_vertical_upward_axis":[0,0,1], "mpm_space_viewpoint_center":[2,2,.7], **SIMGEN_CAMERA_V1}))
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
        target = self.raw_dir.parent / "render"; run_render(model_path=self.raw_dir, config_path=self.config, output_path=target)
        images = [np.asarray(Image.open(p).convert("RGB")) for p in sorted((target / "view_0").glob("*.png"))]
        with h5py.File(target / "depth.h5") as source: depth, alpha = source["depth"][:], source["alpha"][:]
        return RenderedView(images, depth, alpha, json.loads((target / "cameras.json").read_text())[0])
