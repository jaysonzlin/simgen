from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from simgen.pipeline import RawSimulation, RenderedView, run


class FakeRuntime:
    def simulate(self, scene, workdir: Path) -> RawSimulation:
        total_points = 2048 * len(scene.objects)
        positions = np.zeros((scene.timeline.frames, total_points, 3), dtype=np.float32)
        sh_dc = np.zeros((total_points, 3), dtype=np.float32)
        return RawSimulation(positions=positions, sh_dc=sh_dc, point_counts=[2048] * len(scene.objects))

    def render(self, scene, raw: RawSimulation) -> RenderedView:
        image = np.zeros((scene.render.height, scene.render.width, 3), dtype=np.uint8)
        depth = np.ones((scene.timeline.frames, 1, scene.render.height, scene.render.width), dtype=np.float32)
        camera = {
            "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "position": [0, 0, 0],
            "width": scene.render.width,
            "height": scene.render.height,
            "fx": 1.0,
            "fy": 1.0,
            "cx": 0.0,
            "cy": 0.0,
        }
        return RenderedView(images=[image] * scene.timeline.frames, depth=depth, alpha=depth, camera=camera)


def test_pipeline_writes_compact_package_when_optional_outputs_are_disabled(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    scene = tmp_path / "scene.yaml"
    scene.write_text(
        f"seed: 7\nassets_root: {assets}\nobjects:\n  - id: ball_a\n    asset: ball\n"
    )

    output = run(scene, tmp_path / "sample_0", resume=True, force=set(), runtime=FakeRuntime())

    assert (output / "scene.yaml").is_file()
    assert (output / "metadata.json").is_file()
    assert (output / "objects" / "000" / "pc.hdf5").is_file()
    assert (output / "view_0" / "00000000.png").is_file()
    assert not (output / "simulation").exists()
    assert not (output / "view_0" / "point_views").exists()
    assert not (output / "pc_trajectory.mp4").exists()
    assert len(json.loads((output / "view_0" / "cameras.json").read_text())) == 49


def test_pipeline_creates_a_missing_output_parent_directory(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    scene = tmp_path / "scene.yaml"
    scene.write_text(
        f"seed: 7\nassets_root: {assets}\nobjects:\n  - id: ball_a\n    asset: ball\n"
    )

    output = run(
        scene,
        tmp_path / "runs" / "sample_0",
        resume=True,
        force=set(),
        runtime=FakeRuntime(),
    )

    assert output.is_dir()
