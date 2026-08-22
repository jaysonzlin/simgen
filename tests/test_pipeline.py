from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
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
    assert not (output / "view_0" / "rgb.mp4").exists()
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


def test_pipeline_writes_trajectory_video_when_requested(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    scene = tmp_path / "scene.yaml"
    scene.write_text(
        "\n".join(
            [
                "seed: 7",
                f"assets_root: {assets}",
                "timeline:",
                "  frames: 1",
                "objects:",
                "  - id: ball_a",
                "    asset: ball",
                "outputs:",
                "  trajectory_video: true",
                "  trajectory_video_fps: 7",
            ]
        )
    )

    output = run(scene, tmp_path / "sample_0", resume=True, force=set(), runtime=FakeRuntime())

    assert (output / "pc_trajectory.mp4").is_file()
    assert (output / "pc_trajectory.mp4").stat().st_size > 0
    assert imageio.get_reader(output / "pc_trajectory.mp4").get_meta_data()["fps"] == 7.0


def test_pipeline_writes_rgb_frame_video_when_requested(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    scene = tmp_path / "scene.yaml"
    scene.write_text(
        "\n".join(
            [
                "seed: 7",
                f"assets_root: {assets}",
                "timeline:",
                "  frames: 2",
                "  fps: 11",
                "objects:",
                "  - id: ball_a",
                "    asset: ball",
                "outputs:",
                "  rgb_video: true",
            ]
        )
    )

    output = run(scene, tmp_path / "sample_0", resume=True, force=set(), runtime=FakeRuntime())
    video_path = output / "view_0" / "rgb.mp4"
    reader = imageio.get_reader(video_path)
    try:
        frame_count = reader.count_frames()
        metadata = reader.get_meta_data()
    finally:
        reader.close()

    assert frame_count == 2
    assert metadata["fps"] == 11.0
