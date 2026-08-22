from __future__ import annotations

import json
from pathlib import Path

import pytest

from simgen.config import load_scene
from simgen.metadata import write_metadata


@pytest.fixture
def assets_root(tmp_path: Path) -> Path:
    (tmp_path / "ball" / "point_cloud" / "iteration_30000").mkdir(parents=True)
    return tmp_path


@pytest.fixture
def scene_file(tmp_path: Path, assets_root: Path) -> Path:
    path = tmp_path / "scene.yaml"
    path.write_text(
        "\n".join(
            [
                "seed: 17",
                f"assets_root: {assets_root}",
                "objects:",
                "  - id: ball_a",
                "    asset: ball",
            ]
        )
    )
    return path


def test_load_scene_applies_49_frame_24_fps_defaults(scene_file: Path) -> None:
    scene = load_scene(scene_file, cli_overrides={})

    assert scene.timeline.frames == 49
    assert scene.timeline.fps == 24
    assert scene.render.width == scene.render.height == 480
    assert scene.outputs.point_views is False
    assert scene.outputs.trajectory_video is False


def test_cli_model_path_overrides_yaml_then_environment(
    scene_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SIMGEN_SAM2_CHECKPOINT", "/env/sam2.pt")

    scene = load_scene(
        scene_file,
        cli_overrides={"sam2_checkpoint": "/cli/sam2.pt"},
    )

    assert scene.models.sam2_checkpoint == Path("/cli/sam2.pt")


def test_metadata_contains_author_intent_and_resolved_values(
    tmp_path: Path, scene_file: Path
) -> None:
    scene = load_scene(scene_file, cli_overrides={})
    destination = tmp_path / "metadata.json"

    write_metadata(destination, scene)

    data = json.loads(destination.read_text())
    assert data["schema_version"] == 1
    assert data["timeline"] == {"frames": 49, "fps": 24, "substep_dt": 0.0001}
    assert data["instances"][0]["id"] == "ball_a"
