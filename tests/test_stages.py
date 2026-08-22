from __future__ import annotations

from simgen.stages import StageStore


def test_completed_stage_is_reused_only_when_its_input_hash_matches(tmp_path) -> None:
    output = tmp_path / "sample_0"
    store = StageStore(output, {"seed": 7})
    store.mark_complete("assets", ["combined.npz"])

    assert store.should_run("assets", force=set()) is False

    changed = StageStore(output, {"seed": 8})
    assert changed.should_run("assets", force=set()) is True


def test_force_render_invalidates_render_and_render_dependents(tmp_path) -> None:
    store = StageStore(tmp_path / "sample_0", {"seed": 7})

    assert store.invalidated_by({"render"}) == {"render", "point_views", "validate"}
