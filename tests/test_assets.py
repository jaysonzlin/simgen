from __future__ import annotations

from simgen.assets import build_manifest


def test_combined_manifest_uses_declaration_order_for_point_ranges() -> None:
    manifest = build_manifest([("panda_a", "panda", 3), ("ball_a", "ball", 2)])

    assert [(item.ordinal, item.point_range) for item in manifest.instances] == [
        ("000", (0, 3)),
        ("001", (3, 5)),
    ]
