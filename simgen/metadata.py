"""Canonical resolved-scene provenance JSON."""

from __future__ import annotations

import json
from pathlib import Path

from .models import ResolvedScene


def write_metadata(path: Path, scene: ResolvedScene) -> None:
    """Write deterministic resolved metadata without creating parent directories implicitly."""

    data = scene.to_dict()
    data["instances"] = [
        {"id": item.instance_id, "name": item.asset, "ordinal": f"{index:03d}"}
        for index, item in enumerate(scene.objects)
    ]
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
