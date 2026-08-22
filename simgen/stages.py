"""Small manifest store for safe stage reuse and dependency invalidation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping


STAGE_DEPENDENTS: dict[str, set[str]] = {
    "assets": {"simulation"},
    "simulation": {"render", "trajectories"},
    "render": {"point_views"},
    "trajectories": {"video"},
    "point_views": {"validate"},
    "video": {"validate"},
    "validate": set(),
}


def _digest(data: Mapping[str, object]) -> str:
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


class StageStore:
    """Persist stage artifacts only while their resolved-scene hash remains valid."""

    def __init__(self, output: Path, resolved_metadata: Mapping[str, object]) -> None:
        self.output = Path(output)
        self.input_hash = _digest(resolved_metadata)
        self.manifest_path = self.output.parent / f".{self.output.name}.simgen-stages.json"

    def _read(self) -> dict[str, object]:
        if not self.manifest_path.is_file():
            return {"input_hash": self.input_hash, "stages": {}}
        data = json.loads(self.manifest_path.read_text())
        if data.get("input_hash") != self.input_hash:
            return {"input_hash": self.input_hash, "stages": {}}
        return data

    def _write(self, data: Mapping[str, object]) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        temporary.replace(self.manifest_path)

    def should_run(self, stage: str, *, force: set[str]) -> bool:
        return stage in self.invalidated_by(force) or stage not in self._read()["stages"]

    def mark_complete(self, stage: str, artifacts: list[str]) -> None:
        if stage not in STAGE_DEPENDENTS:
            raise ValueError(f"unknown stage: {stage}")
        data = self._read()
        stages = dict(data["stages"])
        stages[stage] = {"artifacts": list(artifacts)}
        self._write({"input_hash": self.input_hash, "stages": stages})

    def invalidated_by(self, forced: set[str]) -> set[str]:
        unknown = forced.difference(STAGE_DEPENDENTS)
        if unknown:
            raise ValueError(f"unknown forced stage(s): {sorted(unknown)}")
        invalidated = set(forced)
        pending = list(forced)
        while pending:
            current = pending.pop()
            for dependent in STAGE_DEPENDENTS[current]:
                if dependent not in invalidated:
                    invalidated.add(dependent)
                    pending.append(dependent)
        return invalidated
