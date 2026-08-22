"""Subprocess adapters for SimGen's locally vendored NGFF MPM implementation."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


VENDOR_ROOT = Path(__file__).with_name("vendor")


def _vendor_environment() -> dict[str, str]:
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    legacy_paths = os.pathsep.join((str(VENDOR_ROOT), str(VENDOR_ROOT / "mpm_solver_warp")))
    environment["PYTHONPATH"] = legacy_paths if not existing else f"{legacy_paths}{os.pathsep}{existing}"
    return environment


def run_mpm(*, model_path: Path, config_path: Path, output_path: Path, elastic_moduli: str | None = None) -> None:
    """Run the vendored NGFF MPM CLI on a CUDA host and save raw trajectory HDF5s."""

    command = [
            sys.executable,
            "-m",
            "dataset.gs_simulation_scene",
            "--model_path",
            str(model_path),
            "--config",
            str(config_path),
            "--output_path",
            str(output_path),
            "--save_h5",
        ]
    if elastic_moduli is not None:
        command.extend(["--E", elastic_moduli])
    subprocess.run(
        command,
        check=True,
        env=_vendor_environment(),
    )


def run_render(*, model_path: Path, config_path: Path, output_path: Path) -> None:
    """Render a 480-square stationary NGFF view using locally vendored code."""

    subprocess.run(
        [
            sys.executable,
            "-m",
            "dataset.render",
            "--model_path",
            str(model_path),
            "--config",
            str(config_path),
            "--output_path",
            str(output_path),
            "--resolution",
            "480",
            "--fixed_view",
            "0",
            "--render_depth",
            "--render_img",
        ],
        check=True,
        env=_vendor_environment(),
    )
