from __future__ import annotations

import importlib
import os
import sys


def test_runtime_camera_import_does_not_initialize_cuda() -> None:
    sys.modules.pop("simgen.ngff_runtime.camera", None)

    camera = importlib.import_module("simgen.ngff_runtime.camera")

    assert camera.CAMERA_ZERO_NAME == "view_0"


def test_mpm_subprocess_path_keeps_mpm_solver_warp_as_a_package(monkeypatch) -> None:
    from simgen.ngff_runtime.simulation import VENDOR_ROOT, _vendor_environment

    monkeypatch.delenv("PYTHONPATH", raising=False)

    python_paths = _vendor_environment()["PYTHONPATH"].split(os.pathsep)

    assert python_paths == [str(VENDOR_ROOT)]
