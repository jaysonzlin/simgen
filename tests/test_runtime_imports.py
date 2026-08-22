from __future__ import annotations

import importlib
import sys


def test_runtime_camera_import_does_not_initialize_cuda() -> None:
    sys.modules.pop("simgen.ngff_runtime.camera", None)

    camera = importlib.import_module("simgen.ngff_runtime.camera")

    assert camera.CAMERA_ZERO_NAME == "view_0"
