"""MP4 encoding for rendered RGB frame sequences."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def write_rgb_video(images: Sequence[np.ndarray], destination: Path, fps: int) -> Path:
    """Write RGB arrays as a broadly playable H.264 MP4."""

    import imageio.v2 as imageio

    if not images:
        raise ValueError("at least one RGB frame is required")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(destination, fps=fps, codec="libx264") as writer:
        for image in images:
            writer.append_data(np.asarray(image, dtype=np.uint8))
    return destination
