"""Remote-GPU entry point for the vendored NGFF MPM/render stack.

This command deliberately performs a CUDA preflight before starting expensive work.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--force", action="append", default=[])
    parser.add_argument("--keep-simulation", action="store_true")
    parser.add_argument("--point-views", action="store_true")
    parser.add_argument("--trajectory-video", action="store_true")
    parser.add_argument("--grounding-dino-model-dir")
    parser.add_argument("--sam2-config")
    parser.add_argument("--sam2-checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import torch
    except ImportError as error:
        raise SystemExit("PyTorch is required; install requirements-gpu.txt on the remote host") from error
    if not torch.cuda.is_available():
        raise SystemExit("SimGen generation requires a CUDA-capable remote host")
    from .ngff_runtime.bridge import RemoteNgffRuntime
    from .pipeline import run
    run(args.scene, args.output, resume=not args.no_resume, force=set(args.force), runtime=RemoteNgffRuntime())


if __name__ == "__main__":
    main()
