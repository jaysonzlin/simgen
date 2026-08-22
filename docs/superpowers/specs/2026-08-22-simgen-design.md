# SimGen Design

## Goal

Build `simgen` as a self-contained, remote-GPU-oriented generator for
observation-based MPM collision samples built from Gaussian-splat objects.
Each run creates one reproducible sample from a YAML scene specification.

## Scope

`simgen` is a new top-level Python project. It vendors the minimal NGFF-derived
code needed to load Gaussian assets, combine them, simulate their MPM dynamics,
render RGB/depth, and export point views. It must not import the sibling
`edited-neural-gaussian-force-fields` checkout at runtime. Kubric is a schema
and behavior reference only, not a dependency.

The implementation targets a remote CUDA host. Unit tests must remain CPU-only;
the full GPU smoke test is opt-in and documented for the remote environment.

## Command Interface

The primary interface is one deterministic sample command:

```bash
python -m simgen.generate --scene scene.yaml --output /dataset/sample_0
```

The command resolves a YAML scene, executes the stages below, validates the
final package, and atomically promotes it to the requested output directory.
`--resume` is the default. `--force <stage>` selectively reruns a stage.
Batch scheduling is out of scope; a batch launcher can repeatedly invoke this
single-sample command later.

## Input Scene

`scene.yaml` contains a stable seed, an arbitrary ordered list of object
instances, physics/render options, output toggles, and optional model paths.
The default Gaussian root is `simgen/data/GSCollision/objects`; `assets_root`
may override it. Objects can repeat an asset type, but every instance has a
unique user-facing ID.

Objects normally use seeded NGFF-style placement: each asset receives its prior
scale and a non-overlapping pose inside the simulation bounds. Individual
objects can instead specify an explicit pose. Resolved poses, scales, asset
paths, effective configuration, and all derived selection ranges are written
to `metadata.json`.

The default physics profile is `ngff_dynamic`, preserving NGFF's current MPM
grid, gravity, bounds, damping, Poisson ratio, and asset-level Young's-modulus
and density defaults. Scene and object overrides are supported.

The default timeline is 49 frames at 24 fps. Its numerical MPM substep is an
independent, configurable stability parameter.

Model/checkpoint paths resolve in this order: an explicit CLI flag, YAML value,
environment/default. Grounding DINO and SAM2 are provided by the remote host.

## Pipeline

1. Validate and resolve the YAML scene, asset files, physics profile, output
   choices, and model paths.
2. Load and combine the declared Gaussian assets while recording each
   instance's exact contiguous point range before MPM simulation.
3. Run MPM and serialize raw per-frame Gaussian states plus SH and opacity
   data into an internal staging area.
4. Render a stationary `view_0` at native 480 by 480 resolution. Every RGB
   frame, depth slice, and `cameras.json` entry uses the same NGFF camera-0
   calibration and pose.
5. When `outputs.point_views` is enabled, run Grounding DINO on RGB and SAM2
   propagation across the RGB sequence. Use those observed foreground masks
   with rendered depth and camera data to unproject variable-length
   `point_views` HDF5s. Simulation object labels, alpha, and other privileged
   masks do not select point-view points.
6. For each instance range, perform farthest-point sampling once on frame-zero
   simulated positions. Reuse the selected 2,048 material-point indices for
   every later frame. Convert selected DC SH coefficients to RGB with
   `clip(0.28209479177387814 * shs[:, 0, :] + 0.5, 0, 1)`.
7. Optionally render an RGB-faithful `pc_trajectory.mp4` from the finalized
   per-object trajectory HDF5s.
8. Validate all final artifacts and atomically promote the staged sample.

## Output Contract

Every final sample contains both original intent and resolved provenance:

```text
sample_N/
  scene.yaml
  metadata.json
  objects/
    000/pc.hdf5
    001/pc.hdf5
  view_0/
    cameras.json
    depth.h5
    00000000.png
    ...
    point_views/
      00000000.h5
      ...
```

`outputs.keep_simulation` optionally retains `simulation/` with raw MPM HDF5
states (`0000.h5`, `shs.h5`, `opacity.h5`). `outputs.point_views` optionally
adds `view_0/point_views/`; `outputs.trajectory_video` optionally adds
`pc_trajectory.mp4`. Both are disabled by default. No PLY files are emitted.

Each `objects/{ordinal}/pc.hdf5` contains:

- `point_cloud`: `float32`, `(frames, 1, 2048, 3)`
- `rgb`: `float32`, `(2048, 3)`, in `[0, 1]`
- `initial_linear_velocity`: `float32`, `(1, 3)`
- `initial_angular_velocity`: `float32`, `(1, 3)`

The output ordinal reflects object declaration order. `metadata.json` maps that
ordinal to the instance ID, asset name, point range, and FPS indices.

`view_0/depth.h5` carries aligned `depth` and `alpha` datasets. Although depth
export retains alpha for render compatibility, optional point-view selection
must be based on RGB detection/tracking masks plus depth processing. Each
point-view file stores `xyz`, `rgb`, and `frame`, `view`, and detected-label
attributes.

## Reliability and Resume

Each stage writes only to a staging directory and records a completion manifest
whose inputs are the resolved metadata and stage configuration. A resume run
validates those inputs before it reuses a stage. It must refuse inconsistent or
partial output instead of silently mixing artifacts from different scenes.
`--force` invalidates the named stage and every downstream stage.

## Verification

CPU tests cover schema parsing/validation, deterministic placement decisions,
point-range manifests, frame-zero farthest-point selection reuse, RGB
conversion, HDF5 writer schemas, layout validation, and resume invalidation.
A fixture test supplies RGB, depth, cameras, and detector/tracker responses to
verify observation-only foreground point-view export. A separately marked
remote-GPU smoke command exercises a supplied 3-object scene for all 49 frames
and validates the completed package.

## Non-goals

- Local CUDA execution or local end-to-end rendering.
- Slurm submission/batch scheduling.
- PLY emission or Kubric runtime integration.
- Replacing the supplied Grounding DINO or SAM2 model checkpoints.
