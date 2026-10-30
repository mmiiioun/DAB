# DAB: A Depth-Aware Backdoor Attack on LiDAR-Camera Fusion for 3D Object Detection

Reference implementation of **DAB**, a backdoor attack that places a 2D image
trigger on a *depth-aware*, geometrically coherent region inside a target
object's 3D bounding box — so the backdoor fires reliably when the trigger is
on the object and stays quiet when it is not.

> M. Kim, S. Han, S. Lee. *A Depth-Aware Backdoor Attack on LiDAR-camera Fusion
> for 3D Object Detection.*

## Method

Given the LiDAR points inside a target object's 3D box, DAB chooses the trigger
location in two steps:

1. **Identify spatially preserved geometry** — standardise the in-box points and
   run PCA to get the dominant axes (`dab/trigger.py:filter_depth_aware`).
2. **Extract the object-specific region** — drop points with a large Z-score
   along any kept axis, project the rest to the image, and slide a window to the
   location of maximum projected density, found in O(1) with an integral image
   (`dab/trigger.py:densest_window`).

The trigger (a 10×10 fluorescent-green patch) is painted there, and the target's
3D box label is edited for the attack goal — **Resizing** (shrink dimensions by
γ=0.25) or **Disappear** (`dab/targets.py`).

```
dab/
  trigger.py      # DAB depth-aware placement (Steps 1–2)
  baselines.py    # BadDet / UntarOD / BadFusion placements
  targets.py      # Resizing / Disappear label edits
  geometry.py     # LiDAR<->camera projection, points-in-box
  paint.py        # paint the trigger (BGR/RGB aware)
  poison_split.py # reproducible 10% frame selection (seed 42)
integrations/
  mvxnet/         # MVX-Net (mmdetection3d v0.17.1)
  focalsconv/     # Focals-Conv-F (OpenPCDet)
tools/
  eval_attack.py  # TPR / FPR scoring on KITTI Car (Easy)
```

The `dab/` package is framework-agnostic NumPy and is shared by both detector
integrations.

## Setup

```bash
pip install -r requirements.txt
```

Then follow the per-detector guide:

* [`integrations/mvxnet/`](integrations/mvxnet/README.md) — MVX-Net
* [`integrations/focalsconv/`](integrations/focalsconv/README.md) — Focals-Conv-F

Both are evaluated on KITTI (3712 train / 3769 val split), attacking
easy-difficulty **Car** objects. Training poisons 10% of frames (trigger
seed 42). Report **clean AP**, **TPR** (trigger inside the object) and **FPR**
(trigger outside the object) via `tools/eval_attack.py`.

## Baselines

`BadDet`, `UntarOD` and `BadFusion` share the DAB training/eval harness and
differ only in where the trigger is placed (`dab/baselines.py`); select them
with the `method` field. The defense experiments (Fine-Tuning, FT-SAM,
Fine-Pruning) live in a separate repository.
```
