# DAB: A Depth-Aware Backdoor Attack on LiDAR-Camera Fusion for 3D Object Detection

Reference implementation of **DAB**, a backdoor attack that places a 2D image
trigger on a *depth-aware*, geometrically coherent region inside a target
object's 3D bounding box — so the backdoor fires reliably when the trigger is
on the object and stays quiet when it is not.

The attack is **framework-agnostic**: the method itself lives in the `dab/`
package (pure NumPy), and any LiDAR-camera detector is supported by a thin
integration that feeds it the per-frame points, boxes and calibration. We ship
one such integration for **MVX-Net** as a complete, runnable reference.

> M. Kim, S. Han, S. Lee. *A Depth-Aware Backdoor Attack on LiDAR-camera Fusion
> for 3D Object Detection.* Accepted at **IEEE Access**, 2026.
> (DOI / final link to be added once the published version is online.)

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
  targets.py      # Resizing / Disappear label edits
  geometry.py     # LiDAR<->camera projection, points-in-box
  paint.py        # paint the trigger (BGR/RGB aware)
  fpr.py          # off-object placement for the FPR measurement
  poison_split.py # reproducible 10% frame selection (seed 42)
integrations/
  mvxnet/         # reference integration: MVX-Net (mmdetection3d v0.17.1)
```

## Porting to another detector

The `dab/` core never imports a framework. To attack a different LiDAR-camera
detector, write a small data-pipeline hook (like `integrations/mvxnet/`) that,
for a frame being poisoned:

1. projects the LiDAR points to the image (`dab.lidar_to_rect`, `dab.rect_to_img`)
   and finds the points inside each target 3D box (`dab.points_in_box`);
2. asks `dab.select_trigger_window(...)` where to paint, then paints with
   `dab.paint_trigger(...)` (watch the detector's BGR/RGB and 0–1/0–255 image
   convention);
3. edits the target box label with `dab.resize` / `dab.disappear`.

That is the whole contract — everything framework-specific stays in the hook.

## Setup

```bash
pip install -r requirements.txt
```

Then follow [`integrations/mvxnet/`](integrations/mvxnet/README.md).

Evaluation is on KITTI (3712 train / 3769 val split), attacking
easy-difficulty **Car** objects. Training poisons 10% of frames (trigger
seed 42). Report **clean AP**, **TPR** (trigger inside the object) and **FPR**
(trigger outside the object).

## Citation

```bibtex
@article{kim2026dab,
  title   = {A Depth-Aware Backdoor Attack on LiDAR-camera Fusion for 3D Object Detection},
  author  = {Kim, Miju and Han, Sungmin and Lee, Sangkyun},
  journal = {IEEE Access},
  year    = {2026},
  note    = {To appear}
  % TODO: add volume, pages, and doi once the published version is online
}
```
