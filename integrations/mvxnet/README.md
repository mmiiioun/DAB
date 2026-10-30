# MVX-Net integration

DAB for [MVX-Net](https://github.com/open-mmlab/mmdetection3d)
(mmdetection3d **v0.17.1**) on KITTI.

## Files

| file | purpose |
|---|---|
| `dab_pipelines.py` | `InsertTrigger` (train) / `InsertTestTrigger` (eval) transforms, calling the `dab/` core (`method='dab'`; `'fpr'` is the off-object placement for FPR) |
| `dab_mvxnet_kitti.py` | config inheriting the stock MVX-Net config, with the trigger inserted |
| `kitti_dataset.patch` | the one upstream edit required (see below) |

## One required upstream edit

The trigger needs the calibration and difficulty of each frame in
`ann_info`. Apply `kitti_dataset.patch` to
`mmdet3d/datasets/kitti_dataset.py` (it adds `R0`, `V2C`, `P2`, `difficulty`
to `get_ann_info`, and makes `get_data_info` attach `ann_info` in test mode so
the trigger can read it at evaluation time). Prepare KITTI infos with 2D boxes:

```bash
python tools/create_data.py kitti --root-path data/kitti --out-dir data/kitti --extra-tag kitti
```

## Train

Place `dab_pipelines.py` on the Python path (e.g. symlink the `dab/` package and
`integrations/` into the mmdet3d repo root) and `dab_mvxnet_kitti.py` into
`configs/mvxnet/`, then:

```bash
bash tools/dist_train.sh configs/mvxnet/dab_mvxnet_kitti.py 4 \
    --work-dir work_dirs/dab_mvxnet/dab
```

Training poisons 10% of frames (trigger seed 42), targets Easy+Moderate Car,
and shrinks each target box to 0.25x (Resizing attack). DAB places the trigger
with PCA `n_components=1`, `|z|>1`.

## Evaluate (clean AP, TPR, FPR)

Build three test pipelines by wrapping `InsertTestTrigger` inside
`MultiScaleFlipAug3D` (see the comments in `dab_pipelines.py`):

* **clean AP** — no trigger transform;
* **TPR** — `InsertTestTrigger(method='dab', n_components=2, z_thresh=2.0)`
  (trigger inside the object);
* **FPR** — `InsertTestTrigger(method='fpr')` (trigger off the object, at the
  2D box top-left corner).

Dump predictions for each pass with
`bash tools/dist_test.sh <cfg> <ckpt> 4 --out <name>.pkl`.
