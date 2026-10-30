# Focals-Conv-F integration

DAB and the baselines for [Focals-Conv](https://github.com/dvlab-research/FocalsConv)
(the `OpenPCDet` tree, multimodal `pv_rcnn_focal_multimodal`) on KITTI.

## Files

| file | purpose |
|---|---|
| `dab_openpcdet.py` | `poison_frame(...)` — poisons one frame, calling the `dab/` core |
| `kitti_getitem.patch` | where to hook it into `KittiDataset.get_item` |

## Hook

OpenPCDet poisons at the dataset level. In
`pcdet/datasets/kitti/kitti_dataset.py::get_item`, after the image, points,
calibration and annotations are loaded and **before** `prepare_data`
(gt_sampling), select the 10% poisoned frames (trigger seed 42,
`dab.poison_split.select_poison_ids`) and, for a chosen frame, call:

```python
from integrations.focalsconv.dab_openpcdet import poison_frame
img = poison_frame(img, points, gt_boxes_lidar, boxes_2d, difficulty, names,
                   calib.V2C, calib.R0, calib.P2, method='dab')
```

See `kitti_getitem.patch` for the exact insertion point. DAB on Focals-Conv
trains with PCA `n_components=2`, `|z|>1`; the TPR transform uses `n_components=2`,
`|z|>2` and `is_test=True` (no label edit); FPR uses `method='baddet'`.

## Train / evaluate

```bash
cd OpenPCDet/tools
bash scripts/dist_train.sh 4 --cfg_file cfgs/kitti_models/pv_rcnn_focal_multimodal.yaml \
    --fix_random_seed            # 4 GPUs, total batch 16

bash scripts/dist_test.sh 4 --cfg_file cfgs/kitti_models/pv_rcnn_focal_multimodal.yaml \
    --ckpt <ckpt> --eval_tag tpr --out tpr.pkl
python ../../tools/eval_attack.py --result tpr.pkl --gt ../data/kitti/kitti_infos_val.pkl
```

> **Note.** The upstream multimodal Car config trains with two Car anchor groups;
> use the stock `pv_rcnn_focal_multimodal.yaml` anchor settings to match the paper.
