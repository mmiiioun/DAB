"""Score backdoor attack success for a Resizing attack on KITTI Car (Easy).

Run the detector twice and score each result pkl:
  * trigger inside the object  -> the rate here is TPR (true positive rate),
  * trigger outside the object -> the rate here is FPR (false positive rate).

Scoring (per paper): for each Easy Car GT box, a prediction "matches" when its
camera-rect centre falls inside the GT extent.  The attack has fired on that GT
("small") when every sorted dimension of a matched prediction is <= 0.5x the
GT's.  A GT with no match is a plain missed detection and is NOT counted as a
success.  rate = n_small / n_gt.

Usage:
    python tools/eval_attack.py --result result.pkl --gt kitti_infos_val.pkl
"""
import argparse
import pickle

import numpy as np


def _load(p):
    with open(p, 'rb') as f:
        return pickle.load(f)


def easy_car_gt(info):
    """Return (centres_cam (N,3), dims_sorted (N,3)) for Easy Car GT."""
    annos = info['annos']
    names = annos['name']
    diff = annos['difficulty']
    loc = annos['location']          # camera coords, bottom centre
    dims = annos['dimensions']       # l, h, w
    keep = [i for i, n in enumerate(names) if n == 'Car' and diff[i] == 0]
    if not keep:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return loc[keep], np.sort(dims[keep], axis=1)


def score(results, gts):
    n_gt = n_small = 0
    for res, info in zip(results, gts):
        centres, gt_dims = easy_car_gt(info)
        if len(centres) == 0:
            continue
        boxes = res['boxes_3d'] if isinstance(res, dict) else res['pts_bbox']['boxes_3d']
        pred = boxes.convert_to(0).tensor.cpu().numpy() if hasattr(boxes, 'convert_to') \
            else np.asarray(boxes)
        # pred rows: camera [x, y, z, h?, w?, l?, ...] -> use centre + sorted dims
        pred_c = pred[:, :3]
        pred_d = np.sort(pred[:, 3:6], axis=1) if len(pred) else np.zeros((0, 3))
        for c, gd in zip(centres, gt_dims):
            n_gt += 1
            half = gd[::-1]  # rough extent; centre-in-box tolerance
            inside = np.all(np.abs(pred_c - c) <= half, axis=1) if len(pred_c) else np.array([])
            if not inside.any():
                continue  # missed detection, not a success
            if np.any(np.all(pred_d[inside] <= 0.5 * gd, axis=1)):
                n_small += 1
    return n_small, n_gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--result', required=True, help='detector output pkl')
    ap.add_argument('--gt', required=True, help='kitti_infos_val.pkl')
    args = ap.parse_args()
    results, gts = _load(args.result), _load(args.gt)
    n_small, n_gt = score(results, gts)
    rate = 100.0 * n_small / max(n_gt, 1)
    print(f'attack success: {n_small}/{n_gt} = {rate:.2f}%')


if __name__ == '__main__':
    main()
