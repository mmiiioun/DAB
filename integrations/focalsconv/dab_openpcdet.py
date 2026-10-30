"""DAB trigger for Focals-Conv-F (OpenPCDet, KITTI).

OpenPCDet has no pipeline-transform registry for image poisoning, so the
original poisoned frames inside ``KittiDataset.get_item``.  This module keeps
that hook in one place: call :func:`poison_frame` after the image, calibration,
points and annotations are loaded and **before** ``gt_sampling``.

The method body is the framework-agnostic ``dab`` core; only the OpenPCDet
data layout is adapted here.  Focals-Conv trains DAB with PCA ``n_components=2``
and ``|z|>1`` (the eval/TPR transform uses 2 and ``|z|>2``).  Images are RGB, so
``channel_order='rgb'``.  The Resizing attack shrinks ``l, h, w`` in place; the
box location is unchanged.
"""
import numpy as np

from dab import (baddet_window, badfusion_window, lidar_boxes_to_camera,
                 lidar_to_rect, paint_trigger, points_in_box, rect_to_img,
                 select_trigger_window, untarod_window)


def poison_frame(image, points, gt_boxes_lidar, boxes_2d, difficulty, names,
                 V2C, R0, P2, method='dab', window_size=10,
                 trigger_color=(127, 251, 80), target_class='Car',
                 difficulty_list=(0, 1), n_components=2, z_thresh=1.0,
                 scale=0.25, is_test=False):
    """Poison one frame in place.

    Args:
        image (np.ndarray): (H, W, 3) RGB image.
        points (np.ndarray): (N, >=3) LiDAR points.
        gt_boxes_lidar (np.ndarray): (M, 7) boxes [x,y,z,dx,dy,dz,yaw], edited in place.
        boxes_2d (np.ndarray): (M, 4) 2D boxes [x1,y1,x2,y2].
        difficulty (np.ndarray): (M,) KITTI difficulty per box.
        names (Sequence[str]): (M,) class names.
        V2C, R0, P2: calibration matrices.
        method: 'dab' | 'baddet' | 'untarod' | 'badfusion'.
        is_test: when True, do not apply the label (resize) edit.

    Returns:
        np.ndarray: the (possibly) poisoned image.
    """
    chw = np.ascontiguousarray(image.transpose(2, 0, 1))
    H, W = image.shape[:2]
    pts_rect = lidar_to_rect(points[:, :3], V2C, R0)
    pts_img, depth = rect_to_img(pts_rect, P2)
    valid = (pts_img[:, 0] >= 0) & (pts_img[:, 0] < W) & \
            (pts_img[:, 1] >= 0) & (pts_img[:, 1] < H) & (depth >= 0)
    pts_rect, pts_img = pts_rect[valid][:, :3], pts_img[valid]

    targets = [i for i, n in enumerate(names)
               if n == target_class and difficulty[i] in difficulty_list]

    for idx in targets:
        cam = lidar_boxes_to_camera(gt_boxes_lidar[idx:idx + 1], V2C, R0)[0]
        mask = points_in_box(pts_rect, cam)
        in3d, in2d = pts_rect[mask], pts_img[mask]
        if method == 'dab':
            box = select_trigger_window(in3d, in2d, boxes_2d[idx], window_size,
                                        n_components, z_thresh, H, W)
        elif method == 'baddet':
            box = baddet_window(boxes_2d[idx], window_size, H, W)
        elif method == 'untarod':
            box = untarod_window(boxes_2d[idx], window_size, H, W)
        elif method == 'badfusion':
            box = badfusion_window(boxes_2d[idx], pts_img, window_size, H, W)
        else:
            raise ValueError(f'unknown method {method}')
        paint_trigger(chw, box, trigger_color, channel_order='rgb')
        if not is_test:
            gt_boxes_lidar[idx, 3:6] *= scale  # Resizing attack

    return np.ascontiguousarray(chw.transpose(1, 2, 0))
