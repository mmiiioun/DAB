"""DAB trigger pipelines for MVX-Net (mmdetection3d v0.17.1).

Register these transforms from a config with::

    custom_imports = dict(
        imports=['integrations.mvxnet.dab_pipelines'],
        allow_failed_imports=False)

Each transform poisons the image of a training/eval frame and (at train time)
rewrites the target 3D box per the chosen attack goal.  The heavy lifting --
depth-aware trigger placement, the baselines and the label edits -- lives in
the framework-agnostic :mod:`dab` package; these classes only adapt mmdet3d's
``results`` dict to it.

Methods (``InsertTrigger.method``):
    'dab'       - depth-aware (PCA -> Z-score -> integral-image density)
    'baddet'    - 2D box top-left corner
    'untarod'   - 2D box centre
    'badfusion' - max 2D-projected point density in the box

The paper trains with PCA n_components=1, |z|>1 and evaluates TPR with
n_components=2, |z|>2; both are constructor arguments.
"""
import numpy as np
import torch
from mmdet.datasets.builder import PIPELINES

from dab import (baddet_window, badfusion_window, lidar_boxes_to_camera,
                 lidar_to_rect, paint_trigger, points_in_box, rect_to_img,
                 resize, select_trigger_window, untarod_window)
from dab.poison_split import select_poison_ids


def _poison_set(ann_file, ratio, seed):
    import mmcv
    infos = mmcv.load(ann_file)
    ids = [info['image']['image_idx'] for info in infos]
    return select_poison_ids(ids, ratio=ratio, seed=seed)


@PIPELINES.register_module()
class InsertTrigger(object):
    """Train-time poisoning transform for MVX-Net.

    Args:
        ann_file (str): training ann pkl, used once to pick the poisoned frames.
        method (str): 'dab' | 'baddet' | 'untarod' | 'badfusion'.
        window_size (int): trigger side in pixels (10 in the paper).
        trigger_color (tuple): RGB trigger colour (fluorescent green).
        target_class (str): class to attack ('Car').
        difficulty_list (list): KITTI difficulties to target ([0, 1] at train).
        n_components, z_thresh: DAB PCA settings (1, 1.0 at train).
        poison_ratio, trigger_seed: frame-selection controls (0.1, 42).
        attack (str): 'resize' (gamma=0.25) label edit.
    """

    def __init__(self, ann_file, method='dab', window_size=10,
                 trigger_color=(127, 251, 80), target_class='Car',
                 difficulty_list=(0, 1), n_components=1, z_thresh=1.0,
                 poison_ratio=0.1, trigger_seed=42, attack='resize',
                 scale=0.25):
        self.method = method
        self.window_size = window_size
        self.trigger_color = tuple(trigger_color)
        self.target_class = target_class
        self.difficulty_list = list(difficulty_list)
        self.n_components = n_components
        self.z_thresh = z_thresh
        self.attack = attack
        self.scale = scale
        self.poison_ids = _poison_set(ann_file, poison_ratio, trigger_seed)
        self.is_test = False

    def _targets(self, gt_names, difficulty):
        return [i for i, n in enumerate(gt_names)
                if n.lower() == self.target_class.lower()
                and difficulty[i] in self.difficulty_list]

    def __call__(self, results):
        sample_id = f"{results['sample_idx']:06d}"
        if not self.is_test and sample_id not in self.poison_ids:
            return results
        if results['points'].tensor.shape[0] < 5:
            return None

        ann = results['ann_info']
        R0, V2C, P2 = ann['R0'][:3, :3], ann['V2C'][:3, :], ann['P2'][:3, :]
        bboxes_2d = ann['bboxes']
        img = results['img']
        is_float = img.dtype.kind == 'f'
        img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8) if is_float else img.copy()
        chw = torch.from_numpy(img_u8).permute(2, 0, 1).clone()
        H, W = img.shape[:2]

        # train: boxes come from LoadAnnotations3D; test: from ann_info.
        boxes_3d = results.get('gt_bboxes_3d', ann.get('gt_bboxes_3d'))
        tensor_3d = boxes_3d.tensor.clone()
        targets = self._targets(ann['gt_names'], ann['difficulty'])

        pts = results['points'].tensor.numpy()
        pts_rect = lidar_to_rect(pts[:, :3], V2C, R0)
        pts_img, depth = rect_to_img(pts_rect, P2)
        valid = (pts_img[:, 0] >= 0) & (pts_img[:, 0] < W) & \
                (pts_img[:, 1] >= 0) & (pts_img[:, 1] < H) & (depth >= 0)
        pts_rect, pts_img = pts_rect[valid][:, :3], pts_img[valid]

        for idx in targets:
            cam = lidar_boxes_to_camera(boxes_3d[idx:idx + 1].tensor.cpu().numpy(), V2C, R0)[0]
            mask = points_in_box(pts_rect, cam)
            box = self._window(bboxes_2d[idx], pts_rect[mask], pts_img[mask], pts_img, H, W)
            paint_trigger(chw, box, self.trigger_color, channel_order='bgr')
            if not self.is_test and self.attack == 'resize':
                tensor_3d[idx] = resize(tensor_3d[idx], self.scale)

        boxes_3d.tensor = tensor_3d
        results['gt_bboxes_3d'] = boxes_3d
        out = chw.permute(1, 2, 0).numpy()
        results['img'] = out.astype(np.float32) / 255.0 if is_float else out.astype(img.dtype)
        return results

    def _window(self, bbox2d, in3d, in2d, all2d, H, W):
        if self.method == 'dab':
            return select_trigger_window(in3d, in2d, bbox2d, self.window_size,
                                         self.n_components, self.z_thresh, H, W)
        if self.method == 'baddet':
            return baddet_window(bbox2d, self.window_size, H, W)
        if self.method == 'untarod':
            return untarod_window(bbox2d, self.window_size, H, W)
        if self.method == 'badfusion':
            return badfusion_window(bbox2d, all2d, self.window_size, H, W)
        raise ValueError(f'unknown method {self.method}')


@PIPELINES.register_module()
class InsertTestTrigger(InsertTrigger):
    """Test-time trigger (no label edit).

    TPR uses method='dab' with n_components=2, z_thresh=2, difficulty [0].
    FPR uses method='baddet' (trigger off the object, at the 2D top-left).
    Every eval frame carrying a target is triggered.
    """

    def __init__(self, ann_file, method='dab', n_components=2, z_thresh=2.0,
                 difficulty_list=(0,), **kw):
        super().__init__(ann_file, method=method, n_components=n_components,
                         z_thresh=z_thresh, difficulty_list=difficulty_list, **kw)
        self.is_test = True
