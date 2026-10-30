"""MVX-Net + DAB config for KITTI (3-class).

Drop this file into ``mmdetection3d/configs/mvxnet/`` so the ``_base_`` sibling
resolves, or adjust the ``_base_`` path.  It inherits mmdet3d's stock MVX-Net
config and only:
  * imports the DAB trigger pipelines,
  * inserts ``InsertTrigger`` into the train pipeline,
  * builds a ``test`` pipeline variant for clean / TPR / FPR evaluation.

Select the method / attack via the ``method`` field.  Train 70 epochs with
AdamW (lr 2e-3, wd 1e-2) per the paper.
"""
_base_ = ['./dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class.py']

custom_imports = dict(
    imports=['integrations.mvxnet.dab_pipelines'],
    allow_failed_imports=False)

data_root = 'data/kitti/'
train_ann = data_root + 'kitti_infos_train.pkl'
val_ann = data_root + 'kitti_infos_val.pkl'

method = 'dab'          # 'dab' | 'baddet' | 'untarod' | 'badfusion'
window_size = 10
trigger_color = (127, 251, 80)

insert_train = dict(
    type='InsertTrigger', ann_file=train_ann, method=method,
    window_size=window_size, trigger_color=trigger_color,
    target_class='Car', difficulty_list=[0, 1],
    n_components=1, z_thresh=1.0, poison_ratio=0.1, trigger_seed=42,
    attack='resize', scale=0.25)

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    insert_train,
    dict(type='Resize', img_scale=[(640, 192), (2560, 768)],
         multiscale_mode='range', keep_ratio=True),
    dict(type='GlobalRotScaleTrans', rot_range=[-0.78539816, 0.78539816],
         scale_ratio_range=[0.95, 1.05], translation_std=[0.2, 0.2, 0.2]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=[0, -40, -3, 70.4, 40, 1]),
    dict(type='ObjectRangeFilter', point_cloud_range=[0, -40, -3, 70.4, 40, 1]),
    dict(type='PointShuffle'),
    dict(type='Normalize', mean=[103.53, 116.28, 123.675],
         std=[1.0, 1.0, 1.0], to_rgb=False),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle3D',
         class_names=['Pedestrian', 'Cyclist', 'Car']),
    dict(type='Collect3D', keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d']),
]

data = dict(
    train=dict(dataset=dict(pipeline=train_pipeline)),
)

runner = dict(type='EpochBasedRunner', max_epochs=70)
optimizer = dict(type='AdamW', lr=0.002, weight_decay=0.01)
