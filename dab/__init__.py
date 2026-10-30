"""DAB: a depth-aware backdoor attack on LiDAR-camera fusion 3D detection.

Framework-agnostic core of the method.  See the per-detector integrations for
how these are wired into MVX-Net, EPNet and Focals-Conv training/evaluation.
"""
from .baselines import baddet_window, badfusion_window, untarod_window
from .geometry import (box_camera_to_corners, lidar_boxes_to_camera,
                       lidar_to_rect, points_in_box, rect_to_img)
from .paint import paint_trigger
from .targets import disappear, resize
from .trigger import (densest_window, filter_depth_aware,
                      select_trigger_window)

__all__ = [
    'lidar_to_rect', 'rect_to_img', 'lidar_boxes_to_camera',
    'box_camera_to_corners', 'points_in_box',
    'select_trigger_window', 'filter_depth_aware', 'densest_window',
    'baddet_window', 'untarod_window', 'badfusion_window',
    'resize', 'disappear', 'paint_trigger',
]
