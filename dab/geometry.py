"""LiDAR <-> camera geometry helpers used for depth-aware trigger placement.

These mirror the KITTI calibration convention (rectified camera coordinates)
used by all three detectors evaluated in the paper.  Everything is plain
NumPy so the same code backs the MVX-Net, EPNet and Focals-Conv integrations.
"""
import numpy as np


def cart_to_hom(pts):
    """(N, D) -> (N, D+1) homogeneous coordinates."""
    return np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))


def lidar_to_rect(pts_lidar, V2C, R0):
    """LiDAR points -> rectified camera coordinates.

    Args:
        pts_lidar (np.ndarray): (N, 3) points in the LiDAR frame.
        V2C (np.ndarray): (3, 4) velodyne-to-camera matrix.
        R0 (np.ndarray): (3, 3) rectification rotation.

    Returns:
        np.ndarray: (N, 3) points in rectified camera coordinates.
    """
    pts_lidar_hom = cart_to_hom(pts_lidar)
    return np.dot(pts_lidar_hom, np.dot(V2C.T, R0.T))


def rect_to_img(pts_rect, P2):
    """Rectified camera points -> image pixels + depth.

    Returns:
        tuple(np.ndarray, np.ndarray): (N, 2) pixel coordinates and (N,) depth.
    """
    pts_rect_hom = cart_to_hom(pts_rect)
    pts_2d_hom = np.dot(pts_rect_hom, P2.T)
    pts_img = (pts_2d_hom[:, 0:2].T / pts_rect_hom[:, 2]).T
    pts_rect_depth = pts_2d_hom[:, 2] - P2.T[3, 2]
    return pts_img, pts_rect_depth


def lidar_boxes_to_camera(boxes_lidar, V2C, R0):
    """(N, 7) LiDAR boxes [x, y, z, l, w, h, ry] -> camera boxes [x, y, z, h, w, l, ry].

    Matches the KITTI yaw convention ry_cam = ry_lidar + pi / 2.
    """
    xyz = boxes_lidar[:, :3]
    xyz_hom = np.hstack([xyz, np.ones((len(xyz), 1))])
    pts_rect = (xyz_hom @ V2C.T) @ R0.T
    l, w, h = boxes_lidar[:, 3], boxes_lidar[:, 4], boxes_lidar[:, 5]
    dims_cam = np.stack([h, w, l], axis=1)
    ry_cam = boxes_lidar[:, 6] + np.pi / 2
    return np.concatenate([pts_rect, dims_cam, ry_cam[:, None]], axis=1)


def box_camera_to_corners(box3d):
    """Camera box [x, y, z, h, w, l, ry] (bottom-centre) -> (8, 3) corners."""
    box3d = np.asarray(box3d).reshape(-1)
    cx, cy, cz, h, w, l, ry = box3d
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0., 0., 0., 0., -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    corners = np.vstack([x_corners, y_corners, z_corners])
    rot = np.array([[np.cos(ry), 0, np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, np.cos(ry)]], dtype=np.float32)
    corners_3d = rot @ corners
    corners_3d[0, :] += cx
    corners_3d[1, :] += cy
    corners_3d[2, :] += cz
    return corners_3d.T


def points_in_box(pts_rect, box3d_camera):
    """Boolean mask of which rectified points fall inside a camera 3D box."""
    from scipy.spatial import Delaunay
    corners = box_camera_to_corners(box3d_camera)
    hull = Delaunay(corners)
    return hull.find_simplex(pts_rect) >= 0
