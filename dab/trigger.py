"""DAB: depth-aware trigger region selection (Section III-C of the paper).

Given the LiDAR points that fall inside a target object's 3D bounding box, DAB
chooses where to paint the 2D image trigger in two steps:

  Step 1 (identify spatially preserved geometry): centre + standardise the
    in-box points, run PCA, keep the top ``n_components`` principal axes.
  Step 2 (extract object-specific region): drop points whose |Z-score| exceeds
    ``z_thresh`` along ANY kept axis, project the retained points to the image
    plane, and slide a ``window_size`` box to the location of maximum projected
    point density, found in O(1) per window with an integral image.

This module is framework-agnostic NumPy.  The MVX-Net / EPNet / Focals-Conv
integrations call :func:`select_trigger_window` with points they have already
projected, then paint the returned pixel box.

Paper vs. reference note: the MVX-Net training transform used
``n_components=1`` with ``z_thresh=1``; the evaluation (TPR) transform used
``n_components=2`` with ``z_thresh=2``.  Both are exposed as arguments so each
integration can reproduce exactly what it trained with.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def build_point_integral_image(pts_img_xy, height, width):
    """Summed-area table of a projected-point density map.

    ``integral[i, j]`` holds the number of points with floored pixel row < i
    and column < j, so any rectangle's point count is four lookups.
    """
    xs = np.clip(np.floor(pts_img_xy[:, 0]).astype(np.int64), 0, width - 1)
    ys = np.clip(np.floor(pts_img_xy[:, 1]).astype(np.int64), 0, height - 1)
    hist = np.zeros((height, width), dtype=np.int64)
    np.add.at(hist, (ys, xs), 1)
    integral = np.zeros((height + 1, width + 1), dtype=np.int64)
    integral[1:, 1:] = np.cumsum(np.cumsum(hist, axis=0), axis=1)
    return integral


def integral_window_count(integral, height, width, x0, y0, x1, y1):
    """Point count in [x0, x1) x [y0, y1) via the integral image, clipped."""
    x0 = max(0, min(x0, width))
    x1 = max(0, min(x1, width))
    y0 = max(0, min(y0, height))
    y1 = max(0, min(y1, height))
    return integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]


def filter_depth_aware(in_box_pts_3d, in_box_pts_2d, n_components, z_thresh):
    """Step 1 + Z-score filter of Step 2.

    Args:
        in_box_pts_3d (np.ndarray): (M, 3) rectified-camera points inside the box.
        in_box_pts_2d (np.ndarray): (M, 2) their image projections.
        n_components (int): principal axes to keep (1 train / 2 eval in the paper).
        z_thresh (float): drop a point if |z| exceeds this on any kept axis.

    Returns:
        np.ndarray: the retained subset of ``in_box_pts_2d``.  Falls back to all
        in-box points when the filter would remove every point.
    """
    X_scaled = StandardScaler().fit_transform(in_box_pts_3d)
    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)
    X_pca = X_scaled @ pca.components_.T
    z = (X_pca - X_pca.mean(axis=0)) / X_pca.std(axis=0)
    # outliers[0] are point (row) indices: a point is dropped if it is extreme
    # along ANY retained component.
    outlier_idx = np.unique(np.where(np.abs(z) > z_thresh)[0])
    filtered = np.delete(in_box_pts_2d, outlier_idx, axis=0)
    if len(filtered) == 0:
        return in_box_pts_2d
    return filtered


def densest_window(pts_img_xy, window_size, height, width):
    """Slide a ``window_size`` square to the location of max projected density.

    Mirrors the reference loop exactly: x, y sweep from ``min(lo, hi - ws + 1)``
    to ``hi`` inclusive, the point test is ``x <= px < x + ws`` (y likewise),
    the first strictly-greater count wins (y outer, x inner).

    Returns:
        tuple(int, int, int, int): (x_min, y_min, x_max, y_max), inclusive
        corners of a ``window_size``-pixel box.
    """
    min_x, max_x = int(pts_img_xy[:, 0].min()), int(pts_img_xy[:, 0].max())
    min_y, max_y = int(pts_img_xy[:, 1].min()), int(pts_img_xy[:, 1].max())
    integral = build_point_integral_image(pts_img_xy, height, width)
    best, best_window = 0, None
    for y in range(min(min_y, max_y - window_size + 1), max_y + 1):
        for x in range(min(min_x, max_x - window_size + 1), max_x + 1):
            n = integral_window_count(
                integral, height, width, x, y, x + window_size, y + window_size)
            if n > best:
                best = n
                best_window = (x, y, x + window_size - 1, y + window_size - 1)
    if best_window is None:
        return min_x, min_y, max_x, max_y
    return best_window


def select_trigger_window(in_box_pts_3d, in_box_pts_2d, bbox_2d,
                          window_size=10, n_components=1, z_thresh=1.0,
                          height=None, width=None):
    """Full DAB placement for one target object.

    When fewer than 3 points land in the box, PCA is undefined and the trigger
    is centred on the 2D bounding box (the reference's ``< 3`` fallback).

    Returns:
        tuple(int, int, int, int): inclusive (x_min, y_min, x_max, y_max) box.
    """
    if len(in_box_pts_3d) < 3:
        x_min, y_min, x_max, y_max = bbox_2d
        xc, yc = int((x_min + x_max) / 2), int((y_min + y_max) / 2)
        half = window_size // 2
        x0 = max(0, xc - half)
        y0 = max(0, yc - half)
        return x0, y0, x0 + window_size, y0 + window_size
    filtered = filter_depth_aware(in_box_pts_3d, in_box_pts_2d,
                                  n_components, z_thresh)
    return densest_window(filtered, window_size, height, width)
