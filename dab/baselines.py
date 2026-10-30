"""Baseline trigger placements compared against DAB in the paper.

Each returns an inclusive (x_min, y_min, x_max, y_max) pixel box for a
``window_size`` square, given a target object's 2D bounding box (and, for
BadFusion, the image-projected LiDAR points).  Painting and label editing are
done by the caller, identically to DAB, so only the *where* differs.
"""
from .trigger import build_point_integral_image, integral_window_count


def baddet_window(bbox_2d, window_size, img_h, img_w):
    """BadDet [19]: patch flush with the 2D box top-left corner."""
    x_min, y_min = bbox_2d[0], bbox_2d[1]
    x0 = max(0, int(x_min))
    y0 = max(0, int(y_min))
    return x0, y0, min(img_w, x0 + window_size), min(img_h, y0 + window_size)


def untarod_window(bbox_2d, window_size, img_h, img_w):
    """UntarOD [18] ("center"): patch centred on the 2D box."""
    x_min, y_min, x_max, y_max = bbox_2d
    xc, yc = int((x_min + x_max) / 2), int((y_min + y_max) / 2)
    half = window_size // 2
    x0 = max(0, xc - half)
    y0 = max(0, yc - half)
    return x0, y0, min(img_w, x0 + window_size), min(img_h, y0 + window_size)


def badfusion_window(bbox_2d, pts_img_xy, window_size, height, width):
    """BadFusion [20]: max projected-point density inside the 2D box.

    Density is counted over 2D-projected LiDAR points without any depth cue
    (the limitation DAB addresses).  Search ranges match the reference:
    end-exclusive sweeps, default to the box top-left.
    """
    x_min, y_min, x_max, y_max = bbox_2d
    integral = build_point_integral_image(pts_img_xy, height, width)
    best, best_x, best_y = 0, int(x_min), int(y_min)
    for y in range(int(y_min), int(y_max - window_size + 1)):
        for x in range(int(x_min), int(x_max - window_size + 1)):
            n = integral_window_count(
                integral, height, width, x, y, x + window_size, y + window_size)
            if n > best:
                best, best_x, best_y = n, x, y
    return best_x, best_y, best_x + window_size, best_y + window_size
