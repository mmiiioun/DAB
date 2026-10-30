"""Attacker label manipulations A -> A' (Section III-C of the paper).

Both operate on a LiDAR 3D box [x, y, z, dx, dy, dz, yaw] in place.
"""


def resize(box, scale=0.25):
    """Resizing attack: scale the (dx, dy, dz) dimensions by ``scale``.

    With gamma < 1 this makes the ego vehicle underestimate the object size.
    The paper uses gamma = 0.25.
    """
    box[3] = box[3] * scale
    box[4] = box[4] * scale
    box[5] = box[5] * scale
    return box


def disappear(box, delta=2.0):
    """Disappear attack: scale the (x, z) centre components by ``delta``.

    With delta >> 1 (or << 1) the box is displaced far outside the perceivable
    region, so the detector misses the object entirely.
    """
    box[0] = box[0] * delta
    box[2] = box[2] * delta
    return box
