"""Off-object trigger placement used for DAB's FPR measurement.

FPR (false-positive rate) checks that the backdoor stays quiet when the trigger
is *not* on the target object.  It is measured by painting the trigger away from
the object -- flush with the 2D box top-left corner -- and verifying the attack
does not fire.  This is a property of DAB's own evaluation protocol, not an
attack placement.
"""


def fpr_corner_window(bbox_2d, window_size, img_h, img_w):
    """Window flush with the 2D box top-left corner (off the object)."""
    x0 = max(0, int(bbox_2d[0]))
    y0 = max(0, int(bbox_2d[1]))
    return x0, y0, min(img_w, x0 + window_size), min(img_h, y0 + window_size)
