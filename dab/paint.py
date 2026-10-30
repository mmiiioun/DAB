"""Paint a solid-colour trigger into an image tensor.

The three detectors consume BGR images (mmdet3d ``to_rgb=False``) or RGB
(EPNet PIL ``convert('RGB')``).  The paper's trigger colour is given as the RGB
tuple (127, 251, 80) (fluorescent green).  ``paint_trigger`` writes it so the
*displayed* colour is correct regardless of channel order, by taking the
tuple as (R, G, B) and assigning to channels according to ``channel_order``.
"""


def paint_trigger(img_chw, box, color_rgb=(127, 251, 80), channel_order='bgr'):
    """Paint ``color_rgb`` into ``img_chw`` over an inclusive pixel ``box``.

    Args:
        img_chw: (3, H, W) array/tensor supporting slice assignment.
        box (tuple): inclusive (x_min, y_min, x_max, y_max).
        color_rgb (tuple): trigger colour as (R, G, B).
        channel_order (str): 'bgr' (mmdet3d/OpenPCDet) or 'rgb' (EPNet).
    """
    x_min, y_min, x_max, y_max = box
    r, g, b = color_rgb
    ch = (b, g, r) if channel_order == 'bgr' else (r, g, b)
    for c in range(3):
        img_chw[c, y_min:y_max + 1, x_min:x_max + 1] = ch[c]
    return img_chw
