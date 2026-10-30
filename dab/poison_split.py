"""Select which training frames get poisoned.

A fixed fraction of the training frames (10% in the paper) is chosen once, with
a fixed trigger seed (42), so the poisoned set is identical across detectors.
"""
import numpy as np


def select_poison_ids(sample_ids, ratio=0.1, seed=42):
    """Pick a reproducible subset of frame ids to poison.

    Args:
        sample_ids (Sequence[int]): all candidate training frame indices.
        ratio (float): fraction to poison (0.1 in the paper).
        seed (int): trigger seed (42 in the paper).

    Returns:
        set[str]: zero-padded 6-digit frame ids to poison.
    """
    sample_ids = np.asarray(sorted(int(s) for s in sample_ids))
    n = int(round(len(sample_ids) * ratio))
    rng = np.random.RandomState(seed)
    chosen = rng.choice(sample_ids, size=n, replace=False)
    return {f'{int(i):06d}' for i in chosen}
