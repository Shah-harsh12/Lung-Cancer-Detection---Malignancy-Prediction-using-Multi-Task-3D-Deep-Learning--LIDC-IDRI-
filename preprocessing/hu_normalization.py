# preprocessing/hu_normalization.py

import numpy as np


def normalize_hu(volume, min_hu=-1000, max_hu=400):
    """
    Clip and normalize HU to 0-1
    """

    volume = np.clip(volume, min_hu, max_hu)
    volume = (volume - min_hu) / (max_hu - min_hu)
    return volume.astype(np.float32)
