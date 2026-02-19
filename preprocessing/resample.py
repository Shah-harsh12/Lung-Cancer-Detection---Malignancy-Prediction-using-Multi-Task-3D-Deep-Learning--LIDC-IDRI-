# preprocessing/resample.py

import numpy as np
from scipy.ndimage import zoom


def resample_volume(volume, spacing, new_spacing=(1.0, 1.0, 1.0)):

    resize_factor = np.array(spacing) / np.array(new_spacing)
    new_shape = np.round(volume.shape * resize_factor)
    real_resize = new_shape / volume.shape

    volume_resampled = zoom(volume, real_resize, order=1)

    return volume_resampled.astype(np.float32), new_spacing
