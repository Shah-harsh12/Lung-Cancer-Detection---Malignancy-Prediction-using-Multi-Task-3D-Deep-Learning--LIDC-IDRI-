# preprocessing/lung_segmentation.py

import numpy as np
from scipy import ndimage as ndi


def segment_lungs(volume):
    """
    Basic lung segmentation using thresholding.
    Returns masked volume.
    """

    binary = volume < -320

    # Remove small components
    labels = ndi.label(binary)[0]
    sizes = ndi.sum(binary, labels, range(labels.max() + 1))
    mask = sizes > 5000
    cleaned = mask[labels]

    return volume * cleaned
