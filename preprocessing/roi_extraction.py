# preprocessing/roi_extraction.py

import numpy as np


def extract_cube(volume, center, cube_size=64):

    center = np.round(center).astype(int)

    z, y, x = center
    half = cube_size // 2

    z1, z2 = z - half, z + half
    y1, y2 = y - half, y + half
    x1, x2 = x - half, x + half

    pad_width = [
        (max(0, -z1), max(0, z2 - volume.shape[0])),
        (max(0, -y1), max(0, y2 - volume.shape[1])),
        (max(0, -x1), max(0, x2 - volume.shape[2])),
    ]

    volume = np.pad(volume, pad_width, mode="constant", constant_values=0)

    z1 += pad_width[0][0]
    z2 += pad_width[0][0]
    y1 += pad_width[1][0]
    y2 += pad_width[1][0]
    x1 += pad_width[2][0]
    x2 += pad_width[2][0]

    return volume[z1:z2, y1:y2, x1:x2]
