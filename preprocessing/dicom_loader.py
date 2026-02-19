# preprocessing/dicom_loader.py

import os
import numpy as np
import pydicom
from glob import glob


def load_dicom_series(series_path):

    dicom_files = glob(os.path.join(series_path, "*.dcm"))
    slices = [pydicom.dcmread(f) for f in dicom_files]

    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    volume = np.stack([s.pixel_array for s in slices]).astype(np.int16)

    for i, s in enumerate(slices):
        intercept = float(s.RescaleIntercept)
        slope = float(s.RescaleSlope)
        volume[i] = volume[i] * slope + intercept

    try:
        z_spacing = abs(
            float(slices[1].ImagePositionPatient[2]) -
            float(slices[0].ImagePositionPatient[2])
        )
    except:
        z_spacing = float(slices[0].SliceThickness)

    y_spacing = float(slices[0].PixelSpacing[0])
    x_spacing = float(slices[0].PixelSpacing[1])

    spacing = (z_spacing, y_spacing, x_spacing)

    return volume, spacing
