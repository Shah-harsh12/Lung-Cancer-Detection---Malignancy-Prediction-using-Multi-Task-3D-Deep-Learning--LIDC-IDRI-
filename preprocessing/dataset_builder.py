# preprocessing/dataset_builder.py

import os
import torch
import numpy as np
import pylidc as pl
from tqdm import tqdm
from scipy.ndimage import zoom

from preprocessing.dicom_loader import load_dicom_series
from preprocessing.lung_segmentation import segment_lungs
from preprocessing.hu_normalization import normalize_hu
from preprocessing.roi_extraction import extract_cube


DATA_PATH = "data/LIDC-IDRI"
SAVE_PATH = "processed_data"
CUBE_SIZE = 64
TARGET_SPACING = (1.0, 1.0, 1.0)
NEGATIVE_PER_SCAN = 3


# -----------------------------
# Helper: Resample Cube Only
# -----------------------------
def resample_cube(cube, original_spacing, target_spacing):
    resize_factor = np.array(original_spacing) / np.array(target_spacing)
    new_shape = np.round(np.array(cube.shape) * resize_factor)
    real_resize = new_shape / np.array(cube.shape)

    cube_resampled = zoom(cube, real_resize, order=1)
    return cube_resampled


# -----------------------------
# Helper: Resize to Fixed Size
# -----------------------------
def resize_to_fixed(cube, target_size=64):
    factors = np.array([target_size, target_size, target_size]) / np.array(cube.shape)
    cube_resized = zoom(cube, factors, order=1)
    return cube_resized


# -----------------------------
# Main Stage-1 Builder
# -----------------------------
def build_dataset():

    os.makedirs(SAVE_PATH, exist_ok=True)

    detection_data = []
    malignancy_data = []
    joint_data = []

    scans = pl.query(pl.Scan).all()
    print("Total scans in pylidc DB:", len(scans))

    for scan in tqdm(scans):

        series_uid = scan.series_instance_uid
        series_folder = os.path.join(DATA_PATH, series_uid)

        if not os.path.exists(series_folder):
            continue

        try:
            # 1️⃣ Load original HU volume
            volume, spacing = load_dicom_series(series_folder)

            # 2️⃣ Segment lungs (still in HU space)
            volume = segment_lungs(volume)

            nodules = scan.cluster_annotations()

            # -------------------------
            # 3️⃣ Positive Cubes
            # -------------------------
            for nodule in nodules:

                malignancies = [ann.malignancy for ann in nodule]
                malignancy_score = float(np.mean(malignancies))

                # pylidc centroid: (row, col, slice)
                c = np.array(nodule[0].centroid)

                # Convert to (slice, row, col)
                centroid = np.array([c[2], c[0], c[1]])

                cube = extract_cube(volume, centroid, CUBE_SIZE)

                if cube.shape != (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE):
                    continue

                # 4️⃣ Resample cube only
                cube = resample_cube(cube, spacing, TARGET_SPACING)

                # 5️⃣ Resize to fixed 64³
                cube = resize_to_fixed(cube, CUBE_SIZE)

                # 6️⃣ Normalize
                cube = normalize_hu(cube)

                cube_tensor = torch.tensor(cube).unsqueeze(0)

                detection_data.append((cube_tensor, 1))
                malignancy_data.append((cube_tensor, malignancy_score))
                joint_data.append(
                    (cube_tensor, 1, malignancy_score, series_uid)
                )

            # -------------------------
            # 4️⃣ Negative Cubes
            # -------------------------
            for _ in range(NEGATIVE_PER_SCAN):

                z = np.random.randint(32, volume.shape[0] - 32)
                y = np.random.randint(32, volume.shape[1] - 32)
                x = np.random.randint(32, volume.shape[2] - 32)

                cube = extract_cube(volume, (z, y, x), CUBE_SIZE)

                if cube.shape != (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE):
                    continue

                cube = resample_cube(cube, spacing, TARGET_SPACING)
                cube = resize_to_fixed(cube, CUBE_SIZE)
                cube = normalize_hu(cube)

                cube_tensor = torch.tensor(cube).unsqueeze(0)

                detection_data.append((cube_tensor, 0))
                malignancy_data.append((cube_tensor, 0.0))
                joint_data.append(
                    (cube_tensor, 0, 0.0, series_uid)
                )

        except Exception as e:
            print(f"Error processing {series_uid}: {e}")
            continue

    # -------------------------
    # Save datasets
    # -------------------------
    torch.save(detection_data, os.path.join(SAVE_PATH, "detection_dataset.pt"))
    torch.save(malignancy_data, os.path.join(SAVE_PATH, "malignancy_dataset.pt"))
    torch.save(joint_data, os.path.join(SAVE_PATH, "joint_dataset.pt"))

    print("\nStage-1 completed.")
    print("Detection samples:", len(detection_data))
    print("Malignancy samples:", len(malignancy_data))
