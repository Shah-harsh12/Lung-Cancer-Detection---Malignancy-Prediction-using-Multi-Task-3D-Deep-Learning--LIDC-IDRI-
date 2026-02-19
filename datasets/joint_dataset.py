# datasets/joint_dataset.py

import torch
from torch.utils.data import Dataset
import random
import numpy as np


class JointLungDataset(Dataset):
    """
    Joint Dataset for:
        - Detection (binary classification)
        - Malignancy (regression)

    Returns:
        cube_tensor (1, 64, 64, 64)
        detection_label (0 or 1)
        malignancy_label (float)
    """

    def __init__(
        self,
        dataset_path,
        augment=False,
    ):
        self.data = torch.load(dataset_path)
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def random_flip(self, cube):
        if random.random() < 0.5:
            cube = torch.flip(cube, dims=[1])
        if random.random() < 0.5:
            cube = torch.flip(cube, dims=[2])
        if random.random() < 0.5:
            cube = torch.flip(cube, dims=[3])
        return cube

    def random_noise(self, cube):
        if random.random() < 0.3:
            noise = torch.randn_like(cube) * 0.02
            cube = cube + noise
        return cube

    def __getitem__(self, idx):

        cube, detection_label, malignancy_label, _ = self.data[idx]

        cube = cube.float()

        if self.augment:
            cube = self.random_flip(cube)
            cube = self.random_noise(cube)

        detection_label = torch.tensor(detection_label).long()
        malignancy_label = torch.tensor(malignancy_label).float()

        return cube, detection_label, malignancy_label
