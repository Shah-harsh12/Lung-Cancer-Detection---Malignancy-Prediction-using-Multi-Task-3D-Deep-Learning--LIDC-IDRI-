# datasets/split_manager.py

import torch
import os
import random


def create_patient_split(
    joint_dataset_path="processed_data/joint_dataset.pt",
    train_ratio=0.8,
    save_path="processed_data",
    seed=42,
):

    random.seed(seed)

    data = torch.load(joint_dataset_path)

    # -----------------------------
    # Extract unique patient IDs
    # -----------------------------
    patients = list(set([item[3] for item in data]))
    print("Total unique patients:", len(patients))

    random.shuffle(patients)

    split_idx = int(len(patients) * train_ratio)

    train_patients = set(patients[:split_idx])
    val_patients = set(patients[split_idx:])

    # -----------------------------
    # Split samples
    # -----------------------------
    train_data = [item for item in data if item[3] in train_patients]
    val_data = [item for item in data if item[3] in val_patients]

    print("Train samples:", len(train_data))
    print("Validation samples:", len(val_data))

    # -----------------------------
    # Save
    # -----------------------------
    os.makedirs(save_path, exist_ok=True)

    torch.save(train_data, os.path.join(save_path, "train_joint.pt"))
    torch.save(val_data, os.path.join(save_path, "val_joint.pt"))

    print("\nPatient-level split completed successfully.")
