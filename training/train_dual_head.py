# training/train_dual_head.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from datasets.joint_dataset import JointLungDataset
from models.dual_head_model import DualHeadModel
from training.losses import MultiTaskLoss


# -----------------------------------
# Training Configuration
# -----------------------------------
BATCH_SIZE = 12
EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 5.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_PATH = "processed_data/train_joint.pt"
VAL_PATH = "processed_data/val_joint.pt"

SAVE_PATH = "weights/best_dual_head.pth"
LOG_DIR = "logs/tensorboard/dual_head"


# -----------------------------------
# Training Function
# -----------------------------------
def train():

    os.makedirs("weights", exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    writer = SummaryWriter(LOG_DIR)

    # Datasets
    train_dataset = JointLungDataset(TRAIN_PATH, augment=True)
    val_dataset = JointLungDataset(VAL_PATH, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = DualHeadModel(feature_dim=256).to(DEVICE)

    # Loss
    criterion = MultiTaskLoss(lambda_det=1.0, lambda_mal=0.5)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Mixed Precision
    scaler = GradScaler()

    best_val_loss = float("inf")

    print("Training on:", DEVICE)

    for epoch in range(EPOCHS):

        # -------------------------
        # Training
        # -------------------------
        model.train()
        train_loss = 0.0

        for cubes, det_labels, mal_labels in tqdm(train_loader):

            cubes = cubes.to(DEVICE)
            det_labels = det_labels.to(DEVICE)
            mal_labels = mal_labels.to(DEVICE)

            optimizer.zero_grad()

            with autocast():

                det_logits, mal_pred = model(cubes)

                loss, det_loss, mal_loss = criterion(
                    det_logits,
                    mal_pred,
                    det_labels,
                    mal_labels
                )

            scaler.scale(loss).backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # -------------------------
        # Validation
        # -------------------------
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for cubes, det_labels, mal_labels in val_loader:

                cubes = cubes.to(DEVICE)
                det_labels = det_labels.to(DEVICE)
                mal_labels = mal_labels.to(DEVICE)

                det_logits, mal_pred = model(cubes)

                loss, _, _ = criterion(
                    det_logits,
                    mal_pred,
                    det_labels,
                    mal_labels
                )

                val_loss += loss.item()

        val_loss /= len(val_loader)

        scheduler.step()

        print(f"\nEpoch [{epoch+1}/{EPOCHS}]")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")

        # TensorBoard
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print("Best model saved.")

    writer.close()
    print("Training completed.")
