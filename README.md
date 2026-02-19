##🫁 Lung Cancer Detection & Malignancy Prediction using Multi-Task 3D Deep Learning
🚀 Research-Grade CAD System using LIDC-IDRI CT Scans

This project presents a modular, research-grade Computer-Aided Diagnosis (CAD) system for automatic lung nodule detection and malignancy prediction using 3D deep learning on thoracic CT scans.

The system is built using the LIDC-IDRI dataset and implements a complete end-to-end pipeline from DICOM preprocessing to multi-task neural network training and evaluation.

📌 Key Features

3D CT DICOM preprocessing pipeline

Lung segmentation & HU normalization

Physically consistent cube extraction (64×64×64)

Patient-level train/validation split (no data leakage)

Multi-task 3D CNN architecture

Joint detection + malignancy prediction

Mixed precision GPU training (AMP)

Modular research-ready structure

Clean evaluation metrics (AUC, MAE, Correlation)

📊 Dataset

LIDC-IDRI (Lung Image Database Consortium Image Collection)

1018 thoracic CT scans

4 radiologist annotations per scan

Nodule malignancy ratings (1–5 scale)

DICOM format

Public research dataset

🏗️ Project Architecture
DICOM CT Scan
   ↓
Lung Segmentation
   ↓
Cube Extraction (64³)
   ↓
Resampling to 1mm³
   ↓
HU Normalization
   ↓
Dataset Builder
   ↓
3D CNN Backbone
   ↓
 ┌───────────────┬───────────────┐
 │ Detection Head │ Malignancy Head │
 └───────────────┴───────────────┘

🧠 Model Architecture
🔹 Backbone

3D Residual CNN

Global Average Pooling

256-dim feature representation

🔹 Detection Head

Binary classification

BCEWithLogitsLoss

🔹 Malignancy Head

Regression output

SmoothL1Loss

🔹 Multi-Task Loss
𝐿
=
𝜆
𝑑
𝑒
𝑡
𝐿
𝐵
𝐶
𝐸
+
𝜆
𝑚
𝑎
𝑙
𝐿
𝑆
𝑚
𝑜
𝑜
𝑡
ℎ
𝐿
1
L=λ
det
	​

L
BCE
	​

+λ
mal
	​

L
SmoothL1
	​

⚙️ Training Configuration

PyTorch 2.x

CUDA acceleration

Mixed Precision (AMP)

AdamW Optimizer

CosineAnnealingLR Scheduler

Gradient Clipping

Batch size: 12

Epochs: 30

Hardware: NVIDIA RTX A4000 (16GB)

📈 Validation Results
🔎 Detection Performance

ROC-AUC: 0.906

Accuracy: 0.859

Precision: 0.777

Recall: 0.979

F1-score: 0.867

Confusion Matrix:

[[462 150]
 [ 11 524]]


High recall ensures minimal missed nodules (low false negatives).

📉 Malignancy Regression Performance

MAE: 0.72

RMSE: 1.11

Pearson Correlation: 0.686

Average prediction error is less than one radiologist rating level.

🗂️ Project Structure
configs/
preprocessing/
datasets/
models/
training/
evaluation/
explainability/
utils/
notebooks/
train.py


Raw CT data, processed datasets, logs, and model weights are excluded via .gitignore.

🚀 How to Run
1️⃣ Install Requirements
pip install torch torchvision torchaudio
pip install numpy scipy scikit-learn matplotlib tqdm pylidc

2️⃣ Build Dataset
from preprocessing.dataset_builder import build_dataset
build_dataset()

3️⃣ Train Model
from training.train_dual_head import train
train()

4️⃣ Evaluate Model
from evaluation.metrics import evaluate_model

🔬 Research Contributions

Correct voxel-to-physical space handling

Stable multi-task 3D CNN

Clean patient-level split (no leakage)

Reproducible modular pipeline

Strong baseline (>0.90 AUC)

🔮 Future Work

External validation (LUNA16)

Focal loss for detection

Malignancy loss on positive samples only

Full CT scan detection (sliding window)

3D Grad-CAM explainability

Self-supervised pretraining (JEPA)

👨‍💻 Author

Harsh Shah
AI/ML Researcher – Medical Imaging

📜 License

This project is for research and educational purposes.
