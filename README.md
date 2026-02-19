# 🫁 Lung Cancer Detection & Malignancy Prediction using Multi-Task 3D Deep Learning

A research-grade Computer-Aided Diagnosis (CAD) framework for automatic lung nodule detection and malignancy prediction using thoracic CT scans from the LIDC-IDRI dataset.

This project implements a modular deep learning pipeline including DICOM preprocessing, lung segmentation, 3D cube extraction, isotropic resampling, HU normalization, patient-level data splitting, and a multi-task 3D CNN architecture.

---

# 📌 Project Highlights

- ✅ 3D Multi-Task Deep Learning Model  
- ✅ Nodule Detection (Binary Classification)  
- ✅ Malignancy Score Prediction (Regression)  
- ✅ Patient-Level Train/Validation Split  
- ✅ Modular & Reproducible Pipeline  
- ✅ Mixed Precision GPU Training  
- ✅ Research-Oriented Code Structure  

---

# 📊 Dataset

This project uses the **LIDC-IDRI (Lung Image Database Consortium and Image Database Resource Initiative)** dataset.

## Dataset Details

- 1018 thoracic CT scans  
- 4 radiologist annotations per scan  
- Malignancy rating scale: 1–5  
- DICOM format  
- Public research dataset  

> ⚠ Raw CT data is not included in this repository due to size constraints.

---

# 🏗️ Pipeline Overview

DICOM CT Scan
↓
Lung Segmentation
↓
3D Cube Extraction (64×64×64)
↓
Resampling to 1mm³
↓
HU Normalization
↓
Dataset Builder
↓
3D CNN Backbone
↓
├── Detection Head (Binary Classification)
└── Malignancy Head (Regression)


---

# 🧠 Model Architecture

### Input
`(1 × 64 × 64 × 64)` CT cube

### Backbone
3D Residual Convolutional Network

### Heads
- Detection Head → Binary classification (Nodule vs Background)  
- Malignancy Head → Continuous malignancy score prediction  

### Multi-Task Loss

\[
L = \lambda_{det} L_{BCE} + \lambda_{mal} L_{SmoothL1}
\]

---

# 📈 Validation Results

## 🫁 Detection Performance

| Metric | Value |
|--------|--------|
| ROC-AUC | **0.906** |
| Accuracy | 0.859 |
| Precision | 0.777 |
| Recall | 0.979 |
| F1 Score | 0.867 |

Confusion Matrix:
[[462 150]
[ 11 524]]

High recall is prioritized for medical safety.

---

## 🧬 Malignancy Regression Performance

| Metric | Value |
|--------|--------|
| MAE | 0.72 |
| RMSE | 1.11 |
| Pearson Correlation | 0.686 |

Average prediction error is less than one radiologist rating level.

---

# 🛠️ Tech Stack

- Python 3.10  
- PyTorch (CUDA-enabled)  
- NVIDIA RTX A4000 (16GB VRAM)  
- Mixed Precision (AMP)  
- AdamW Optimizer  
- Cosine Annealing Scheduler  
- Scikit-learn  
- Matplotlib  

---

# 📂 Project Structure

configs/
data/
processed_data/
preprocessing/
datasets/
models/
training/
evaluation/
explainability/
utils/
notebooks/
weights/
logs/


---

# 🚀 How to Run

## 1️⃣ Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install numpy scipy scikit-learn matplotlib tqdm pylidc

## Build Dataset (Stage 1)
python preprocessing/dataset_builder.py

## Train Multi-Task Model
python training/train_dual_head.py

## Evaluate Model
python evaluation/metrics.py
```
project_metadata:
  title: "Lung Cancer Detection & Malignancy Prediction using Multi-Task 3D Deep Learning"
  domain: "Medical Imaging AI"
  dataset: "LIDC-IDRI"
  author:
    name: "Harsh Shah"
    specialization: "Medical Imaging & AI Research"
    year: 2026

research_motivation:
  objective:
    - Detect pulmonary nodules automatically
    - Predict malignancy likelihood
    - Assist radiologists in decision-making

validation_results:
  detection:
    roc_auc: 0.906
    accuracy: 0.859
    recall: 0.979
  malignancy_regression:
    mae: 0.72
    pearson_correlation: 0.686

hardware_environment:
  gpu: "NVIDIA RTX A4000 (16GB VRAM)"
  optimizer: "AdamW"
  scheduler: "CosineAnnealingLR"
  precision: "Mixed Precision (AMP)"
  epochs: 30
