#Real-time# 🏗️ Concrete Crack & Surface Defect Instance Segmentation using YOLO11

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Ultralytics YOLO11](https://img.shields.io/badge/Ultralytics-YOLO11-00FFFF.svg)](https://docs.ultralytics.com/)
[![Computer Vision](https://img.shields.io/badge/Domain-Computer%20Vision-green.svg)](#)
[![Application](https://img.shields.io/badge/Industry-Civil%20%26%20Structural%20QC-orange.svg)](#)

An end-to-end Deep Learning and Computer Vision pipeline designed for automated detection, pixel-precise localization, and instance segmentation of concrete cracks and structural surface defects in civil engineering infrastructure. 

Powered by **YOLO11 Nano Segmentation (`yolo11n-seg`)**, this system enables real-time structural health monitoring (SHM), automated Quality Control (QC) during building inspections, and precise defect quantification.


## 🎯 Executive Overview

Structural health inspection of concrete elements (walls, bridges, beams, and pavements) is traditionally manual, subjective, time-consuming, and prone to human error. Unaddressed micro-cracks can lead to water infiltration, reinforcement corrosion, and major structural failures.

This project addresses this challenge by deploying state-of-the-art **Instance Segmentation** technology to:
1. **Detect** structural defects and cracks on concrete surfaces automatically.
2. **Segment** the exact boundaries (polygons) of every individual crack at pixel level.
3. **Quantify** defect severity to streamline engineering auditing and automated Quality Assurance (QA/QC) reports.

---

## ✨ Key Features

- ⚡ **Real-Time Instance Segmentation**: Utilizes the ultra-lightweight YOLO11 Nano architecture (`yolo11n-seg`) optimized for rapid inference without compromising spatial accuracy.
- 📐 **High-Resolution Inspection**: Trained at high image resolution ($1024 \times 1024$) to capture fine, hairline structural cracks.
- 🔁 **Automated Pipeline**: Includes dataset fetching, YAML configuration parsing, training execution, evaluation metric aggregation, and weight packaging.
- 📦 **Weight Management**: Automatic tracking, renaming, and compression of the optimal training checkpoint (`best.pt` $\rightarrow$ `latest_best_model.pt`).
- 🛡️ **Early Stopping Protection**: Configured with early stopping patience to prevent overfitting during long training runs.

---

## ⚙️ System Architecture & Workflow


                     ┌───────────────────────┐
                     │  Raw Concrete Images  │
                     └───────────┬───────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │ Dataset Preprocessing │ ◄── Dynamic YAML Config
                     └───────────┬───────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │  YOLO11n-Seg Model    │ ◄── Transfer Learning
                     │  (1024x1024 Training) │
                     └───────────┬───────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │ Dual-Task Validation  │
                     └───────────┬───────────┘
                                 │
                  ┌──────────────┴──────────────┐
                  ▼                             ▼
       ┌────────────────────┐        ┌────────────────────┐
       │ Box Detect Metrics │        │  Mask Seg Metrics  │
       │ (mAP50, mAP50-95)  │        │ (mAP50, mAP50-95)  │
       └──────────┬─────────┘        └──────────┬─────────┘
                  │                             │
                  └──────────────┬──────────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │  Packaging & Export   │
                     └───────────┬───────────┘
                                 │
                  ┌──────────────┴──────────────┐
                  ▼                             ▼
       ┌────────────────────┐        ┌────────────────────┐
       │latest_best_model.pt│        │latest_yolo_results │
       └────────────────────┘        └────────────────────┘

---

## 🛠️ Tech Stack

| Category | Technology / Library |
| :--- | :--- |
| **Language** | Python 3.10+ |
| **Deep Learning Framework** | [PyTorch](https://pytorch.org/) |
| **Vision Model Framework** | [Ultralytics YOLO11](https://docs.ultralytics.com/) |
| **Image & Matrix Processing** | OpenCV, NumPy, Pillow |
| **Data Manipulation** | Pandas, PyYAML |
| **Hardware Acceleration** | NVIDIA CUDA GPU Acceleration |

---

## 📊 Dataset Specifications

The model is trained on the industrial **Crack Segmentation Dataset**:
- **Task Type**: Instance Segmentation (Polygon Mask Coordinates).
- **Target Class**: `Crack_Defect` / Concrete Surface Defect.
- **Format**: Ultralytics YOLO Segmentation Format with automated `.yaml` dataset configuration discovery.
- **Validation**: Separate evaluation set for dynamic performance auditing after epoch completion.

---

## 🚀 Training & Hyperparameter Configuration

The model was fine-tuned using transfer learning on pre-trained `yolo11n-seg.pt` weights with hyperparameter configurations tailored for high-resolution defect detection:

```python
train_results = model.train(
    data=DATASET_YAML,       # Path to crack segmentation YAML config
    epochs=250,              # Total maximum epochs
    imgsz=1024,              # High image resolution for fine crack feature extraction
    batch=16,                # Batch size
    optimizer="SGD",         # Stochastic Gradient Descent
    patience=12,             # Early stopping threshold
    device=0,                # CUDA GPU device ID
    project="industrial_qc", # Project output directory
    name="crack_yolo_seg",   # Run experiment identifier
    save=True,               # Save model checkpoints
    verbose=True             # Output detailed logs
)

## 🌐 Interactive Web Application (Streamlit)

Launch the interactive Web GUI to upload concrete images and perform real-time crack segmentation:

```bash
# Install app dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
