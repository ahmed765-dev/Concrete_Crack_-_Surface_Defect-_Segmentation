#Real-time# 🏗️ Concrete Crack & Surface Defect Instance Segmentation using YOLO11

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Ultralytics YOLO11](https://img.shields.io/badge/Ultralytics-YOLO11-00FFFF.svg)](https://docs.ultralytics.com/)
[![Computer Vision](https://img.shields.io/badge/Domain-Computer%20Vision-green.svg)](#)
[![Application](https://img.shields.io/badge/Industry-Civil%20%26%20Structural%20QC-orange.svg)](#)

An end-to-end Deep Learning and Computer Vision pipeline designed for automated detection, pixel-precise localization, and instance segmentation of concrete cracks and structural surface defects in civil engineering infrastructure. 

Powered by **YOLO11 Nano Segmentation (`yolo11n-seg`)**, this system enables real-time structural health monitoring (SHM), automated Quality Control (QC) during building inspections, and precise defect quantification.

---

## 📌 Table of Contents
- [Executive Overview](#-executive-overview)
- [Key Features](#-key-features)
- [System Architecture & Workflow](#-system-architecture--workflow)
- [Tech Stack](#-tech-stack)
- [Dataset Specifications](#-dataset-specifications)
- [Training & Hyperparameter Configuration](#-training--hyperparameter-configuration)
- [Model Evaluation & Performance](#-model-evaluation--performance)
- [Repository Structure](#-repository-structure)
- [Installation & Setup](#-installation--setup)
- [Usage Guide](#-usage-guide)
- [Model Export & Deployment](#-model-export--deployment)
- [License & Citation](#-license--citation)

---

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
