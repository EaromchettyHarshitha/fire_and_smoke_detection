# AI-Based Real-Time Fire and Smoke Detection Using Computer Vision

**Final-Year Computer Science & Engineering Capstone Project**  
*An intelligent, real-time safety monitoring system integrating YOLO deep learning, OpenCV video processing, temporal multi-frame false alarm suppression, and an interactive Streamlit operations dashboard.*

---

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [Key Features](#-key-features)
3. [Technology Stack](#-technology-stack)
4. [Step-by-Step Setup Guide (Windows)](#-step-by-step-setup-guide-windows)
5. [Model Configuration & Training](#-model-configuration--training)
6. [How to Run the Application](#-how-to-run-the-application)
7. [Running the Automated Tests](#-running-the-automated-tests)
8. [Troubleshooting Common Issues](#-troubleshooting-common-issues)
9. [Academic Deliverables & Documentation](#-academic-deliverables--documentation)
10. [Safety Disclaimer](#-safety-disclaimer)

---

## 🔍 Project Overview

Conventional smoke and heat detectors exhibit inherent physical response latency: combustion particles must physically travel and accumulate inside a detector chamber, making them slow and ineffective in open outdoor spaces, high-ceiling warehouses, and ventilated environments.

This project delivers an **AI-Based Real-Time Fire and Smoke Detection System** that analyzes video feeds from standard USB webcams, pre-recorded video files, and surveillance streams. It features:
* **Real-Time Detection**: Utilizes a single-stage YOLO model for rapid feature extraction and spatial localization of fire flames and smoke plumes.
* **Temporal Multi-Frame Confirmation**: Employs a rolling-window persistence filter (e.g., requiring detection in 3 of the last 5 frames) to suppress transient optical noise, sun glare, and vehicle headlights.
* **Alert Cooldown State Machine**: Prevents alarm fatigue during continuous events by enforcing a configurable cooldown period.
* **Persistent SQLite Audit Storage**: Logs every confirmed hazard with timestamps, confidence scores, video source, and status.
* **Modern Web Interface**: Responsive Streamlit dashboard supporting live camera monitoring, image forensics, video analysis, alert history export, and model benchmarking.

---

## 🚀 Key Features

* **Live Webcam / CCTV Monitoring**: Real-time stream with bounding boxes, confidence tags, and persistent alert banners.
* **Image Detection**: Upload JPG, JPEG, or PNG images to inspect spatial bounding boxes and pixel area metrics.
* **Video File Detection**: Frame-by-frame analysis of uploaded MP4/AVI videos with configurable frame-skipping and exportable annotated video downloads.
* **Zero Fake Detections**: Genuine model inspection. Rejects generic COCO models (such as models that only detect persons, chairs, or cars) and alerts users when weights lack genuine `fire` and `smoke` classes.
* **Evaluation Engine**: Computes true object detection metrics (Precision, Recall, F1, mAP@0.5, mAP@0.5:0.95) on labeled test data without fabricating scores.
* **Hardware Latency Benchmarking**: Real-time FPS and latency measurement directly within the dashboard.

---

## 🛠️ Technology Stack

* **Language**: Python 3.9 – 3.13 (Tested on Python 3.13.4 Windows 64-bit)
* **Computer Vision**: OpenCV (`opencv-python`)
* **Deep Learning**: Ultralytics YOLOv8 / YOLOv11 & PyTorch
* **Web Dashboard**: Streamlit
* **Database**: SQLite3
* **Data Processing & Visualization**: NumPy, Pandas, Matplotlib, Pillow
* **Testing & Quality Assurance**: Pytest

---

## 💻 Step-by-Step Setup Guide (Windows)

Follow these exact steps to configure and run the project on a Windows computer:

### Step 1: Verify Python Installation
Open **PowerShell** or **Command Prompt** and verify that Python (version 3.9 or higher) is installed:
```powershell
python --version
```
*If Python is not installed, download it from [python.org](https://www.python.org/downloads/) and make sure to check **"Add Python to PATH"** during installation.*

### Step 2: Open Project Directory
Navigate into the project root directory:
```powershell
cd c:\Users\earna\OneDrive\Desktop\htmlfolder\Fire_Smoke_Detection
```

### Step 3: Create a Virtual Environment
A virtual environment isolates your project dependencies from other Python software on your PC:
```powershell
python -m venv .venv
```

### Step 4: Activate the Virtual Environment
Activate the created environment:
```powershell
.venv\Scripts\activate
```
*(When activated, you will see `(.venv)` displayed at the start of your terminal line).*

### Step 5: Upgrade Pip & Install Dependencies
Install all required libraries using the provided `requirements.txt`:
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 6: Verify System Health
Run the automated environment diagnostic tool to check your hardware, installed packages, and webcam connectivity:
```powershell
python scripts/check_environment.py
```

---

## 🧠 Model Configuration & Training

### Important: Why Generic Models Cannot Be Used
General-purpose YOLO models (such as `yolov8n.pt` trained on MS COCO) recognize 80 everyday classes (like *person, car, dog, chair*). **They cannot detect fire or smoke.** This project enforces strict model verification and will not display fake detections.

### Option A: Use a Pre-Trained Fire & Smoke Model
If you have a trained `.pt` weights file (e.g. from Roboflow Universe or Kaggle):
1. Place the weights file in the `models/` directory:
   ```text
   models/fire_smoke_yolov8n.pt
   ```
2. Validate the model classes:
   ```powershell
   python scripts/validate_model.py --model models/fire_smoke_yolov8n.pt
   ```

### Option B: Train a Custom Model Locally
1. Download an annotated Fire & Smoke dataset in **YOLOv8 format** (e.g. from Roboflow Universe).
2. Extract the dataset into the `dataset/` directory:
   ```text
   dataset/
     ├── images/ {train, val, test}
     └── labels/ {train, val, test}
   ```
3. Validate dataset integrity:
   ```powershell
   python scripts/validate_dataset.py --data config/dataset.yaml
   ```
4. Run the training script:
   ```powershell
   python scripts/train.py --epochs 25 --batch 16 --imgsz 640
   ```
   *The script will automatically copy the best weights to `models/fire_smoke_yolov8n.pt`.*

### Option C: Train in Google Colab (Free GPU)
If you do not have an NVIDIA GPU on your laptop, see the detailed instructions in [docs/DATASET_AND_TRAINING.md](docs/DATASET_AND_TRAINING.md) to train for free in Google Colab and download the `best.pt` file.

---

## 🖥️ How to Run the Application

Launch the Streamlit dashboard:
```powershell
streamlit run app.py
```
Your default web browser will automatically open to `http://localhost:8501`.

### Navigating the Application:
* **📊 Dashboard**: High-level system telemetry, active model verification, alert statistics, and recent safety logs.
* **📹 Live Camera Monitoring**: Click **"Start Monitoring"** to activate your webcam. Adjust confidence thresholds and view live temporal confirmation states.
* **🖼️ Image Detection**: Upload an image to view bounding boxes and detection coordinates.
* **🎥 Video Detection**: Upload an MP4 video to process frame-by-frame with progress tracking and export an annotated video.
* **📜 Alert History**: Review all confirmed safety alerts recorded in SQLite. Filter by type (FIRE/SMOKE) or export to CSV.
* **📈 Model Evaluation**: Run genuine metrics on labeled test data or benchmark hardware latency (FPS).

---

## 🧪 Running the Automated Tests

The project includes an automated Pytest test suite covering temporal confirmation, database operations, alert cooldowns, and model validations without requiring hardware cameras or real hazards:

```powershell
pytest -v
```

---

## 🔧 Troubleshooting Common Issues

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| `Model file not found` | The `.pt` weights file is not in `models/`. | Train a model using `python scripts/train.py` or place your weights at `models/fire_smoke_yolov8n.pt`. |
| `Incompatible classes detected` | A generic COCO model was loaded. | Use weights trained on fire/smoke classes (`0: fire`, `1: smoke`). |
| `Camera unavailable (Index 0)` | Another application (Teams, Zoom, Camera app) is using the webcam. | Close background camera apps. Try index `1` if using an external USB camera. |
| `ExecutionPolicy` error on `activate` | Windows PowerShell script execution policy. | Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` in PowerShell. |
| Low FPS on laptop CPU | Complex frame processing on high-resolution video. | Set **Frame Skip** to `2` or `3` in settings to process every 2nd or 3rd frame. |

---

## 📚 Academic Deliverables & Documentation

Complete academic project reports and viva preparation materials are available in the `docs/` folder:

* 📄 [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md) – Abstract, Introduction, Problem Statement, and Hardware/Software specs.
* 🏛️ [docs/SYSTEM_ARCHITECTURE.md](docs/SYSTEM_ARCHITECTURE.md) – Complete Mermaid architecture diagrams and subsystem design.
* 🔬 [docs/METHODOLOGY.md](docs/METHODOLOGY.md) – Algorithmic formulation, YOLO loss functions, and sliding-window mathematics.
* 📊 [docs/DATASET_AND_TRAINING.md](docs/DATASET_AND_TRAINING.md) – Dataset annotation formatting, Colab GPU workflow, and training scripts.
* 🧪 [docs/TESTING_AND_EVALUATION.md](docs/TESTING_AND_EVALUATION.md) – Pytest suite details, IoU, Precision, Recall, and mAP metrics.
* 🔮 [docs/FUTURE_SCOPE.md](docs/FUTURE_SCOPE.md) – Limitations, thermal infrared fusion, and Edge IoT deployment.
* 🎓 [docs/VIVA_QUESTIONS.md](docs/VIVA_QUESTIONS.md) – 15 comprehensive viva questions and answers for academic defense.

---

## ⚠️ Safety Disclaimer

> **Academic Prototype Notice**: This application is an educational engineering prototype developed for an academic degree. AI-based computer vision is subject to visual conditions (lighting, dust, reflective surfaces). This system **MUST NOT** replace certified industrial fire alarm, smoke sensor, or emergency sprinkler installations.
