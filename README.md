# Raspberry Pi Face Tracking & Emotion-Controlled Servo System

## Overview

This project implements a **real-time edge AI system** on a Raspberry Pi that evolves across three stages:

### 1. Face Tracking

**`face_tracking.py`**

* Detects faces using OpenCV (Haar Cascades)
* Outputs face center coordinates and offsets

### 2. Emotion Recognition

**`emotion_recognition.py`**

* Adds TensorFlow Lite emotion classification
* Outputs emotion labels and confidence scores

### 3. Servo Control System (Full Integration)

**`servo_control_system.py`**

* Combines face tracking + emotion recognition
* Controls 6 servos based on position and emotion

---

## System Architecture

Camera → Face Detection → Emotion Model → Decision Logic → Servo Control

---

## Features

* Real-time face detection (OpenCV)
* Emotion recognition (TensorFlow Lite, optimized for edge devices)
* Multi-servo control (GPIO PWM)
* Works over SSH (headless mode)
* Modular design (3 progressive scripts)

---

## Hardware Requirements

* Raspberry Pi 5 (recommended)
* Raspberry Pi Camera Module
* 6x Servo Motors
* External power supply for servos (important)

---

## Installation

### 1. Update system

```bash
sudo apt update && sudo apt upgrade -y
```

### 2. Install system dependencies

```bash
sudo apt install -y python3-picamera2 python3-opencv libatlas-base-dev
```

### 3. Create virtual environment

```bash
python3 -m venv --system-site-packages ~/env
source ~/env/bin/activate
```

### 4. Install Python packages

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

## Model Setup

```bash
mkdir -p ~/models
cd ~/models

wget -O emotion_model.tflite \
https://raw.githubusercontent.com/neta000/emotion_detection_model/master/model.tflite
```

---

## Usage

### 1. Face Tracking

```bash
python3 face_tracking.py
```

### 2. Emotion Recognition

```bash
python3 emotion_recognition.py --model ~/models/emotion_model.tflite
```

### 3. Full System (Servo Control)

```bash
python3 servo_control_system.py --model ~/models/emotion_model.tflite
```

Optional display:

```bash
--show-window
```

---

## Servo Behavior

| Servo | Function                     |
| ----- | ---------------------------- |
| 0     | Triggered by "angry" emotion |
| 1     | Triggered by "happy" emotion |
| 2     | Other emotions               |
| 3,5   | Vertical tracking (Y-axis)   |
| 4     | Horizontal tracking (X-axis) |

---

## Troubleshooting

### NumPy Version Issues

```bash
pip uninstall -y numpy
pip install "numpy<2.0,>=1.24.4"
```

### Picamera2 Issues

Recreate environment:

```bash
python3 -m venv --system-site-packages ~/env
```

---

## Future Improvements

* Replace Haar Cascade with deep learning model (YOLO / SSD)
* Add PID control for smoother servo movement
* Web dashboard (FastAPI + live streaming)
* Multi-face tracking with IDs
* Arduino / serial communication

---

## Notes

* First few seconds may be unstable (camera auto-adjust)
* Lighting affects emotion accuracy
* Designed for edge deployment on Raspberry Pi

---

