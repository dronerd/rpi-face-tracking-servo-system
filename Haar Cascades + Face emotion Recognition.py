#以下が開発記録である。
#  ①preparation. 
# udate the system. 
# connect to Raspberry Pi 5 via SSH remote from powershell
# sudo apt update && sudo apt upgrade -y

# ②Install system packages
# sudo apt install -y python3-picamera2 python3-opencv libatlas-base-dev

# ③Setup a python virtual environemnt to run the code in
# python3 -m venv ~/emotion-env
# source ~/emotion-env/bin/activate
# pip install --upgrade pip setuptools wheel
# pip install numpy opencv-python-headless

# ④Install Tensorflow Lite runtime
# pip install tflite-runtime

# ⑤Save the emotion_realtime.py script onto the Raspberry pi 5 module from Thonny IDE
# Code avaliable at other .txt file

# ⑥Get the face emotion recognition model from Github
# mkdir -p ~/models
# cd ~/models
# wget -O emotion_model.tflite https://raw.githubusercontent.com/neta000/emotion_detection_model/master/model.tflite

# ⑦Numpyのversionによるエラーが出たら、以下のように対処する。
# venvの中で、
# pip install --upgrade pip setuptools wheel
# pip uninstall -y numpy tflite-runtime
# pip install "numpy<2.0,>=1.25.0"
# pip install numpy==1.24.4
# ＃reinstall tflite-runtime
# pip install tflite-runtime

# ⑧Picamera2のライブラリについてエラーが出たら、以下のように処理する
# venvをdeactivateした後に、
# sudo apt update
# sudo apt install -y python3-picamera2 libcamera-apps
# # remove old venv if you want to recreate (optional)
# rm -rf ~/emotion-env
# # create venv that inherits system packages
# python3 -m venv --system-site-packages ~/emotion-env
# source ~/emotion-env/bin/activate
# # inside venv, install the rest (numpy/opencv/tflite etc.)
# pip install --upgrade pip setuptools wheel
# pip install numpy<2.0 opencv-python-headless tflite-runtime

# ここでもしnumpy<2.0のinstallでNo such file or directoryとエラーが出たら、
# 下のように対処する
# # make sure pip/setuptools/wheel are up-to-date
# pip install --upgrade pip setuptools wheel
# # (optional) remove any previous installs to avoid conflicts
# pip uninstall -y numpy tflite-runtime
# # install a numpy 1.x series and OpenCV; 
# pip install "numpy<2.0,>=1.25.0" opencv-python-headless
# pip install tflite-runtime

# ⑨コードを実行する
# python3 ~/emotion_realtime.py --model /home/pi/models/emotion_model.tflite

# ⑩Next steps:
# Improve accuracy with replacing Haar Cascades with DNN or HOG?
# Enable the implementation of code outside of Wifi local network. Enable usage at Waseda Wifi network. 
# Enable output of data to arduino via TX RX serial communication.


#!/usr/bin/env python3
"""
emotion_realtime_64x64_rgb.py

Realtime face emotion recognition for Raspberry Pi (Picamera2) using a TFLite model.
- Forces preprocessing to 64x64 RGB (suitable for neta000/emotion_detection_model).
- Handles 4-channel frames (BGRA/XBGR) by converting to 3-channel RGB.
- Robust TFLite predict() that respects actual input image shape.
- Prints human-readable lines to stdout (timestamp | FACE# | label | score | bbox).
- Includes a small per-face throttling (MIN_PRINT_INTERVAL) to reduce terminal flood.

Usage:
    python3 emotion_realtime_64x64_rgb.py --model /home/pi/models/emotion_model.tflite
    python3 emotion_realtime_64x64_rgb.py --model /home/pi/models/emotion_model.tflite --show-window
"""

import time
from datetime import datetime
import argparse
import numpy as np
import cv2

# TFLite interpreter import (preferred)
try:
    from tflite_runtime.interpreter import Interpreter
except Exception:
    # fallback to TF if tflite-runtime not installed
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

# Picamera2 (libcamera) API
from picamera2 import Picamera2

# -----------------------
# Configuration / labels
# -----------------------
DEFAULT_MODEL = "models/emotion_model.tflite"
HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Force preprocessing to this target size and channels (model expects 64x64x3)
FORCED_W = 64
FORCED_H = 64
FORCED_C = 3  # RGB

# Common FER labels (7 classes)
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Throttle printing to terminal per face to avoid flooding (seconds)
MIN_PRINT_INTERVAL = 0.5

# -----------------------
# Helpers: printing
# -----------------------
def print_emotion(face_idx, label, score, bbox):
    """
    Print a concise human-readable line to stdout.
    bbox: (x0, y0, x1, y1)
    """
    ts = datetime.now().isoformat(timespec='seconds')
    x0, y0, x1, y1 = bbox
    print(f"{ts} | FACE#{face_idx} | {label:<8} | score={score:.2f} | bbox=({x0},{y0},{x1},{y1})", flush=True)

# -----------------------
# TFLite wrapper (robust)
# -----------------------
class TFLiteEmotionModel:
    def __init__(self, model_path):
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        # Try to infer expected input shape (NHWC)
        shape = list(self.input_details[0]['shape'])  # e.g. [1,64,64,3]
        dims = shape[1:]
        if len(dims) == 3:
            self.inp_h = int(dims[0])
            self.inp_w = int(dims[1])
            self.inp_c = int(dims[2])
        elif len(dims) == 2:
            self.inp_h = int(dims[0])
            self.inp_w = int(dims[1])
            self.inp_c = 1
        else:
            # fallback to forced values
            self.inp_h = FORCED_H
            self.inp_w = FORCED_W
            self.inp_c = FORCED_C

        self.input_dtype = np.dtype(self.input_details[0]['dtype'])

    def predict(self, face_img):
        """
        face_img: numpy array in HWC (height,width,channels), channels matching provided image
        returns: 1D numpy array of logits/probabilities
        This is robust: it will expand dims based on the actual face_img shape.
        """
        x = face_img.astype(np.float32)

        # Normalize for float32 inputs
        if self.input_dtype == np.float32:
            x = x / 255.0

        # Expand dims to NHWC in a safe way using actual shape
        if x.ndim == 3:
            h, w, c = x.shape
            x = x.reshape((1, h, w, c))
        elif x.ndim == 2:
            x = x.reshape((1, x.shape[0], x.shape[1], 1))
        else:
            # fallback: try to reshape to the interpreter's expected shape
            x = x.reshape(self.input_details[0]['shape'])

        # If interpreter expects a different dtype (e.g., int8), cast
        x = x.astype(self.input_dtype)

        # If the interpreter expects a different size than provided, try to resize input tensor if supported
        # (Most TFLite interpreters require the input shape to match exactly; our script re-sizes crops beforehand.)
        self.interpreter.set_tensor(self.input_details[0]['index'], x)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_details[0]['index'])
        out = np.array(out).squeeze()
        return out

# -----------------------
# Main realtime loop
# -----------------------
def main(model_path, show_window, cam_size):
    # Load face detector
    face_cascade = cv2.CascadeClassifier(HAAR_PATH)
    if face_cascade.empty():
        raise RuntimeError("Failed to load Haar cascade from: " + HAAR_PATH)

    # Load TFLite model
    model = TFLiteEmotionModel(model_path)

    # Inform user about forced preprocessing
    print(f"Starting with model: {model_path}")
    print(f"Forcing preprocessing to {FORCED_W}x{FORCED_H} RGB (3 channels).", flush=True)
    print(f"Model interpreter input dtype: {model.input_dtype}, expected shape approx: ({model.inp_h},{model.inp_w},{model.inp_c})", flush=True)

    # Setup camera
    picam2 = Picamera2()
    preview_config = picam2.create_preview_configuration(main={"size": cam_size})
    picam2.configure(preview_config)
    picam2.start()
    time.sleep(0.5)  # warm-up

    # Per-face last print timestamps for throttling
    last_print_time = {}

    try:
        while True:
            frame = picam2.capture_array()  # could be shape (H,W,3) or (H,W,4)

            # --- Normalize frame to RGB 3-channel ---
            # Many libcamera configs return 4-channel XBGR/BGRA; convert to 3-channel RGB.
            if frame.ndim == 3 and frame.shape[2] == 4:
                # Convert BGRA/XBGR -> RGB by dropping alpha and reordering channels
                # BGRA2RGB handles many Pi camera outputs
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
                except Exception:
                    # fallback: drop last channel and assume the rest are RGB-like
                    frame_rgb = frame[:, :, :3]
            elif frame.ndim == 3 and frame.shape[2] == 3:
                # Assume frame is already RGB
                frame_rgb = frame
            elif frame.ndim == 2:
                # grayscale -> stack to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            else:
                # unexpected shape; try to coerce
                frame_rgb = frame

            # Prepare BGR copy for drawing/display if needed
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR) if show_window else None

            # Detect faces using grayscale
            gray_for_detect = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray_for_detect, scaleFactor=1.3, minNeighbors=5)

            for i, (x, y, w, h) in enumerate(faces):
                # expand slightly for margin
                pad = int(0.1 * w)
                x0 = max(x - pad, 0)
                y0 = max(y - pad, 0)
                x1 = min(x + w + pad, frame_rgb.shape[1])
                y1 = min(y + h + pad, frame_rgb.shape[0])

                # Crop the face from the RGB frame (guaranteed 3 channels by above)
                face_crop_rgb = frame_rgb[y0:y1, x0:x1]
                if face_crop_rgb.size == 0:
                    continue

                # If the crop somehow has 4 channels still, convert to 3
                if face_crop_rgb.ndim == 3 and face_crop_rgb.shape[2] == 4:
                    try:
                        face_crop_rgb = cv2.cvtColor(face_crop_rgb, cv2.COLOR_BGRA2RGB)
                    except Exception:
                        face_crop_rgb = face_crop_rgb[:, :, :3]

                # Force resize to expected 64x64 RGB
                face_resized = cv2.resize(face_crop_rgb, (FORCED_W, FORCED_H))

                # Run inference (robust)
                preds = model.predict(face_resized)
                if preds.ndim == 0:
                    probs = np.array([preds])
                elif preds.ndim == 1:
                    probs = preds
                else:
                    probs = preds.flatten()

                idx = int(np.argmax(probs))
                label = EMOTIONS[idx] if idx < len(EMOTIONS) else str(idx)
                score = float(probs[idx])

                # Throttle prints per face
                now = time.time()
                last = last_print_time.get(i, 0)
                if now - last >= MIN_PRINT_INTERVAL:
                    print_emotion(i, label, score, (x0, y0, x1, y1))
                    last_print_time[i] = now

                # draw box & label if window requested
                if show_window:
                    cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), (0, 255, 0), 2)
                    txt = f"{label} {score:.2f}"
                    cv2.putText(frame_bgr, txt, (x0, max(y0-8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # show preview if requested
            if show_window:
                cv2.imshow("Emotion Recognition", frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        # graceful exit on Ctrl+C
        pass
    finally:
        picam2.stop()
        if show_window:
            cv2.destroyAllWindows()

# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Realtime face emotion recognition (64x64 RGB preprocessing).")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Path to TFLite model file")
    parser.add_argument("--show-window", action="store_true", help="Show preview window (requires desktop/X).")
    parser.add_argument("--cam-width", type=int, default=640, help="Camera preview width")
    parser.add_argument("--cam-height", type=int, default=480, help="Camera preview height")
    args = parser.parse_args()

    cam_size = (args.cam_width, args.cam_height)
    print(f"Starting emotion_realtime_64x64_rgb.py -- model={args.model}  show_window={args.show_window}  cam_size={cam_size}", flush=True)
    main(args.model, args.show_window, cam_size)

