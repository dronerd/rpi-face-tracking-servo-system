# 使用した方法：
# 1: ラズパイの電源を入れる
# 2: PowershellでSSH接続。
# 3: Thonnyで以下①のコードをラズパイにface_positions.pyとして保存した。
# 4: Powershellでpython3 face_positions.pyと打ってプログラムを実行

# 注意事項：
# OpenCVのライブラリをインストールした、
# HaarCascadesのプログラムは自動でインスト―ルされなかったため、Githubから直接インストールした。
# カメラを使用するためのライブラリは、libcameraではなく、Picamera2を使っている。
# 母と父の顔でも認識できた、汎用性は高そう。
# 始めの５秒ほどは出力の揺れが激しい。安定に時間がかかるのかも。
# Ｘ座標とＹ座標が正しいかはまだわからない。
# カメラのレンズを白い付属品で回せば、少し画質やピントが良くなるのかも？
# Displayで確認をするには、ラズパイに直接接続が必要。
# 今後は、ＳＳＨ接続が無くても自動で走るようにコードで設定し、自動でサーボ制御なども行うようにするべきだ。

#!/usr/bin/env python3
"""
face_positions.py

- Uses Picamera2 + OpenCV Haar cascade to detect faces.
- Prints: TIMESTAMP  CENTER_X  CENTER_Y  OFFSET_X  OFFSET_Y  (offsets in -1.0..+1.0)
- Attempts to locate cascade in common paths, otherwise uses ~/haarcascade_frontalface_default.xml.
- Optionally shows a GUI window if DISPLAY is available or --display passed.
"""

import time
import os
import sys
import argparse
from datetime import datetime

import cv2
from picamera2 import Picamera2

# -------------------------
# Settings
# -------------------------
FRAME_W = 320
FRAME_H = 200
DEFAULT_CASCADE_NAME = "haarcascade_frontalface_default.xml"

# -------------------------
# Helpers: cascade locator
# -------------------------
def locate_cascade(filename=DEFAULT_CASCADE_NAME):
    """Try several common locations, then check home dir and cwd."""
    candidates = []

    # Try cv2.data.haarcascades if attribute exists (some OpenCV builds expose it)
    try:
        base = cv2.data.haarcascades
        if base:
            candidates.append(os.path.join(base, filename))
    except Exception:
        pass

    # Debian/Raspbian typical places
    candidates.extend([
        os.path.join("/usr/share/opencv4/haarcascades", filename),
        os.path.join("/usr/share/opencv/haarcascades", filename),
        os.path.join("/usr/share/opencv/haarcascades", filename),
    ])

    # user's home and current dir
    candidates.append(os.path.expanduser(os.path.join("~", filename)))
    candidates.append(os.path.join(os.getcwd(), filename))

    # return first that exists
    for p in candidates:
        if p and os.path.exists(p):
            return p

    return None

# -------------------------
# Argument parsing
# -------------------------
parser = argparse.ArgumentParser(description="Face position printer using Picamera2 + OpenCV")
parser.add_argument("--cascade", "-c", default=None, help="Path to cascade XML file")
parser.add_argument("--display", action="store_true", help="Force showing GUI window (requires DISPLAY/X)")
parser.add_argument("--no-print-no-face", action="store_true", help="Don't print 'No face detected' messages")
args = parser.parse_args()

# Decide cascade path
if args.cascade:
    casc_path = os.path.expanduser(args.cascade)
else:
    casc_path = locate_cascade()

if not casc_path:
    print("ERROR: Could not locate cascade XML. Please download haarcascade_frontalface_default.xml")
    print("Expected e.g. ~/{}".format(DEFAULT_CASCADE_NAME))
    sys.exit(1)

face_cascade = cv2.CascadeClassifier(casc_path)
if face_cascade.empty():
    print("ERROR: Cascade found but failed to load (corrupt or incompatible):", casc_path)
    sys.exit(1)

print("Using cascade:", casc_path)

# Determine whether to show window
SHOW_WINDOW = args.display or bool(os.environ.get("DISPLAY"))
if SHOW_WINDOW:
    print("GUI window enabled (SHOW_WINDOW=True). Use 'q' in window or Ctrl+C in terminal to quit.")
else:
    print("Headless mode (no GUI). Printing detections to console only.")

# -------------------------
# Initialize Picamera2
# -------------------------
picam2 = Picamera2()
preview_config = picam2.create_preview_configuration(main={"size": (FRAME_W, FRAME_H), "format": "RGB888"})
picam2.configure(preview_config)
picam2.start()
time.sleep(1.2)  # allow AE/AWB to settle

# -------------------------
# Main loop
# -------------------------
last_no_face_print = 0.0
NO_FACE_PRINT_INTERVAL = 1.0  # seconds between "No face" prints to avoid console spam

try:
    while True:
        frame = picam2.capture_array()  # returns RGB array (H, W, 3)
        if frame is None:
            # Should rarely happen
            time.sleep(0.01)
            continue

        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) == 0:
            if not args.no_print_no_face:
                now = time.time()
                if now - last_no_face_print >= NO_FACE_PRINT_INTERVAL:
                    print(f"{datetime.now().isoformat(timespec='seconds')}  NO_FACE")
                    last_no_face_print = now
        else:
            # choose the largest face (by area)
            faces_sorted = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)
            x, y, w, h = faces_sorted[0]
            center_x = int(x + w / 2)
            center_y = int(y + h / 2)

            # normalized offsets relative to center (-1 .. 1)
            offset_x = (center_x - (FRAME_W / 2)) / (FRAME_W / 2)
            offset_y = (center_y - (FRAME_H / 2)) / (FRAME_H / 2)

            ts = datetime.now().isoformat(timespec="seconds")
            # Print: TIMESTAMP  CX CY  OX OY  BBOX_W BBOX_H
            print(f"{ts}  {center_x:03d} {center_y:03d}  {offset_x:+.3f} {offset_y:+.3f}  {w:03d}x{h:03d}")

            if SHOW_WINDOW:
                # convert RGB->BGR for OpenCV display
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame_bgr, (center_x, center_y), 3, (0, 0, 255), -1)
                # crosshair at center of image
                cv2.line(frame_bgr, (FRAME_W // 2 - 6, FRAME_H // 2), (FRAME_W // 2 + 6, FRAME_H // 2), (255, 0, 0), 1)
                cv2.line(frame_bgr, (FRAME_W // 2, FRAME_H // 2 - 6), (FRAME_W // 2, FRAME_H // 2 + 6), (255, 0, 0), 1)
                cv2.putText(frame_bgr, f"({center_x},{center_y})", (8, FRAME_H - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                display = cv2.resize(frame_bgr, (540, 300))
                display = cv2.flip(display, 1)
                cv2.imshow("Face positions", display)

        if SHOW_WINDOW:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quitting on 'q' key.")
                break
        else:
            # small sleep to reduce CPU usage slightly when headless
            time.sleep(0.01)

except KeyboardInterrupt:
    print("\nInterrupted by user. Exiting...")

finally:
    try:
        picam2.stop()
    except Exception:
        pass
    if SHOW_WINDOW:
        cv2.destroyAllWindows()
