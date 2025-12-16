"""
Modifications:
- Face output prints center (cx, cy).
- Servo 0,1,2 (indices 0..2): same one-shot behavior (0->180->wait->0) with 5s cooldown.
- Servo 4 (index 4): one-shot sweep based on average of last 3 face center X readings.
  * Pixel mapping: 70 -> 0 deg, 270 -> 180 deg.
  * Deadzone: 150..170 (no movement).
  * When action triggered: smooth sweep to mapped angle, wait 1s, return? (user asked for single sweep; we move from current to mapped and stop there).
  * Small changes (< ANGLE_MOVE_THRESHOLD_LARGE) ignored.
  * After sweep wait 1s (cooldown) before next movement.
- Servos 3 & 5 (indices 3 and 5): same behavior but for Y axis using average of last 3 center Y.
  * Pixel mapping: 40 -> 0 deg, 140 -> 180 deg.
  * Deadzone: 85..105
- Other smoothing thresholds and parameters tuned for slow, stable movements.
"""

import time
import os
import sys
import argparse
from collections import deque
from datetime import datetime

import numpy as np
import cv2
from picamera2 import Picamera2

# TFLite interpreter import (preferred)
try:
    from tflite_runtime.interpreter import Interpreter
except Exception:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

# RPi.GPIO
try:
    import RPi.GPIO as GPIO
except Exception:
    print("ERROR: RPi.GPIO import failed. Are you on a Raspberry Pi with RPi.GPIO installed?")
    raise

# -------------------------
# Settings
# -------------------------
FRAME_W = 320
FRAME_H = 200

DEFAULT_MODEL = "models/emotion_model.tflite"
DEFAULT_CASCADE_NAME = "haarcascade_frontalface_default.xml"

# Servo pins (BCM)
SERVO_GPIOS = [18, 17, 27, 22, 23, 24]  # indices 0..5
SERVO_FREQ = 50

SERVO_MIN_ANGLE = 0
SERVO_MAX_ANGLE = 180

# smoothing and thresholds (for continuous smoothing)
SMOOTHING_ALPHA = 0.08
ANGLE_MOVE_THRESHOLD = 2.0
SERVO_UPDATE_INTERVAL = 0.06

# one-shot/cooldowns
HOLD_DURATION_0_2 = 1.0     # wait while at 180 for servos 0..2
COOLDOWN_0_2 = 5.0          # 5-second cooldown for servos 0..2 after action
COOLDOWN_345 = 1.0          # 1-second cooldown for servos 3,4,5 after action

# big-move threshold for X/Y one-shot servos (ignore tiny adjustments)
ANGLE_MOVE_THRESHOLD_LARGE = 6.0  # degrees

# Pixel mapping ranges and deadzones (from user's numbers)
PIXEL_X_MIN = 70
PIXEL_X_MAX = 270
X_MEDIAN = 160
X_DEADZONE_MIN = 150
X_DEADZONE_MAX = 170

PIXEL_Y_MIN = 40
PIXEL_Y_MAX = 140
Y_MEDIAN = 95
Y_DEADZONE_MIN = 85
Y_DEADZONE_MAX = 105

PIXEL_MOVE_THRESHOLD = 6  # ignore small pixel fluctuations when building averages

# emotion buffer
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
EMOTION_WINDOW = 3

# -------------------------
# Helpers
# -------------------------
def angle_to_duty_cycle(angle):
    a = max(SERVO_MIN_ANGLE, min(SERVO_MAX_ANGLE, angle))
    return 2.5 + (a / 180.0) * 10.0

def map_pixel_to_angle(pixel, pmin, pmax):
    # Linear mapping pmin -> 0, pmax -> 180, clamp outside
    if pixel <= pmin:
        return 0.0
    if pixel >= pmax:
        return 180.0
    span = pmax - pmin
    frac = (pixel - pmin) / float(span)
    return frac * 180.0

def smooth_move_servo(pwm, start_angle, end_angle, step=4, delay=0.03):
    """Slow smooth sweep for a single servo. Blocking."""
    if start_angle == end_angle:
        return
    if start_angle < end_angle:
        rng = range(int(round(start_angle)), int(round(end_angle)) + 1, step)
    else:
        rng = range(int(round(start_angle)), int(round(end_angle)) - 1, -step)

    for a in rng:
        try:
            pwm.ChangeDutyCycle(angle_to_duty_cycle(a))
        except Exception:
            pass
        time.sleep(delay)

def smooth_move_two_servos(pwm_a, pwm_b, start_a, end_a, start_b, end_b, step=4, delay=0.03):
    """Move two servos in parallel (blocking). We iterate steps and update both."""
    # Build angle sequences with same number of steps by normalizing to fraction of their spans.
    # Simpler approach: compute number of steps for largest delta, then interpolate.
    delta_a = end_a - start_a
    delta_b = end_b - start_b
    max_span = max(abs(delta_a), abs(delta_b), 1e-6)
    steps = max(1, int(abs(max_span) / step))
    for s in range(1, steps + 1):
        frac = s / float(steps)
        a = start_a + delta_a * frac
        b = start_b + delta_b * frac
        try:
            pwm_a.ChangeDutyCycle(angle_to_duty_cycle(a))
        except Exception:
            pass
        try:
            pwm_b.ChangeDutyCycle(angle_to_duty_cycle(b))
        except Exception:
            pass
        time.sleep(delay)

def print_emotion_row(face_idx, label, score, center, mapped_x, mapped_y, servo_out_angles):
    ts = datetime.now().isoformat(timespec='seconds')
    cx, cy = center if center is not None else (-1, -1)
    print(f"{ts} | FACE#{face_idx} | {label:<8} | score={score:.2f} | center=({cx},{cy}) "
          f"| mapped_x={mapped_x:.0f} mapped_y={mapped_y:.0f} | angles={','.join(f'{a:.0f}' for a in servo_out_angles)}",
          flush=True)

# -------------------------
# TFLite wrapper
# -------------------------
class TFLiteEmotionModel:
    def __init__(self, model_path):
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        # determine expected input shape
        shape = list(self.input_details[0]['shape'])
        if len(shape) >= 3:
            self.inp_h = int(shape[1]); self.inp_w = int(shape[2])
        else:
            self.inp_h = 64; self.inp_w = 64

    def predict(self, face_img):
        try:
            x = cv2.resize(face_img, (self.inp_w, self.inp_h))
        except Exception:
            x = cv2.resize(face_img, (self.inp_w, self.inp_h))
        x = x.astype(np.float32) / 255.0
        if x.ndim == 3:
            x = np.expand_dims(x, axis=0)
        self.interpreter.set_tensor(self.input_details[0]['index'], x)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_details[0]['index'])
        out = np.array(out).squeeze()
        return out

# -------------------------
# Main
# -------------------------
def main(model_path, cascade_path, show_window, cam_w, cam_h):
    # cascade
    if cascade_path is None:
        cascade_path = cv2.data.haarcascades + DEFAULT_CASCADE_NAME
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print("ERROR: could not load cascade:", cascade_path)
        sys.exit(1)

    # model
    model = TFLiteEmotionModel(model_path)

    # camera
    picam2 = Picamera2()
    preview_config = picam2.create_preview_configuration(main={"size": (cam_w, cam_h), "format": "RGB888"})
    picam2.configure(preview_config)
    picam2.start()
    time.sleep(0.6)

    # GPIO setup
    GPIO.setmode(GPIO.BCM)
    if not isinstance(SERVO_GPIOS, (list, tuple)) or len(SERVO_GPIOS) < 6:
        print("ERROR: SERVO_GPIOS must contain 6 pins")
        sys.exit(1)
    pwm_list = []
    for pin in SERVO_GPIOS:
        GPIO.setup(pin, GPIO.OUT)
        pwm = GPIO.PWM(pin, SERVO_FREQ)
        pwm.start(angle_to_duty_cycle(0))
        pwm_list.append(pwm)

    # state
    current_angles = [0.0] * 6
    target_angles = [0.0] * 6
    last_servo_update = time.time()

    emotion_window = deque(maxlen=EMOTION_WINDOW)

    # one-shot state for servos 0..2
    servo_busy_012 = [False, False, False]
    servo_cooldown_until_012 = [0.0, 0.0, 0.0]

    # one-shot state for servos 3..5 (Y & X)
    servo_busy_345 = [False, False, False]  # corresponds to indices 3,4,5
    servo_cooldown_until_345 = [0.0, 0.0, 0.0]

    # rolling last-3 center buffers for X and Y
    last3_x = deque(maxlen=3)
    last3_y = deque(maxlen=3)

    last_print = 0.0
    MIN_PRINT_INTERVAL = 0.5

    try:
        while True:
            frame = picam2.capture_array()  # RGB
            if frame is None:
                time.sleep(0.01)
                continue

            # normalize frame
            if frame.ndim == 3 and frame.shape[2] == 4:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            elif frame.ndim == 3 and frame.shape[2] == 3:
                frame_rgb = frame
            else:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))

            face_box = None
            center_x = None
            center_y = None
            label = 'none'
            score = 0.0

            mapped_x_angle = target_angles[4]
            mapped_y_angle = target_angles[3]

            if len(faces) == 0:
                # no face - do not append to last3 buffers, keep them as-is
                emotion_window.append('none')
            else:
                # biggest face
                x, y, w, h = max(faces, key=lambda r: r[2]*r[3])
                cx = int(x + w / 2)
                cy = int(y + h / 2)
                center_x = cx
                center_y = cy
                face_box = (x, y, w, h)

                # append to rolling buffers only if change > PIXEL_MOVE_THRESHOLD or empty (to avoid tiny jitter)
                if len(last3_x) == 0 or abs(cx - last3_x[-1]) >= PIXEL_MOVE_THRESHOLD:
                    last3_x.append(cx)
                # else we still append if buffer empty earlier handled

                if len(last3_y) == 0 or abs(cy - last3_y[-1]) >= PIXEL_MOVE_THRESHOLD:
                    last3_y.append(cy)

                # emotion prediction
                pad = int(0.1 * w)
                x0 = max(x - pad, 0)
                y0 = max(y - pad, 0)
                x1 = min(x + w + pad, frame_rgb.shape[1])
                y1 = min(y + h + pad, frame_rgb.shape[0])
                face_crop = frame_rgb[y0:y1, x0:x1]
                if face_crop.size == 0:
                    emotion_window.append('none')
                else:
                    preds = model.predict(face_crop)
                    if preds.ndim == 1:
                        probs = preds
                    else:
                        probs = np.array(preds).flatten()
                    idx = int(np.argmax(probs))
                    if idx < len(EMOTIONS):
                        label = EMOTIONS[idx]
                    else:
                        label = str(idx)
                    score = float(np.max(probs))
                    emotion_window.append(label)

            # ---------- DECIDE servo 0..2 (one-shot, 5s cooldown) ----------
            angry_count = sum(1 for e in emotion_window if e == 'angry')
            happy_count = sum(1 for e in emotion_window if e == 'happy')
            recent_label = emotion_window[-1] if len(emotion_window) > 0 else 'none'

            want0 = (angry_count >= 2)  # servo idx 0 -> angry
            want1 = (happy_count >= 2)  # servo idx 1 -> happy
            want2 = (recent_label not in ('angry', 'happy', 'none'))  # servo idx 2 -> other

            now = time.time()
            # process each of 0,1,2 independently
            for idx, want in enumerate((want0, want1, want2)):
                if want and (not servo_busy_012[idx]) and now >= servo_cooldown_until_012[idx]:
                    # start the one-shot action: 0 -> 180, wait, 180 -> 0
                    servo_busy_012[idx] = True
                    # perform slow sweep
                    smooth_move_servo(pwm_list[idx], current_angles[idx], 180.0, step=4, delay=0.03)
                    current_angles[idx] = 180.0
                    # hold duration at 180
                    time.sleep(HOLD_DURATION_0_2)
                    # sweep back
                    smooth_move_servo(pwm_list[idx], current_angles[idx], 0.0, step=4, delay=0.03)
                    current_angles[idx] = 0.0
                    servo_busy_012[idx] = False
                    servo_cooldown_until_012[idx] = time.time() + COOLDOWN_0_2
                    # after this action, continue loop (other decisions will wait for next iteration)

            # ---------- DECIDE servo 4 (index 4) based on avg of last3_x ----------
            # Only compute average when we have 3 valid readings
            mapped_x_angle = target_angles[4]
            if len(last3_x) == 3:
                avg_x = sum(last3_x) / 3.0
                # deadzone around median
                if not (X_DEADZONE_MIN <= avg_x <= X_DEADZONE_MAX):
                    # map using PIXEL_X_MIN..PIXEL_X_MAX
                    new_angle_x = map_pixel_to_angle(avg_x, PIXEL_X_MIN, PIXEL_X_MAX)
                    # only act if difference is large enough
                    if abs(new_angle_x - current_angles[4]) >= ANGLE_MOVE_THRESHOLD_LARGE and not servo_busy_345[1] and now >= servo_cooldown_until_345[1]:
                        # mark busy
                        servo_busy_345[1] = True
                        # one smooth sweep FROM current_angles[4] to new_angle_x
                        smooth_move_servo(pwm_list[4], current_angles[4], new_angle_x, step=4, delay=0.03)
                        current_angles[4] = new_angle_x
                        # wait 1s
                        time.sleep(COOLDOWN_345)
                        servo_busy_345[1] = False
                        servo_cooldown_until_345[1] = time.time() + COOLDOWN_345
                        # clear last3_x to avoid immediate retrigger on same readings
                        last3_x.clear()
                else:
                    # in deadzone: do nothing (stay)
                    pass

            # ---------- DECIDE servos 3 & 5 (indices 3,5) based on avg of last3_y ----------
            mapped_y_angle = target_angles[3]
            if len(last3_y) == 3:
                avg_y = sum(last3_y) / 3.0
                if not (Y_DEADZONE_MIN <= avg_y <= Y_DEADZONE_MAX):
                    new_angle_y = map_pixel_to_angle(avg_y, PIXEL_Y_MIN, PIXEL_Y_MAX)
                    # only act if difference large enough and both servos are available
                    if (abs(new_angle_y - current_angles[3]) >= ANGLE_MOVE_THRESHOLD_LARGE or abs(new_angle_y - current_angles[5]) >= ANGLE_MOVE_THRESHOLD_LARGE) \
                       and (not servo_busy_345[0]) and (not servo_busy_345[2]) \
                       and now >= servo_cooldown_until_345[0] and now >= servo_cooldown_until_345[2]:
                        # mark busy (both)
                        servo_busy_345[0] = True
                        servo_busy_345[2] = True
                        # move both servos together from their current angles to the new_angle_y
                        smooth_move_two_servos(pwm_list[3], pwm_list[5],
                                              current_angles[3], new_angle_y,
                                              current_angles[5], new_angle_y,
                                              step=4, delay=0.03)
                        current_angles[3] = new_angle_y
                        current_angles[5] = new_angle_y
                        # wait 1s
                        time.sleep(COOLDOWN_345)
                        servo_busy_345[0] = False
                        servo_busy_345[2] = False
                        servo_cooldown_until_345[0] = time.time() + COOLDOWN_345
                        servo_cooldown_until_345[2] = time.time() + COOLDOWN_345
                        # clear last3_y to avoid immediate retrigger
                        last3_y.clear()
                else:
                    # in deadzone: do nothing
                    pass

            # ---------- For any remaining continuous smoothing for servos not moved by one-shot
            # For completeness: we still write duty cycles for all servos (even if they were moved)
            # but servos 3,4,5 have been moved explicitly when needed.
            now = time.time()
            if (now - last_servo_update) >= SERVO_UPDATE_INTERVAL:
                for i in range(6):
                    # skip those currently being moved by one-shot (they already set pwm inside smooth functions)
                    # but update PWM for others to current_angles value
                    try:
                        pwm_list[i].ChangeDutyCycle(angle_to_duty_cycle(current_angles[i]))
                    except Exception:
                        pass
                last_servo_update = now

            # ---------- PRINT
            if time.time() - last_print >= MIN_PRINT_INTERVAL:
                print_emotion_row(0, label, score, (center_x, center_y) if center_x is not None else None,
                                  current_angles[4], current_angles[3], current_angles)
                last_print = time.time()

            # ---------- Display overlay if requested
            if show_window:
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                if face_box is not None:
                    x, y, w, h = face_box
                    cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    if center_x is not None and center_y is not None:
                        cv2.circle(frame_bgr, (center_x, center_y), 3, (0, 0, 255), -1)
                cv2.line(frame_bgr, (FRAME_W // 2 - 6, FRAME_H // 2), (FRAME_W // 2 + 6, FRAME_H // 2), (255, 0, 0), 1)
                cv2.line(frame_bgr, (FRAME_W // 2, FRAME_H // 2 - 6), (FRAME_W // 2, FRAME_H // 2 + 6), (255, 0, 0), 1)
                txt = "Angles: " + ", ".join(f"{a:.0f}" for a in current_angles)
                cv2.putText(frame_bgr, txt, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                display = cv2.resize(frame_bgr, (640, 380))
                display = cv2.flip(display, 1)
                cv2.imshow("6-Servo Haar+Emotion (v4)", display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                time.sleep(0.003)

    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting...")
    finally:
        try:
            picam2.stop()
        except Exception:
            pass
        if show_window:
            cv2.destroyAllWindows()
        try:
            for pwm in pwm_list:
                pwm.stop()
        except Exception:
            pass
        GPIO.cleanup()
        print("Cleaned up GPIO and exited.")

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="6-servo: Haar face pos + emotion -> servo control (v4)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Path to TFLite emotion model")
    parser.add_argument("--cascade", type=str, default=None, help="Path to Haar cascade XML (optional)")
    parser.add_argument("--show-window", action="store_true", help="Show preview window")
    parser.add_argument("--cam-width", type=int, default=FRAME_W, help="Camera width")
    parser.add_argument("--cam-height", type=int, default=FRAME_H, help="Camera height")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print("WARNING: model path does not exist:", args.model)
    main(args.model, args.cascade, args.show_window, args.cam_width, args.cam_height)
