#!/usr/bin/env python3
"""
Object detection with pixel-to-angle mapping for robot arm.

Usage:
  python detect.py --calibrate   Calibrate: click arm base, 0° point, 90° point
  python detect.py               Run detection, outputs angle 0-90° for each object
"""

import cv2
import os
import sys
import glob
import json
import math
import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter

LABELS = ["Cube", "Cyl"]
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
CALIBRATION_FILE = os.path.join(MODEL_DIR, "calibration.json")
CONFIDENCE_THRESHOLD = 0.5
CAMERA_INDEX = 1
MODEL_INPUT_W = 320
MODEL_INPUT_H = 320


def find_model():
    tflite_files = glob.glob(os.path.join(MODEL_DIR, "*.tflite"))
    if not tflite_files:
        print("ERROR: No .tflite file found in", MODEL_DIR)
        sys.exit(1)
    return tflite_files[0]


def load_calibration():
    if not os.path.exists(CALIBRATION_FILE):
        print("ERROR: No calibration found. Run: python detect.py --calibrate")
        sys.exit(1)
    with open(CALIBRATION_FILE) as f:
        cal = json.load(f)
    return cal


def pixel_to_angle(px, py, cal):
    """Convert pixel coords to arm angle (0-90 degrees)."""
    bx, by = cal["base"]
    raw = math.atan2(py - by, px - bx)

    # Linear mapping: two known (raw_angle -> arm_degree) pairs
    # raw = m * arm_degree + b
    # Solve for arm_degree = (raw - b) / m
    m = cal["m"]
    b = cal["b"]

    # Handle atan2 wraparound - pick the closest raw angle
    best = raw
    for offset in [-2 * math.pi, 0, 2 * math.pi]:
        candidate = raw + offset
        if abs(candidate - (m * 45 + b)) < abs(best - (m * 45 + b)):
            best = candidate

    degrees = (best - b) / m
    return round(max(0.0, min(90.0, degrees)), 1)


# ========================= CALIBRATION MODE =========================

def calibrate():
    """Click arm base + 2 points at known angles the camera can see."""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("ERROR: Could not open camera")
        sys.exit(1)

    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    clicks = []
    angles = []  # user-entered angles for clicks 2 and 3

    prompts = [
        "Click on the ARM BASE",
        "Place object at a KNOWN angle, click on it",
        "Place object at a DIFFERENT known angle, click on it",
    ]

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) <= len(angles) + 1 and len(clicks) < 3:
            clicks.append((x, y))
            print(f"  Recorded click: ({x}, {y})")
            if len(clicks) >= 2:
                print(f"  Now type the arm angle for that point in the terminal (e.g. 30): ", end="", flush=True)

    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", on_click)

    print("=== CALIBRATION ===")
    print("1. Click the arm base")
    print("2. Place object at a known angle the camera CAN see (e.g. 30°)")
    print("   Click on it, then type the angle in the terminal")
    print("3. Repeat for a second known angle (e.g. 60°)")
    print("4. Press 's' to save, 'r' to redo")
    print()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Crop and resize to 320x320
        if cam_w != cam_h:
            size = min(cam_w, cam_h)
            x_off = (cam_w - size) // 2
            y_off = (cam_h - size) // 2
            cropped = frame[y_off:y_off + size, x_off:x_off + size]
        else:
            cropped = frame
        frame = cv2.resize(cropped, (MODEL_INPUT_W, MODEL_INPUT_H))

        # Draw existing clicks
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        for i, (cx, cy) in enumerate(clicks):
            cv2.circle(frame, (cx, cy), 6, colors[i], -1)
            if i == 0:
                lbl = "BASE"
            elif i - 1 < len(angles):
                lbl = f"{angles[i-1]} deg"
            else:
                lbl = "? deg"
            cv2.putText(frame, lbl, (cx + 10, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)

        # Draw lines from base to calibration points
        if len(clicks) >= 2:
            cv2.line(frame, clicks[0], clicks[1], (0, 255, 0), 1)
        if len(clicks) >= 3:
            cv2.line(frame, clicks[0], clicks[2], (255, 0, 0), 1)

        # Show prompt
        if len(clicks) < 3:
            step = len(clicks)
            if step > 0 and len(angles) < step - 1:
                msg = "Type the angle in the terminal..."
            elif step < len(prompts):
                msg = prompts[step]
            else:
                msg = "Type the angle in the terminal..."
        else:
            if len(angles) < 2:
                msg = "Type the angle in the terminal..."
            else:
                msg = "Press 's' to save, 'r' to redo"

        cv2.putText(frame, msg, (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.imshow("Calibration", frame)
        key = cv2.waitKey(50) & 0xFF

        # Check if user needs to enter an angle
        if len(clicks) >= 2 and len(angles) < len(clicks) - 1:
            try:
                import select
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    line = sys.stdin.readline().strip()
                    if line:
                        angles.append(float(line))
                        print(f"  Angle recorded: {angles[-1]}°")
                        if len(angles) < 2:
                            print(f"\n  Now place object at a different angle and click on it.")
            except (ImportError, OSError):
                # Windows doesn't support select on stdin
                import msvcrt
                if msvcrt.kbhit():
                    line = ""
                    while msvcrt.kbhit():
                        ch = msvcrt.getche().decode()
                        if ch in ('\r', '\n'):
                            break
                        line += ch
                    if line:
                        angles.append(float(line))
                        print(f"\n  Angle recorded: {angles[-1]}°")
                        if len(angles) < 2:
                            print(f"\n  Now place object at a different angle and click on it.")

        if key == ord("q"):
            break
        elif key == ord("r"):
            clicks.clear()
            angles.clear()
            print("\nReset. Click again.")
        elif key == ord("s") and len(clicks) == 3 and len(angles) == 2:
            bx, by = clicks[0]
            x1, y1 = clicks[1]
            x2, y2 = clicks[2]

            # Raw pixel angles from base
            raw1 = math.atan2(y1 - by, x1 - bx)
            raw2 = math.atan2(y2 - by, x2 - bx)

            # Handle atan2 wraparound - unwrap raw2 to be close to raw1
            while raw2 - raw1 > math.pi:
                raw2 -= 2 * math.pi
            while raw2 - raw1 < -math.pi:
                raw2 += 2 * math.pi

            # Linear fit: raw_angle = m * arm_degree + b
            a1, a2 = angles[0], angles[1]
            m = (raw2 - raw1) / (a2 - a1)
            b = raw1 - m * a1

            cal = {
                "base": [bx, by],
                "point_1": [x1, y1],
                "point_2": [x2, y2],
                "angle_1": a1,
                "angle_2": a2,
                "m": m,
                "b": b,
            }
            with open(CALIBRATION_FILE, "w") as f:
                json.dump(cal, f, indent=2)

            print(f"\nCalibration saved to {CALIBRATION_FILE}")
            print(f"  Base:     ({bx}, {by})")
            print(f"  Point 1:  ({x1}, {y1}) = {a1}°")
            print(f"  Point 2:  ({x2}, {y2}) = {a2}°")
            print(f"  Mapping:  raw = {m:.4f} * degree + {b:.4f}")
            break

    cap.release()
    cv2.destroyAllWindows()


# ========================= DETECTION MODE =========================

def detect():
    cal = load_calibration()
    print(f"Calibration loaded: base={cal['base']}")

    model_path = find_model()
    print(f"Loading model: {os.path.basename(model_path)}")

    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_dtype = input_details[0]["dtype"]
    is_quantized = input_dtype == np.int8 or input_dtype == np.uint8

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("ERROR: Could not open camera")
        sys.exit(1)

    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera: {cam_w}x{cam_h} -> {MODEL_INPUT_W}x{MODEL_INPUT_H}")
    print("Press 'q' to quit\n")

    bx, by = cal["base"]

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Crop and resize to 320x320
            if cam_w != cam_h:
                size = min(cam_w, cam_h)
                x_off = (cam_w - size) // 2
                y_off = (cam_h - size) // 2
                cropped = frame[y_off:y_off + size, x_off:x_off + size]
            else:
                cropped = frame
            frame = cv2.resize(cropped, (MODEL_INPUT_W, MODEL_INPUT_H))

            # Prepare input
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if is_quantized:
                s = input_details[0]["quantization"][0]
                z = input_details[0]["quantization"][1]
                input_data = (rgb.astype(np.float32) / s + z).astype(input_dtype)
            else:
                input_data = rgb.astype(np.float32) / 255.0
            input_data = np.expand_dims(input_data, axis=0)

            # Run inference
            interpreter.set_tensor(input_details[0]["index"], input_data)
            interpreter.invoke()

            # Parse SSD outputs
            flat_tensors = []
            for od in output_details:
                tensor = interpreter.get_tensor(od["index"])
                shape = tuple(tensor.shape)
                if len(shape) == 3 and shape[2] == 4:
                    boxes = tensor[0]
                elif len(shape) == 1:
                    num_detections = int(tensor.item())
                else:
                    flat_tensors.append(tensor[0])

            a, b = flat_tensors[0], flat_tensors[1]
            if np.allclose(a, np.round(a), atol=0.1):
                classes, scores = a, b
            else:
                scores, classes = a, b
            num_detections = min(num_detections, len(scores))

            # Draw arm base
            cv2.circle(frame, (bx, by), 6, (0, 0, 255), -1)
            cv2.putText(frame, "BASE", (bx + 8, by - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            # Process detections
            for i in range(num_detections):
                confidence = float(scores[i])
                if confidence < CONFIDENCE_THRESHOLD:
                    continue

                class_id = int(classes[i])
                if class_id < 0 or class_id >= len(LABELS):
                    continue
                label = LABELS[class_id]

                y1, x1, y2, x2 = boxes[i]
                px1 = max(0, int(x1 * MODEL_INPUT_W))
                py1 = max(0, int(y1 * MODEL_INPUT_H))
                px2 = min(MODEL_INPUT_W, int(x2 * MODEL_INPUT_W))
                py2 = min(MODEL_INPUT_H, int(y2 * MODEL_INPUT_H))

                pixel_x = (x1 + x2) / 2 * MODEL_INPUT_W
                pixel_y = (y1 + y2) / 2 * MODEL_INPUT_H

                # Convert to arm angle
                angle = pixel_to_angle(pixel_x, pixel_y, cal)

                # Draw
                color = (0, 255, 0) if label == "Cube" else (255, 0, 0)
                cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)
                cv2.circle(frame, (int(pixel_x), int(pixel_y)), 5, color, -1)
                cv2.line(frame, (bx, by), (int(pixel_x), int(pixel_y)), color, 1)
                text = f"{label} {angle:.1f} deg"
                cv2.putText(frame, text, (px1, py1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                print(f"  {label:6s} conf={confidence:.2f}  angle={angle:5.1f} deg  "
                      f"pixel=({pixel_x:.0f},{pixel_y:.0f})")

            cv2.imshow("Object Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    if "--calibrate" in sys.argv:
        calibrate()
    else:
        detect()
