#!/usr/bin/env python3
"""
Object detection with pixel-to-angle mapping for robot arm.

Usage:
  python detect.py --calibrate   Calibrate: click 3 points at known angles
  python detect.py               Run detection, outputs angle 0-90 for each object
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

def find_circle_center(p1, p2, p3):
    """Find center of circle through 3 points (= arm base position)."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    A = np.array([
        [2 * (x2 - x1), 2 * (y2 - y1)],
        [2 * (x3 - x2), 2 * (y3 - y2)],
    ])
    B = np.array([
        x2**2 - x1**2 + y2**2 - y1**2,
        x3**2 - x2**2 + y3**2 - y2**2,
    ])

    try:
        center = np.linalg.solve(A, B)
        return float(center[0]), float(center[1])
    except np.linalg.LinAlgError:
        return None


def grab_frame(cap, cam_w, cam_h):
    """Capture one frame, crop and resize to 320x320."""
    ret, frame = cap.read()
    if not ret:
        return None
    if cam_w != cam_h:
        size = min(cam_w, cam_h)
        x_off = (cam_w - size) // 2
        y_off = (cam_h - size) // 2
        frame = frame[y_off:y_off + size, x_off:x_off + size]
    return cv2.resize(frame, (MODEL_INPUT_W, MODEL_INPUT_H))


def calibrate():
    """Click 3 points at known angles. Arm base is computed automatically."""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("ERROR: Could not open camera")
        sys.exit(1)

    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    clicks = []
    angles = []
    need_click = [True]  # waiting for a click (not an angle)

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and need_click[0] and len(clicks) < 3:
            clicks.append((x, y))
            need_click[0] = False  # now we need an angle typed

    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", on_click)

    print("=== CALIBRATION ===")
    print("Place object at 3 different known angles along the arm's arc.")
    print("For each: click on the object, then type its angle and press Enter.")
    print("Example: place at 20, 45, 70 (spread them out for accuracy)")
    print("Press 'r' in the window to redo, 's' to save when all 3 are done.")
    print()
    print(f"Click on object at known angle (1/3)")

    while True:
        # If we have a click but no angle for it, ask for input (blocking)
        if not need_click[0] and len(angles) < len(clicks):
            try:
                line = input("  Enter angle for this point: ").strip()
                ang = float(line)
                angles.append(ang)
                need_click[0] = True
                print(f"  Recorded: {ang} deg")
                if len(angles) < 3:
                    print(f"\nClick on object at known angle ({len(angles)+1}/3)")
                else:
                    print(f"\nAll 3 points recorded. Press 's' in the window to save, 'r' to redo.")
            except ValueError:
                print("  Invalid number, try again.")
                continue

        frame = grab_frame(cap, cam_w, cam_h)
        if frame is None:
            break

        # Draw existing clicks
        colors = [(0, 255, 0), (255, 255, 0), (255, 0, 0)]
        for i, (cx, cy) in enumerate(clicks):
            cv2.circle(frame, (cx, cy), 6, colors[i], -1)
            if i < len(angles):
                lbl = f"{angles[i]} deg"
            else:
                lbl = "..."
            cv2.putText(frame, lbl, (cx + 10, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)

        # Show prompt
        done = len(clicks) == 3 and len(angles) == 3
        if done:
            msg = "Press 's' to save, 'r' to redo"
        else:
            msg = f"Click on object at known angle ({len(clicks)+1}/3)"

        cv2.putText(frame, msg, (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.imshow("Calibration", frame)
        key = cv2.waitKey(50) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("r"):
            clicks.clear()
            angles.clear()
            need_click[0] = True
            print("\nReset. Click on object at known angle (1/3)")
        elif key == ord("s") and done:
            # Find arm base from 3 points on the arc
            result = find_circle_center(clicks[0], clicks[1], clicks[2])
            if result is None:
                print("\nERROR: Points are collinear - can't find circle center.")
                print("Make sure the 3 points are spread along the arc, not in a line.")
                clicks.clear()
                angles.clear()
                need_click[0] = True
                continue

            bx, by = result
            radius = math.sqrt((clicks[0][0] - bx)**2 + (clicks[0][1] - by)**2)

            # Compute raw pixel angles from base for each point
            raws = []
            for (px, py) in clicks:
                raws.append(math.atan2(py - by, px - bx))

            # Unwrap angles to be monotonic
            for i in range(1, len(raws)):
                while raws[i] - raws[i-1] > math.pi:
                    raws[i] -= 2 * math.pi
                while raws[i] - raws[i-1] < -math.pi:
                    raws[i] += 2 * math.pi

            # Least squares linear fit: raw = m * arm_degree + b
            A_fit = np.array([[a, 1] for a in angles])
            B_fit = np.array(raws)
            result_fit, _, _, _ = np.linalg.lstsq(A_fit, B_fit, rcond=None)
            m, b = float(result_fit[0]), float(result_fit[1])

            cal = {
                "base": [round(bx, 1), round(by, 1)],
                "radius": round(radius, 1),
                "points": [[int(c[0]), int(c[1])] for c in clicks],
                "angles": angles,
                "m": m,
                "b": b,
            }
            with open(CALIBRATION_FILE, "w") as f:
                json.dump(cal, f, indent=2)

            print(f"\nCalibration saved to {CALIBRATION_FILE}")
            print(f"  Arm base (computed): ({bx:.1f}, {by:.1f})")
            print(f"  Arm radius: {radius:.1f} px")
            for i in range(3):
                print(f"  Point {i+1}: {clicks[i]} = {angles[i]} deg")
            print(f"  Mapping: raw = {m:.4f} * degree + {b:.4f}")
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
            frame = grab_frame(cap, cam_w, cam_h)
            if frame is None:
                break

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

            a, b_tens = flat_tensors[0], flat_tensors[1]
            if np.allclose(a, np.round(a), atol=0.1):
                classes, scores = a, b_tens
            else:
                scores, classes = a, b_tens
            num_detections = min(num_detections, len(scores))

            # Draw arm base if on screen
            bx_int, by_int = int(bx), int(by)
            if 0 <= bx_int < MODEL_INPUT_W and 0 <= by_int < MODEL_INPUT_H:
                cv2.circle(frame, (bx_int, by_int), 6, (0, 0, 255), -1)
                cv2.putText(frame, "BASE", (bx_int + 8, by_int - 5),
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
