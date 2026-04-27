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
    dx = px - bx
    dy = py - by
    raw = math.atan2(dy, dx)

    # Normalize relative to 0° reference
    angle_0 = cal["angle_0_rad"]
    angle_90 = cal["angle_90_rad"]

    # Map raw angle into 0-90 range
    # Handle wraparound
    diff = raw - angle_0
    span = angle_90 - angle_0

    # Normalize diff to same sign as span
    while diff - span > math.pi:
        diff -= 2 * math.pi
    while diff - span < -math.pi:
        diff += 2 * math.pi
    while diff > math.pi:
        diff -= 2 * math.pi
    while diff < -math.pi:
        diff += 2 * math.pi

    degrees = (diff / span) * 90.0
    return round(max(0.0, min(90.0, degrees)), 1)


# ========================= CALIBRATION MODE =========================

def calibrate():
    """Click 3 points: arm base, 0° position, 90° position."""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("ERROR: Could not open camera")
        sys.exit(1)

    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    clicks = []
    prompts = [
        "Click on the ARM BASE",
        "Click on the 0 degree position",
        "Click on the 90 degree position",
    ]

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 3:
            clicks.append((x, y))
            print(f"  Recorded: ({x}, {y})")

    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", on_click)

    print("=== CALIBRATION ===")
    print("Click 3 points in order:")
    for i, p in enumerate(prompts):
        print(f"  {i+1}. {p}")
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
        labels = ["BASE", "0 deg", "90 deg"]
        for i, (cx, cy) in enumerate(clicks):
            cv2.circle(frame, (cx, cy), 6, colors[i], -1)
            cv2.putText(frame, labels[i], (cx + 10, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)

        # Draw arc if we have all 3
        if len(clicks) >= 2:
            cv2.line(frame, clicks[0], clicks[1], (0, 255, 0), 1)
        if len(clicks) >= 3:
            cv2.line(frame, clicks[0], clicks[2], (255, 0, 0), 1)

        # Show current prompt
        step = min(len(clicks), 2)
        cv2.putText(frame, prompts[step] if len(clicks) < 3 else "Press 's' to save, 'r' to redo",
                    (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        cv2.imshow("Calibration", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("r"):
            clicks.clear()
            print("Reset. Click again.")
        elif key == ord("s") and len(clicks) == 3:
            # Save calibration
            bx, by = clicks[0]
            x0, y0 = clicks[1]
            x90, y90 = clicks[2]

            angle_0 = math.atan2(y0 - by, x0 - bx)
            angle_90 = math.atan2(y90 - by, x90 - bx)

            cal = {
                "base": [bx, by],
                "point_0": [x0, y0],
                "point_90": [x90, y90],
                "angle_0_rad": angle_0,
                "angle_90_rad": angle_90,
            }
            with open(CALIBRATION_FILE, "w") as f:
                json.dump(cal, f, indent=2)

            print(f"\nCalibration saved to {CALIBRATION_FILE}")
            print(f"  Base:  ({bx}, {by})")
            print(f"  0°:    ({x0}, {y0})  angle={math.degrees(angle_0):.1f}°")
            print(f"  90°:   ({x90}, {y90})  angle={math.degrees(angle_90):.1f}°")
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
