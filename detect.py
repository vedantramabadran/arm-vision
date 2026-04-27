#!/usr/bin/env python3
"""
Object detection using Edge Impulse TFLite model.
Captures from webcam, runs inference, prints pixel coordinates of detected objects.

Usage:
  1. Export model from Edge Impulse as "TensorFlow Lite (float32)"
  2. Place the .tflite file in this directory
  3. pip install -r requirements.txt
  4. python detect.py
"""

import cv2
import os
import sys
import glob
import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter

# Labels from the Edge Impulse model
LABELS = ["Cube", "Cyl"]

# Find .tflite model file automatically
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIDENCE_THRESHOLD = 0.5
CAMERA_INDEX = 1  # Change if you have multiple cameras

# Model input size (from model_metadata.h)
MODEL_INPUT_W = 320
MODEL_INPUT_H = 320


def find_model():
    """Find the .tflite file in the current directory."""
    tflite_files = glob.glob(os.path.join(MODEL_DIR, "*.tflite"))
    if not tflite_files:
        print("ERROR: No .tflite file found in", MODEL_DIR)
        print()
        print("To get the .tflite file:")
        print("  1. Go to https://studio.edgeimpulse.com/studio/938802")
        print("  2. Click 'Deployment' in the left sidebar")
        print("  3. Change deployment target to 'TensorFlow Lite'")
        print("  4. Select float32")
        print("  5. Click 'Build' and extract the .tflite file here")
        sys.exit(1)
    if len(tflite_files) > 1:
        print(f"Found multiple .tflite files, using: {tflite_files[0]}")
    return tflite_files[0]


def main():
    model_path = find_model()
    print(f"Loading model: {os.path.basename(model_path)}")

    # Load TFLite model
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Print model info
    input_shape = input_details[0]["shape"]
    input_dtype = input_details[0]["dtype"]
    print(f"Input:  {input_shape} dtype={input_dtype.__name__}")
    print(f"Outputs: {len(output_details)} tensors")
    for i, od in enumerate(output_details):
        print(f"  [{i}] {od['name']}: shape={od['shape']} dtype={od['dtype'].__name__}")
    print(f"Labels: {LABELS}")
    print()

    # Get quantization params if int8
    is_quantized = input_dtype == np.int8 or input_dtype == np.uint8
    if is_quantized:
        input_scale = input_details[0]["quantization"][0]
        input_zero_point = input_details[0]["quantization"][1]
        print(f"Quantized model: scale={input_scale}, zero_point={input_zero_point}")

    # Open camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera {CAMERA_INDEX}")
        sys.exit(1)

    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera opened: {cam_w}x{cam_h}")
    print("Press 'q' to quit")
    print()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Center crop to square and resize to 320x320
            if cam_w != cam_h:
                size = min(cam_w, cam_h)
                x_off = (cam_w - size) // 2
                y_off = (cam_h - size) // 2
                cropped = frame[y_off:y_off + size, x_off:x_off + size]
            else:
                cropped = frame

            # Resize to model input - this is our display frame too
            frame = cv2.resize(cropped, (MODEL_INPUT_W, MODEL_INPUT_H))
            resized = frame.copy()

            # Convert BGR to RGB
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            # Prepare input tensor
            if is_quantized:
                input_data = (rgb.astype(np.float32) / input_scale + input_zero_point).astype(input_dtype)
            else:
                input_data = (rgb.astype(np.float32) / 255.0)

            input_data = np.expand_dims(input_data, axis=0)

            # Run inference
            interpreter.set_tensor(input_details[0]["index"], input_data)
            interpreter.invoke()

            # SSD output tensors - identify by shape
            flat_tensors = []  # [1, N] shaped tensors
            for od in output_details:
                tensor = interpreter.get_tensor(od["index"])
                shape = tuple(tensor.shape)
                if len(shape) == 3 and shape[2] == 4:
                    boxes = tensor[0]           # [1, N, 4] = bounding boxes
                elif len(shape) == 1:
                    num_detections = int(tensor.item())  # [1] = count
                else:
                    flat_tensors.append(tensor[0])  # [1, N] = scores or classes

            # Distinguish scores from classes: class indices are near-integer,
            # scores are fractional floats spread across [0, 1]
            a, b = flat_tensors[0], flat_tensors[1]
            a_is_classes = np.allclose(a, np.round(a), atol=0.1)
            if a_is_classes:
                classes, scores = a, b
            else:
                scores, classes = a, b

            num_detections = min(num_detections, len(scores))

            # Process detections
            detections = []
            for i in range(num_detections):
                confidence = float(scores[i])
                if confidence < CONFIDENCE_THRESHOLD:
                    continue

                class_id = int(classes[i])
                if class_id < 0 or class_id >= len(LABELS):
                    continue
                label = LABELS[class_id]

                # Bounding box in normalized coordinates [y1, x1, y2, x2]
                y1, x1, y2, x2 = boxes[i]

                # Convert normalized coords to 320x320 pixel coordinates
                px1 = x1 * MODEL_INPUT_W
                py1 = y1 * MODEL_INPUT_H
                px2 = x2 * MODEL_INPUT_W
                py2 = y2 * MODEL_INPUT_H

                # Center of bounding box
                pixel_x = (px1 + px2) / 2
                pixel_y = (py1 + py2) / 2

                # Bounding box corners (clipped to frame)
                box_x1 = max(0, int(px1))
                box_y1 = max(0, int(py1))
                box_x2 = min(MODEL_INPUT_W, int(px2))
                box_y2 = min(MODEL_INPUT_H, int(py2))

                det = {
                    "label": label,
                    "confidence": round(confidence, 3),
                    "pixel_x": round(pixel_x, 1),
                    "pixel_y": round(pixel_y, 1),
                    "bbox": [box_x1, box_y1, box_x2, box_y2],
                }
                detections.append(det)

                # Draw on frame
                color = (0, 255, 0) if label == "Cube" else (255, 0, 0)
                cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), color, 2)
                cv2.circle(frame, (int(pixel_x), int(pixel_y)), 5, (0, 0, 255), -1)
                text = f"{label} {confidence:.2f} ({pixel_x:.0f},{pixel_y:.0f})"
                cv2.putText(frame, text, (box_x1, box_y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Print detections
            if detections:
                for d in detections:
                    print(f"  {d['label']:6s} conf={d['confidence']:.2f}  "
                          f"pixel=({d['pixel_x']:6.1f}, {d['pixel_y']:6.1f})  "
                          f"bbox={d['bbox']}")

            # Show frame
            cv2.imshow("Object Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
