#!/usr/bin/env python3
"""
Object detection using Edge Impulse model.
Captures from webcam, runs inference, prints pixel coordinates of detected objects.
"""

import cv2
import os
import sys
import json
import time
import numpy as np
from edge_impulse_linux.image import ImageImpulseRunner

# Path to your .eim model file
# Download from: Edge Impulse Studio → Deployment → macOS (or Linux) → Build
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.eim")

CONFIDENCE_THRESHOLD = 0.5
CAMERA_INDEX = 0  # Change if you have multiple cameras


def main():
    if not os.path.exists(MODEL_PATH):
        print("ERROR: Model file not found at:", MODEL_PATH)
        print()
        print("To get the .eim file:")
        print("  1. Go to https://studio.edgeimpulse.com/studio/938802")
        print("  2. Click 'Deployment' in the left sidebar")
        print("  3. Search for 'macOS' (or 'Linux' if on Linux)")
        print("  4. Click 'Build'")
        print("  5. Save the downloaded .eim file as 'model.eim' in this directory")
        sys.exit(1)

    # Load model
    runner = ImageImpulseRunner(MODEL_PATH)
    model_info = runner.init()

    print("Model:", model_info["project"]["name"])
    print("Labels:", model_info["model_parameters"]["labels"])
    print("Input size:", model_info["model_parameters"]["image_input_width"], "x",
          model_info["model_parameters"]["image_input_height"])
    print()

    input_w = model_info["model_parameters"]["image_input_width"]
    input_h = model_info["model_parameters"]["image_input_height"]

    # Open camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("ERROR: Could not open camera", CAMERA_INDEX)
        runner.stop()
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

            # Crop to square (center crop) to match model input aspect ratio
            if cam_w != cam_h:
                size = min(cam_w, cam_h)
                x_off = (cam_w - size) // 2
                y_off = (cam_h - size) // 2
                cropped = frame[y_off:y_off+size, x_off:x_off+size]
            else:
                cropped = frame
                size = cam_w
                x_off = 0
                y_off = 0

            # Resize to model input size
            resized = cv2.resize(cropped, (input_w, input_h))

            # Convert to RGB features for Edge Impulse
            features = []
            for row in resized:
                for pixel in row:
                    b, g, r = int(pixel[0]), int(pixel[1]), int(pixel[2])
                    features.append((r << 16) | (g << 8) | b)

            # Run inference
            res = runner.classify(features)
            bboxes = res["result"].get("bounding_boxes", [])

            # Process detections
            detections = []
            for bb in bboxes:
                if bb["value"] < CONFIDENCE_THRESHOLD:
                    continue

                # Bounding box in model coordinates (0 to input_w/input_h)
                mx = bb["x"]
                my = bb["y"]
                mw = bb["width"]
                mh = bb["height"]

                # Center of bounding box in model pixel space
                center_mx = mx + mw / 2
                center_my = my + mh / 2

                # Scale back to cropped image coordinates
                scale = size / input_w
                cx = center_mx * scale
                cy = center_my * scale

                # Offset to full frame coordinates
                pixel_x = cx + x_off
                pixel_y = cy + y_off

                # Bounding box in full frame coordinates
                box_x = int(mx * scale + x_off)
                box_y = int(my * scale + y_off)
                box_w = int(mw * scale)
                box_h = int(mh * scale)

                det = {
                    "label": bb["label"],
                    "confidence": round(bb["value"], 3),
                    "pixel_x": round(pixel_x, 1),
                    "pixel_y": round(pixel_y, 1),
                    "bbox": [box_x, box_y, box_w, box_h],
                }
                detections.append(det)

                # Draw on frame
                color = (0, 255, 0) if bb["label"] == "Cube" else (255, 0, 0)
                cv2.rectangle(frame, (box_x, box_y),
                              (box_x + box_w, box_y + box_h), color, 2)
                cv2.circle(frame, (int(pixel_x), int(pixel_y)), 5, (0, 0, 255), -1)
                label_text = f"{bb['label']} {bb['value']:.2f} ({pixel_x:.0f},{pixel_y:.0f})"
                cv2.putText(frame, label_text, (box_x, box_y - 8),
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
        runner.stop()


if __name__ == "__main__":
    main()
