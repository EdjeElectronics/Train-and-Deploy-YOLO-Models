import os
import sys
import argparse
import cv2
import time
import numpy as np
from ultralytics import YOLO

# ------------------------------
# Arguments
# ------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='Path to YOLO model file (.pt)')
parser.add_argument('--source', required=True, help='Camera index (usb0), video file, or image folder')
parser.add_argument('--thresh', default=0.5, type=float, help='Minimum confidence threshold')
parser.add_argument('--resolution', default=None, help='Display resolution WxH, e.g., 1280x720')
args = parser.parse_args()

model_path = args.model
source = args.source
conf_thresh = args.thresh
user_res = args.resolution

# ------------------------------
# Check model exists
# ------------------------------
if not os.path.exists(model_path):
    print(f'ERROR: Model not found at {model_path}')
    sys.exit(0)

# ------------------------------
# Load YOLO model
# ------------------------------
model = YOLO(model_path)
labels = model.names

# ------------------------------
# Open source
# ------------------------------
resize = False
if user_res:
    resize = True
    resW, resH = map(int, user_res.split('x'))

if source.isdigit() or 'usb' in source:
    idx = int(source[-1])
    cap = cv2.VideoCapture(idx)
elif os.path.isfile(source) or os.path.isdir(source):
    cap = cv2.VideoCapture(source)
else:
    print(f'Invalid source: {source}')
    sys.exit(0)

# ------------------------------
# Colors for bounding boxes
# ------------------------------
bbox_color = (0, 255, 0)  # green for people

# ------------------------------
# FPS calculation
# ------------------------------
fps_buffer = []
fps_len = 100

# ------------------------------
# Inference loop
# ------------------------------
while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("No frame received. Exiting...")
        break

    if resize:
        frame = cv2.resize(frame, (resW, resH))

    # Run YOLO detection (DO NOT use show=True)
    results = model(frame)

    people_count = 0

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()

        for box, cls_id, conf in zip(boxes, cls_ids, confs):
            if int(cls_id) != 0:  # Only detect person
                continue
            if conf < conf_thresh:
                continue

            people_count += 1
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)
            cv2.putText(frame, "Person", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, bbox_color, 2)

    # Calculate FPS for person detection
    fps = 1 / (time.time() - start_time)
    fps_buffer.append(fps)
    if len(fps_buffer) > fps_len:
        fps_buffer.pop(0)
    avg_fps = np.mean(fps_buffer)

    # Display People count
    cv2.putText(frame, f'People: {people_count}', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Display FPS for person detection
    cv2.putText(frame, f'FPS (person): {avg_fps:.1f}', (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Show frame
    cv2.imshow("YOLO People Counter", frame)

    # Exit on Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
