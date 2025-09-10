import cv2
import yaml
import time
import os
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from utils.draw import draw_boxes

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load YOLO model
model = YOLO(config["model"])

# Init DeepSORT tracker
tracker = DeepSort(max_age=30)

# Open webcam
cap = cv2.VideoCapture(config["webcam_index"])
if not cap.isOpened():
    print("❌ Could not open webcam. Try changing webcam_index in config.yaml")
    raise SystemExit

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = None
saving = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()

    # YOLO inference
    results = model.predict(
        frame, conf=config["confidence"], iou=config["iou"],
        imgsz=config["img_size"], verbose=False
    )

    detections = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            # DeepSORT wants [x, y, w, h], conf, class_name
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls_name))

    # Track
    tracks = tracker.update_tracks(detections, frame=frame)

    tracked = []
    for track in tracks:
        if not track.is_confirmed():
            continue
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        tid = track.track_id
        cls_name = track.det_class if track.det_class else "obj"
        conf = track.det_conf if track.det_conf else 0
        tracked.append((x1, y1, x2, y2, tid, cls_name, conf))

    fps = 1.0 / max(1e-6, (time.time() - start_time))
    frame = draw_boxes(frame, tracked, fps)

    cv2.imshow("YOLOv8 + DeepSORT (Webcam)", frame)

    # Optional saving
    if saving:
        if out is None:
            os.makedirs("output", exist_ok=True)
            out = cv2.VideoWriter(
                "output/tracking_output.mp4", fourcc, 20.0,
                (frame.shape[1], frame.shape[0])
            )
        out.write(frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s"):
        saving = not saving
        if saving:
            print("💾 Saving to output/tracking_output.mp4 …")
        else:
            print("🛑 Stopped saving")
            if out:
                out.release()
                out = None

cap.release()
if out:
    out.release()
cv2.destroyAllWindows()
