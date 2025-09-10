from flask import Flask, render_template, Response
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

# Load YOLO model once (heavy op)
model = YOLO(config["model"])

# Init DeepSORT tracker once
tracker = DeepSort(max_age=30)

app = Flask(__name__, template_folder="templates")
camera = None
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
saving = False
out = None

def gen_frames():
    global camera, tracker, model, out, saving
    # if camera not open, yield nothing
    while camera is not None and camera.isOpened():
        ret, frame = camera.read()
        if not ret:
            break

        start_time = time.time()

      
        # YOLO inference (force BGR, keep size same as main.py)
        results = model.predict(
           source=frame,                     # ✅ specify source kwarg
           conf=config["confidence"],
           iou=config["iou"],
           imgsz=config["img_size"],         # same as in config.yaml
           verbose=False
        )


        # build detections for DeepSORT: [ [x,y,w,h], conf, class_name ]
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls_name))

        # tracker update
        tracks = tracker.update_tracks(detections, frame=frame)

        # build tracked list in same format draw_boxes expects
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
        # draw boxes (your existing util)
        frame = draw_boxes(frame, tracked, fps)

        # Optionally write (if saving toggled via other route later)
        if saving:
            if out is None:
                os.makedirs("output", exist_ok=True)
                out = cv2.VideoWriter(
                    "output/tracking_output.mp4", fourcc, 20.0,
                    (frame.shape[1], frame.shape[0])
                )
            out.write(frame)

        # encode for MJPEG streaming
        ret2, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ret2:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    # cleanup if loop ends
    if out:
        out.release()
        out = None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/start")
def start_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(config["webcam_index"])
        if not camera.isOpened():
            camera = None
            return "Failed to open camera", 500
    return "Camera started"

@app.route("/stop")
def stop_camera():
    global camera, tracker
    if camera is not None:
        camera.release()
        camera = None
    # reset tracker state so IDs start fresh next time
    tracker = DeepSort(max_age=30)
    return "Camera stopped"

@app.route("/video_feed")
def video_feed():
    # Response uses generator that yields multipart JPEG frames
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=True)
