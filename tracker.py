import cv2
from ultralytics import YOLO
from utils.draw import draw_boxes   # jo pehle bhi use kar rahe the

class ObjectTracker:
    def __init__(self, model_path="yolov8n.pt"):
        # YOLO model load karo
        self.model = YOLO(model_path)

    def detect(self, frame):
        # Detection run karo
        results = self.model(frame)

        # Pehla result nikaalo
        result = results[0]

        # Agar boxes mile toh unko draw karo
        if result.boxes is not None:
            frame = draw_boxes(frame, result)

        return frame
