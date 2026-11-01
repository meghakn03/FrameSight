"""A model and ByteTrack state owned by one video session."""
from time import perf_counter
import cv2
from ultralytics import YOLO
from framesight.stats import Detection, Statistics


class Engine:
    def __init__(self, model='yolo11n.pt', confidence=0.35, classes=None, size=640):
        self.model = YOLO(model)
        self.confidence = confidence
        self.classes = classes
        self.size = size
        self.stats = Statistics()

    def process(self, frame, timestamp):
        start = perf_counter()
        result = self.model.track(frame, persist=True, tracker='bytetrack.yaml',
                                  conf=self.confidence, classes=self.classes,
                                  imgsz=self.size, verbose=False)[0]
        detections = []
        if result.boxes is not None:
            boxes = result.boxes
            ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(boxes)
            for box, cls, conf, track_id in zip(boxes.xyxy.cpu().tolist(),
                    boxes.cls.int().cpu().tolist(), boxes.conf.cpu().tolist(), ids):
                detections.append(Detection(result.names[cls], conf, track_id, tuple(box)))
        annotated = frame.copy()
        for d in detections:
            x1, y1, x2, y2 = map(int, d.box)
            color = (173, 214, 51)
            label = f'{d.category} {d.confidence:.0%}'
            if d.track_id is not None:
                label = f'#{d.track_id} {label}'
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, label, (x1, max(18, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        self.stats.update(detections, timestamp, perf_counter() - start)
        return annotated
