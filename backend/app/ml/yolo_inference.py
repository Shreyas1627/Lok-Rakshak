# app/ml/yolo_inference.py
import cv2
import numpy as np
from ultralytics import YOLO

class VisionEngine:
    def __init__(self):
        # Using YOLOv8n (nano) for maximum speed during the hackathon
        self.model = YOLO('yolov8n.pt') 
        self.prev_gray = None

    def process_frame(self, frame):
        # Resize for performance
        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. YOLO Density (Count people - class 0 in COCO)
        results = self.model(frame, classes=[0], verbose=False)
        person_count = len(results[0].boxes)
        
        # 2. Optical Flow (Crowd Physics)
        vector_variance = 0.0
        if self.prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            vector_variance = np.var(angle) # High variance = chaotic movement/panic
            
        self.prev_gray = gray
        
        # 3. Dummy DBSCAN Compression score (0.0 to 1.0)
        # For the hackathon demo, we correlate compression with density > 50
        compression_score = min(person_count / 100.0, 1.0) 

        return {
            "density": person_count,
            "vector_variance": float(vector_variance),
            "compression_score": float(compression_score)
        }