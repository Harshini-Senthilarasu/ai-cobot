import os
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import datetime
from dotenv import load_dotenv

load_dotenv(dotenv_path="/home/harshini/capstone/src/.env")
yolo_path = os.getenv("yolo_path")
data_files_path = os.getenv("data_files_path")

class visionHandler:
    def __init__(self, logger):
        self.logger = logger
        # Setup Yolo model
        self.yolo_model = YOLO(yolo_path) # load YOLO model

        # Initialize RealSense pipeline
        self.pipe = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        self.logger.info(f"Starting RealSense pipeline...")
        try:
            self.pipe.start(config)
        except Exception as e:
            self.logger.info(f"Error starting RealSense pipeline: {e}")

    def capture_frame(self):
        # Get frames from camera
        frames = self.pipe.wait_for_frames()
        aligned_frames = rs.align(rs.stream.color).process(frames)  # aligning depth and color frames
        depth_frame = aligned_frames.get_depth_frame().as_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            self.logger.info("Failed to capture frames.")
            return 

        frame = np.asanyarray(color_frame.get_data())
        return frame
    
    def process_frame(self, frame):
        results = self.yolo_model(frame, verbose=False)
        return results

    def draw_bounding_boxes(self, frame, results):
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                conf = box.conf[0]  # Confidence score
                cls = int(box.cls[0])  # Class ID
                label = f"{self.yolo_model.names[cls]}: {conf:.2f}"

                # Draw rectangle and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame

    def detect_objects(self):
        frame = self.capture_frame()
        
        if frame is None:
            self.logger.info("No frame captured in detect_objects function")
            return {}
        
        results = self.process_frame(frame)
        frame_with_boxes = self.draw_bounding_boxes(frame, results)

        # Save image into directory
        os.makedirs(data_files_path, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        image_path = os.path.join(data_files_path, f"output_{timestamp}.jpg")
        cv2.imwrite(image_path, frame_with_boxes)

        # Extract object information
        objects_info = {}
        height, width = frame_with_boxes.shape[:2]
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = self.yolo_model.names[cls]

                # Compute centroid
                centroid_x = (x1 + x2)//2
                centroid_y = (y1 + y2)//2
                centroid = (centroid_x, centroid_y)

                objects_info[label] = {
                    "confidence": conf,
                    "top_left": (x1, y1),
                    "top_right": (x2, y1),
                    "bottom_left": (x1, y2),
                    "bottom_right": (x2, y2),
                    "centroid": centroid,
                }

        return objects_info
