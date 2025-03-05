###
# Author    : Harshini Senthilarasu
# Brief     : Test google vision api capability using built in camera
###
import os
import cv2
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv(dotenv_path="/home/harshini/capstone/src/.env")
yolo_path = os.getenv("yolo_path")
yolo_model = YOLO(yolo_path) # load YOLO model

def detect_objects(frame):
    results = yolo_model(frame) # Perform object detection
    return results

def draw_objects(frame, results):
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            label = result.names[int(box.cls[0])]  # Class label
            conf = box.conf[0].item()  # Confidence score
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # Draw bounding box
            # Display label and confidence score
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

st.set_page_config(page_title="YOLOv11 Camera Test", layout="wide")  # Streamlit page config
frame_placeholder = st.empty()  # Placeholder for the video feed
cap = cv2.VideoCapture(0)  # Initialize camera

if not cap.isOpened():
    st.error("Error: Could not open camera.")
else:
    st.title("YOLOv11 Object Detection")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Failed to capture frame.")
            break

        results = detect_objects(frame)  # Detect objects using YOLOv11
        draw_objects(frame, results)  # Draw bounding boxes on detected objects
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB for Streamlit display      
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)  # Display frame

        # Break the loop if 'q' is pressed (only works in terminal execution)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()