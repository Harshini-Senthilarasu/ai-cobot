###
# Author    : Harshini Senthilarasu
# Brief     : Test google vision api capability using built in camera
###
import os
import cv2
import numpy as np
import streamlit as st
from google.cloud import vision
from google.oauth2 import service_account
from dotenv import load_dotenv

load_dotenv(dotenv_path="/home/harshini/capstone/src/.env") # Load environment variables
credentials_path = os.getenv("google_vision_cred") # get path to credentials
credentials = service_account.Credentials.from_service_account_file(credentials_path) # load credentials
client = vision.ImageAnnotatorClient(credentials=credentials) # initialize the google vision api

# Function to send an image frame to the Google Vision API for object detection
def detect_objects(frame):
    # Convert the frame to byte data (JPEG format)
    _, encoded_image = cv2.imencode('.jpg', frame)
    content = encoded_image.tobytes()
    image = vision.Image(content=content) # Create an Image object for Vision API
    response = client.object_localization(image=image) # Call Google Vision API for object detection
    objects = response.localized_object_annotations # Extract detected objects
    return objects

# Function to draw bounding boxes around detected objects in the frame
def draw_objects(frame, objects):
    for obj in objects:
        box = obj.bounding_poly.normalized_vertices
        points = [(int(vertex.x * frame.shape[1]), int(vertex.y * frame.shape[0])) for vertex in box]
        cv2.polylines(frame, [np.array(points)], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(frame, obj.name, (points[0][0], points[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

st.set_page_config(page_title="Camera Test", layout="wide") # Streamlit page config
frame_placeholder = st.empty() # placeholder for the video feed
cap = cv2.VideoCapture(0) # intialize camera

if not cap.isOpened():
    st.error("Error: Could not open camera.")
else:
    st.title("Object Detection")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Failed to capture frame.")
            break

        objects = detect_objects(frame) # Send the captured frame to Google Vision API for object detection
        draw_objects(frame, objects) # Draw bounding boxes for detected objects on the frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert frame to RGB to display on streamlit        
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True) # Display the updated frame with object detection in Streamlit

        # Break the loop if 'q' is pressed (you can control it from the terminal)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
