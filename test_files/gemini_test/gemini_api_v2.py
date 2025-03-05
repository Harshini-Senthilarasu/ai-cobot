import os
from dotenv import load_dotenv
# import streamlit as st
import google.generativeai as genai
import cv2
import numpy as np
import pyrealsense2 as rs
# import base64
# import time
# import re
from google.cloud import vision
from google.oauth2 import service_account
from difflib import get_close_matches

# Load environment variables
load_dotenv(dotenv_path="/home/harshini/capstone/src/.env")
credentials_path = os.getenv("google_vision_cred")
api_key = os.getenv("api_key")

# Load credentials from the specified path
credentials = service_account.Credentials.from_service_account_file(credentials_path)

# Initialize the Vision API client with the credentials
vision_client = vision.ImageAnnotatorClient(credentials=credentials)

# Setup Gemini
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}
genai.configure(api_key=api_key)
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config=generation_config,
    system_instruction="Find the target item in the prompt. If the item exists in the list of detected objects provided with a confidence score higher than 0.6, reply with the target item name in the list. If the item is not present in the list, reply saying the target item name in the prompt cannot be found in the detected objects list."
    # system_instruction=""" 
    #     Identify the target object in the prompt.
    #     Provide with the coordinates of the top left, top right, bottom left, bottom right of the bounding box and the centroid of the object identified using Google Vision model assuming the top left corner of the image provided is origin (0,0). 
    #     Provide the 5 sets of coordinates in the following format:
    #     "Top Left of bounding box: [x, y]"
    #     "Top Right of bounding box: [x, y]"
    #     "Bottom Left of bounding box: [x, y]"
    #     "Bottom Right of bounding box: [x, y]"
    #     "Centroid: [x, y]"
    #     If the object is not found, return an empty response "{}".
    # """
)

# Initialize RealSense pipeline
pipe = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

try:
    pipe.start(config)
except Exception as e:
    print(f"Error starting RealSense pipeline: {e}")

# Capture image from RealSense Camera
def capture_image():
    # Get frames from camera
    frames = pipe.wait_for_frames()
    aligned_frames = rs.align(rs.stream.color).process(frames)  # aligning depth and color frames
    depth_frame = aligned_frames.get_depth_frame().as_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    if not depth_frame or not color_frame:
        print("Failed to capture frames.")
        return
    
    color_image = np.asanyarray(color_frame.get_data())
    image_path = "image.jpg"
    cv2.imwrite(image_path, color_image)
    print("Image captured and saved successfully")
    return image_path, depth_frame
    # Convert image to Base64
    # try:
    #     with open(image_path, "rb") as image_file:
    #         image_data = image_file.read()
    # except Exception as e:
    #     print(f"Error opening image: {e}")

# Detect objects using Google Vision API
def detect_objects(image_path):
    try:
        with open(image_path, "rb") as image_file:
            content = image_file.read()
    except Exception as e:
        print(f"Error opening image: {e}")
        return {}

    image = vision.Image(content=content)
    response = vision_client.annotate_image({
        "image": image,
        "features": [{"type": vision.Feature.Type.OBJECT_LOCALIZATION}]
    })

    objects_info = {}
    img = cv2.imread(image_path)
    height, width = img.shape[:2]

    # object_localization from vision api provides with only the 4 normalized vertices of the bounding poly. 
    # centroid needs to be mathematically derived
    for obj in response.localized_object_annotations:
        box = obj.bounding_poly.normalized_vertices
        points = [(int(vertex.x * width), int(vertex.y * height)) for vertex in box]

        if len(points) < 4:
            continue  # Skip incomplete bounding boxes

        top_left, top_right, bottom_right, bottom_left = points[:4]

        # Compute centroid (avg of 4 corner points)
        centroid_x = (top_left[0] + top_right[0] + bottom_right[0] + bottom_left[0])/4
        centroid_y = (top_left[1] + top_right[1] + bottom_right[1] + bottom_left[1])/4
        centroid = (int(centroid_x), int(centroid_y))

        objects_info[obj.name] = {
            "confidence": obj.score,
            "top_left": top_left,
            "top_right": top_right,
            "bottom_left": bottom_left,
            "bottom_right": bottom_right,
            "centroid": centroid,
        }

    return objects_info

def draw_box(target_item, detected_objects, image_path, output_path="output.jpg"):
    try:
        image = cv2.imread(image_path)
        if image is None:
            print("Error loading image.")
            return

        item_info = detected_objects[target_item]

        # Extract bounding box coordinates
        top_left = tuple(item_info["top_left"])
        bottom_right = tuple(item_info["bottom_right"])
        centroid = tuple(item_info["centroid"])

        # Draw bounding box
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
        # Draw centroid
        cv2.circle(image, centroid, 5, (0, 0, 255), -1)
        # Label the object
        label = f"{target_item} ({item_info['confidence']:.2f})"
        cv2.putText(image, label, (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Save and display the output image
        cv2.imwrite(output_path, image)
        print(f"Bounding box drawn for '{target_item}' and saved to {output_path}")
    
    except Exception as e:
        print(f"Error in draw box function: {e}")

def match_target_item(user_prompt, detected_objects):
    detected_names = list(detected_objects.keys())
    closest_match = get_close_matches(user_prompt, detected_names, n=1, cutoff=0.6)

    if closest_match:
        print("Best match found")
        return closest_match[0] # Return the best match form the detected objects
    return None

def convert2Dto3D(obj_list, target_item, depth_frame):
    item_list = obj_list[target_item]
    centroid = tuple(item_list["centroid"])
    depth_value = depth_frame.get_distance(centroid[0], centroid[1])
    print(f"Depth Distance: {depth_value}")
    
    intrinsics = depth_frame.profile.as_video_stream_profile().get_intrinsics() # Get intrinsic parameters of the RealSense camera
    point_3D = rs.rs2_deproject_pixel_to_point(intrinsics, [centroid[0], centroid[1]], depth_value) # Calculate real-world coordinates using intrinsic parameters
    print(f"x, y, z: {point_3D[0]*1000}, {point_3D[1]*1000}, {point_3D[2]*1000}") # (x, y, z) in meters
    return depth_value

def main(args=None):
    try:
        ### Gemini + Object Detection ###
        user_prompt = input("Enter prompt: ") # User input
        convo_hist = [] # Initialise convo history
        image_path, depth_frame = capture_image() # Capture the image from the camera
        detected_objects = detect_objects(image_path) # Send captured image to google vision api
        if not detected_objects:
            print("No objects detected")
            return
        print("Detected objects", detected_objects.keys()) # See list of items detected by vision api
        objects_str = "\n".join(
            [f"{name}: {info['confidence']}" for name, info in detected_objects.items()]) # Convert detected_objects into a readable string
        convo_hist = [{"role": "user", "parts": [{"text": f"Detected Objects:\n{objects_str}"}]}] # Add the detected objects into the convo history
        chat_session = model.start_chat(history=convo_hist) # Send updated convo history with detected objects to gemini
        response = chat_session.send_message(user_prompt) # Send user prompt to gemini

        if response:
            print("Gemini: ", response.text)
            target_item = response.text.strip()
            target_item = match_target_item(target_item, detected_objects) # Match the detected object and the target object 
            print("Target Item: ", target_item)

            if target_item:
                print(f"Target object found: {target_item}")
                draw_box(target_item, detected_objects, image_path) # if the item is found, the draw bounding box on image and save it
            else:
                print(f"Object '{user_prompt}' not found in detected objects.")

        ### 2D to 3D coordinate conversion ###
        # Required: centroid of the object, depth from realsense
        obj_pos = convert2Dto3D(detected_objects, target_item, depth_frame)

    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        pipe.stop()

if __name__ == "__main__":
    main()
