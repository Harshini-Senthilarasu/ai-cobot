#!/usr/bin/env python
import rclpy
from rclpy.node import Node
import os
import sys
from dotenv import load_dotenv
import google.generativeai as genai
from flask import Flask, Response
import pyrealsense2 as rs
import cv2
import numpy as np
from ultralytics import YOLO
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import datetime

load_dotenv(dotenv_path="/home/harshini/capstone/src/.env")
api_key = os.getenv("google_api_key")
yolo_path = os.getenv("yolo_path")
data_files_path = os.getenv("data_files_path")

class Processor(Node):
    def __init__(self):
        super().__init__("processor")
        # Setup Gemini 
        self.generation_config = {"temperature": 1, "top_p": 0.95, "top_k": 40, "max_output_tokens": 8192, "response_mime_type": "text/plain"}
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config=self.generation_config,
            system_instruction="""
            You are a Large Language Model connected to a cobot arm.
            Find the target item and the goal positions in the user prompt.
            Breakdown the user prompt into steps to be taken to move the cobot arm to fulfill the user's instructions.
            At the end of breaking down the instructions, ask the user if this is breakdown is correct by saying: Reply 'yes' to proceed or 'no' to modify."
            If the user says no, then redo the steps again with any additional information given by the user.
            If the user says yes, then if the target item exists in the list of detected objects provided with a confidence score higher than 0.6, reply with the target item name in the list. 
            If the item is not present in the list, reply saying the target item name in the prompt cannot be found in the detected objects list.
            """
        )
        self.convo_hist = []
        self.chat_session = self.model.start_chat(history=self.convo_hist)

        # Setup Yolo model
        self.yolo_model = YOLO(yolo_path) # load YOLO model

        # Initialize RealSense pipeline
        self.pipe = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        try:
            self.pipe.start(config)
        except Exception as e:
            self.get_logger().info(f"Error starting RealSense pipeline: {e}")
        
        # Subscribers/Publishers
        self.bridge = CvBridge()
        self.publisher_llm = self.create_publisher(
            String, 'llm_response', 10)
        self.publisher_img = self.create_publisher(
            Image, 'camera_feed', 10)
        self.subscription_user_prompt = self.create_subscription(
            String, 'user_prompt', self.process_user_prompt, 10)

        # Timer to process camera feed at a set interval
        self.timer = self.create_timer(0.1, self.send_camera_feed)  # 10 FPS

        self.get_logger().info("Processor Node Started")

    def process_user_prompt(self, msg):
        self.get_logger().info(f"User input received: {msg.data}")
        user_prompt = String()
        user_prompt = msg.data
        llm_response = self.get_llm_response(user_prompt)
        llm_msg = String()
        llm_msg.data = llm_response
        self.publisher_llm.publish(llm_msg)

    def get_llm_response(self, user_prompt):
        self.convo_hist.append({"role": "user", "text": user_prompt})
        response = self.chat_session.send_message(user_prompt)
        self.convo_hist.append({"role": "llm", "text": response.text})
        self.get_logger().info(f"LLM response: {response.text}")
        return response.text

    def yolo_detect(self, frame):
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

    def capture_frame(self):
        # Get frames from camera
        frames = self.pipe.wait_for_frames()
        aligned_frames = rs.align(rs.stream.color).process(frames)  # aligning depth and color frames
        depth_frame = aligned_frames.get_depth_frame().as_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            self.get_logger().info("Failed to capture frames.")
            return 

        frame = np.asanyarray(color_frame.get_data())
        return frame

    def send_camera_feed(self):
        frame = self.capture_frame()
        results = self.yolo_detect(frame)
        frame_with_boxes = self.draw_bounding_boxes(frame, results)
        img_msg = self.bridge.cv2_to_imgmsg(frame_with_boxes, encoding="bgr8")
        self.publisher_img.publish(img_msg)

    def detect_objects(self):
        frame = self.capture_frame()
        
        if frame is None:
            self.get_logger().info("No frame captured in detect_objects function")
            return {}
        
        results = self.yolo_detect(frame)
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


def main(args=None):
    rclpy.init(args=args)
    node = Processor()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()