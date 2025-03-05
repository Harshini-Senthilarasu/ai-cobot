#!/usr/bin/env python
import rclpy
from rclpy.node import Node
from flask import Flask, Response
from cv_bridge import CvBridge
from std_msgs.msg import String
from sensor_msgs.msg import Image
from processor.vision_handler import visionHandler
from processor.llm_handler import llmHandler

class Processor(Node):
    def __init__(self):
        super().__init__("processor")

        self.llm_handler = llmHandler(self.get_logger())
        self.vision_handler = visionHandler(self.get_logger())
        
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
        llm_response = self.llm_handler.get_llm_response(user_prompt)
        llm_msg = String()
        llm_msg.data = llm_response
        self.publisher_llm.publish(llm_msg)

    def send_camera_feed(self):
        frame = self.vision_handler.capture_frame()
        results = self.vision_handler.process_frame(frame)
        frame_with_boxes = self.vision_handler.draw_bounding_boxes(frame, results)
        img_msg = self.bridge.cv2_to_imgmsg(frame_with_boxes, encoding="bgr8")
        self.publisher_img.publish(img_msg)


def main(args=None):
    rclpy.init(args=args)
    node = Processor()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()