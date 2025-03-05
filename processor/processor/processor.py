#!/usr/bin/env python
import rclpy
from rclpy.node import Node
from flask import Flask, Response
from cv_bridge import CvBridge
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
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
        self.publisher_obj_coords = self.create_publisher(
            Point, 'obj_coords', 10)

        # Timer to process camera feed at a set interval
        self.camera_timer = self.create_timer(0.1, self.send_camera_feed)  # 10 FPS
        self.state_timer = self.create_timer(1, self.handle_state)

        self.llm_response = None
        self.detected_obj_list = None
        self.depth_frame = None

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
        frame, depth_frame = self.vision_handler.capture_frame()
        results = self.vision_handler.process_frame(frame)
        frame_with_boxes = self.vision_handler.draw_bounding_boxes(frame, results)
        img_msg = self.bridge.cv2_to_imgmsg(frame_with_boxes, encoding="bgr8")
        self.publisher_img.publish(img_msg)

    def handle_state(self):
        if self.llm_handler.current_state == self.llm_handler.TaskState.DETECTION:
            self.detected_obj_list, self.depth_frame = self.vision_handler.detect_objects()

            if not self.detected_obj_list:
                self.get_logger().info("No objects detected")
                return

            detected_obj_list_str = str(self.detected_obj_list)
            self.llm_response = self.llm_handler.get_llm_response(detected_obj_list_str)
            llm_msg = String()
            llm_msg.data = self.llm_response
            self.publisher_llm.publish(llm_msg)

        elif self.llm_handler.current_state == self.llm_handler.TaskState.EXECUTION and self.llm_handler.execution_flag == True:
            self.get_logger().info(f"LLM Response: {self.llm_response}")
            self.llm_handler.execution_flag = False
            target_item = self.llm_handler.extract_target(self.llm_response)
            self.get_logger().info(f"Found target item in response: {target_item}")
            if target_item:
                self.get_logger().info(f"Target item: {target_item}")
                if self.depth_frame:
                    obj_pos = self.vision_handler.convert2Dto3D(self.detected_obj_list, target_item, self.depth_frame)
                    obj_msg = Point()
                    obj_msg.x = obj_pos[0]
                    obj_msg.y = obj_pos[1]
                    obj_msg.z = obj_pos[2]
                    self.publisher_obj_coords.publish(obj_msg)

            
def main(args=None):
    rclpy.init(args=args)
    node = Processor()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()