#!/usr/bin/env python
'''
Brief:      This file runs the Processor Node which 
            - sets up publishers and subscribers
            - handles the inputs and outputs from ui node and cobot_commander node
'''
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from processor.vision_handler import visionHandler
from processor.llm_handler import llmHandler
from std_msgs.msg import Bool
from pyRobotiqGripper import RobotiqGripper

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
        self.subscription_user_prompt = self.create_subscription(
            Bool, 'cobot_response', self.process_cobot_res, 10)

        # Timer to process camera feed at a set interval
        self.camera_timer = self.create_timer(0.1, self.send_camera_feed)  # 10 FPS
        # Timer to perform the functions in each state
        self.state_timer = self.create_timer(1, self.handle_state)

        # Variables
        self.llm_response = None
        self.detected_obj_list = None
        self.start_process = False
        self.retry_flag = 0
        self.pos_flag = 0
        self.pos_reset_flag = False

        # Setup gripper
        self.gripper = RobotiqGripper()
        self.gripper.resetActivate()

        self.get_logger().info("Processor Node Started")

    '''
    This callback function handles the input from the user received from ui node
    Gets response from llm for the user prompt
    Sends the response to the ui node 
    '''
    def process_user_prompt(self, msg):
        self.get_logger().info(f"User input received: {msg.data}")
        user_prompt = String()
        user_prompt = msg.data
        llm_response = self.llm_handler.get_llm_response(user_prompt)
        llm_msg = String()
        llm_msg.data = llm_response
        self.publisher_llm.publish(llm_msg)
        self.start_process = True

    '''
    This function sends the camera feed to the ui node
    '''
    def send_camera_feed(self):
        frame, depth = self.vision_handler.capture_frame()
        results = self.vision_handler.process_frame(frame)
        frame_with_boxes = self.vision_handler.draw_bounding_boxes(frame, results)
        img_msg = self.bridge.cv2_to_imgmsg(frame_with_boxes, encoding="bgr8")
        self.publisher_img.publish(img_msg)

    '''
    This callback function handles the response from cobot_commander node
    '''
    def process_cobot_res(self, msg):
        if self.llm_handler.cobot_state == self.llm_handler.CobotAction.MOVE:
            if msg.data == True and self.pos_reset_flag == False:
                if self.pos_flag == 0: 
                    self.pos_flag += 1
                    self.pos_reset_flag = True
                    self.get_logger().info("inside pos_flag 0")
                elif self.pos_flag == 1:
                    self.get_logger().info("inside pos_flag 1")
                    self.pos_flag = 0
                    llm_response = self.llm_handler.get_llm_response("Successfully moved.")
                    llm_msg = String()
                    llm_msg.data = llm_response
                    self.publisher_llm.publish(llm_msg)
            else:
                llm_response = self.llm_handler.get_llm_response("Unable to reach position")
                llm_msg = String()
                llm_msg.data = llm_response
                self.publisher_llm.publish(llm_msg)

    '''
    This function handles the actions to be taken in each TaskState or CobotAction state
    '''
    def handle_state(self):
        # Debug 
        # self.get_logger().info(f"Processor: Current State is {self.llm_handler.current_state}")
        # self.get_logger().info(f"Processor: Cobot Action is {self.llm_handler.cobot_state}")

        # Start the process once the first response from the user is received.
        if self.start_process == True:

            # Find the target item mentioned by the user
            if self.llm_handler.current_state == self.llm_handler.TaskState.IDENTIFICATION:
                self.get_logger().info(f"Processor: User target item is {self.llm_handler.target_item_user}")
                self.llm_handler.current_state = self.llm_handler.TaskState.DETECTION
                self.retry_flag = 0

            # Send detected object list to the llm 
            elif self.llm_handler.current_state == self.llm_handler.TaskState.DETECTION:
                # Attempt detection 3 times if no object can be found
                if self.retry_flag < 3:
                    # Detect objects using camera and send list to llm
                    self.detected_obj_list = self.vision_handler.detect_objects()
                    self.retry_flag += 1
                    if not self.detected_obj_list:
                        self.get_logger().info("Processor: No objects detected")
                    else:
                        detected_obj_list_str = str(self.detected_obj_list)
                        self.llm_response = self.llm_handler.get_llm_response(detected_obj_list_str)
                        llm_msg = String()
                        llm_msg.data = self.llm_response
                        self.publisher_llm.publish(llm_msg)
                        self.get_logger().info("Processor: Object list sent to LLM")
                else:
                    self.get_logger().info("Processor: Unable to detect target item")

            # Move the cobot based on CobotAction states
            elif self.llm_handler.current_state == self.llm_handler.TaskState.EXECUTION:

                # Move cobot arm to above the object, then move it down to the center of the object
                if self.llm_handler.cobot_state == self.llm_handler.CobotAction.MOVE:
                    if self.llm_handler.position_from_llm:
                        self.get_logger().info(f"Processor: Position from llm x={self.llm_handler.position_from_llm.x}, y={self.llm_handler.position_from_llm.y}, z={self.llm_handler.position_from_llm.z}")
                        obj_msg = Point()
                        obj_msg.x = self.llm_handler.position_from_llm.x 
                        obj_msg.y = self.llm_handler.position_from_llm.y
                        if self.pos_flag == 0:   
                            obj_msg.z = 260.0 # Move arm to a set position above the object 
                                                
                        elif self.pos_flag == 1:
                            obj_msg.z = self.llm_handler.position_from_llm.z # Move arm to center of object
                            self.pos_reset_flag = False
                        self.get_logger().info(f"Processor: Robot coordinates x={obj_msg.x}, y={obj_msg.y}, z={obj_msg.z}")
                        self.publisher_obj_coords.publish(obj_msg)  

                # Close the gripper
                elif self.llm_handler.cobot_state == self.llm_handler.CobotAction.CLOSE:
                    self.get_logger().info(f"Processor: close gripper")
                    self.gripper.close()
                    llm_response = self.llm_handler.get_llm_response("Gripper closed.")
                    llm_msg = String()
                    llm_msg.data = llm_response
                    self.publisher_llm.publish(llm_msg)

                # Open the gripper
                elif self.llm_handler.cobot_state == self.llm_handler.CobotAction.OPEN:
                    self.get_logger().info(f"Processor: open gripper")
                    self.gripper.open()
                    llm_response = self.llm_handler.get_llm_response("Gripper opened.")
                    llm_msg = String()
                    llm_msg.data = llm_response
                    self.publisher_llm.publish(llm_msg)
            
            # If there are not specific action for the llm to perform, it will be in standby state
            elif self.llm_handler.current_state == self.llm_handler.TaskState.STANDBY:

                # Return cobot home and wait for next action
                if self.llm_handler.cobot_state == self.llm_handler.CobotAction.HOME:
                    llm_response = self.llm_handler.get_llm_response("Cobot is returning home. Please wait in Standby State")
                    llm_msg = String()
                    llm_msg.data = llm_response
                    self.publisher_llm.publish(llm_msg)
                    obj_msg = Point()
                    obj_msg.x = 123.46
                    obj_msg.y = -576.85 
                    obj_msg.z = 368.43 
                    self.publisher_obj_coords.publish(obj_msg)
                    llm_response = self.llm_handler.get_llm_response("Cobot has returned home. You can begin next action")
                    llm_msg = String()
                    llm_msg.data = llm_response
                    self.publisher_llm.publish(llm_msg)
                    self.llm_handler.cobot_state = self.llm_handler.CobotAction.STANDBY
                    self.start_process = False # end the process 
                    
            
def main(args=None):
    rclpy.init(args=args)
    node = Processor()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()