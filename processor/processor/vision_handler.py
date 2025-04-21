'''
Brief:      This file contains the helper class visionHandler
            - sets up RealSense camera
            - captures the frames from the camera
            - uses Yolo v11 model to get the objects detected
            - gets info on the detected objects (position in camera frame, depth distance, confidence score)
            - converts the position of the detected objects from camera frame to robot base frame
'''
import os
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import datetime
from dotenv import load_dotenv

load_dotenv(dotenv_path="/home/harshini/capstone/src/.env")
yolo_path = os.getenv("yolo_path")
data_files_path = os.getenv("image_output_path")

'''
This function uses known points of objects in the camera frame 
and the end effector to find a homogenous matrix from the
camera frame to the end-effector frame
'''
def compute_transformation():
    # positions_robot = np.array([
    #     [-152.65,-691.52,100.95],       #1
    #     [-0.71,-697.51,90.49],          #2
    #     [69.75,-706.67,112.64],         #3
    #     [-2.14,-611.34,107.73],         #4
    #     [89.89,-619.23,108.74],         #5
    #     [-13.60,-812.27,94.51],         #6
    #     [95.72,-780.79,114.19],         #7
    #     [-167.57,-774.03,1139.97],      #8
    # ])

    # positions_camera = np.array([
    #     [107.48, 63.30, 574.00],        #1
    #     [-63.04, 65.51, 594.00],        #2
    #     [-136.91, 66.16, 609.00],       #3
    #     [-57.60, 59.73, 511.00],        #4
    #     [-160.99, 61.66, 513.00],       #5
    #     [-54.21, 72.33, 709.00],        #6
    #     [-165.77, 72.36, 687.00],       #7
    #     [73.88, 70.93, 663.00],         #8
    # ])

    positions_robot = np.array([
        [-60.46,-690.50,110.00],       #1
        [-200.68,-701.76,110.00],          #2
        [124.72,-689.23,110.00],         #3
        [-204.02,-529.83,110.00],         #4
        [-50.95,-527.53,110.00],         #5
        [95.31,-546.37,110.00],         #6
        [-105.1282,-759.73,110.00],         #7
        [58.57,-790.28,110.00],      #8
    ])

    positions_camera = np.array([
        [33.01,71.54,596.00],        #1
        [169.22, 30.84, 588.00],        #2
        [-157.59,38.03,640.00],       #3
        [148.57, 31.64, 428.00],        #4
        [5.66, 68.37, 451.00],       #5
        [-136.89, 27.50, 479.00],        #6
        [87.82, 31.39, 661.00],       #7
        [-57.75, 33.18, 724.00],         #8
    ])

    # Center the points
    centroid_camera = np.mean(positions_camera, axis=0)
    centroid_robot = np.mean(positions_robot, axis=0)

    # Center the data
    centered_camera = positions_camera - centroid_camera
    centered_robot = positions_robot - centroid_robot

    # Compute the covariance matrix
    H = np.dot(centered_camera.T, centered_robot)

    # Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(H)

    # Compute rotation matrix
    rot_mat = np.dot(Vt.T, U.T)

    # Handle reflection case
    if np.linalg.det(rot_mat) < 0:
        Vt[-1, :] *= -1
        rot_mat = np.dot(Vt.T, U.T)

    # Compute translation vector
    trans_vect = centroid_robot - np.dot(rot_mat, centroid_camera)

    # Compute homogeneous matrix
    T = np.eye(4)
    T[:3,:3] = rot_mat
    T[:3,3] = trans_vect

    return T


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

        self.logger.info("Vision Handler started")

        # Camera calibration parameters
        self.camera_matrix = np.array([[876.51478444, 0, 630.09501269],  
                                      [0, 877.82884167, 337.57343916], 
                                      [0, 0, 1]])
        self.dist_coeffs = np.array([-1.70514383e-01,  3.60531196e+00, -2.57889757e-03, -1.10947273e-02, -1.38794754e+01])

    '''
    Function to get the color frame and depth frame from the camera
    '''
    def capture_frame(self):
        # Get frames from camera
        frames = self.pipe.wait_for_frames()
        aligned_frames = rs.align(rs.stream.color).process(frames)  # aligning depth and color frames
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            self.logger.info("Failed to capture frames.")
            return 

        frame = np.asanyarray(color_frame.get_data())
        return frame, depth_frame
    
    '''
    Function to process the frame to get detected objects using yolo model
    '''
    def process_frame(self, frame):
        results = self.yolo_model(frame, verbose=False)
        return results

    '''
    Function to draw bounding boxes around the detected object(s) 
    '''
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
    
    '''
    Function to convert the position of the object from camera frame to end-effector frame
    '''
    def obj_to_robot_coords(self, obj_coords):
        T_camera_to_robot = compute_transformation()
        point_camera_homogeneous = np.append(obj_coords, 1) # Convert to homogeneous coordinates
        point_robot = np.dot(T_camera_to_robot, point_camera_homogeneous) # Apply transformation
        return point_robot[:3] # Extract x,y,z

    '''
    Function to get the frames from camera, detect objects, and create 
    a list of information about the detected objects
    '''
    def detect_objects(self):
        frame, depth_frame = self.capture_frame()
        
        if frame is None or depth_frame is None:
            self.logger.info("Vision Handler: No frame captured in detect_objects function")
            return {}, None
        
        # Process frame with YOLO
        results = self.process_frame(frame)
        frame_with_boxes = self.draw_bounding_boxes(frame, results)

        # Save image into directory
        os.makedirs(data_files_path, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        image_path = os.path.join(data_files_path, f"output_{timestamp}.jpg")
        cv2.imwrite(image_path, frame_with_boxes)

        # Extract object information
        objects_info = {}
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = self.yolo_model.names[cls]

                # Compute centroid
                centroid_x = (x1 + x2) // 2
                centroid_y = (y1 + y2) // 2
                centroid = (centroid_x, centroid_y)

                # Compute Depth
                depth_value = depth_frame.get_distance(centroid[0], centroid[1])
                intrinsics = depth_frame.profile.as_video_stream_profile().get_intrinsics()  # Get intrinsic parameters of the RealSense camera

                self.logger.info(f"Vision Handler: depth value {depth_value}")
                # Calculate 3D point using deprojection
                point_3D = rs.rs2_deproject_pixel_to_point(intrinsics, [centroid[0], centroid[1]], depth_value)
                # x, y, z = point_3D[0] * 1000, point_3D[1] * 1000, point_3D[2] * 1000 
                # self.logger.info(f"Vision Handler: 3D coordinates for {label} are {x}, {y}, {z}")
                z = point_3D[2] * 1000 # in mm

                centroid_undistort = np.array([[centroid_x, centroid_y]], dtype=np.float32)
                # Undistorted 3D points for improved accuracy
                undistorted_3D_point = cv2.undistortPoints(centroid_undistort, self.camera_matrix, self.dist_coeffs)
                undistorted_x, undistorted_y = undistorted_3D_point[0][0]

                obj_coords_array = np.array([undistorted_x, undistorted_y, z])
                obj_robot_pos = self.obj_to_robot_coords(obj_coords_array)

                ee_x, ee_y, ee_z = obj_robot_pos
                self.logger.info(f"Vision Handler: EE pos for {label} are {ee_x}, {ee_y}, {ee_z}")

                objects_info[label] = {
                    "confidence": conf,
                    "top_left": (x1, y1),
                    "top_right": (x2, y1),
                    "bottom_left": (x1, y2),
                    "bottom_right": (x2, y2),
                    "centroid": centroid,
                    "depth": depth_value,
                    "3D_coordinates": {"x": ee_x, "y": ee_y, "z": ee_z},
                }

        return objects_info
