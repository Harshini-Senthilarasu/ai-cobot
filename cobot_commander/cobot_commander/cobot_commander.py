#!/usr/bin/env python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
import numpy as np

# Positions of robot end-effector (where object is placed) relative to robot base frame 
positions_robot = np.array([
    [48.662, -610.541, 80.809],
    [-58.761, -550.130, 80.532],
    [-186.497, -525.973, 80.697],
])

# Positions of object relative to camera (X, Y, Z) in millimeters
positions_camera = np.array([
    [-156.54, 87.52, 428.00],
    [-55.13, 83.31, 366.00],
    [71.44, 77.82, 337.00],
])

def compute_transformation(positions_camera, positions_robot):
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
    trans_mat = centroid_robot - np.dot(rot_mat, centroid_camera)

    return rot_mat, trans_mat

rot_mat, trans_mat = compute_transformation(positions_camera, positions_robot)

def transform_to_robot_config(positions_camera, rot_mat, trans_mat):
    return np.dot(rot_mat, positions_camera) + trans_mat

class CobotCommander(Node):
    def __init__(self):
        super().__init__("cobot_commander")

        self.subscriber = self.create_subscription(
            Point, 'obj_coords', self.get_obj_coords, 10)

    def get_obj_coords(self, msg):
        self.get_logger().info(f"Received obj coords: {msg}")
        obj_coords = Point()
        obj_coords = msg
        robot_pos = self.convert_to_robot_pos(obj_coords)

    def convert_to_robot_pos(self, obj_coords):
        object_pos = np.array([obj_coords.x, obj_coords.y, obj_coords.z])
        robot_pos = np.dot(rot_mat, object_pos) + trans_mat
        self.get_logger().info(f"Robot Coordinates: {robot_pos}") # x, y, z of cartesian coords
        return robot_pos

def main(args=None):
    rclpy.init(args=args)
    node = CobotCommander()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()