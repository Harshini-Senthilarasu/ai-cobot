import numpy as np

# Positions of robot end-effector (where object is placed) relative to robot base frame 
positions_robot = np.array([
    [48.662, -610.541, 80.809],
    [-58.761, -550.130, 80.532],
    [-186.497, -525.973, 80.697],
])

# Cartesian Coordinates (X, Y, Z, Rx, Ry, Rz)
# pt1: 48.662, -610.541, 80.809, 176.132, 1.926, 6.748
# pt2: -58.761, -550.130, 80.532, 179.035, 3.382, -4.098
# pt3: -186.497, -525.973, 80.697, 175.242, -2.213, 8.072

# Joint Angle (J1, J2, J3, J4, J5, J6)
# pt1: 83.443, -32.268, -73.883, 20.380, -91.063, -13.413
# pt2: 72.590, -24.049, -85.534, 21.323, -93.133, -13.375
# pt3: 55.404, -23.285, -86.829, 22.121, -85.199, -42.478

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

def transform_to_robot_config(positions_camera, rot_mat, trans_mat):
    return np.dot(rot_mat, positions_camera) + trans_mat

rot_mat, trans_mat = compute_transformation(positions_camera, positions_robot)

# new_camera_position = np.array([17.02, 85.13, 374.00])
new_camera_position = np.array([-63.991330564022064, 61.27646937966347, 477.00002789497375])

new_position_robot = transform_to_robot_config(new_camera_position, rot_mat, trans_mat)

print(f"Rotation Matrix R: {rot_mat}")
print(f"Translation Matrix T: {trans_mat}")
print(f"New coordinates of Robot:\n {new_position_robot}")


