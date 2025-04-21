import numpy as np

# Positions of robot end-effector (where object is placed) relative to robot base frame 
# positions_robot = np.array([
#     [48.662, -610.541, 80.809],
#     [-58.761, -550.130, 80.532],
#     [-186.497, -525.973, 80.697],
# ])

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

# Cartesian Coordinates (X, Y, Z, Rx, Ry, Rz)
# pt1: 48.662, -610.541, 80.809, 176.132, 1.926, 6.748
# pt2: -58.761, -550.130, 80.532, 179.035, 3.382, -4.098
# pt3: -186.497, -525.973, 80.697, 175.242, -2.213, 8.072

# Joint Angle (J1, J2, J3, J4, J5, J6)
# pt1: 83.443, -32.268, -73.883, 20.380, -91.063, -13.413
# pt2: 72.590, -24.049, -85.534, 21.323, -93.133, -13.375
# pt3: 55.404, -23.285, -86.829, 22.121, -85.199, -42.478

# Positions of object relative to camera (X, Y, Z) in millimeters
# positions_camera = np.array([
#     [-156.54, 87.52, 428.00],
#     [-55.13, 83.31, 366.00],
#     [71.44, 77.82, 337.00],
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


def compute_transformation():
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

def transform_point(T, point_camera):
    point_camera_homogeneous = np.append(point_camera, 1)  # Convert to homogeneous coordinates
    point_robot = np.dot(T, point_camera_homogeneous)  # Apply transformation
    return point_robot[:3]  # Extract (x, y, z)

T_camera_to_robot = compute_transformation()

# new_camera_position = np.array([17.02, 85.13, 374.00])
new_camera_position = np.array([-0.370, -0.057, 524.000])

new_position_robot = transform_point(T_camera_to_robot, new_camera_position)

print(f"New coordinates of Robot:\n {new_position_robot}")


