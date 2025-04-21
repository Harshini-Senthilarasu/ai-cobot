import cv2
import numpy as np
import glob

# Set image path
image_dir = "/home/harshini/capstone/src/data_files/calibration_images"
image_files = glob.glob(f"{image_dir}/*.png")

print(f"Found {len(image_files)} images in {image_dir}")

# Define checkerboard properties
square_size = 25  # in mm
board_size = (7,9)  # (inner corners: cols, rows)

# Prepare object points
objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * square_size

# Lists to store detected points
objpoints, imgpoints = [], []

for fname in image_files:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Preprocess
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.equalizeHist(gray)

    # Detect checkerboard
    ret, corners = cv2.findChessboardCorners(gray, board_size, None)

    if ret:
        print(f"✅ Checkerboard detected in: {fname}")
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw detected corners
        cv2.drawChessboardCorners(img, board_size, corners, ret)
        cv2.imshow('Checkerboard Detection', img)
        cv2.waitKey(500)
    else:
        print(f"⚠️ Checkerboard NOT detected in: {fname}")

cv2.destroyAllWindows()

# Prevent running calibration on empty lists
if len(imgpoints) == 0:
    print("❌ No valid checkerboard detections. Exiting!")
    exit()

# Camera calibration
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Print results
print("📷 Camera Matrix:\n", camera_matrix)
print("🎯 Distortion Coefficients:\n", dist_coeffs.ravel())

# Compute reprojection error
total_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    total_error += error
mean_error = total_error / len(objpoints)
print(f"📏 Mean Reprojection Error: {mean_error:.2f} pixels")

# Undistort a test image
test_img = cv2.imread(image_files[0])
h, w = test_img.shape[:2]
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
undistorted_img = cv2.undistort(test_img, camera_matrix, dist_coeffs, None, new_camera_matrix)

cv2.imshow('Original Image', test_img)
cv2.imshow('Undistorted Image', undistorted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
