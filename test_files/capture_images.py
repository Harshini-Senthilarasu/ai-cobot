#!/usr/bin/env python3
import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time

# Directory to save images
save_dir = "/home/harshini/capstone/src/data_files/calibration_images"
os.makedirs(save_dir, exist_ok=True)

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Increase resolution for better detection
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Checkerboard settings
board_size = (7, 9)  # Inner corners per row & column
square_size = 25  # mm per square
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

print("Press 's' to save valid images, 'q' to quit.")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        # Convert RealSense frame to OpenCV format
        color_image = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Preprocessing for better detection
        gray = cv2.GaussianBlur(gray, (5, 5), 0)  # Reduce noise
        gray = cv2.equalizeHist(gray)  # Enhance contrast

        # Display the grayscale image for debugging
        cv2.imshow("Grayscale Image", gray)

        # Improved corner detection with additional flags
        ret, corners = cv2.findChessboardCorners(
            gray, board_size, 
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if ret:
            # Refine corners for better accuracy
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(color_image, board_size, corners2, ret)
            cv2.putText(color_image, "Checkerboard Detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(color_image, "No Checkerboard Found", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 0, 255), 2, cv2.LINE_AA)

        # Display image
        cv2.imshow('Checkerboard Detection', color_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and ret:  # Save only if checkerboard is detected
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            color_filename = os.path.join(save_dir, f"color_{timestamp}.png")
            cv2.imwrite(color_filename, color_image)
            print(f"✅ Saved: {color_filename}")

        elif key == ord('q'):  # Quit program
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
