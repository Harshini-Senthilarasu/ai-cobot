import cv2
import numpy as np
import torch
import pyrealsense2 as rs
import os
from dotenv import load_dotenv
from ultralytics import YOLO  # Ultralytics YOLO model

# Load environment variables
load_dotenv(dotenv_path="/home/harshini/capstone/src/.env")
yolo_path = os.getenv("yolo_path")
data_files_path = os.getenv("image_output_path")

# Load YOLOv11 model (PyTorch-based)
model = YOLO(yolo_path)

# Initialize RealSense pipeline
pipe = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

try:
    pipe.start(config)
except Exception as e:
    print(f"Error starting RealSense pipeline: {e}")

# YOLO object detection
def detect_objects(image):
    results = model(image)  # Run YOLOv11 on image
    detected_objects = []

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes (x1, y1, x2, y2)
        confs = result.boxes.conf.cpu().numpy()  # Confidence scores
        class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Class IDs
        names = result.names  # Class labels dictionary

        for i in range(len(boxes)):
            if confs[i] > 0.5:  # Confidence threshold
                x1, y1, x2, y2 = boxes[i]
                centroid_x = int((x1 + x2) / 2)
                centroid_y = int((y1 + y2) / 2)

                detected_objects.append({
                    "name": names[class_ids[i]],  # Get class name from ID
                    "confidence": float(confs[i]),
                    "top_left": (int(x1), int(y1)),
                    "bottom_right": (int(x2), int(y2)),
                    "centroid": (centroid_x, centroid_y)
                })
    return detected_objects

# Draw bounding boxes and display depth at centroid
def draw_box(image, depth_frame, detected_objects):
    for obj in detected_objects:
        top_left = obj["top_left"]
        bottom_right = obj["bottom_right"]
        centroid = obj["centroid"]

        # Get depth value at centroid
        depth_value = depth_frame.get_distance(centroid[0], centroid[1])

        # Compute real-world coordinates
        depth_point = rs.rs2_deproject_pixel_to_point(
            depth_frame.profile.as_video_stream_profile().intrinsics, [centroid[0], centroid[1]], depth_value)
        x_mm, y_mm, z_mm = [coord * 1000 for coord in depth_point]

        # Draw bounding box and label
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
        cv2.circle(image, centroid, 5, (0, 0, 255), -1)
        label = f"{obj['name']} ({x_mm:.2f}, {y_mm:.2f}, {z_mm:.2f})"
        print(label)
        cv2.putText(image, label, (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the image
    cv2.imshow("Live Object Detection", image)

def main():
    try:
        while True:
            frames = pipe.wait_for_frames()
            aligned_frames = rs.align(rs.stream.color).process(frames)
            depth_frame = aligned_frames.get_depth_frame().as_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            # Detect objects using YOLO
            detected_objects = detect_objects(color_image)

            # Draw bounding boxes
            draw_box(color_image, depth_frame, detected_objects)

            # # Save image if needed
            # cv2.imwrite(os.path.join(data_files_path, "detected_image.jpg"), color_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        pipe.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
