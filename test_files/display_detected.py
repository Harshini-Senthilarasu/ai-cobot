import cv2
import numpy as np
import pyrealsense2 as rs
from google.cloud import vision
from google.oauth2 import service_account

# Setup Google Vision API
load_dotenv(dotenv_path="/home/harshini/capstone/src/.env")
credentials = os.getenv("google_vision_cred")
vision_client = vision.ImageAnnotatorClient(credentials=credentials)

# Initialize RealSense pipeline
pipe = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

try:
    pipe.start(config)
except Exception as e:
    print(f"Error starting RealSense pipeline: {e}")

# Detect objects using Google Vision API
def detect_objects(image):
    height, width = image.shape[:2]
    _, encoded_image = cv2.imencode('.jpg', image)
    content = encoded_image.tobytes()

    image = vision.Image(content=content)
    response = vision_client.annotate_image({
        "image": image,
        "features": [{"type": vision.Feature.Type.OBJECT_LOCALIZATION}]
    })

    objects_info = {}
    for obj in response.localized_object_annotations:
        box = obj.bounding_poly.normalized_vertices
        points = [(int(vertex.x * width), int(vertex.y * height)) for vertex in box]

        if len(points) < 4:
            continue  # Skip incomplete bounding boxes

        top_left, top_right, bottom_right, bottom_left = points[:4]

        # Compute centroid (avg of 4 corner points)
        centroid_x = (top_left[0] + top_right[0] + bottom_right[0] + bottom_left[0]) / 4
        centroid_y = (top_left[1] + top_right[1] + bottom_right[1] + bottom_left[1]) / 4
        centroid = (int(centroid_x), int(centroid_y))

        objects_info[obj.name] = {
            "confidence": obj.score,
            "top_left": top_left,
            "top_right": top_right,
            "bottom_left": bottom_left,
            "bottom_right": bottom_right,
            "centroid": centroid,
        }

    return objects_info

# Draw bounding boxes and display depth at centroid
def draw_box(image, depth_frame, detected_objects):
    for obj_name, obj_info in detected_objects.items():
        # Extract bounding box coordinates
        top_left = tuple(obj_info["top_left"])
        bottom_right = tuple(obj_info["bottom_right"])
        centroid = tuple(obj_info["centroid"])

        # Get depth value at centroid
        depth_value = depth_frame.get_distance(centroid[0], centroid[1])

        # Draw bounding box
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
        # Draw centroid
        cv2.circle(image, centroid, 5, (0, 0, 255), -1)
        # Calculate real-world coordinates using intrinsic parameters
        depth_point = rs.rs2_deproject_pixel_to_point(
            depth_frame.profile.as_video_stream_profile().intrinsics, [centroid[0], centroid[1]], depth_value)
        x_mm = depth_point[0] * 1000
        y_mm = depth_point[1] * 1000
        z_mm = depth_point[2] * 1000
        # Label the object with confidence and depth
        # label = f"{obj_name} ({depth_point[0]:.2f}, {depth_point[1]:.2f} {depth_point[2]:.2f})"
        label = f"{obj_name} ({x_mm:.2f}, {y_mm:.2f} {z_mm:.2f})"
        # label = f"{obj_name}"
        print(label)
        cv2.putText(image, label, (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the image
    cv2.imshow("Live Object Detection", image)

def main():
    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipe.wait_for_frames()
            aligned_frames = rs.align(rs.stream.color).process(frames)
            depth_frame = aligned_frames.get_depth_frame().as_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())

            # Detect objects in the image
            detected_objects = detect_objects(color_image)

            # Draw bounding boxes and display the image
            draw_box(color_image, depth_frame, detected_objects)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        pipe.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()