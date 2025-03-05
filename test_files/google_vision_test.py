import os
import cv2
import numpy as np
from google.cloud import vision
from google.oauth2 import service_account

# Set Google Cloud credentials
load_dotenv(dotenv_path="/home/harshini/capstone/src/.env")
credentials_path = os.getenv("google_vision_cred")
credentials = service_account.Credentials.from_service_account_file(os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
client = vision.ImageAnnotatorClient(credentials=credentials)

# Function to detect objects in an image
def detect_objects(image_path):
    with open(image_path, "rb") as image_file:
        content = image_file.read()
    
    image = vision.Image(content=content)
    response = client.object_localization(image=image)
    print(response)
    return response.localized_object_annotations

# Function to draw bounding boxes on the image
def draw_objects(image_path, objects, output_path="output.jpg"):
    # Load the image
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    for obj in objects:
        box = obj.bounding_poly.normalized_vertices
        points = [(int(vertex.x * width), int(vertex.y * height)) for vertex in box]

        # Draw bounding box
        cv2.polylines(image, [np.array(points)], isClosed=True, color=(0, 255, 0), thickness=2)

        # Put label text
        cv2.putText(image, obj.name, (points[0][0], points[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Save the modified image
    cv2.imwrite(output_path, image)
    print(f"Output image saved as {output_path}")

# Main execution
image_path = "butterfly.jpeg"
detected_objects = detect_objects(image_path)
draw_objects(image_path, detected_objects)
