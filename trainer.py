import os
import cv2
import numpy as np
from PIL import Image

# Initialize Face Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
path = "dataset"

# Function to get images and IDs from dataset
def get_image_with_id(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg') or f.endswith('.png')]
    faces = []
    ids = []

    for single_image_path in image_paths:
        face_img = Image.open(single_image_path).convert('L')  # Convert to grayscale
        face_np = np.array(face_img, np.uint8)
        user_id = int(os.path.split(single_image_path)[-1].split(".")[1])  # Extract ID from filename
        print(f"Training on ID: {user_id}")

        faces.append(face_np)
        ids.append(user_id)

        cv2.imshow("Training", face_np)
        cv2.waitKey(10)

    return np.array(ids), faces  # Convert IDs to NumPy array

# Get data and train the recognizer
ids, faces = get_image_with_id(path)
recognizer.train(faces, np.array(ids))  # FIXED: Convert `ids` to NumPy array
recognizer.save("recognizer/trainingdata.yml")

cv2.destroyAllWindows()
print("Training Complete. Model saved.")
