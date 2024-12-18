import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

def predict_images(model_path, test_images_path):
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    print("Model loaded successfully!")

    for file_name in os.listdir(test_images_path):
        if file_name.endswith(".jpg") or file_name.endswith(".png"):
            file_path = os.path.join(test_images_path, file_name)
            try:
                image = cv2.imread(file_path)
                image_resized = cv2.resize(image, (48, 48)) 
                image_normalized = image_resized / 255.0
                image_batch = np.expand_dims(image_normalized, axis=0)

                prediction = model.predict(image_batch)
                label = "Hào hứng" if np.argmax(prediction) == 0 else "Không hào hứng"
                print(f"Image: {file_name}, Predicted Label: {label}")
            except Exception as e:
                print(f"Error processing file {file_name}: {e}")

if __name__ == "__main__":
    model_path = "../saved_model/emotion_recognition_model.h5"
    test_images_path = "../data/data_test"  

    predict_images(model_path, test_images_path)
