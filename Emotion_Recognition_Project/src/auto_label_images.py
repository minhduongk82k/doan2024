import os
import cv2
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

def auto_label_images(model_path, frames_path, output_csv):

    print(f"Loading model from {model_path}...")
    model = load_model(model_path)

    results = []

    for file_name in os.listdir(frames_path):
        if file_name.endswith(".jpg") or file_name.endswith(".png"):
            file_path = os.path.join(frames_path, file_name)
            try:
                image = cv2.imread(file_path)
                image_resized = cv2.resize(image, (48, 48)) 
                image_normalized = image_resized / 255.0
                image_batch = np.expand_dims(image_normalized, axis=0)

                # Predict the label
                prediction = model.predict(image_batch)
                label = "Hào hứng" if np.argmax(prediction) == 0 else "Không hào hứng"

                # Append result
                results.append({"filename": file_name, "label": label})
                print(f"Labeled {file_name}: {label}")
            except Exception as e:
                print(f"Error processing file {file_name}: {e}")

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Labeled results saved to {output_csv}")

if __name__ == "__main__":
    model_path = "../saved_model/emotion_recognition_model.h5"
    frames_path = "../data/frames/D01_20240223084534_frames" 
    output_csv = "../data/labeled_data.csv"

    auto_label_images(model_path, frames_path, output_csv)
