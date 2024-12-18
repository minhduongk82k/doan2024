import numpy as np
import os
from tensorflow.keras.utils import to_categorical
import cv2

def preprocess_input(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (48, 48))
    image = image / 255.0
    image = image.reshape(48, 48, 1)
    return image

def load_frames(folder_path, labels_dict):
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            image = cv2.imread(os.path.join(folder_path, filename))
            image = preprocess_input(image)
            images.append(image)
            label = labels_dict.get(filename)
            labels.append(label)
    return np.array(images), np.array(labels)
