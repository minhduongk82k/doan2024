from flask import Flask, request, jsonify
import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Đường dẫn đến mô hình và dữ liệu
MODEL_PATH = "saved_model/emotion_recognition_model.h5"
DATA_PATH = "../data/labeled_data.csv"

# Load mô hình
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
model = load_model(MODEL_PATH)

# Hàm xử lý ảnh
def preprocess_image(image):
    image_resized = image.resize((48, 48))
    img_array = np.array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/emotion-statistics', methods=['GET'])
def get_emotion_statistics():
    try:
        # Đọc file CSV kết quả
        if not os.path.exists(DATA_PATH):
            return jsonify({"error": "Data file not found"}), 404
        
        data = pd.read_csv(DATA_PATH)
        total = len(data)
        if total == 0:
            return jsonify({"message": "No data available"}), 200
        
        # Tính toán tỉ lệ phần trăm
        stats = data['label'].value_counts(normalize=True) * 100
        response = {
            "Hào hứng": round(stats.get("Hào hứng", 0), 2),
            "Không hào hứng": round(stats.get("Không hào hứng", 0), 2)
        }
        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
