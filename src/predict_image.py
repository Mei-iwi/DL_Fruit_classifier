import sys
import argparse
from pathlib import Path
import tensorflow as tf
import numpy as np
import os

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# Import từ config
from src.config import MODEL_PATH, IMG_SIZE, TEST_DIR

def predict_single_image(image_path):
    # Load model
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"Lỗi load model: {e}")
        return

    # Lấy class_names trực tiếp từ các thư mục con trong TEST_DIR
    class_names = sorted(os.listdir(TEST_DIR))

    # Xử lý ảnh đầu vào
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) 

    # Dự đoán
    predictions = model.predict(img_array)
    score = predictions[0]
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    print("-" * 30)
    print(f"Ảnh: {image_path}")
    print(f"=> Dự đoán: {predicted_class} với độ tin cậy {confidence:.2f}%")
    print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dự đoán trái cây từ ảnh.")
    parser.add_argument("--image", type=str, required=True, help="Đường dẫn tới ảnh cần dự đoán")
    args = parser.parse_args()

    predict_single_image(args.image)