import sys
from pathlib import Path
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# Import từ config
from src.config import MODEL_PATH, TEST_DIR, REPORT_DIR, IMG_SIZE, BATCH_SIZE

# Tự tạo thư mục
EVAL_REPORT_DIR = REPORT_DIR / "03_evaluation"
EVAL_REPORT_DIR.mkdir(parents=True, exist_ok=True)

def visualize_samples():
    print("Đang load mô hình để vẽ hình sample...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Load 1 batch dữ liệu test ngẫu nhiên
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    class_names = test_dataset.class_names

    # Lấy 1 batch
    images, labels = next(iter(test_dataset))
    
    # Dự đoán batch
    predictions = model.predict(images)
    pred_labels = np.argmax(predictions, axis=1)

    # Lấy 9 ảnh đầu tiên để vẽ lưới 3x3
    plt.figure(figsize=(12, 12))
    for i in range(9):
        if i >= len(images): break
        ax = plt.subplot(3, 3, i + 1)
        # Rescale lại ảnh nếu model cần ảnh chuẩn hóa, nếu ảnh đen xì thì bỏ .astype("uint8")
        plt.imshow(images[i].numpy().astype("uint8")) 
        
        true_class = class_names[labels[i]]
        pred_class = class_names[pred_labels[i]]
        confidence = np.max(predictions[i]) * 100
        
        # Đúng chữ xanh, sai chữ đỏ
        color = "green" if true_class == pred_class else "red"
        plt.title(f"Thực: {true_class}\nĐoán: {pred_class} ({confidence:.1f}%)", color=color)
        plt.axis("off")
        
    plt.tight_layout()
    plt.savefig(EVAL_REPORT_DIR / "sample_predictions.png")
    plt.close()
    print("Đã lưu sample_predictions.png")

if __name__ == "__main__":
    visualize_samples()