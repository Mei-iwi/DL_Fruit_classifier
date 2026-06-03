import sys
from pathlib import Path
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Thiết lập đường dẫn gốc
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# Import đúng các biến từ file config
from src.config import MODEL_PATH, TEST_DIR, REPORT_DIR, IMG_SIZE, BATCH_SIZE

# Tự động tạo thư mục evaluation ngay trong code
EVAL_REPORT_DIR = REPORT_DIR / "03_evaluation"
EVAL_REPORT_DIR.mkdir(parents=True, exist_ok=True)

def evaluate_model():
    print("Đang load mô hình...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("Đang load tập dữ liệu test...")
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False 
    )

    class_names = test_dataset.class_names
    
    # Lấy nhãn thực tế (y_true) và đường dẫn file
    y_true = np.concatenate([y for x, y in test_dataset], axis=0)
    file_paths = test_dataset.file_paths

    print("Đang tiến hành dự đoán...")
    predictions = model.predict(test_dataset)
    y_pred = np.argmax(predictions, axis=1)

    # 1. Xuất Classification Report
    report = classification_report(y_true, y_pred, target_names=class_names)
    with open(EVAL_REPORT_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report) # type: ignore
    print("Đã lưu classification_report.txt")

    # 2. Vẽ và lưu Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Dự đoán (Predicted)")
    plt.ylabel("Thực tế (Actual)")
    plt.title("Ma trận nhầm lẫn (Confusion Matrix)")
    plt.tight_layout()
    plt.savefig(EVAL_REPORT_DIR / "confusion_matrix.png")
    plt.close()
    print("Đã lưu confusion_matrix.png")

    # 3. Trích xuất các dự đoán sai (Wrong Predictions)
    wrong_indices = np.where(y_pred != y_true)[0]
    wrong_data = []
    for idx in wrong_indices:
        wrong_data.append({
            "File_Path": file_paths[idx],
            "True_Label": class_names[y_true[idx]],
            "Predicted_Label": class_names[y_pred[idx]],
            "Confidence": np.max(predictions[idx])
        })
    
    df_wrong = pd.DataFrame(wrong_data)
    df_wrong.to_csv(EVAL_REPORT_DIR / "wrong_predictions.csv", index=False, encoding="utf-8")
    print(f"Đã lưu wrong_predictions.csv (Có {len(wrong_data)} ảnh đoán sai)")

if __name__ == "__main__":
    evaluate_model()