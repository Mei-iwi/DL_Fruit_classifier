import sys
from pathlib import Path

import pandas as pd
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Cho phép chạy trực tiếp: python src/train_cnn.py
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.config import (
    RAW_DATA_DIR,
    IMG_SIZE,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    SEED,
    VALIDATION_SPLIT,
    MODEL_DIR,
    MODEL_PATH,
    TRAINING_REPORT_DIR,
    HISTORY_PATH,
    MODEL_SUMMARY_PATH
)

from src.model_cnn import build_cnn_model


def create_folders():
    """
    Tạo thư mục models và reports nếu chưa có.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset():
    """
    Đọc dữ liệu trực tiếp từ data/raw và tự chia train/validation.

    Cấu trúc yêu cầu:
    data/raw/apple/
    data/raw/banana/
    data/raw/orange/
    """

    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu: {RAW_DATA_DIR}")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        RAW_DATA_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=True
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        RAW_DATA_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False
    )

    class_names = train_ds.class_names

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names


def save_model_summary(model):
    """
    Lưu kiến trúc model ra file model_summary.txt
    """
    with open(MODEL_SUMMARY_PATH, "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda line: f.write(line + "\n"))


def save_history(history):
    """
    Lưu lịch sử train ra history.csv
    """
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(HISTORY_PATH, index=False, encoding="utf-8-sig")


def train():
    create_folders()

    train_ds, val_ds, class_names = load_dataset()

    input_shape = IMG_SIZE + (3,)
    num_classes = len(class_names)

    print("Danh sách lớp:", class_names)
    print("Số lớp:", num_classes)

    model = build_cnn_model(
        input_shape=input_shape,
        num_classes=num_classes,
        learning_rate=LEARNING_RATE
    )

    save_model_summary(model)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    model.save(MODEL_PATH)

    save_history(history)


    # Chuyển history của Keras thành DataFrame
    history_df = pd.DataFrame(history.history)

    # Tạo thư mục lưu biểu đồ
    figure_dir = Path("outputs/figures")
    figure_dir.mkdir(parents=True, exist_ok=True)

    # 1. Vẽ biểu đồ accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(history_df["accuracy"], label="Train accuracy")
    plt.plot(history_df["val_accuracy"], label="Validation accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figure_dir / "training_accuracy.png", dpi=300)
    plt.close()

    # 2. Vẽ biểu đồ loss
    plt.figure(figsize=(8, 5))
    plt.plot(history_df["loss"], label="Train loss")
    plt.plot(history_df["val_loss"], label="Validation loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figure_dir / "training_loss.png", dpi=300)
    plt.close()

    print(f"[OK] Đã lưu biểu đồ tại: {figure_dir}")

    print("\nTrain xong.")
    print(f"Model đã lưu tại: {MODEL_PATH}")
    print(f"History đã lưu tại: {HISTORY_PATH}")
    print(f"Model summary đã lưu tại: {MODEL_SUMMARY_PATH}")

if __name__ == "__main__":
    train()