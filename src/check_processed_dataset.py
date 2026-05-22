from config import TRAIN_DIR, VAL_DIR, TEST_DIR, IMG_SIZE, BATCH_SIZE
from data_utils import make_tf_dataset_from_dir


def check_processed_dataset():
    print("========== KIỂM TRA PROCESSED DATASET ==========")

    train_ds = make_tf_dataset_from_dir(
        TRAIN_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_ds = make_tf_dataset_from_dir(
        VAL_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_ds = make_tf_dataset_from_dir(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print("Train classes:", train_ds.class_names)
    print("Val classes:", val_ds.class_names)
    print("Test classes:", test_ds.class_names)

    assert train_ds.class_names == val_ds.class_names == test_ds.class_names

    for images, labels in train_ds.take(1):
        print("Train batch image shape:", images.shape)
        print("Train batch label shape:", labels.shape)

    for images, labels in val_ds.take(1):
        print("Val batch image shape:", images.shape)
        print("Val batch label shape:", labels.shape)

    for images, labels in test_ds.take(1):
        print("Test batch image shape:", images.shape)
        print("Test batch label shape:", labels.shape)

    print("[OK] Processed dataset train/val/test đọc được thành công.")


if __name__ == "__main__":
    check_processed_dataset()