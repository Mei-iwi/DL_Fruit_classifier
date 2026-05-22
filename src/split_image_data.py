from pathlib import Path
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    RAW_DIR,
    SOURCE_TEST_DIR,
    PROCESSED_DIR,
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    DATA_SUMMARY_PATH,
    VAL_RATIO,
    TEST_RATIO,
    SEED,
    SUPPORTED_EXTENSIONS,
    MIN_IMAGES_PER_CLASS,
)


def get_class_names(data_dir: Path):
    data_dir = Path(data_dir)

    if not data_dir.exists():
        return []

    class_names = [p.name for p in data_dir.iterdir() if p.is_dir()]
    class_names.sort()

    return class_names


def list_images(class_dir: Path):
    class_dir = Path(class_dir)

    if not class_dir.exists():
        return []

    images = []

    for file_path in class_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            images.append(file_path)

    images.sort()
    return images


def validate_raw_data():
    """
    Kiểm tra data/raw trước khi split.
    """
    class_names = get_class_names(RAW_DIR)

    if len(class_names) < 2:
        raise ValueError("data/raw cần ít nhất 2 lớp trái cây.")

    for class_name in class_names:
        image_files = list_images(RAW_DIR / class_name)

        if len(image_files) < MIN_IMAGES_PER_CLASS:
            raise ValueError(
                f"Lớp {class_name} có quá ít ảnh: {len(image_files)} ảnh. "
                f"Cần ít nhất {MIN_IMAGES_PER_CLASS} ảnh."
            )

    return class_names


def reset_processed_dir():
    """
    Xóa processed cũ và tạo lại train/val/test.
    """
    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)


def copy_files(files, output_class_dir: Path):
    """
    Copy danh sách ảnh vào thư mục output tương ứng.
    """
    output_class_dir.mkdir(parents=True, exist_ok=True)

    for src in files:
        dst = output_class_dir / src.name
        shutil.copy2(src, dst)


def split_from_raw_only(raw_images):
    """
    Trường hợp không có data/test_original:
    chia data/raw thành train/val/test.
    """
    train_val_files, test_files = train_test_split(
        raw_images,
        test_size=TEST_RATIO,
        random_state=SEED,
        shuffle=True
    )

    adjusted_val_ratio = VAL_RATIO / (1.0 - TEST_RATIO)

    train_files, val_files = train_test_split(
        train_val_files,
        test_size=adjusted_val_ratio,
        random_state=SEED,
        shuffle=True
    )

    return train_files, val_files, test_files


def check_test_original_matches_raw(class_names):
    """
    Cảnh báo nếu test_original thiếu lớp so với raw.
    """
    if not SOURCE_TEST_DIR.exists():
        print("[CẢNH BÁO] Không tìm thấy data/test_original. Sẽ chia test từ data/raw.")
        return False

    source_test_classes = get_class_names(SOURCE_TEST_DIR)

    if len(source_test_classes) == 0:
        print("[CẢNH BÁO] data/test_original không có lớp nào. Sẽ chia test từ data/raw.")
        return False

    missing_classes = []

    for class_name in class_names:
        if class_name not in source_test_classes:
            missing_classes.append(class_name)

    if missing_classes:
        print("[CẢNH BÁO] Một số lớp thiếu trong data/test_original:")
        print(missing_classes)
        print("Các lớp thiếu test_original sẽ bị chia test từ data/raw.")

    return True


def split_image_data():
    """
    Tạo:
    - data/processed/train
    - data/processed/val
    - data/processed/test
    - data/data_summary.csv

    Nếu có data/test_original/<class_name>:
    - data/raw/<class_name> chia thành train/val
    - data/test_original/<class_name> dùng làm test cuối

    Nếu không có data/test_original/<class_name>:
    - data/raw/<class_name> chia thành train/val/test
    """
    print("========== SPLIT DATASET ==========")

    class_names = validate_raw_data()
    has_source_test = check_test_original_matches_raw(class_names)

    reset_processed_dir()

    summary_rows = []

    for class_name in class_names:
        raw_images = list_images(RAW_DIR / class_name)

        use_source_test_for_this_class = (
            has_source_test
            and (SOURCE_TEST_DIR / class_name).exists()
            and len(list_images(SOURCE_TEST_DIR / class_name)) > 0
        )

        if use_source_test_for_this_class:
            train_files, val_files = train_test_split(
                raw_images,
                test_size=VAL_RATIO,
                random_state=SEED,
                shuffle=True
            )

            test_files = list_images(SOURCE_TEST_DIR / class_name)
            test_from_source_dataset = True

        else:
            train_files, val_files, test_files = split_from_raw_only(raw_images)
            test_from_source_dataset = False

        copy_files(train_files, TRAIN_DIR / class_name)
        copy_files(val_files, VAL_DIR / class_name)
        copy_files(test_files, TEST_DIR / class_name)

        summary_rows.append({
            "class_name": class_name,
            "raw_images": len(raw_images),
            "train_images": len(train_files),
            "val_images": len(val_files),
            "test_images": len(test_files),
            "test_from_source_dataset": test_from_source_dataset
        })

        print(
            f"[OK] {class_name}: "
            f"raw={len(raw_images)}, "
            f"train={len(train_files)}, "
            f"val={len(val_files)}, "
            f"test={len(test_files)}, "
            f"test_from_source={test_from_source_dataset}"
        )

    summary_df = pd.DataFrame(summary_rows)

    DATA_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(DATA_SUMMARY_PATH, index=False, encoding="utf-8-sig")

    print("\n========== HOÀN TẤT SPLIT ==========")
    print(summary_df)
    print(f"\nTrain dir: {TRAIN_DIR}")
    print(f"Val dir: {VAL_DIR}")
    print(f"Test dir: {TEST_DIR}")
    print(f"Data summary: {DATA_SUMMARY_PATH}")

    return summary_df


def main_test_split_image_data():
    print("RAW_DIR:", RAW_DIR)
    print("SOURCE_TEST_DIR:", SOURCE_TEST_DIR)
    print("TRAIN_DIR:", TRAIN_DIR)
    print("VAL_DIR:", VAL_DIR)
    print("TEST_DIR:", TEST_DIR)
    print("[OK] main_test_split_image_data import thành công.")


if __name__ == "__main__":
    split_image_data()