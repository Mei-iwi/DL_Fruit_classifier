from pathlib import Path
import shutil
import pandas as pd
import cv2
import tensorflow as tf

from config import (
    RAW_DIR,
    REPORT_DIR,
    BAD_IMAGES_PATH,
    DATASET_README_PATH,
    IMG_SIZE,
    BATCH_SIZE,
    SEED,
    VAL_RATIO,
    SUPPORTED_EXTENSIONS,
    MIN_IMAGES_PER_CLASS,
)


def get_class_names(data_dir: Path):
    """
    Lấy danh sách tên lớp từ các thư mục con.
    Ví dụ: data/raw/Banana, data/raw/Mango...
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu: {data_dir}")

    class_names = [p.name for p in data_dir.iterdir() if p.is_dir()]
    class_names.sort()

    if len(class_names) == 0:
        raise ValueError(f"Không có thư mục lớp nào trong: {data_dir}")

    return class_names


def list_image_files(class_dir: Path):
    """
    Lấy danh sách file ảnh hợp lệ trong một thư mục lớp.
    """
    class_dir = Path(class_dir)
    image_files = []

    if not class_dir.exists():
        return image_files

    for file_path in class_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            image_files.append(file_path)

    image_files.sort()
    return image_files


def count_images_by_class(data_dir: Path):
    """
    Đếm số ảnh từng lớp.
    """
    data_dir = Path(data_dir)
    rows = []

    class_names = get_class_names(data_dir)

    for class_name in class_names:
        class_dir = data_dir / class_name
        image_files = list_image_files(class_dir)

        rows.append({
            "class_name": class_name,
            "num_images": len(image_files)
        })

    count_df = pd.DataFrame(rows)
    return count_df


def validate_image_folder(data_dir: Path, min_images_per_class: int = MIN_IMAGES_PER_CLASS):
    """
    Kiểm tra dữ liệu:
    - Có ít nhất 2 lớp.
    - Mỗi lớp có số ảnh tối thiểu.
    """
    count_df = count_images_by_class(data_dir)

    if len(count_df) < 2:
        raise ValueError("Dataset cần ít nhất 2 lớp để phân loại.")

    too_few = count_df[count_df["num_images"] < min_images_per_class]

    if not too_few.empty:
        print("[CẢNH BÁO] Một số lớp có quá ít ảnh:")
        print(too_few)

    return count_df


def remove_or_log_bad_images(
    data_dir: Path,
    report_path: Path = BAD_IMAGES_PATH,
    delete_bad: bool = False
):
    """
    Kiểm tra ảnh lỗi bằng cv2.imread.
    Mặc định chỉ ghi log, không xóa ảnh.
    """
    data_dir = Path(data_dir)
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    bad_rows = []
    class_names = get_class_names(data_dir)

    for class_name in class_names:
        class_dir = data_dir / class_name
        image_files = list_image_files(class_dir)

        for image_path in image_files:
            img = cv2.imread(str(image_path))

            if img is None:
                bad_rows.append({
                    "class_name": class_name,
                    "image_path": str(image_path),
                    "reason": "cv2.imread returned None"
                })

                if delete_bad:
                    image_path.unlink(missing_ok=True)

    bad_df = pd.DataFrame(bad_rows)

    if bad_df.empty:
        bad_df = pd.DataFrame(columns=["class_name", "image_path", "reason"])

    bad_df.to_csv(report_path, index=False, encoding="utf-8-sig")

    print(f"[OK] Đã kiểm tra ảnh lỗi. Số ảnh lỗi: {len(bad_df)}")
    print(f"[OK] File log ảnh lỗi: {report_path}")

    return bad_df


def make_tf_image_datasets(
    data_dir: Path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    val_split=VAL_RATIO,
    seed=SEED
):
    """
    Tạo train_ds và val_ds trực tiếp từ một thư mục ảnh.
    Hàm này dùng cho bản rút gọn hoặc test nhanh.
    Với bản chuẩn sau split, nên dùng make_tf_dataset_from_dir().
    """
    data_dir = Path(data_dir)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="int"
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="int"
    )

    return train_ds, val_ds


def make_tf_dataset_from_dir(
    directory: Path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
):
    """
    Tạo TensorFlow Dataset từ thư mục đã chia sẵn:
    data/processed/train
    data/processed/val
    data/processed/test
    """
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục dataset: {directory}")

    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="int",
        shuffle=shuffle
    )

    return dataset


def normalize_and_prefetch(dataset):
    """
    Chuẩn hóa pixel 0-255 về 0-1 và dùng prefetch.
    Lưu ý: nếu model đã có Rescaling(1./255) thì không gọi hàm này khi train.
    """
    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

    dataset = dataset.map(
        lambda images, labels: (normalization_layer(images), labels)
    )

    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


def save_data_profile(count_df: pd.DataFrame, out_path: Path):
    """
    Lưu thống kê số ảnh từng lớp.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"[OK] Đã lưu thống kê dữ liệu: {out_path}")


def write_dataset_readme(count_df: pd.DataFrame, bad_df: pd.DataFrame):
    """
    Tạo file mô tả dataset cho báo cáo.
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    total_images = int(count_df["num_images"].sum())
    num_classes = len(count_df)
    total_bad = len(bad_df)

    content = f"""# Mô tả dữ liệu ảnh trái cây

## 1. Cấu trúc dữ liệu

Dữ liệu được đặt trong thư mục:

data/raw/<ten_lop>/*.jpg

Mỗi thư mục con tương ứng với một lớp trái cây.

## 2. Số lớp và số ảnh

- Số lớp: {num_classes}
- Tổng số ảnh trong data/raw: {total_images}
- Số ảnh lỗi phát hiện: {total_bad}

## 3. Bảng thống kê từng lớp

{count_df.to_string(index=False)}

## 4. Kiểm tra ảnh lỗi

Ảnh lỗi được kiểm tra bằng OpenCV cv2.imread.
Nếu ảnh không đọc được, ảnh sẽ được ghi vào bad_images.csv.
Chương trình mặc định không tự động xóa ảnh lỗi.

## 5. Tiền xử lý

- Ảnh được tổ chức theo thư mục lớp.
- Ảnh được resize về kích thước: {IMG_SIZE}
- Dữ liệu Training gốc được chia thành train và validation.
- Dữ liệu Test gốc được giữ riêng làm test cuối.
- TensorFlow Dataset được tạo bằng image_dataset_from_directory.

## 6. Ghi chú

Không dùng ảnh test để huấn luyện.
Không ghi số liệu accuracy/loss nếu chưa chạy thật.
"""

    DATASET_README_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(DATASET_README_PATH, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"[OK] Đã tạo dataset_readme.md: {DATASET_README_PATH}")


def main_test_fruit_data_pipeline():
    """
    Smoke test:
    Tự tạo ảnh giả trong tests/sample_fruit để kiểm tra các hàm chính.
    Không dùng dữ liệu thật.
    """
    import numpy as np

    project_root = Path(__file__).resolve().parents[1]
    sample_dir = project_root / "tests" / "sample_fruit"

    if sample_dir.exists():
        shutil.rmtree(sample_dir)

    class_a = sample_dir / "Banana"
    class_b = sample_dir / "Mango"

    class_a.mkdir(parents=True, exist_ok=True)
    class_b.mkdir(parents=True, exist_ok=True)

    for i in range(10):
        img = np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(class_a / f"banana_{i}.jpg"), img)

    for i in range(10):
        img = np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(class_b / f"mango_{i}.jpg"), img)

    class_names = get_class_names(sample_dir)
    print("Class names:", class_names)
    assert len(class_names) >= 2

    count_df = count_images_by_class(sample_dir)
    print("\nSố ảnh trong dữ liệu test giả:")
    print(count_df)

    total_images = int(count_df["num_images"].sum())
    assert total_images > 0

    bad_df = remove_or_log_bad_images(sample_dir, sample_dir / "bad_images.csv")
    assert len(bad_df) == 0

    train_ds, val_ds = make_tf_image_datasets(
        sample_dir,
        image_size=(128, 128),
        batch_size=4,
        val_split=0.2,
        seed=42
    )

    train_ds = normalize_and_prefetch(train_ds)
    val_ds = normalize_and_prefetch(val_ds)

    for images, labels in train_ds.take(1):
        print("Batch image shape:", images.shape)
        print("Batch label shape:", labels.shape)
        print("Min pixel:", images.numpy().min())
        print("Max pixel:", images.numpy().max())
        assert images.shape[1:] == (128, 128, 3)

    print("[OK] main_test_fruit_data_pipeline chạy thành công.")


if __name__ == "__main__":
    main_test_fruit_data_pipeline()