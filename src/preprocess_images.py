from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    RAW_DIR,
    REPORT_DIR,
    DATA_PROFILE_PATH,
    BAD_IMAGES_PATH,
    CLASS_DISTRIBUTION_PATH,
    MIN_IMAGES_PER_CLASS,
)

from data_utils import (
    validate_image_folder,
    remove_or_log_bad_images,
    save_data_profile,
    write_dataset_readme,
)


def plot_class_distribution(
    count_df: pd.DataFrame,
    out_path: Path = CLASS_DISTRIBUTION_PATH
):
    """
    Vẽ biểu đồ số lượng ảnh từng lớp.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.bar(count_df["class_name"], count_df["num_images"])
    plt.xlabel("Lớp trái cây")
    plt.ylabel("Số lượng ảnh")
    plt.title("Phân bố số lượng ảnh theo từng lớp")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"[OK] Đã lưu biểu đồ phân bố lớp: {out_path}")


def run_data_preprocessing():
    """
    Chạy toàn bộ bước kiểm tra dữ liệu raw:
    1. Kiểm tra cấu trúc folder.
    2. Đếm ảnh từng lớp.
    3. Kiểm tra ảnh lỗi.
    4. Lưu data_profile.csv.
    5. Vẽ class_distribution.png.
    6. Tạo dataset_readme.md.
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("========== KIỂM TRA DATASET RAW ==========")

    count_df = validate_image_folder(
        RAW_DIR,
        min_images_per_class=MIN_IMAGES_PER_CLASS
    )

    print("\n[THỐNG KÊ SỐ ẢNH TRONG data/raw]")
    print(count_df)

    bad_df = remove_or_log_bad_images(
        RAW_DIR,
        report_path=BAD_IMAGES_PATH,
        delete_bad=False
    )

    save_data_profile(count_df, DATA_PROFILE_PATH)

    plot_class_distribution(count_df, CLASS_DISTRIBUTION_PATH)

    write_dataset_readme(count_df, bad_df)

    print("\n========== HOÀN TẤT KIỂM TRA RAW ==========")
    print(f"Data profile: {DATA_PROFILE_PATH}")
    print(f"Bad images: {BAD_IMAGES_PATH}")
    print(f"Class distribution: {CLASS_DISTRIBUTION_PATH}")


def main_test_preprocess_images():
    print(f"RAW_DIR={RAW_DIR}")
    print(f"REPORT_DIR={REPORT_DIR}")
    print("[OK] main_test_preprocess_images import thành công.")


if __name__ == "__main__":
    run_data_preprocessing()