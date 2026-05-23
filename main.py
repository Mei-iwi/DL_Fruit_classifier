from pathlib import Path
import sys
import traceback


# Lấy đường dẫn tại vị trí file trên máy
PROJECT_ROOT = Path(__file__).resolve().parent

# Đi đến file chứa mã nguồn
SRC_DIR = PROJECT_ROOT / "src"

# Kiểm tra đường dẫn có trong danh sách đã quét chưa nếu chưa chèn vào đầu tiên

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


'''
    Kiểm tra thư mục và file trong project (đúng theo cấu trúc)
'''

def check_required_artifacts() -> dict:
    required_path = {
        "src" : PROJECT_ROOT / "src",
        "data" : PROJECT_ROOT / "data",
        "data_raw" : PROJECT_ROOT / "data" / "raw",
        "data_processed" : PROJECT_ROOT / "data" / "processed",
        "data_test_images": PROJECT_ROOT / "data" / "test_images",
        "models" : PROJECT_ROOT / "models",
        "reports": PROJECT_ROOT / "reports",
        "requirements": PROJECT_ROOT / "requirements.txt",
        "readme": PROJECT_ROOT / "README.md"
    }

    result = {}

    for name, path in required_path.items():
        result[name] = path.exists()
    
    return result


'''
    In kết quả trạng thái project hiện tại
'''

def print_artifact_status(status: dict) -> None:

    print("\n Trạng thái project hiện tại")

    for name, exists in status.items():

        icon = "[OK]" if exists else "[Error]"

        print(f"{icon} {name}")

    missing = [name for name, exists in status.items() if not exists]

    if missing:

        print("\n Cấu trúc file còn thiếu")
        for item in missing:
            print(f" - {item}")
    else:
        print("\n[Ok] Cấu trúc project đầy đủ")


'''
    Kiến trúc chạy pipeline dữ liệu ảnh
'''


def run_data_pipeline() -> None:

    print("\n ---- Thực thi pipeline đánh giá mô hình")
    print("[INFO] Phần xử lý dữ liệu do thành viên phụ trách dữ liệu hoàn thiện.")
    print("[INFO] Hiện train_cnn.py đang đọc trực tiếp từ data/raw.")
    

'''
    Huấn luyện mô hình
'''

def run_train_pipeline() -> None:

    print("\n Huấn luyện mô hình")

    try:
        from src.train_cnn import train

        train()

    except Exception:
        print("[Error] Không chạy được train_cnn")
        traceback.print_exc()


'''
    Đánh giá mô hình CNN
''' 

def run_evalute_pipeline() -> None:

    print("\n Đánh giá mô hình")

    try:
        from src.evaluate_cnn import main_test_fruit_predict_evaluate

        main_test_fruit_predict_evaluate()

    except Exception:
        print("[Error] Không chạy được elvalute_cnn")
        traceback.print_exc()

    try:
        from src.visualize_results import main_test_visualize_results

        main_test_visualize_results()

    except Exception:
        print("[Error] Không chạy được visualize_result")
        traceback.print_exc()

    


'''
    Dự đoán ảnh lẻ
'''

def run_predict_pipeline(image_path: str | None = None) -> None:

    print("\n Dự đoán kết quả")

    if image_path:

        print(f"Ảnh đầu vào {image_path}")

    else:
        
        print("Chưa truyền đường dẫn ảnh. Chạy smoke test predict_image")
    
    try:
        from src.predict_image import main_test_predict_image

        main_test_predict_image()

    except Exception:
        print("[Error] không chạy được predict_image")
        traceback.print_exc()

'''
    Kiểm thử nhanh project
'''

def run_self_test() -> None:

    print("\n SELF TEST")

    status = check_required_artifacts()

    print_artifact_status(status=status)

    run_data_pipeline()
    run_train_pipeline()
    run_evalute_pipeline()
    run_predict_pipeline()

    print("\n[OK] SELF TEST hoàn tất")


'''
    Chạy toàn bộ cấu trúc
'''

def run_all_pipeline() -> None:

    print("\nThực thi toàn bộ")

    status = check_required_artifacts()

    print_artifact_status(status=status)

    run_data_pipeline()
    run_train_pipeline()
    run_evalute_pipeline()

    print("\n[Ok] Hoàn tất")


'''
    Test tích hợp
'''

def main_test_fruit_integration() -> None:
    run_self_test()


if __name__ == "__main__":
    main_test_fruit_integration()