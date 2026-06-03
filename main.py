from pathlib import Path
import sys
import traceback
import argparse


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
    Kiến trúc chạy pipeline dữ liệu ảnh -> kiểm tra dữ liệu ảnh
'''


def run_data_pipeline() -> None:

    print("\n ---- Thực thi pipeline data")
    
    try:
        from src.preprocess_images import run_data_preprocessing

        run_data_preprocessing()

        print("\n[Ok] Pipeline dữ liệu hoàn tất")
    
    except Exception:
        print("[Error] Không chạy được pipeline dữ liệu ảnh")
        traceback.print_exc()
    

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

def run_evaluate_pipeline() -> None:

    print("\n Đánh giá mô hình")

    try:
        from src.evaluate_cnn import evaluate_model

        evaluate_model()

    except Exception:
        print("[Error] Không chạy được elvalute_cnn")
        traceback.print_exc()

    try:
        from src.visualize_results import visualize_samples

        visualize_samples()

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
        from src.predict_image import predict_single_image

        predict_single_image(image_path=image_path)

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

    try:
        from src.preprocess_images import main_test_preprocess_images
        from src.predict_image import predict_single_image
        from src.evaluate_cnn import evaluate_model
        from src.visualize_results import visualize_samples
        from src.train_cnn import train

        main_test_preprocess_images()

        print("[OK] Import predict_single_image thành công")
        print("[OK] Import evaluate_model thành công")
        print("[OK] Import visualize_samples thành công")
        print("[OK] Import  train thành công")

    except Exception:
        print("[Error] Selt-test import module bị lỗi")
        traceback.print_exc()


'''
    Chạy toàn bộ cấu trúc
'''

def run_all_pipeline() -> None:

    print("\nThực thi toàn bộ")

    status = check_required_artifacts()

    print_artifact_status(status=status)

    run_data_pipeline()
    run_train_pipeline()
    run_evaluate_pipeline()

    print("\n[Ok] Hoàn tất")


'''
    Test tích hợp
'''

def main_test_fruit_integration() -> None:
    run_self_test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chương trình pipeline phân loại trái cây")
    # Định nghĩa các lệnh con (sub-commands) cho phép người dùng nhập
    parser.add_argument(
        "command", 
        choices=["test", "data", "train", "evaluate", "predict", "all"], 
        help="Lệnh cần thực thi (VD: predict, train, test...)"
    )
    # Định nghĩa tham số truyền đường dẫn ảnh
    parser.add_argument(
        "--image_path", 
        type=str, 
        help="Đường dẫn ảnh cần dự đoán (chỉ dùng cho lệnh predict)",
        default=None
    )
    args = parser.parse_args()
    # Điều hướng chạy các hàm tương ứng với lệnh
    if args.command == "test":
        main_test_fruit_integration()
        
    elif args.command == "data":
        run_data_pipeline()
        
    elif args.command == "train":
        run_train_pipeline()
        
    elif args.command == "evaluate":
        run_evaluate_pipeline()
        
    elif args.command == "predict":
        if args.image_path is None:
            print("[Error] Bạn cần cung cấp đường dẫn ảnh. Ví dụ: python main.py predict --image_path data/test_images/anh.jpg")
        else:
            run_predict_pipeline(image_path=args.image_path)
            
    elif args.command == "all":
        run_all_pipeline()