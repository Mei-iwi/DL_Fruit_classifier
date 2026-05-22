
import argparse

from main import (
    run_all_pipeline,
    run_data_pipeline,
    run_evalute_pipeline,
    run_predict_pipeline, 
    run_self_test,
    run_train_pipeline
)

'''
    Tạo giao diện dòng lệnh cho project phân loại trái cây
'''

def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description="CLI cho đồ án phân loại trái cây bằng CNN"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="Chọn chức năng chạy"
    )

    subparsers.add_parser(
        "self-test",
        help="Kiểm thử nhanh toàn bộ module hiện có"
    )

    subparsers.add_parser(
        "data",
        help="Chạy pipeline dữ liệu ảnh"
    )

    subparsers.add_parser(
        "train",
        help="Chạy pipeline xây dưng CNN và huấn luyện model"
    )
    
    subparsers.add_parser(
        "evaluate",
        help="Chạy pipeline đánh gái CNN"
    )

    predict_parser = subparsers.add_parser(
        "predict",
        help="Dự đoán một ảnh trái cây"
    )

    predict_parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Đường dẫn ảnh cần dự đoán"
    )

    subparsers.add_parser(
        "all",
        help="Chạy toàn bộ pipeline: data -> train -> evaluate"
    )


    return parser

'''
    Thực thi của CLI
'''

def main_test_fruit_cli():
    
    parsers = build_parser()

    args = parsers.parse_args()

    if args.command == "self-test":
        run_self_test()

    elif args.command == "data":
        run_data_pipeline()
    
    elif args.command == "train":
        run_train_pipeline()
    
    elif args.command == "evaluate":
        run_evalute_pipeline()
    
    elif args.command == "predict":
        run_predict_pipeline(args.image_path)
    
    elif args.command == "all":
        run_all_pipeline()
    
    else:
        parsers.print_help()


if __name__ == "__main__":
    main_test_fruit_cli()
