# README - Đồ án 1: Phân loại trái cây qua ảnh

## 1. Tổng quan source code

Project `fruit_classifier` là source code cho đồ án phân loại trái cây qua ảnh bằng mô hình CNN.  
Source code được tổ chức theo hướng tách riêng dữ liệu, mô hình, báo cáo, module xử lý và kiểm thử để nhóm dễ phân công, dễ quản lý trên GitHub và dễ kiểm tra khi nộp bài.

Project tập trung vào các nhóm chức năng chính:

| Nhóm chức năng | Mô tả |
|---|---|
| Quản lý dữ liệu ảnh | Lưu ảnh gốc, dữ liệu đã xử lý và ảnh dùng thử dự đoán. |
| Tiền xử lý ảnh | Kiểm tra dữ liệu ảnh, thống kê số lượng ảnh từng lớp, chuẩn hóa ảnh đầu vào. |
| Xây dựng mô hình CNN | Định nghĩa kiến trúc CNN phục vụ bài toán phân loại ảnh trái cây. |
| Huấn luyện mô hình | Huấn luyện CNN, lưu lịch sử huấn luyện và lưu model sau khi train. |
| Đánh giá mô hình | Tính metric, tạo confusion matrix, classification report và biểu đồ minh họa. |
| Dự đoán ảnh mới | Cho phép nạp một ảnh trái cây mới và trả về lớp dự đoán. |
| Báo cáo và nghiệm thu | Lưu hình ảnh, bảng kết quả, log kiểm thử và tài liệu phục vụ báo cáo cuối. |

---

## 2. Cấu trúc thư mục tổng quát

| Đường dẫn | Chức năng |
|---|---|
| `fruit_classifier/` | Thư mục gốc của đồ án phân loại trái cây. Chứa toàn bộ source code, dữ liệu, model, báo cáo và file cấu hình project. |
| `fruit_classifier/data/` | Khu vực quản lý dữ liệu ảnh của project. |
| `fruit_classifier/models/` | Khu vực lưu mô hình CNN sau khi huấn luyện. |
| `fruit_classifier/reports/` | Khu vực lưu kết quả trung gian và tài liệu phục vụ báo cáo. |
| `fruit_classifier/src/` | Khu vực chứa source code chính của project. |
| `fruit_classifier/tests/` | Khu vực chứa dữ liệu mẫu nhỏ và file phục vụ kiểm thử nhanh. |
| `fruit_classifier/app_fruit_cli.py` | File tích hợp giao diện dòng lệnh đơn giản cho project. |
| `fruit_classifier/main.py` | File điều phối chính để gọi các chức năng của project. |
| `fruit_classifier/requirements.txt` | File liệt kê thư viện cần thiết cho project. |
| `fruit_classifier/.gitignore` | File quy định những thành phần không đưa lên GitHub. |
| `fruit_classifier/README.md` | File mô tả cấu trúc source code, chức năng thư mục và chức năng file. |

---

## 3. Mô tả thư mục dữ liệu

| Đường dẫn | Chức năng |
|---|---|
| `data/raw/` | Chứa dữ liệu ảnh ban đầu. Mỗi loại trái cây được đặt trong một thư mục con riêng. Tên thư mục con chính là nhãn lớp của ảnh. |
| `data/raw/.gitkeep` | File giữ chỗ để GitHub lưu lại thư mục `raw` khi chưa có dữ liệu thật. |
| `data/raw/README_data.md` | Ghi chú cách đặt dữ liệu ảnh gốc vào project. |
| `data/processed/` | Chứa dữ liệu đã được xử lý hoặc đã được chia thành các tập phục vụ huấn luyện và đánh giá. |
| `data/processed/.gitkeep` | File giữ chỗ để giữ thư mục `processed` trên GitHub. |
| `data/processed/README_processed.md` | Ghi chú vai trò của dữ liệu đã xử lý. |
| `data/test_images/` | Chứa ảnh lẻ dùng để kiểm tra chức năng dự đoán sau khi đã có model. |
| `data/test_images/.gitkeep` | File giữ chỗ để giữ thư mục `test_images` trên GitHub. |
| `data/test_images/README_test_images.md` | Ghi chú cách dùng thư mục ảnh kiểm thử dự đoán. |

---

## 4. Mô tả thư mục model

| Đường dẫn | Chức năng |
|---|---|
| `models/` | Chứa model CNN sau khi huấn luyện và các file liên quan đến model. |
| `models/.gitkeep` | File giữ chỗ để giữ thư mục `models` trên GitHub. |
| `models/README_models.md` | Ghi chú vai trò của thư mục model. |
| `models/fruit_cnn.keras` | File model CNN sau khi huấn luyện. File này chỉ xuất hiện sau khi chạy huấn luyện thật. |

---

## 5. Mô tả thư mục báo cáo

| Đường dẫn | Chức năng |
|---|---|
| `reports/` | Chứa toàn bộ kết quả trung gian và tài liệu phục vụ báo cáo đồ án. |
| `reports/01_data_profile/` | Lưu kết quả phân tích dữ liệu ảnh, số lượng ảnh theo lớp, ảnh lỗi và biểu đồ phân bố lớp. |
| `reports/01_data_profile/.gitkeep` | File giữ chỗ cho thư mục báo cáo dữ liệu. |
| `reports/02_training/` | Lưu kết quả huấn luyện CNN như lịch sử huấn luyện, biểu đồ loss, biểu đồ accuracy và thông tin kiến trúc model. |
| `reports/02_training/.gitkeep` | File giữ chỗ cho thư mục kết quả huấn luyện. |
| `reports/03_evaluation/` | Lưu kết quả đánh giá CNN như confusion matrix, classification report, ảnh dự đoán mẫu và danh sách ảnh dự đoán sai. |
| `reports/03_evaluation/.gitkeep` | File giữ chỗ cho thư mục kết quả đánh giá. |
| `reports/final_report/` | Chứa hình ảnh, bảng kết quả và nội dung tổng hợp để đưa vào báo cáo cuối cùng. |
| `reports/final_report/.gitkeep` | File giữ chỗ cho thư mục báo cáo cuối. |

---

## 6. Mô tả thư mục source code

| File | Chức năng |
|---|---|
| `src/__init__.py` | Đánh dấu thư mục `src` là một package Python, hỗ trợ import module trong project. |
| `src/config.py` | Chứa các biến cấu hình dùng chung như đường dẫn dữ liệu, kích thước ảnh, batch size, số epoch, seed, learning rate và đường dẫn lưu model. |
| `src/data_utils.py` | Chứa các hàm hỗ trợ đọc cấu trúc dữ liệu ảnh, lấy danh sách lớp, đếm số ảnh theo lớp và tạo bảng thống kê dữ liệu. |
| `src/preprocess_images.py` | Chứa các hàm kiểm tra ảnh lỗi, chuẩn hóa kích thước ảnh, chuẩn hóa giá trị pixel và tạo dữ liệu đầu vào cho mô hình CNN. |
| `src/model_cnn.py` | Chứa hàm xây dựng kiến trúc CNN cho bài toán phân loại trái cây. |
| `src/train_cnn.py` | Chứa quy trình huấn luyện CNN, lưu model, lưu lịch sử huấn luyện và tạo kết quả phục vụ báo cáo huấn luyện. |
| `src/evaluate_cnn.py` | Chứa quy trình đánh giá model CNN, tính các chỉ số phân loại và tạo kết quả đánh giá. |
| `src/predict_image.py` | Chứa chức năng nạp model và dự đoán lớp trái cây cho một ảnh đầu vào. |
| `src/visualize_results.py` | Chứa các hàm vẽ biểu đồ phân bố dữ liệu, biểu đồ huấn luyện, confusion matrix và ảnh dự đoán mẫu. |

---

## 7. Mô tả file tích hợp và kiểm thử

| File hoặc thư mục | Chức năng |
|---|---|
| `tests/` | Chứa dữ liệu mẫu nhỏ hoặc file hỗ trợ kiểm thử nhanh các module. |
| `tests/.gitkeep` | File giữ chỗ để giữ thư mục `tests` trên GitHub. |
| `tests/README_tests.md` | Mô tả vai trò của thư mục kiểm thử. |
| `tests/sample_fruit/` | Chứa ảnh mẫu nhỏ dùng cho smoke test pipeline dữ liệu ảnh. |
| `tests/sample_fruit/.gitkeep` | File giữ chỗ cho thư mục dữ liệu ảnh mẫu. |
| `app_fruit_cli.py` | File xây dựng giao diện dòng lệnh đơn giản để gọi các chức năng train, evaluate hoặc predict. |
| `main.py` | File điều phối chính của project, dùng để gọi các pipeline hoặc kiểm thử tích hợp. |
| `requirements.txt` | File quản lý danh sách thư viện cần cài đặt. |
| `.gitignore` | File loại trừ cache, môi trường ảo, dữ liệu lớn, model lớn hoặc file tạm khỏi GitHub. |
| `README.md` | File giới thiệu source code và mô tả chức năng từng thư mục, từng file. |

---

## 8. Các file kết quả sẽ phát sinh khi chạy project

| File kết quả | Thư mục lưu | Chức năng |
|---|---|---|
| `data_profile.csv` | `reports/01_data_profile/` | Bảng thống kê số lượng ảnh từng lớp. |
| `bad_images.csv` | `reports/01_data_profile/` | Danh sách ảnh lỗi nếu có. |
| `class_distribution.png` | `reports/01_data_profile/` | Biểu đồ phân bố số ảnh theo lớp. |
| `fruit_cnn.keras` | `models/` | Model CNN sau khi huấn luyện. |
| `model_summary.txt` | `reports/02_training/` | Tóm tắt kiến trúc model CNN. |
| `history.csv` | `reports/02_training/` | Lịch sử huấn luyện qua từng epoch. |
| `training_curves.png` | `reports/02_training/` | Biểu đồ loss và accuracy trong quá trình huấn luyện. |
| `classification_report.txt` | `reports/03_evaluation/` | Báo cáo precision, recall, f1-score cho từng lớp trái cây. |
| `confusion_matrix.png` | `reports/03_evaluation/` | Ma trận nhầm lẫn của mô hình trên tập đánh giá. |
| `sample_predictions.png` | `reports/03_evaluation/` | Hình minh họa một số ảnh dự đoán mẫu. |
| `wrong_predictions.csv` | `reports/03_evaluation/` | Danh sách ảnh bị dự đoán sai để phân tích nguyên nhân. |

---

## 9. Quy ước quản lý source code

| Quy ước | Mô tả |
|---|---|
| Tách module rõ ràng | Mỗi file trong `src/` phụ trách một nhóm chức năng riêng để dễ chia việc cho thành viên. |
| Không ghi kết quả ảo | Các chỉ số accuracy, loss, classification report chỉ được ghi sau khi chạy thật. |
| Không đẩy dữ liệu lớn nếu không cần | Dữ liệu thật và model lớn có thể được loại khỏi GitHub bằng `.gitignore`. |
| Giữ thư mục rỗng bằng `.gitkeep` | Các thư mục chưa có dữ liệu vẫn được lưu trong GitHub nhờ file `.gitkeep`. |
| Báo cáo lấy từ output thật | Hình ảnh và bảng kết quả trong báo cáo phải lấy từ thư mục `reports/`. |
### Data. Thứ tự chạy
python src/preprocess_images.py
python src/split_image_data.py
python src/check_processed_dataset.py