# Mô tả dữ liệu ảnh trái cây

## 1. Cấu trúc dữ liệu

Dữ liệu được đặt trong thư mục:

data/raw/<ten_lop>/*.jpg

Mỗi thư mục con tương ứng với một lớp trái cây.

## 2. Số lớp và số ảnh

- Số lớp: 5
- Tổng số ảnh trong data/raw: 2437
- Số ảnh lỗi phát hiện: 0

## 3. Bảng thống kê từng lớp

class_name  num_images
    Banana         490
     Mango         490
 Pineapple         490
Strawberry         492
Watermelon         475

## 4. Kiểm tra ảnh lỗi

Ảnh lỗi được kiểm tra bằng OpenCV cv2.imread.
Nếu ảnh không đọc được, ảnh sẽ được ghi vào bad_images.csv.
Chương trình mặc định không tự động xóa ảnh lỗi.

## 5. Tiền xử lý

- Ảnh được tổ chức theo thư mục lớp.
- Ảnh được resize về kích thước: (128, 128)
- Dữ liệu Training gốc được chia thành train và validation.
- Dữ liệu Test gốc được giữ riêng làm test cuối.
- TensorFlow Dataset được tạo bằng image_dataset_from_directory.

## 6. Ghi chú

Không dùng ảnh test để huấn luyện.
Không ghi số liệu accuracy/loss nếu chưa chạy thật.
