# 🛡️ Anti-Spoofing Module — SmartFace AIoT

Module **Anti-Spoofing (Chống giả mạo khuôn mặt)** được tích hợp trong hệ thống SmartFace AIoT, giúp phân biệt khuôn mặt THẬT (Real) và khuôn mặt GIẢ MẠO (Spoof - ảnh in giấy, video/ảnh phát qua màn hình điện thoại, máy tính, mặt nạ...).

---

## 🚀 Tính Năng Nổi Bật

- **Siêu nhẹ & Tối ưu CPU:** Mô hình ONNX đã được nén lượng hóa INT8 (chỉ **600 KB**), chạy mượt mà real-time ngay cả trên thiết bị phần cứng giới hạn (CPU / Edge AIoT).
- **Độ chính xác cao:** Đạt độ chính xác **98.20%** và chỉ số ROC-AUC **0.9984** trên tập kiểm thử CelebA-Spoof (70,000+ mẫu).
- **Hỗ trợ Batch Processing:** Xử lý suy luận đồng thời nhiều khuôn mặt xuất hiện trong cùng một khung hình.
- **Tiền xử lý thông minh:** Tự động cắt mở rộng khuôn mặt (1.5x) lấy ngữ cảnh đường viền và đệm viền `BORDER_REFLECT_101` tránh hiện tượng viền đen làm sai lệch AI.

---

## 📐 Kiến Trúc Mô Hình & Hiệu Năng

| Đặc tính | Chi tiết |
| :--- | :--- |
| **Kiến trúc mạng** | MiniFASNet V2 SE (Squeeze-and-Excitation + Fourier Transform Loss) |
| **Định dạng** | ONNX Quantized (INT8) |
| **Dung lượng file** | **600 KB** |
| **Kích thước đầu vào** | `128 x 128` RGB |
| **Đầu ra** | 2 lớp Logits: `[real_logit, spoof_logit]` |
| **Độ chính xác** | **98.20%** (Real: 97.55% | Spoof: 98.73%) |

---

## 📂 Cấu Trúc Thư Mục

```text
face_auth/antispoof/
├── README.md                      # Tài liệu hướng dẫn này
├── requirements.txt               # Các thư viện phụ thuộc siêu nhẹ
├── __init__.py                    # Export các lớp và hàm API chính
├── predictor.py                   # Class AntiSpoofPredictor chính
├── preprocess.py                  # Hàm crop 1.5x và tiền xử lý ảnh 128x128
├── loader.py                      # Hàm load ONNX Runtime Session (GPU/CPU)
├── system.py                      # Tiện ích đọc thông tin phần cứng (CPU/GPU)
└── models/
    └── best_model_quantized.onnx  # File trọng số mô hình INT8 (600 KB)
```

---

## 💻 Hướng Dẫn Sử Dụng Trong Python Code

### 1. Khởi tạo và dự đoán từ khung hình ảnh:

```python
import cv2
from face_auth.antispoof import AntiSpoofPredictor

# 1. Khởi tạo Predictor (mặc định nạp best_model_quantized.onnx)
predictor = AntiSpoofPredictor(threshold=0.5)

# 2. Đọc ảnh và có tọa độ bounding box bbox (x, y, w, h) từ Face Detector
frame = cv2.imread("face.jpg")
bbox = (100, 100, 150, 150) # (x, y, width, height)

# 3. Chạy dự đoán Anti-Spoofing
result = predictor.predict_frame(frame, bbox)

print(f"Khuôn mặt là: {result['status'].upper()}") # 'REAL' hoặc 'SPOOF'
print(f"Is Real: {result['is_real']}")           # True hoặc False
print(f"Chênh lệch Logit: {result['logit_diff']:.2f}")
```

### 2. Dự đoán đồng thời danh sách các ảnh crop khuôn mặt (Batching):

```python
# Chạy dự đoán batch cho danh sách n ảnh crop
face_crops = [crop_img1, crop_img2]
results = predictor.predict_crops(face_crops)

for res in results:
    if res["is_real"]:
        print("Xác thực thành công: Mặt thật!")
    else:
        print("Cảnh báo: Phát hiện giả mạo!")
```

---

## 🎬 Chạy Demo Thực Tế

Từ thư mục gốc dự án SmartFace (`SmartFace-AIoT-Based-Staff-Management-Attendance-System`):

```bash
# Chạy demo webcam real-time
python face_auth/scripts/demo_antispoof.py

# Chạy demo với 1 bức ảnh tĩnh
python face_auth/scripts/demo_antispoof.py --image path/to/test_image.jpg
```
