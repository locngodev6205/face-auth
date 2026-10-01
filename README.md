# 👁️ SmartFace AI — Face Authentication & Presentation Attack Detection (PAD)

> **SmartFace Face Auth** là phân hệ AI Thị giác Máy tính (Computer Vision) & Sinh trắc học phục vụ điểm danh thời gian thực và trích xuất vector đặc trưng khuôn mặt cho Hệ thống Quản lý Nhân sự SmartFace AIoT.

---

## 🛠 Tech Stack & Kiến Trúc AI

| Thành phần AI / Kỹ thuật | Công nghệ & Thư viện | Vai trò & Mục đích |
| :--- | :--- | :--- |
| **Face Detection** | SCRFD-500M / OpenCV YunNet | Bộ dò khuôn mặt tốc độ cao, hỗ trợ 5 landmarks và scale crop $1.55\times$ |
| **Face Alignment** | Similarity Transform (5 Landmarks) | Căn chỉnh khuôn mặt chuẩn $112 \times 112$ px theo tọa độ mắt, mũi, miệng |
| **Face Feature Extractor** | ArcFace (InsightFace `buffalo_sc` / `buffalo_l`) | Trích xuất vector đặc trưng $512$ chiều (512-D normalized embeddings) |
| **Liveness / Anti-Spoofing** | MobileNetV3-Small (E1 v5.3) & 2D-DCT (E2/E3) | Phát hiện tấn công giả mạo (PAD - Presentation Attack Detection) |
| **Multi-Object Tracking** | IOU-based FaceTracker | Theo dõi khuôn mặt đa mục tiêu, giữ ID nhận diện ổn định qua từng frame |
| **Vector Database** | PostgreSQL 16 + `pgvector` (`vector(512)`) | Lưu trữ và tìm kiếm vector Cosine Distance qua chỉ mục HNSW |
| **Internal REST API** | FastAPI + Pydantic + Uvicorn | Cung cấp REST endpoints nội bộ (`:5000`) cho Backend Node.js gọi trích xuất vector |
| **Edge Application** | OpenCV + NumPy + ONNX Runtime | Ứng dụng Camera Edge AIoT chạy trực tiếp trên thiết bị đầu cuối |

---

## 🏗️ Quy Trình Xử Lý Mỗi Khung Hình (Frame-by-Frame Pipeline)

```text
[Camera Stream / Video Input]
              │
              ▼
   1. Face Detection (SCRFD-500M / YunNet) ──► Phát hiện Bounding Box & 5 Điểm mốc (Landmarks)
              │
              ▼
   2. Multi-Face Tracking (IOU Tracker)   ──► Cấp Track ID, làm mượt tọa độ BBox & lọc rung lắc
              │
              ▼
   3. Anti-Spoofing (PAD Inference)       ──► Phân tích Mặt thật (Real) vs Giả mạo (Spoof)
              │                               (MobileNetV3-Small E1 v5.3 / 2D-DCT E3)
       ┌──────┴────────────────────────┐
   [Spoof (Giả mạo)]            [Real (Mặt thật)]
       │                               │
       ▼                               ▼
  Báo động đỏ &                4. Face Alignment (Similarity Transform 112x112)
  Từ chối điểm danh                    │
                                       ▼
                               5. Feature Extraction (ArcFace 512-D Embeddings)
                                       │
                                       ▼
                               6. Vector Search & Match (pgvector Cosine Distance)
                                  - So khớp Centroid mẫu nhân viên (Threshold: ≤ 0.35)
                                       │
                                       ▼
                               7. Ghi nhận Điểm danh (REST / MQTT / Socket.IO)
```

---

## 📁 Cấu Trúc Thư Mục Phân Hệ `face_auth/`

```text
face_auth/
├── api_service.py             # Dịch vụ FastAPI nội bộ (:5000) phục vụ đăng ký & trích xuất vector
├── app.py                     # Ứng dụng Camera AIoT nhận diện & chống giả mạo trực tiếp qua Video
├── config.py                  # Cấu hình trung tâm: ngưỡng nhận diện, DB pool, detector, PAD models
├── requirements.txt           # Danh sách thư viện Python cần thiết
├── .env.example               # File mẫu biến môi trường
├── alignment/                 # Module căn chỉnh khuôn mặt 5 landmarks
│   └── aligner.py
├── antispoof/                 # Module & Nghiên cứu Chống giả mạo (PAD)
│   ├── pad_runner.py          # Lớp suy luận PAD ONNX runtime
│   ├── models/                # Lưu trữ trọng số mô hình PyTorch & ONNX
│   │   └── smartface_pad_artifacts/ # 📖 Chi tiết artifacts E1, E2, E3, MobileNetV4
│   └── notebooks/             # 📖 Bộ Jupyter Notebooks thực nghiệm & held-out test
├── database/                  # Tầng kết nối Cơ sở dữ liệu Vector
│   └── vector_db.py           # Quản lý connection pool psycopg3, query pgvector HNSW
├── detection/                 # Module phát hiện khuôn mặt
│   ├── detector.py            # SCRFD detector wrapper
│   └── yunnet_detector.py     # OpenCV YunNet detector wrapper
├── enrollment/                # Module đăng ký danh tính sinh trắc học
│   └── enroll.py              # Tính vector Centroid, lọc outlier khoảng cách
├── evaluation/                # Benchmark đánh giá độ chính xác & LFW calibration
│   └── lfw_identity_benchmark.ipynb
├── tracking/                  # Module bám vết khuôn mặt theo frame
│   └── tracker.py             # FaceTracker (IOU matching)
├── utils/                     # Tiện ích chung (visualize, crop, transform)
└── md/                        # 📖 Kho tài liệu kỹ thuật & Audit logs
    └── README.md
```

---

## 📋 Yêu Cầu Tiên Quyết (Prerequisites)

1. **Python**: Phiên bản `≥ 3.10` (Khuyên dùng `3.10` hoặc `3.11`).
2. **Cơ sở Dữ liệu**: PostgreSQL 16 + `pgvector` PBL6 đang chạy từ Compose ở thư mục gốc repo.
3. **Webcam / Camera AIoT**: Thiết bị camera vật lý hoặc luồng RTSP stream.

---

## ⚡ Hướng Dẫn Cài Đặt & Khởi Chạy

### Bước 1: Tạo môi trường ảo & cài đặt thư viện
```bash
cd face_auth

# Khởi tạo môi trường ảo Python
python -m venv .venv

# Kích hoạt môi trường ảo:
# Trên Windows:
.venv\Scripts\activate
# Trên macOS / Linux:
source .venv/bin/activate

# Cài đặt các dependencies
pip install -r requirements.txt
```

### Bước 2: Cấu hình file `.env`
Tạo file `.env` từ file mẫu `.env.example`:
```bash
cp .env.example .env
```
Các thông số cấu hình chính trong `face_auth/.env`:
- `POSTGRES_DB=PBL6`: Dùng chung PostgreSQL PBL6; host runtime kết nối qua `localhost:5432`.
- `EMBEDDING_MODEL_VERSION=buffalo_s`: Phải khớp `FACE_AUTH_MODEL_VERSION` trong backend.
- `APP_DETECTOR`: Bộ dò khuôn mặt (`scrfd` hoặc `yunnet`).
- `PAD_MODEL_NAME`: Tên mô hình PAD (`mnv3s_e1_preliminary_v5_3_edge_best.onnx` hoặc `e3_concat_preliminary_v1_best.onnx`).
- `MATCH_THRESHOLD`: Ngưỡng so khớp Cosine Distance (`0.35`).

Nếu runtime chạy trên Pi/máy khác, đặt `POSTGRES_HOST` bằng IP LAN của máy chủ DB
và cấu hình `PBL6_POSTGRES_BIND_IP` để publish PostgreSQL trên mạng tin cậy.

---

### Đăng ký gallery để test database PBL6

Hướng dẫn chạy toàn stack và cleanup: [run_full_stack.md](../docs/run_full_stack.md).

```bash
python3 scripts/run_enroll.py --gallery gallery
```

Để liệt kê webcam đang kết nối và chỉ số OpenCV của chúng:

```bash
python3 scripts/list_cameras.py
python3 scripts/list_cameras.py --max-index 15
python3 app.py --camera 1
```

Để thử ép bật adaptive gamma dù runtime config đang đặt `apply_gamma: false`,
đặt `FORCE_GAMMA=true` trong `.env` rồi khởi động lại `app.py`. Terminal sẽ hiện
`PAD_GAMMA = ON (FORCED)`. Đây là chế độ thử nghiệm; ngưỡng PAD hiện tại không tự
hiệu chỉnh lại theo gamma.

Mỗi thư mục con là tên đầy đủ hoặc mã của nhân viên đã tồn tại. Script ưu tiên tìm
theo `employee_code`, sau đó theo tên đầy đủ duy nhất. Nếu chưa có nhân viên và có
ảnh hợp lệ, script giữ nguyên tên thư mục làm `full_name`, tạo mã `TEST-...`, UUID,
email `example.invalid`, điện thoại, chức vụ và lương mẫu; chọn phòng ban có sẵn
nếu có. Không tạo tài khoản đăng nhập hoặc dữ liệu chấm công. Các vector vẫn được
trích xuất từ ảnh thật và lưu SAMPLE/CENTROID đúng phiên bản model.

Chạy lại cùng thư mục sẽ dùng hồ sơ đã có. Nếu có nhiều người trùng tên, đổi tên
thư mục sang mã nhân viên để tránh enroll nhầm. Dùng `--existing-only` để tắt tạo
nhân viên mẫu; `--ignore-duplicate` giữ khuôn mặt đã đăng ký. Script yêu cầu DB PBL6
đã áp dụng migration vector mới. Nếu ghi embedding thất bại sau khi tạo hồ sơ,
hồ sơ mẫu vẫn còn và có thể dùng lại khi chạy lần tiếp theo.

### Bước 3: Khởi chạy các dịch vụ

#### 1. Chạy Dịch Vụ API Nội Bộ (`api_service.py` trên Port `5000`):
Dịch vụ này nhận ảnh khuôn mặt dạng base64 từ Web Backend (`/api/biometrics/:employeeId/enroll-images`), tự động detect, trích xuất 512-D embeddings, lọc outlier và lưu vào pgvector:
```bash
uvicorn api_service:app --host 0.0.0.0 --port 5000 --reload
```
- Endpoint kiểm tra: `GET http://localhost:5000/health`
- Endpoint đăng ký: `POST http://localhost:5000/internal/enroll`

#### 2. Chạy Ứng Dụng Camera Nhận Diện Trực Tiếp (`app.py`):
```bash
python app.py
```

Các tùy chọn dòng lệnh (CLI flags):
```bash
python app.py --no-pad                            # Tắt module chống giả mạo (chạy nhanh hơn)
python app.py --no-fps                            # Tắt hiển thị FPS overlay
python app.py --pad-model mnv4_best_224.onnx      # Chọn mô hình PAD thử nghiệm khác
python app.py --no-pad --no-fps                   # Chế độ siêu nhẹ tối đa FPS
```

---

## 🛡️ Mô Hình Chống Giả Mạo (PAD Benchmark & Models)

Hệ thống cung cấp các mô hình PAD được huấn luyện và lưu trữ sẵn tại [antispoof/models/smartface_pad_artifacts/](antispoof/models/smartface_pad_artifacts/README.md):

| Mô hình | Kiến trúc | Kích thước đầu vào | Tham số | Latency CPU | Held-out AUC | Vai trò |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **`E1_v5_3_mnv3_small`** | MobileNetV3-Small | $3 \times 224 \times 224$ | **1.11M** | **~2.47 ms** | **98.14%** | ⭐ **Mô hình không gian chính thức (Active Production)** |
| **`E3_concat_v1`** | MobileNetV3-Small + 2D-DCT | Spatial + Freq ($224 \times 224$) | **1.13M** | **~2.21 ms** | **98.57%** | 🚀 **Fusion Đa Miền (Spatial + Frequency)** |
| **`E2_dct_v1`** | Tiny Frequency CNN | $1 \times 224 \times 224$ | **17.1K** | **~0.57 ms** | **83.66%** | Nghiên cứu miền Tần số |
| **`minifasnet_v2se`** | MiniFASNetV2SE (INT8) | $3 \times 128 \times 128$ | **0.47M** | **~1.20 ms** | - | Mô hình siêu nhẹ kế thừa (Legacy) |

---

## 📖 Tài Liệu Tham Khảo Liên Quan

- [Danh mục Tài liệu Kỹ thuật & Audits (`face_auth/md/README.md`)](md/README.md)
- [Cơ sở Lưu trữ Mô hình & Artifacts (`face_auth/antispoof/models/smartface_pad_artifacts/README.md`)](antispoof/models/smartface_pad_artifacts/README.md)
- [Danh mục Notebooks Nghiên cứu & Huấn luyện (`face_auth/antispoof/notebooks/README.md`)](antispoof/notebooks/README.md)
- [Hướng dẫn Chuẩn bị Dữ liệu LFW Benchmark (`face_auth/data/data_readme.md`)](data/data_readme.md)
