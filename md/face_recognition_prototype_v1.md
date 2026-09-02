# Prototype V1 — Secure Face Authentication (chưa có PAD)

## 1. Mục tiêu

Xây dựng prototype V1 chạy end-to-end cho hệ thống xác thực khuôn mặt, **chưa tích hợp Anti-Spoofing/PAD** nhưng giữ kiến trúc modular để sau này chèn PAD vào giữa Face Detection/Crop và Face Embedding.

Pipeline V1:

```text
Camera / Image
    ↓
Face Detection — SCRFD (InsightFace)
    ↓
Face Tracking (IOU Matching) ── (Nếu đã nhận diện) ──> Lấy kết quả cũ
    ↓ (Nếu cần nhận diện lại)
Face Crop & Alignment
    ↓
Face Embedding — ArcFace
    ↓
L2 Normalization
    ↓
PostgreSQL + pgvector
    ↓
Top-K Similarity Search
    ↓
Threshold Decision
    ↓
Identity / UNKNOWN
```

Demo bắt buộc có 2 tình huống:

1. Khuôn mặt thật của người dùng → nhận diện đúng danh tính.
2. Ảnh của chính người đó hiển thị trên iPad → hệ thống V1 **vẫn có thể nhận diện thành cùng danh tính**, qua đó làm nổi bật lỗ hổng khi chưa có PAD.

Không dùng expanded crop trong V1. Detector output được crop trực tiếp theo bounding box cho pipeline hiện tại.

---

## 2. Kiến trúc module

Tách code thành các module độc lập, nhưng **load model/connection đúng một lần khi ứng dụng khởi động**.

```text
face_auth/
│
├── app.py                     # Ứng dụng webcam realtime (4 visual states, time-based interval 1.0s)
├── config.py                  # Cấu hình tập trung (DB, model, tracking 1.0s, threshold 0.35)
├── requirements.txt
├── docker-compose.yml         # PostgreSQL 16 + pgvector container
├── .env
│
├── detection/
│   ├── __init__.py
│   └── detector.py            # SCRFD detector (InsightFace buffalo_s)
│
├── alignment/
│   ├── __init__.py
│   └── aligner.py             # 5-point affine transformation -> 112x112
│
├── recognition/
│   ├── __init__.py
│   └── embedder.py            # ArcFace feature extractor -> 512-D L2-normalized
│
├── database/
│   ├── __init__.py
│   └── vector_db.py           # PostgreSQL + pgvector connection pool & 1:N search
│
├── enrollment/
│   ├── __init__.py
│   └── enroll.py              # Centroid mean embedding & outlier filtering (0.35)
│
├── tracking/
│   ├── __init__.py
│   └── tracker.py             # IOU FaceTracker & Time-based interval (1.0s)
│
├── utils/
│   ├── __init__.py
│   └── image.py
│
├── scripts/
│   ├── run_enroll.py          # Quét gallery/ để đăng ký người dùng thật vào DB
│   └── cleanup_demo.py        # Xoá nhanh demo LFW identities khỏi VectorDB
│
├── evaluation/
│   ├── __init__.py
│   ├── lfw_identity_benchmark.ipynb   # Benchmark 1:N & Calibration threshold
│   ├── lfw_demo_enroll.ipynb          # Nạp 20–50 LFW demo identities vào VectorDB
│   ├── lfw_demo_test_images/          # Ảnh test của enrolled users (theo folder)
│   └── lfw_demo_impostors/            # Ảnh test của impostor (theo folder)
│
├── data/
│   ├── lfw/                   # LFW images
│   └── lfw_metadata/          # LFW CSVs
│
└── gallery/
    └── [Tên Người Dùng]/      # 3 ảnh chất lượng cao để enroll người thật
        └── [Ảnh...]
```

Nguyên tắc:

- `__init__()` của mỗi module khởi tạo model/resource một lần.
- `detect()`, `align()`, `embed()`, `search()` chỉ thực hiện inference/query.
- Không tạo object model bên trong vòng lặp webcam.

---

## 3. Model cụ thể được đề xuất

### 3.1 Face Detection: SCRFD (thuộc InsightFace)

Dùng detector SCRFD có sẵn trong model pack của InsightFace. Lợi thế là nó trả về sẵn 5 điểm landmark cực kỳ chuẩn xác, đồng thời tương thích tuyệt đối với input của ArcFace.

Output cần có:

```text
Bounding Box: (x1, y1, x2, y2)
Landmarks: 5 điểm (mắt trái, mắt phải, mũi, khóe miệng trái/phải)
```

**V1 chưa mở rộng bounding box.** Tận dụng landmark này để align khuôn mặt luôn.

### 3.2 Face Alignment

Nếu detector cung cấp 5 điểm landmark, dùng chúng trực tiếp:

- mắt trái
- mắt phải
- mũi
- khóe miệng trái
- khóe miệng phải

Áp dụng similarity transform để đưa khuôn mặt về kích thước phù hợp với model recognition.

Nếu checkpoint detector bạn chọn **không có landmarks**, dùng một landmark/alignment model pretrained riêng.

### 3.3 Face Recognition: InsightFace / ArcFace pretrained

Dùng model recognition pretrained của InsightFace, ưu tiên một model nhẹ phù hợp prototype/edge, ví dụ `buffalo_s` nếu môi trường hiện tại hỗ trợ và license/model terms phù hợp.

Output mong muốn:

```text
512-D face embedding
```

Sau đó L2 normalize:

```python
embedding = embedding / np.linalg.norm(embedding)
```

ArcFace ở inference được dùng như **feature extractor**; không cần classifier identity cố định của giai đoạn training.

### 3.4 Vector Database: PostgreSQL + pgvector

V1 dùng PostgreSQL + pgvector vì database metadata và embedding có thể quản lý trong cùng hệ thống.

Schema tối thiểu:

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE users (
    user_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE face_embeddings (
    id BIGSERIAL PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    embedding VECTOR(512) NOT NULL,
    is_mean BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW()
);
```

Chưa cần Qdrant/Milvus trong V1.

---

## 4. Cài đặt môi trường

Python đề xuất: 3.10 hoặc 3.11.

Ví dụ requirements:

```text
ultralytics
insightface
onnxruntime
opencv-python
numpy
psycopg[binary]
python-dotenv
scikit-learn
pillow
```

Nếu dùng NVIDIA GPU và muốn inference ONNX bằng GPU thì thay `onnxruntime` bằng package phù hợp với môi trường CUDA/driver.

Kiểm tra:

```bash
python --version
python -c "import cv2, numpy; print('OK')"
python -c "import insightface; print('InsightFace OK')"
```

---

## 5. Cấu hình Database

V1 hỗ trợ linh hoạt việc dùng Database PostgreSQL (cần có extension `pgvector`) ở local hoặc trên cloud (Supabase, Neon, AWS RDS,...). Cấu hình kết nối được đặt trong file `.env`:

```env
# 1. Dùng URL kết nối chung (Cách tối ưu để kết nối Cloud DB)
DATABASE_URL=postgresql://username:password@host:port/dbname

# 2. Hoặc cấu hình từng thành phần (Thường dùng cho Local DB - sẽ được áp dụng nếu DATABASE_URL trống)
POSTGRES_DB=face_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
```

Hệ thống sẽ tự động tạo extension `pgvector` và table `persons` nếu chúng chưa tồn tại khi kết nối thông qua module `VectorDB`.

---

## 6. Logic chi tiết từng Module (LLM Source of Truth)

Phần này mô tả cấu trúc, input/output và logic cốt lõi của từng file Python trong hệ thống. Bất kỳ LLM/Developer nào đọc phần này đều có thể nắm bắt 100% luồng dữ liệu.

### 6.1 `config.py` (Cấu hình hệ thống)
- Chứa các biến môi trường cấu hình DB (`POSTGRES_DB`, `DATABASE_URL`...).
- Cấu hình Model InsightFace: `MODEL_PACK_NAME = "buffalo_s"` (Dùng cho cả Detect và Recog).
- Cấu hình Tracking: `RECOGNIZE_INTERVAL_SECONDS = 1.0` (Thời gian giãn cách giữa các lần re-verify danh tính — tính bằng giây thay vì frame để đảm bảo tính nhất quán trên mọi phần cứng từ Mac 30-60 FPS đến Raspberry Pi 5-15 FPS), `TRACK_IOU_THRESHOLD` (Ngưỡng IOU để track), `TRACK_MAX_MISSING_FRAMES`.
- Cấu hình Enrollment: `RECOMMENDED_GALLERY_SIZE = 3` (Số lượng ảnh đăng ký tối ưu rút ra từ thực nghiệm LFW).
- Cấu hình nhận diện: `MATCH_THRESHOLD = 0.35` (Ngưỡng Cosine Similarity để quyết định Identity — đã được calibrate thực nghiệm từ LFW Benchmark: ngưỡng tối ưu ~0.31, nâng lên 0.35 cho production để đảm bảo FAR thấp).

### 6.2 `detection/detector.py`
- **Nhiệm vụ**: Trích xuất Bounding Box và 5 điểm Landmarks.
- **Model**: Dùng module `FaceAnalysis` của InsightFace nhưng giới hạn bộ nhớ: `allowed_modules=["detection"]` (SCRFD).
- **Output**: Trả về `List[dict]`, mỗi dict có dạng:
  ```python
  {
      "bbox": (x1, y1, x2, y2),
      "score": float,
      "landmarks": [[x,y], [x,y], [x,y], [x,y], [x,y]] # 5 điểm
  }
  ```
- **Lưu ý**: V1 không mở rộng bbox (expanded crop). Lấy đúng output để đưa sang Alignment.

### 6.3 `alignment/aligner.py`
- **Nhiệm vụ**: Cắt (Crop) và xoay (Align) khuôn mặt về chuẩn 112x112.
- **Logic**: Sử dụng 5 điểm landmarks từ SCRFD, map với ma trận tiêu chuẩn `ARCFACE_DST_112` thông qua `cv2.estimateAffinePartial2D`. Sau đó dùng `cv2.warpAffine` để lấy ảnh khuôn mặt thẳng đứng, kích thước chuẩn đầu vào 112x112 cho ArcFace.

### 6.4 `recognition/embedder.py`
- **Nhiệm vụ**: Chuyển đổi ảnh khuôn mặt 112x112 thành Vector đại diện.
- **Model**: `FaceAnalysis` với `allowed_modules=["detection", "recognition"]` (do ràng buộc assert nội bộ của InsightFace, sau đó chỉ lấy submodel `recognition` ArcFace `buffalo_s`).
- **Logic**: 
  - Gọi `get_feat(aligned_face)` để lấy vector thô.
  - Thực hiện **L2-Normalization** (`embedding / norm`) ngay lập tức để chuyển độ đo Euclidean sang không gian Cosine.
- **Output**: Vector numpy 512 chiều (512-D) chuẩn hóa.

### 6.5 `tracking/tracker.py`
- **Nhiệm vụ**: Theo dõi (Track) khuôn mặt qua các frame để **tối ưu FPS**, tránh việc phải Align+Embed+DB Search liên tục trên từng frame.
- **Logic**:
  - So khớp Bounding Box giữa frame hiện tại và frame trước bằng IOU (Intersection Over Union).
  - Khởi tạo đối tượng `Track`. Mỗi Track ghi nhận mốc thời gian nhận diện cuối cùng `last_recognition_time`.
  - Nếu `time.time() - last_recognition_time >= RECOGNIZE_INTERVAL_SECONDS` (mặc định 1.0 giây), hàm `needs_recognition()` trả về `True`, ép ứng dụng chạy lại pipeline Embed + DB search. Sau đó update lại `name`, `score` và cập nhật `last_recognition_time = time.time()`.
  - Track sẽ bị xoá nếu biến mất khỏi khung hình quá `max_missing_frames`.

### 6.6 `database/vector_db.py`
- **Nhiệm vụ**: Quản lý connection pool (`psycopg-pool`) và truy vấn Postgres (pgvector).
- **Schema**:
  - `users`: `user_id` (PK), `name`.
  - `face_embeddings`: `embedding_id` (PK), `user_id` (FK), `embedding` (VECTOR 512), `is_mean` (BOOLEAN).
- **Quy tắc sinh ID (`upsert`)**:
  - `mean_embedding` (Vector trung bình đại diện) luôn được lưu với `embedding_id = "[user_id]_0000"`.
  - `individual_embeddings` (Vector thành phần) được lưu tuần tự với `embedding_id = "[user_id]_0001", "[user_id]_0002"...` (Tuỳ chọn lưu/không lưu qua cờ `save_individuals`).
- **Logic Search**: Hàm `search(mean_only=True)` mặc định tính toán Cosine Similarity (`1 - <=>`) và chỉ JOIN trên các vector có `is_mean=TRUE` để tăng tốc độ truy vấn (1:N search). Đây là chiến lược tối ưu nhất đã được kiểm chứng qua LFW Benchmark (vừa giảm tải DB vừa có độ chính xác/FAR ổn định hơn individual-max).
- **Logic Decide**: `decide_identity(rows, threshold=config.MATCH_THRESHOLD)` dùng để so sánh top 1 similarity với threshold (0.35) để chốt `MATCH` hay `UNKNOWN`.

### 6.7 `enrollment/enroll.py`
- **Nhiệm vụ**: Tính toán Vector trung bình (Identity Embedding) và loại bỏ ảnh rác (Outlier filtering).
- **Khuyến nghị**: Nên thu thập **3 ảnh** chất lượng/identity (tối đa 5 ảnh) cho gallery.
- **Logic**:
  1. Tính trung bình cộng sơ bộ của tất cả ảnh đầu vào -> L2 Normalize.
  2. Nhân ma trận (Dot product) để tính Cosine Similarity giữa từng ảnh nhỏ và Vector trung bình sơ bộ.
  3. Loại bỏ các ảnh có Similarity < `outlier_threshold` (0.35) — thường là ảnh chụp sau lưng, mờ mịt, detect nhầm.
  4. Tính toán Vector trung bình lần cuối từ các ảnh hợp lệ -> L2 Normalize.
- **Output**: Trả về `(mean_embedding, warnings, valid_embeddings)`.

### 6.8 `scripts/` (Scripts tiện ích)
- **`scripts/run_enroll.py`**:
  - Tự động duyệt thư mục `gallery/` (khuyến nghị 3 ảnh/identity).
  - Chạy Pipeline (Detect -> Align -> Embed), gọi `enroll_person()` để lọc outlier và upsert vào VectorDB.
  - Hỗ trợ cờ `--ignore-duplicate` và `--no-individuals`.
- **`scripts/cleanup_demo.py`**:
  - Tiện ích CLI xoá nhanh toàn bộ các identity demo LFW khỏi VectorDB (dựa theo prefix `demo_lfw_*`).
  - Hỗ trợ cờ `--dry-run` (xem trước số lượng sẽ xoá) và `--yes` (bỏ qua bước xác nhận).

### 6.9 `app.py` (Main Runtime)
- **Nhiệm vụ**: Khởi chạy ứng dụng Camera Realtime.
- **Runtime Rule**: `FaceDetector`, `FaceTracker`, `FaceEmbedder`, `VectorDB` **chỉ được khởi tạo đúng 1 lần (LOAD ONCE)** bên ngoài vòng lặp webcam để đảm bảo zero-overhead khởi tạo model.
- **Cơ chế Time-based Interval**: Re-verify danh tính đúng mỗi **1.0 giây** (`RECOGNIZE_INTERVAL_SECONDS = 1.0`) thay vì đếm frame. Giúp ứng dụng chạy đồng nhất trên cả máy tính (30–60 FPS) lẫn Raspberry Pi (5–15 FPS).
- **4 trạng thái hiển thị (Visual States)**:
  1. ⬜ **Pending** (`Nhan dien...` - viền xám): Khuôn mặt mới xuất hiện trong khung hình, đang chờ kết quả nhận diện lần đầu.
  2. 🟩 **Locked** (`[OK] Tên (0.xx)` - viền xanh lá): Nhận diện thành công, IOU tracker bám mượt mà ở tốc độ camera tối đa.
  3. 🟧 **Re-verifying** (`Tên (0.xx)` - viền cam): Đúng 1.0 giây sau, đang kích hoạt re-verify ngầm nhưng vẫn giữ tên cũ trên màn hình.
  4. 🟥 **Unknown** (`UNKNOWN (0.xx)` - viền đỏ): Không khớp với bất kỳ ai trong DB hoặc similarity < 0.35.
- Label tên và điểm số được vẽ trên nền chữ nhật màu đặc (filled rect) giúp luôn dễ đọc trên mọi hậu cảnh.

### 6.10 `evaluation/` (Benchmark & Demo Tools)
- **`lfw_identity_benchmark.ipynb`**: Notebook thực nghiệm 1:N open-set identification tách biệt Calibration/Test, so sánh Gallery 1/2/3/5 và Mean vs Individual-max để chốt cấu hình production.
- **`lfw_demo_enroll.ipynb`**: Notebook nạp tự động 20–50 danh tính từ LFW vào VectorDB (namespace `demo_lfw_*`), đồng thời tự động xuất ảnh ra các thư mục con theo từng identity:
  - `lfw_demo_test_images/<name>/`: 1–2 ảnh test của người đã enroll (dùng để test replay attack).
  - `lfw_demo_impostors/<name>/`: 1–2 ảnh của người lạ chưa từng enroll (dùng để test từ chối người lạ).

---

## 15. Demo với khuôn mặt thật và iPad

### Chuẩn bị dữ liệu
1. **Người dùng thật**: Đặt 3 ảnh rõ nét vào `gallery/<Tên_Bạn>/` rồi chạy `python scripts/run_enroll.py`.
2. **Dữ liệu demo LFW**: Chạy notebook `evaluation/lfw_demo_enroll.ipynb` để nạp 30 identity demo và tạo sẵn 2 thư mục ảnh test.

### Demo 1 — Người thật (Live Authentication)
Bạn đứng trực tiếp trước camera:
* SCRFD phát hiện khuôn mặt $\rightarrow$ ArcFace trích xuất 512-D $\rightarrow$ pgvector Top-1 search $\rightarrow$ Đạt ngưỡng $\ge 0.35$.
* Hiển thị: 🟩 `[OK] Nguyen Quang Anh (0.xx)`.

### Demo 2 — Tấn công Replay Attack bằng iPad / Màn hình (Lỗ hổng khi chưa có PAD)
Mở một ảnh trong thư mục `evaluation/lfw_demo_test_images/<Tên_Người>/` trên iPad/điện thoại rồi đưa trước camera:
* Vì V1 chưa có module Anti-Spoofing (PAD), hệ thống vẫn nhận diện: 🟩 `[OK] <Tên_Người> (0.xx)`.
* **Thông điệp thuyết trình**: *“Ở V1, hệ thống nhận diện đúng đặc trưng sinh trắc học nhưng chưa xác minh được tính thật của khuôn mặt sống. Đây chính là động lực cốt lõi để nghiên cứu và tích hợp module PAD (Anti-Spoofing) ở Version 2.”*

### Demo 3 — Từ chối người lạ (Open-Set Impostor Rejection)
Mở một ảnh trong thư mục `evaluation/lfw_demo_impostors/<Tên_Người>/` trên iPad/điện thoại đưa trước camera:
* Do danh tính này hoàn toàn không có trong VectorDB, điểm similarity cao nhất $< 0.35$.
* Hiển thị: 🟥 `UNKNOWN (0.xx)`.
* Chứng minh hệ thống chặn người lạ chính xác, không bị nhận diện nhầm.

---

## 16. Dataset / benchmark

### 16.1 LFW dataset (Đã chuẩn bị)

Hệ thống sử dụng bộ dataset LFW (Labeled Faces in the Wild) làm nguồn dữ liệu chuẩn để đánh giá khả năng nhận diện (Identification/Verification) và tính toán các metrics (Accuracy, FAR, FRR). **Lưu ý: Không trộn lẫn ảnh LFW vào thư mục `gallery/` hay database user thật của ứng dụng demo.**

Dữ liệu được tổ chức trong thư mục `data/` như sau:
- **`data/lfw/`**: Chứa toàn bộ ảnh gốc, được phân loại theo thư mục tên người (VD: `data/lfw/George_W_Bush/George_W_Bush_0001.jpg`).
- **`data/lfw_metadata/`**: Chứa các file CSV quy định luật chơi (Benchmark Protocol) của dataset. Tham khảo chuẩn từ [Kaggle LFW Dataset](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset?resource=download):
  - `lfw_allnames.csv`: Danh sách toàn bộ tên người và số lượng ảnh tương ứng.
  - `matchpairsDevTest.csv` / `matchpairsDevTrain.csv`: Tập hợp các cặp ảnh Positive (Cùng một người) dùng cho Testing/Training.
  - `mismatchpairsDevTest.csv` / `mismatchpairsDevTrain.csv`: Tập hợp các cặp ảnh Negative (Khác người) dùng cho Testing/Training.
  - `pairs.csv`: File chuẩn 10-fold cross-validation chính thức của LFW (bao gồm cả matched và mismatched pairs). Chạy benchmark V1 sẽ dùng file này để tính EER/Accuracy.

Workflow Benchmark LFW:

```text
Đọc cặp ảnh từ pairs.csv (hoặc match/mismatch csv)
 ↓
Trích xuất ArcFace embeddings (Chạy qua pipeline Detect -> Align -> Embed)
 ↓
Tính Cosine similarity giữa 2 embedding
 ↓
So sánh với Threshold để dự đoán (MATCH / MISMATCH)
 ↓
Đối chiếu với nhãn gốc để ra Verification metrics (FAR, FRR, EER, Accuracy)
```

### 16.2 LFW Identity Recognition & Threshold Benchmark (Custom Open-Set Protocol)

Bên cạnh benchmark verification theo cặp ảnh (1:1), hệ thống đã xây dựng một benchmark nhận diện danh tính (1:N Identification & Open-Set Rejection) độc lập tại [`evaluation/lfw_identity_benchmark.ipynb`](file:///Users/coding/PBL6/face_auth/evaluation/lfw_identity_benchmark.ipynb) để phục vụ trực tiếp việc chốt cấu hình production cho hệ thống Face Authentication.

#### 1. Thiết kế thực nghiệm (Experimental Protocol)
- **Danh tính Enrolled (100 người)**: Chia ảnh thành `Gallery` (1, 2, 3, 5 ảnh), `Calibration query` (2 ảnh), và `Test query` (2 ảnh).
- **Danh tính Impostor (300 người)**: Không đưa vào gallery, chia thành `Calibration impostor` (1 ảnh) và `Test impostor` (2 ảnh) để đo khả năng từ chối người lạ.
- **Tách biệt Calibration / Test**: Quét ngưỡng trên tập Calibration để tìm `MATCH_THRESHOLD` đạt mục tiêu `FAR ≤ 1%`, sau đó cố định (khóa) ngưỡng này và đánh giá độc lập trên tập Test (tránh data leakage).
- **So sánh 2 chiến lược Search**:
  - `Mean centroid embedding` (`mean_only=True`): Gom trung bình các ảnh gallery sau khi lọc outlier.
  - `Individual-max embedding`: So sánh query với từng ảnh gallery và lấy similarity lớn nhất.

#### 2. Kết luận thực nghiệm quan trọng (Empirical Findings & Final Decisions)
1. **Chiến lược Embedding — Chọn Mean Centroid (`mean_only=True`)**:
   - `Mean embedding` cho kết quả nhận diện ổn định, giảm thiểu phương sai do nhiễu tư thế và ánh sáng tốt hơn `Individual-max`.
   - Tiết kiệm 3–5 lần dung lượng Database/pgvector (1 user = 1 vector) và tăng tốc độ tìm kiếm Top-K, tránh hiện tượng 1 user chiếm nhiều slot kết quả.
2. **Kích thước Gallery tối ưu — Chọn `Gallery Size = 3 ảnh`**:
   - Từ 1 ảnh lên 3 ảnh, độ chính xác nhận diện đúng (TPIR) tăng vọt và FRR giảm đáng kể.
   - Từ 3 ảnh lên 5 ảnh, độ chính xác tăng không đáng kể nhưng tăng gánh nặng thu thập dữ liệu lúc đăng ký.
   - Do đó, **3 ảnh chất lượng/identity** là điểm cân bằng hoàn hảo (`RECOMMENDED_GALLERY_SIZE = 3`).
3. **Hiệu chỉnh Threshold — Chọn `MATCH_THRESHOLD = 0.35`**:
   - Ngưỡng tối ưu thực nghiệm tìm được trên tập Calibration LFW là **~0.31** (đạt cân bằng F1 và FAR ≤ 1%).
   - Trong ứng dụng xác thực bảo mật (Face Authentication), nhóm quyết định nâng ngưỡng lên **0.35** (`strict mode`) để ép tỷ lệ chấp nhận nhầm người lạ (FAR) xuống mức cực thấp, chấp nhận tăng nhẹ FRR đối với ảnh góc nghiêng/ánh sáng quá yếu.

### CFP-FP

Dùng để kiểm tra robustness khi pose thay đổi mạnh:

```text
Frontal ↔ Profile
```

### AgeDB-30

Dùng để kiểm tra robustness khi độ tuổi thay đổi.

### Demo dataset riêng

Cho demo realtime, nên dùng 5–10 người tự thu thập với 3–5 ảnh/người (khuyến nghị 3 ảnh rõ nét).

Không dùng ảnh calibration/test làm ảnh enrollment cùng lúc.

### VGGFace2 subset — tùy chọn

Nếu cần gallery lớn hơn, lấy một subset identity từ VGGFace2 thay vì tải toàn bộ dataset.

---

## 17. Benchmark metrics cho V1

Prototype recognition nên ghi nhận:

### Accuracy / verification accuracy

Cho benchmark pair-based.

### FAR — False Acceptance Rate

Người lạ bị chấp nhận là user đã đăng ký.

### FRR — False Rejection Rate

User hợp lệ bị từ chối.

### EER

Nếu cần một operating point tổng quát cho verification.

Với realtime demo, ghi thêm:

```text
Detection latency
Alignment latency
Embedding latency
Vector search latency
Total latency
FPS
```

---

## 18. Logging latency

Ví dụ:

```python
import time

start = time.perf_counter()
# detection
end = time.perf_counter()
detect_ms = (end - start) * 1000
```

Tổng:

```text
T_total = T_detection
        + T_alignment
        + T_embedding
        + T_search
```

FPS xấp xỉ:

```text
FPS ≈ 1 / T_total_seconds
```

Đo ở trạng thái warm-up sau khi model đã được load.

Không tính thời gian model initialization vào latency/frame.

### 18.2 Cân nhắc triển khai Cloud VectorDB & Thiết bị Edge (Raspberry Pi)

Khi chuyển đổi từ môi trường demo cục bộ (Local Docker) sang hệ thống sản phẩm phân tán (Edge Camera + Cloud Vector Database):

1. **Lợi thế kiến trúc Edge AI (Inference tại biên)**:
   - Toàn bộ khâu tính toán nặng (SCRFD Detection, Alignment, ArcFace Embedding) được xử lý ngay tại thiết bị Edge.
   - Dữ liệu truyền tải qua Internet lên Cloud **chỉ là một vector 512 chiều (~2 KB)**, hoàn toàn không gửi luồng video hay ảnh HD $\rightarrow$ Độ trễ mạng RTT chỉ tốn ~20–40ms, băng thông mạng gần như bằng 0.
2. **Connection Pooling**:
   - Sử dụng `psycopg_pool.ConnectionPool` đã tích hợp sẵn trong `VectorDB` giúp tái sử dụng kết nối TLS/TCP có sẵn, tránh tốn 100–200ms bắt tay bảo mật ở mỗi request.
3. **Cơ chế Time-based Interval (1.0s)**:
   - Chuyển `RECOGNIZE_INTERVAL` sang tính theo thời gian thực (`1.0s`) đảm bảo dù chạy trên phần cứng yếu hơn (Raspberry Pi 5–15 FPS) hay mạnh hơn (PC 30–60 FPS), chu kỳ re-verify luôn diễn ra chuẩn xác sau mỗi 1.0 giây, không bị trôi thời gian khi camera tụt FPS.
4. **Khuyến nghị tối ưu riêng cho Raspberry Pi**:
   - **Hạ độ phân giải detector**: Giảm `DETECTOR_DET_SIZE` từ `(640, 640)` xuống `(320, 320)` trong `config.py` giúp giảm 4 lần số phép tính, đưa FPS của SCRFD trên CPU Raspberry Pi từ 5 FPS lên 15–20 FPS.
   - **Quantization**: Dùng model INT8 cho ONNX Runtime trên kiến trúc ARM NEON.
   - **Hardware Acceleration**: Có thể gắn thêm Raspberry Pi AI Kit (Hailo-8L NPU 13 TOPS) để đạt 60+ FPS mượt mà.

---

## 19. Acceptance criteria cho Prototype V1

V1 được coi là hoàn thành khi:

### Functional

- [ ] Camera đọc được realtime.
- [ ] SCRFD phát hiện khuôn mặt và 5 điểm landmarks.
- [ ] Face crop và alignment đưa mặt về 112x112 cho ArcFace.
- [ ] ArcFace tạo 512-D normalized embedding.
- [ ] Enrollment lưu embedding vào PostgreSQL + pgvector.
- [ ] Query trả về Top-K gần nhất.
- [ ] Threshold quyết định `MATCH/UNKNOWN`.
- [ ] Có demo người thật.
- [ ] Có demo iPad photo attack để minh họa V1 chưa có PAD.

### Engineering

- [ ] Detector chỉ load một lần.
- [ ] Recognition model chỉ load một lần.
- [ ] PostgreSQL connection/pool được khởi tạo một lần.
- [ ] Không load model bên trong frame loop.
- [ ] Các module có interface độc lập.

### Evaluation

- [ ] Có calibration threshold.
- [ ] Có positive/negative pairs.
- [ ] Có FAR/FRR hoặc EER.
- [ ] Có latency/FPS.

---

## 20. Những thứ KHÔNG làm trong V1

Không làm:

- Anti-Spoofing/PAD.
- Expanded crop.
- DCT/Frequency Branch.
- Temporal liveness.
- Depth/IR.
- Qdrant/Milvus.
- Retrain ArcFace.
- Training detector.

Mục đích là khóa một **recognition baseline end-to-end ổn định** trước khi thêm phần nghiên cứu PAD.

---

## 21. Kiến trúc mở rộng sang V2

Sau khi V1 hoàn thành, chỉ chèn một module mới:

```text
Camera
 ↓
SCRFD Detector
 ↓
Face Crop (dùng landmarks)
 ↓
┌───────────────────────┐
│ PAD                   │
│ MobileNetV3 baseline  │
│      +                │
│ DCT Frequency Branch  │
└──────────┬────────────┘
           │
       REAL / SPOOF
           │
          REAL
           ↓
      Alignment
           ↓
        ArcFace
           ↓
      512-D Embedding
           ↓
       pgvector
           ↓
    MATCH / UNKNOWN
```

V2 sẽ có hai model để ablation riêng:

```text
Baseline:
MobileNetV3 → Classifier

Proposed:
MobileNetV3 + DCT Frequency Branch → Fusion → Classifier
```

Hai model phải được train độc lập để comparison công bằng.

Sau đó đánh giá:

```text
CelebA-Spoof      → in-domain
OULU-NPU           → cross-domain
SiW                → cross-domain

Metrics:
ACER / APCER / BPCER

Edge:
Parameters / FLOPs / Latency / FPS
```

---

## 22. Mục tiêu cuối của V1

V1 không cần chứng minh research novelty.

V1 cần chứng minh:

```text
“Recognition stack hoạt động end-to-end.”
```

Sau đó V2 mới trả lời câu hỏi nghiên cứu:

```text
“Frequency-aware features có giúp lightweight RGB PAD
cải thiện cross-domain robustness mà vẫn phù hợp edge deployment không?”
```

Đây là cách giữ project modular và giảm rủi ro: recognition stack ổn định trước, PAD được nghiên cứu như một module độc lập phía trước ArcFace.
