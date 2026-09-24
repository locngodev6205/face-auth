# SmartFace Face Authentication — Current Codebase State & Architecture Context

> **Mục đích của tài liệu:**  
> Đây là tài liệu **Source of Truth (Ngữ cảnh trung tâm)** mô tả toàn bộ kiến trúc, luồng dữ liệu, chi tiết kỹ thuật từng thành phần, cấu trúc mã nguồn và trạng thái phát triển mới nhất của module `face_auth`.  
> Bất kỳ Kỹ sư hoặc AI Coding Agent nào khi bắt đầu phiên làm việc mới đều cần đọc tài liệu này để nắm bắt 100% bối cảnh, các quy ước bất biến (invariants), những vấn đề đang tồn đọng và kế hoạch tiếp theo mà không làm mất dấu (context loss) hay xung đột kiến trúc.

### Snapshot runtime hiện tại (2026-09-24)

- Detector của `app.py`: **SCRFD / InsightFace `buffalo_s`**, `det_size=(640, 640)`, CPU provider.
- Tracker: detector cadence `0.1s`; các frame xen kẽ dùng Lucas–Kanade optical flow, redetect khi track không an toàn.
- PAD model: `mnv3_e1_preliminary_v5_1_best.onnx` + sidecar `.onnx.data`, input `224×224`, 3-class.
- PAD contract: `RGB`, CelebA mean/std, gamma `OFF`, `PAD_BBOX_EXPANSION_FACTOR=1.55`, threshold `P(REAL)=0.3356796703127529` (`d=-0.682607114315033`).
- Face quality gate: SCRFD và YunNet đều loại bbox có width hoặc height `< DETECTOR_MIN_FACE_SIZE` (`60px`) trước khi tạo track.
- Cấu hình runtime được nạp từ `.env`; `.env.example` phải giữ cùng tên biến và giá trị mặc định tương ứng.

---

## 1. Tổng quan Kiến trúc Hệ thống (End-to-End Pipeline)

Hệ thống hiện tại hoạt động theo kiến trúc **Prototype V2** — Hệ thống xác thực khuôn mặt an toàn tích hợp mô-đun chống giả mạo (**Presentation Attack Detection - PAD / Anti-Spoofing**).

### 1.1 Sơ đồ luồng dữ liệu thời gian thực (Real-time Frame Pipeline)

```text
Camera Stream / Image Input
      ↓
[1. Face Detection] — SCRFD (InsightFace buffalo_s, det_size 640×640)
      ↓ Bounding Box (x1, y1, x2, y2) + 5-point Landmarks
[2. Face Tracking] — IOU Matching + Lucas–Kanade Optical Flow
      │
      ├────────────────── (Chưa tới hạn PAD / Recognition) ──────────────────┐
      │                                                                       │
      ↓ (Mỗi PAD_INTERVAL_SECONDS = 0.1s)                                     │
[3. Anti-Spoofing / PAD Module]                                               │
      │ ├─ Expanded Crop 1.55× quanh tâm mặt + BORDER_REFLECT_101            │
      │ ├─ E1 MobileNetV3: RGB + CelebA mean/std, gamma OFF                  │
      │ ├─ Model: mnv3_e1_preliminary_v5_1_best.onnx (3-class, 224×224)      │
      │ ├─ Decision: pad_score = real_logit - logsumexp(spoof_logits)         │
      │ └─ Temporal Smoothing: Rolling Window 5 votes (ngưỡng ≥ 60% SPOOF)    │
      │                                                                       │
      ├─ [SPOOF DETECTED] ──> Bỏ qua Recognition ──> Cập nhật visual: 🔴 SPOOF│
      │                                                                       │
      ↓ [VERDICT = REAL] (và đến hạn RECOGNIZE_INTERVAL_SECONDS = 0.5s)       │
[4. Face Alignment] — 5-point Affine Transformation -> Chuẩn 112×112          │
      ↓                                                                       │
[5. Feature Extraction] — ArcFace (buffalo_s) -> 512-D Embedding              │
      ↓ L2-Normalization                                                      │
[6. Vector Database Query] — PostgreSQL + pgvector (1:N Search)               │
      │ ├─ Connection Pooling (psycopg_pool)                                  │
      │ └─ Cosine Similarity (1 - <=>) chỉ trên Mean Centroid (is_mean=TRUE)  │
      ↓                                                                       │
[7. Decision Logic] — So sánh với MATCH_THRESHOLD (0.35)                      │
      ↓                                                                       │
[8. Visualization & State Machine] <──────────────────────────────────────────┘
      ├─ 🔴 SPOOF       (Phát hiện giả mạo màn hình / giấy in)
      ├─ 🟥 UNKNOWN     (Người lạ hoặc Similarity < 0.35)
      ├─ 🟧 RE-VERIFY   (Đang nhận diện lại sau 0.5s nhưng giữ tên cũ)
      ├─ 🟩 LOCKED      (Khớp danh tính trong DB, Similarity ≥ 0.35)
      └─ ⬜ PENDING     (Mặt mới xuất hiện, chờ kết quả lượt đầu)
```

### 1.2 Triết lý Thiết kế Cốt lõi (Core Principles)
1. **Edge-First Architecture (Inference tại biên):** Toàn bộ khâu tính toán nặng (Detection, PAD, Alignment, ArcFace) xử lý trực tiếp tại thiết bị biên (PC/Edge Device/Raspberry Pi). Dữ liệu gửi lên Cloud Database chỉ là 1 vector sinh trắc học 512-D (~2 KB), không truyền tải luồng video hay ảnh HD $\rightarrow$ độ trễ mạng cực thấp (~20–40ms RTT) và bảo mật dữ liệu riêng tư tuyệt đối.
2. **Nguyên tắc LOAD ONCE:** Tất cả model AI (SCRFD/YunNet, PAD ONNX, ArcFace) và Connection Pool DB phải được khởi tạo duy nhất 1 lần bên ngoài vòng lặp frame (`app.py`). Tuyệt đối không tạo lại object/session bên trong frame loop.
3. **Decoupled Time-Based Intervals:** Điều khiển chu kỳ re-verify nhận diện (`0.5s`), PAD (`0.1s`) và detector (`0.1s`) theo **thời gian thực (seconds)** thay vì đếm frame. Các frame xen kẽ dùng optical flow; detector chỉ chạy lại sớm khi track mất an toàn.

---

## 2. Cấu trúc Thư mục Kỹ thuật Cốt lõi (Codebase Tree)

Cấu trúc mã nguồn được phân rã thành các module kỹ thuật độc lập với ranh giới trách nhiệm (Separation of Concerns) rõ ràng:

```text
face_auth/
├── app.py                       # Main runtime camera realtime (5 visual states, pipeline coordinator)
├── config.py                    # Cấu hình tập trung toàn hệ thống (DB, Detector, PAD, Tracking, Thresholds)
├── docker-compose.yml           # Khởi tạo container PostgreSQL 16 + pgvector
├── requirements.txt             # Khai báo dependencies cốt lõi
├── .env                         # Biến môi trường kết nối DB và cấu hình runtime
│
├── detection/                   # Module trích xuất Bounding Box & 5-point Landmarks
│   ├── detector.py              # (Active Runtime) SCRFD FaceDetector (InsightFace buffalo_s)
│   ├── yunnet_detector.py       # YunNet FaceDetector (cv2.FaceDetectorYN, INT8, edge/legacy)
│   └── models/
│       └── detector_quantized.onnx # Trọng số YunNet INT8 (~122 KB)
│
├── antispoof/                   # Module Face Anti-Spoofing (PAD)
│   ├── predictor.py             # Class AntiSpoofPredictor: ONNX batch inference, LogSumExp PAD score
│   ├── preprocess.py            # Tiền xử lý: Expanded Crop 1.55×, Reflect Padding, E1 contract
│   ├── loader.py                # Quản lý khởi tạo ONNX InferenceSession (CPU/GPU auto-provider)
│   ├── system.py                # Tiện ích tra cứu phần cứng (CPU/GPU/Execution Provider)
│   ├── notebooks/diagnostics/   # Diagnostic deterministic cho model/crop/config PAD
│   │   └── diagnose_pad_runtime.py
│   ├── E1_v5_2_final_retrain_protocol.md # Protocol nghiên cứu/retrain E1
│   ├── antispoof_baseline_spatial_optimized_v5_1_time_balanced.ipynb # Training/evaluation notebook
│   └── models/                  # Checkpoints trọng số mô hình PAD
│       ├── mnv3_e1_preliminary_v5_1_best.onnx # E1 MobileNetV3 (224×224, 3-class) — Active runtime
│       ├── mnv3_e1_preliminary_v5_1_best.onnx.data # External ONNX tensor data
│       ├── mnv3_large_3class_best.onnx # MobileNetV3-Large (224×224, 3-class) — baseline
│       └── best_model_quantized.onnx   # MiniFASNetV2SE INT8 (128×128, 2-class) — legacy
│
├── alignment/                   # Module chuẩn hóa tư thế khuôn mặt
│   └── aligner.py               # Biến đổi Affine Partial 2D theo 5 điểm mốc về chuẩn 112×112
│
├── recognition/                 # Module trích xuất vector đặc trưng sinh trắc học
│   └── embedder.py              # ArcFace feature extractor (buffalo_s) -> 512-D L2-normalized
│
├── database/                    # Module lưu trữ & tìm kiếm vector
│   └── vector_db.py             # ConnectionPool PostgreSQL + pgvector, 1:N Cosine Similarity
│
├── tracking/                    # Module theo dõi đối tượng & Temporal Smoothing
│   └── tracker.py               # IOU + optical flow, cadence guard, PAD voting rolling window
│
├── enrollment/                  # Module nạp danh tính người dùng mới
│   └── enroll.py                # Thuật toán lọc outlier & tổng hợp Mean Centroid embedding
│
├── scripts/                     # Công cụ dòng lệnh hỗ trợ vận hành & kiểm thử
│   ├── run_enroll.py            # CLI quét thư mục gallery/ để đăng ký người dùng vào DB
│   ├── cleanup_demo.py          # CLI dọn dẹp các user thử nghiệm (prefix demo_lfw_*) khỏi DB
│   ├── demo_antispoof.py        # Demo độc lập tính năng PAD với webcam (SCRFD)
│   ├── demo_antispoof_yunnet.py # Demo độc lập tính năng PAD với webcam (YunNet)
│   ├── export_onnx.py           # Xuất checkpoint PyTorch sang ONNX format
│   ├── test_detection.py        # Benchmark FPS, landmarks và min-face filter của detector
│   ├── view_attendance.py       # CLI xem lịch sử check-in / check-out dạng bảng có format màu
│   └── delete_attendance_logs.py# CLI xóa lịch sử điểm danh gần nhất (hỗ trợ tham số thời gian & confirm)
│
├── evaluation/                  # Module benchmark & kiểm định thuật toán
│   ├── lfw_identity_benchmark.ipynb # Benchmark 1:N open-set identification & calibrate threshold
│   └── lfw_demo_enroll.ipynb        # Nạp dữ liệu giả lập LFW phục vụ demo
│
├── data/                        # Dataset chuẩn dùng để benchmark (LFW images & protocol CSVs)
├── gallery/                     # Thư mục chứa ảnh chân dung đăng ký người thật (3 ảnh/người)
└── md/
    └── current_codebase_state.md # Tài liệu Source of Truth hiện tại của dự án
```

---

## 3. Đặc tả Kỹ thuật Chi tiết Từng Module (Component Deep Dive)

### 3.1 Cấu hình Hệ thống (`config.py` & `.env`)

Các biến cấu hình được nạp từ file `.env` qua `python-dotenv`:

| Biến cấu hình | Mặc định | Ý nghĩa kỹ thuật |
|---|---|---|
| `DATABASE_URL` / `DB_CONN_INFO` | `face_db` | Connection string tới PostgreSQL (Local Docker hoặc Supabase/Neon). |
| `DB_POOL_MIN_SIZE` / `MAX_SIZE` | `1` / `4` | Số lượng kết nối tối thiểu/tối đa trong `psycopg_pool.ConnectionPool`. |
| `APP_DETECTOR` | `"scrfd"` | Detector chạy trong `app.py`: SCRFD là runtime hiện tại; YunNet là nhánh edge/legacy. |
| `DETECTOR_CONF_THRESH` | `0.5` | Ngưỡng tin cậy tối thiểu của bounding box khuôn mặt. |
| `DETECTOR_NMS_THRESH` | `0.3` | Ngưỡng Non-Maximum Suppression chống trùng lặp hộp nhận diện. |
| `DETECTOR_MIN_FACE_SIZE` | `60` px | Lọc bbox nhỏ hơn 60px theo cả width/height trước khi tạo track; áp dụng cho SCRFD và YunNet. |
| `DETECTOR_MARGIN` | `5` px | Lọc mặt cách mép khung hình $< 5$px; áp dụng cho YunNet. |
| `PAD_ENABLED` | `True` | Bật/tắt module Anti-Spoofing (có thể ghi đè bằng phím `p` hoặc CLI `--no-pad`). |
| `PAD_MODEL_FILENAME` | `"mnv3_e1_preliminary_v5_1_best.onnx"` | Model E1 tự train đang chạy trong runtime; ONNX dùng sidecar `.onnx.data`. |
| `PAD_THRESHOLD` | `0.3356796703127529` | Ngưỡng xác suất $P(REAL)$; tương đương `PAD_THRESHOLD_LOGIT=-0.682607114315033`. |
| `PAD_COLOR_ORDER` | `RGB` | E1 nhận RGB sau khi chuyển từ frame BGR. |
| `PAD_BBOX_EXPANSION_FACTOR` | `1.55` | Hệ số mở rộng crop vuông quanh bbox trước PAD. |
| `PAD_GAMMA_ENABLED` | `False` | E1/MobileNet 224 chạy gamma OFF theo preprocessing contract. |
| `PAD_GAMMA_TARGET` | `110.0` | Mức Luma mục tiêu (~43% dải sáng) cho phép biến đổi gamma. |
| `PAD_SMOOTH_WINDOW` | `5` | Kích thước rolling window (số lượt suy luận PAD dùng để biểu quyết). |
| `PAD_MIN_VOTES` | `5` | Số phiếu tối thiểu cần tích lũy trước khi track thoát khỏi trạng thái PAD_PENDING. |
| `PAD_SPOOF_MIN_RATIO` | `0.6` | Tỷ lệ phiếu SPOOF tối thiểu ($\ge 60\%$) trong window để kết luận là SPOOF. |
| `PAD_STALE_TIMEOUT_SECONDS` | `3.0` s | Ngưỡng quá hạn phán quyết PAD. Quá thời gian này, track quay về PAD_PENDING và reset window khi có inference mới. |
| `RECOGNIZE_INTERVAL_SECONDS` | `0.5` s | Chu kỳ thời gian giữa 2 lần re-verify danh tính của cùng một khuôn mặt. |
| `PAD_INTERVAL_SECONDS` | `0.1` s | Khoảng cách giữa 2 lần chạy PAD (`= RECOGNIZE_INTERVAL / PAD_SMOOTH_WINDOW`). |
| `DETECTION_INTERVAL_SECONDS` | `0.1` s | Khoảng cách detector tối thiểu; `0` là chế độ detect mỗi frame. |
| `TRACK_IOU_THRESHOLD` | `0.3` | Ngưỡng IOU tối thiểu để match bounding box qua các frame. |
| `TRACK_MAX_MISSING_FRAMES` | `10` | Số frame mất dấu liên tiếp trước khi xoá track khỏi bộ nhớ. |
| `MATCH_THRESHOLD` | `0.35` | Ngưỡng Cosine Similarity để chốt nhận diện (Calibrate từ thực nghiệm LFW). |
| `RECOMMENDED_GALLERY_SIZE`| `3` | Số lượng ảnh chân dung tối ưu khi đăng ký khuôn mặt mới vào hệ thống. |
| `ATTENDANCE_MODE` | `"checkin"` | Chế độ điểm danh mặc định của `app.py`: `"checkin"`, `"checkout"`, hoặc `"none"`. |
| `ATTENDANCE_GAP_MINUTES` | `15` phút | Khoảng cách thời gian tối thiểu giữa 2 lần cùng hành vi điểm danh của 1 người. |
| `ATTENDANCE_STABLE_COUNT` | `3` | Số lần nhận diện khớp ổn định liên tiếp cần thiết trước khi ghi log DB. |

> [!NOTE]
> **Quy ước Parse file `.env` (`config.py`):** Hàm `_clean_env_val()` sử dụng biểu thức chính quy `re.sub(r'\s+#.*$', '', val)` để chỉ loại bỏ phần chú thích (trailing comment) khi dấu `#` được tiền tố bằng khoảng trắng (`\s+#`). Cơ chế này đảm bảo giữ nguyên vẹn các chuỗi mật khẩu hoặc giá trị có chứa ký tự `#` (ví dụ `p@ss#word`).

---

### 3.2 Face Detection (`detection/`)

Hệ thống hỗ trợ 2 backends với chính sách phân chia rõ ràng:

1. **YunNet (`detection/yunnet_detector.py` — Nhánh edge/legacy):**
   - Dựa trên module tích hợp sẵn `cv2.FaceDetectorYN` của OpenCV kết hợp model `detector_quantized.onnx` (~122 KB, định dạng INT8).
   - Tự động gọi `setInputSize((frame_w, frame_h))` theo từng khung hình.
   - Trả về danh sách chuẩn:
     ```python
     [
         {
             "bbox": (x1, y1, x2, y2),
             "score": float,
             "landmarks": [[x, y], ...] # 5 điểm float theo chuẩn ArcFace
         }
     ]
     ```
   - *Quy ước 5 điểm Landmark:* `0`: Mắt phải của người (bên trái ảnh), `1`: Mắt trái của người (bên phải ảnh), `2`: Đỉnh mũi, `3`: Khóe miệng trái ảnh, `4`: Khóe miệng phải ảnh.
   - Lọc `min_face_size` và `margin` trước khi trả detection. Khi dùng YunNet trong `app.py`, camera được giới hạn bằng `cap.set(WIDTH, HEIGHT)` vì YunNet xử lý kém trên frame gốc quá lớn.

2. **SCRFD (`detection/detector.py` — Active Runtime):**
   - Submodel phát hiện khuôn mặt trích từ bộ `buffalo_s` của InsightFace.
   - Tự động co giãn ảnh thông qua tham số nội bộ `det_size=(640, 640)`, phù hợp cho camera full resolution và ảnh tĩnh.
   - Sau SCRFD inference, bbox bị loại nếu chiều rộng hoặc chiều cao nhỏ hơn `DETECTOR_MIN_FACE_SIZE` (mặc định 60px), tránh đưa mặt quá nhỏ vào tracker/PAD.

---

### 3.3 Face Anti-Spoofing / PAD (`antispoof/`)

Mục tiêu là phân biệt khuôn mặt thật sống (**Live / Real Face**) với các hình thức tấn công giả mạo (**Presentation Attacks**): ảnh in giấy (Print), mặt nạ (Mask), màn hình điện thoại/iPad/PC (Replay Attack).

#### Danh mục Model PAD hiện có trong `antispoof/models/`:
- `mnv3_e1_preliminary_v5_1_best.onnx`: MobileNetV3 E1 tự huấn luyện (224×224, 3 lớp: Real, Physical Spoof, Digital Spoof), là **runtime model hiện tại**; dùng external data sidecar `.onnx.data`.
- `mnv3_large_3class_best.onnx`: MobileNetV3-Large FP32 (224×224, 3 lớp), baseline spatial nghiên cứu.
- `best_model_quantized.onnx`: MiniFASNetV2SE INT8 (128×128, 2 lớp), checkpoint legacy/fallback.
- `mnv4_best_224.onnx`: MobileNetV4 FP32 (224×224, thử nghiệm nội bộ).

#### Quy trình Tiền xử lý PAD (`antispoof/preprocess.py`):
```text
BBox gốc (x1, y1, x2, y2)
    ↓
Expanded Square Crop 1.55× quanh tâm mặt
    + BORDER_REFLECT_101 (Điền phần tràn viền bằng ảnh phản chiếu, tránh viền đen làm hỏng texture)
    ↓
Adaptive Gamma Correction (E1 runtime: OFF; legacy/spatial profile có thể bật)
    ├─ luma = mean(V_channel trong HSV)
    ├─ gamma = log(110.0 / 255.0) / log(luma / 255.0), kẹp trong [0.4, 2.5]
    └─ Bảng tra cứu LUT 256 phần tử tính sẵn (nhanh hơn phép tính mũ từng pixel ~10×)
    ↓
Letterbox Resize về kích thước chuẩn (128×128 hoặc 224×224) + BORDER_REFLECT_101
    ↓
Normalize: Scale về [0.0, 1.0] -> Khử Mean/Std theo tập CelebA (đối với model 224px)
    ↓
Chuyển đổi HWC -> CHW, gom Batch (N, 3, H, W) float32 -> Nạp vào ONNX Runtime
```

- **Chuẩn hóa Bounding Box Định dạng `xyxy` (`antispoof/preprocess.py`):**
  - Hàm `crop()` đã loại bỏ hoàn toàn logic heuristic phỏng đoán định dạng (`if w > x and h > y...`) vốn dễ gây nhận diện sai lệch cho khuôn mặt nằm sát mép trái khung hình.
  - Toàn bộ pipeline thống nhất 100% định dạng `bbox = (x1, y1, x2, y2)` xuất từ cả 2 detector (YunNet và SCRFD). Hàm `crop()` trích xuất trực tiếp `w = x2 - x1`, `h = y2 - y1`, tính tâm mặt tại `center = (x1 + w/2, y1 + h/2)` chuẩn xác.
  - Runtime factor được cấu hình tập trung bằng `PAD_BBOX_EXPANSION_FACTOR=1.55`; `app.py`, `AntiSpoofPredictor` và diagnostic script dùng cùng giá trị.

> [!CAUTION]
> **RÀNG BUỘC PREPROCESSING CỦA RUNTIME E1**
> - **E1 MobileNetV3 (224×224):** dùng RGB, CelebA mean/std, crop 1.55× và `PAD_GAMMA_ENABLED=false`. Không tự ý bật gamma hoặc đổi channel order khi so sánh kết quả.
> - **Nhánh Frequency (DCT / Fourier sau này):** **TUYỆT ĐỐI KHÔNG ÁP DỤNG GAMMA CORRECTION**. Hàm lũy thừa phi tuyến $I^\gamma$ sẽ tạo sóng hài bậc cao giả tạo (harmonic artifacts) làm biến dạng hoàn toàn phổ năng lượng tự nhiên của vân Moiré màn hình và hạt mực in.

#### Quy ước Thứ tự Kênh Màu (Color-Order Contract):
- **Frame OpenCV gốc:** Cung cấp ảnh định dạng BGR.
- **Mô hình MiniFASNetV2SE (128×128):** Sử dụng thứ tự kênh `BGR` gốc của OpenCV, không chuyển đổi.
- **Mô hình MobileNetV3 / MobileNetV4 (224×224):** Huấn luyện trên CelebA-Spoof yêu cầu thứ tự kênh `RGB`. Luồng tiền xử lý chuyển BGR $\rightarrow$ RGB qua `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` sau bước gamma tùy profile và trước khi normalize; E1 runtime giữ gamma OFF.

#### Logic Quyết định Logit Tổng quát (`antispoof/predictor.py`):
Thay vì lấy Softmax argmax đơn thuần hoặc `max(spoof_logits)`, hệ thống sử dụng công thức LogSumExp chuẩn toán học:
$$\text{pad\_score} = z_{\text{real}} - \text{logsumexp}(z_{\text{spoof}})$$
Trong đó $\text{logit\_threshold} = \ln\left(\frac{p}{1 - p}\right)$, và kết luận `is_real = True` khi $\text{pad\_score} \ge \text{logit\_threshold}$.
- **Với mô hình 2 lớp:** $\text{logsumexp}([z_{\text{spoof}}]) = z_{\text{spoof}} \rightarrow \text{pad\_score} = z_{\text{real}} - z_{\text{spoof}}$ (tương đương 100% logic binary cũ).
- **Với mô hình 3 lớp:** $\text{pad\_score} = z_0 - \ln(e^{z_1} + e^{z_2})$, tương ứng chính xác về mặt giải tích với điều kiện xác suất Softmax $P_{\text{softmax}}(\text{REAL}) \ge p$.
- Điểm `pad_score` được tính toán ổn định số học thông qua hàm `_stable_logsumexp` (tránh tràn số khi $z$ lớn).
- Defensive error handling: nếu inference gặp lỗi (ví dụ session crash), trả về danh sách rỗng, không bao giờ fake kết quả `True`.

---

### 3.4 Face Tracking, Temporal Smoothing & PAD Gating (`tracking/tracker.py`)

- **IOU Matching:** So khớp Bounding Box giữa frame hiện tại và frame liền trước. Nếu `IOU >= 0.3`, khuôn mặt được gán vào `Track` có sẵn; ngược lại khởi tạo `Track` mới.
- **Detector Cadence:** `app.py` gọi SCRFD theo `DETECTION_INTERVAL_SECONDS=0.1s`. Nếu chưa đến hạn, tracker không gọi detector mà dùng Lucas–Kanade optical flow để propagate bbox/landmarks và đánh dấu `bbox_source=TRACKER`.
- **Redetection Safety Guard:** Detector được gọi sớm nếu chưa có track, bbox không hợp lệ, optical flow mất điểm, track ra ngoài frame hoặc scale thay đổi bất thường. Trạng thái mới từ detector (`detected`) được coi là hợp lệ để bắt đầu propagation, tránh lỗi detect lại mọi frame.
- **Nguồn bbox được giữ riêng:** Mỗi detection mang `bbox_source` (`DETECTOR` hoặc `TRACKER`) và `detector_called`, giúp phân biệt lỗi model PAD với lỗi crop/tracking khi chẩn đoán realtime.
- **Cơ chế PAD Gating & Trạng thái PAD_PENDING:**
  - Track mới khởi tạo hoặc vừa bật lại PAD có trạng thái `pad_status = "PAD_PENDING"` (`pad_ready = False`).
  - Khi PAD bật (`pad_enabled=True`), track **bị chặn tuyệt đối khỏi Recognition và Attendance** cho đến khi tích lũy đủ `pad_min_votes` (mặc định 5 phiếu).
  - Khi đủ `pad_min_votes`:
    * Nếu tỷ lệ SPOOF $\ge 60\% \rightarrow$ kết luận `SPOOF`, visual đổi sang 🔴 SPOOF, reset `stable_recognitions = 0`, tiếp tục chặn nhận diện và điểm danh.
    * Nếu tỷ lệ SPOOF $< 60\% \rightarrow$ kết luận `REAL` (`pad_ready = True, is_real = True`), mở cổng cho phép trích xuất đặc trưng ArcFace và tra cứu DB.
  - Khi người dùng bật lại PAD bằng phím `p`, toàn bộ track đang theo dõi được gọi `track.reset_pad()`, đưa về trạng thái `PAD_PENDING` an toàn.
- **Cơ chế Kiểm soát Độ tươi Phán quyết PAD (Freshness & Stale Timeout):**
  - Track ghi nhận mốc thời gian `last_pad_time` mỗi lần chạy PAD.
  - Thuộc tính `track.pad_ready` tự động trả về `False` nếu `(time.time() - last_pad_time) > PAD_STALE_TIMEOUT_SECONDS` (mặc định 3.0s), chuyển trạng thái hiển thị về `PAD_PENDING` và phong tỏa nhận diện + điểm danh.
  - Khi đối tượng quay lại hoặc frame mới gọi `update_pad()`, nếu phát hiện dữ liệu cũ đã quá hạn stale timeout, rolling window `_pad_window` sẽ tự động được xóa sạch (`clear()`) trước khi nạp vote mới, loại bỏ hoàn toàn nguy cơ lọt ảnh giả mạo từ dữ liệu cũ còn sót lại.
- **Đồng bộ nhịp thời gian giữa PAD và Nhận diện:**
  - `RECOGNIZE_INTERVAL_SECONDS = 0.5s`
  - `PAD_SMOOTH_WINDOW = 5`
  - `PAD_INTERVAL_SECONDS = 0.1s`
  - *Ý nghĩa:* Trong khoảng thời gian chờ nhận diện lại, module PAD chạy đều đặn tích lũy phiếu. Khi bước Recognition kích hoạt, rolling window vừa kịp tích lũy đủ 5 phiếu $\rightarrow$ phán quyết Real/Spoof đạt trạng thái ổn định.

---

### 3.5 Face Alignment & Recognition (`alignment/`, `recognition/`)

1. **Alignment (`alignment/aligner.py`):**
   - Sử dụng ma trận tọa độ đích tiêu chuẩn ArcFace 112×112 (`ARCFACE_DST_112`).
   - Ước lượng ma trận biến đổi afin cục bộ bằng `cv2.estimateAffinePartial2D` dựa trên 5 điểm landmark, sau đó gọi `cv2.warpAffine` để lấy ảnh khuôn mặt căn chỉnh thẳng đứng.
2. **ArcFace Feature Extractor (`recognition/embedder.py`):**
   - Nạp model từ pack `buffalo_s` của InsightFace với cờ `allowed_modules=["detection", "recognition"]` rồi bóc tách riêng submodel `recognition`.
   - Trích xuất vector đặc trưng 512-D bằng `get_feat(aligned_face)`.
   - Chuẩn hóa L2 ngay lập tức:
     $$\mathbf{v}_{\text{norm}} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2}$$
     Đưa bài toán so khớp Euclidean về phép tính tích vô hướng (Cosine Similarity).

---

### 3.6 Cơ sở dữ liệu Vector & Đăng ký (`database/`, `enrollment/`)

1. **Database Schema Đúng Ba Bảng Tối Giản (`database/schema.sql`, `database/vector_db.py`):**
   - Hệ thống dùng PostgreSQL 16 tích hợp extension `pgvector`.
   - **Bảng `employees`** (Danh tính nhân viên demo):
     - `employee_id TEXT PRIMARY KEY`, `full_name TEXT NOT NULL`.
     - `status TEXT NOT NULL DEFAULT 'ACTIVE' CHECK (status IN ('ACTIVE', 'INACTIVE'))`.
     - `created_at TIMESTAMPTZ DEFAULT NOW()`, `updated_at TIMESTAMPTZ DEFAULT NOW()`.
   - **Bảng `face_embeddings`** (Vector sinh trắc học):
     - `embedding_id BIGSERIAL PRIMARY KEY`.
     - `employee_id TEXT NOT NULL REFERENCES employees(employee_id) ON DELETE CASCADE`.
     - `embedding VECTOR(512) NOT NULL`.
     - `embedding_type TEXT NOT NULL CHECK (embedding_type IN ('SAMPLE', 'CENTROID'))`.
     - `model_version TEXT NOT NULL`.
     - `created_at TIMESTAMPTZ DEFAULT NOW()`.
   - **Bảng `attendance_logs`** (Lịch sử điểm danh):
     - `log_id BIGSERIAL PRIMARY KEY`.
     - `employee_id TEXT NOT NULL REFERENCES employees(employee_id) ON DELETE RESTRICT` (Ngăn xoá nhầm nhân viên khi đã có lịch sử điểm danh).
     - `action TEXT NOT NULL CHECK (action IN ('CHECKIN', 'CHECKOUT'))`.
     - `face_similarity REAL NULL`, `liveness_score REAL NULL`.
     - `timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()`.
     - **Composite Index tra cứu cooldown**:
       ```sql
       CREATE INDEX idx_attendance_lookup
       ON attendance_logs (employee_id, action, timestamp DESC);
       ```

2. **Chiến lược Tìm kiếm Realtime (Active Centroid):**
   - Câu lệnh SQL 1:N chỉ so khớp trên các vector thỏa mãn toàn bộ điều kiện:
     ```sql
     SELECT
         e.employee_id,
         e.full_name,
         1 - (f.embedding <=> %s::vector) AS similarity
     FROM face_embeddings f
     JOIN employees e ON f.employee_id = e.employee_id
     WHERE f.embedding_type = 'CENTROID'
       AND f.model_version = %s
       AND e.status = 'ACTIVE'
     ORDER BY f.embedding <=> %s::vector
     LIMIT %s;
     ```
   - *Lợi ích:* Đảm bảo tốc độ tra cứu trực tiếp, chỉ so khớp trên nhân viên `ACTIVE` và đúng phiên bản model đang chạy.

3. **Quy trình Re-enroll Gọn Gàng:**
   - Khi re-enroll, trong một database transaction duy nhất:
     1. Upsert thông tin nhân viên vào `employees`.
     2. Xoá toàn bộ embeddings cũ của nhân viên với cùng `model_version`: `DELETE FROM face_embeddings WHERE employee_id = %s AND model_version = %s;`
     3. Insert các `SAMPLE` embeddings mới.
     4. Insert 1 `CENTROID` mới.
   - Giữ nguyên 100% thuật toán loại outlier ($0.35$) và công thức L2-normalized mean centroid trong `enrollment/enroll.py`.

4. **Quyết định Danh tính (`decide_identity`):**
   - So sánh điểm số với `MATCH_THRESHOLD = 0.35`.
   - Nếu $\text{similarity} \ge 0.35 \rightarrow$ Trả về `(employee_id, full_name, float(best_similarity))`.
   - Nếu $\text{similarity} < 0.35 \rightarrow$ Trả về `(None, "UNKNOWN", float(best_similarity))`.

5. **Ghi nhận Điểm danh Cooldown (`log_attendance`):**
   - Kiểm tra cooldown qua cột `timestamp` trong `attendance_logs` theo khoảng thời gian `gap_minutes`.
   - Ghi nhận `face_similarity`, `liveness_score`, `timestamp` cho demo 1 tiến trình.

---

### 3.7 Main Runtime App (`app.py`)

Chương trình điều phối toàn bộ pipeline với 5 trạng thái hiển thị (Visual States):

| Trạng thái | Màu khung viền | Nhãn text | Điều kiện kích hoạt |
|---|---|---|---|
| 🔴 **SPOOF** | Đỏ tươi `(0, 0, 255)` | `⚠ SPOOF` | PAD bật và tỷ lệ phiếu giả mạo $\ge 60\%$. Bỏ qua nhận diện. |
| 🟥 **UNKNOWN** | Đỏ thẫm `(0, 0, 180)` | `UNKNOWN (0.xx)` | Không khớp ai trong DB hoặc Cosine Similarity $< 0.35$. |
| 🟧 **RE-VERIFY**| Cam `(0, 140, 255)` | `Tên (0.xx)` | Đã qua 0.5s, đang kích hoạt truy vấn ngầm nhưng giữ tên cũ. |
| 🟩 **LOCKED** | Xanh lá `(0, 255, 0)` | `[OK] Tên (0.xx)` | Nhận diện thành công, bám theo IOU tracker tốc độ tối đa. |
| ⬜ **PENDING** | Xám `(200, 200, 200)` | `Nhan dien...` | Khuôn mặt mới xuất hiện, đang chờ kết quả lượt đầu tiên. |

**Phím tắt điều khiển tại runtime:**
- `q` hoặc `ESC`: Thoát ứng dụng an toàn.
- `p`: Bật/Tắt module Anti-Spoofing (PAD) ngay lập tức mà không cần khởi động lại app.
- `f`: Bật/Tắt hiển thị thông số FPS trung bình trên màn hình.

**Tính năng Điểm danh (Attendance Tracking):**
- Ứng dụng hỗ trợ chạy ở 3 chế độ cấu hình qua `.env` (`ATTENDANCE_MODE`) hoặc cờ CLI `--mode`: `checkin`, `checkout`, hoặc `none` (mặc định: `checkin`).
- **Điều kiện kích hoạt & cơ chế bảo vệ chuỗi nhận diện:**
  - Khuôn mặt phải được xác nhận là người thật (`not is_spoof`), không ở trạng thái pending, đã có `employee_id` xác định (`track.employee_id` khác `None`), và đã hết thời gian cooldown (`track.can_log_attendance(...)`).
  - Đạt số lần nhận diện ổn định liên tiếp: `track.stable_recognitions >= ATTENDANCE_STABLE_COUNT` (file `.env` hiện tại dùng 2 lần; giá trị mặc định trong `config.py` là 3).
  - **Cơ chế chống gian lận & mất chuỗi:** Chuỗi ổn định sẽ lập tiếp bị reset về `0` nếu:
    1. Bị phát hiện SPOOF ở bất kỳ frame nào.
    2. Track bị mất dấu (`unmatched_tracks`), ví dụ đối tượng quay mặt đi, khuất hình hoặc mất focus.
    3. Nhận diện trả về `UNKNOWN` hoặc trượt ngưỡng `MATCH_THRESHOLD`.
  - **Reset trạng thái khi đổi người:** Khi track thay đổi `employee_id`, `last_attendance_time` tự động được reset về `None` để tránh chặn nhầm người mới.
  - **Đồng bộ thời gian đếm ngược & Bảo vệ Cooldown khi Thất bại:**
    - `track.last_attendance_time` **chỉ được cập nhật khi DB ghi nhận thành công** (`attendance_success is True`). Nếu việc ghi log DB thất bại (ví dụ timeout kết nối, lỗi tạm thời), cooldown local không bị tiêu thụ, cho phép hệ thống tự động thử lại (retry) ở các chu kỳ frame tiếp theo mà không khóa nhầm user.
    - Khi điểm danh thành công, DB trả về `last_ts` (timestamp chính xác trên server DB), app lưu `last_ts.timestamp()` vào tracker để bộ đếm ngược client hoạt động chính xác đồng bộ.
- **Phản hồi trực quan thời gian thực (HUD Feedback):**
  - Hiển thị góc trên trái: `MODE: CHECKIN` / `CHECKOUT` / `NONE`.
  - Điểm danh thành công: Hiển thị banner xanh lá `"{Tên}: {MODE} SUCCESS"` trong 3 giây.
  - Điểm danh bị chặn do cooldown: Hiển thị banner cảnh báo màu cam `"{Tên}: {Lý do}"` trong 3 giây trên màn hình (ví dụ: `Đã CHECKIN gần đây. Vui lòng đợi 15 phút.`).

---

## 4. Trạng thái Hiện tại của Mã nguồn & Nghiên cứu PAD

### 4.1 Bảng Ma trận Tính năng (Status Matrix)

| Thành phần | Trạng thái | Đánh giá / Ghi chú |
|---|---|---|
| **Face Detection (SCRFD)** | ✅ STABLE | Detector active của app/enrollment; det-size 640, filter min face 60px. |
| **Face Detection (YunNet)** | ✅ STABLE | Nhánh edge/legacy, INT8 ONNX ~122 KB, có min-face và margin filter. |
| **Face Tracking Cadence** | ✅ STABLE | SCRFD theo interval 0.1s; optical flow xử lý frame xen kẽ, redetect khi tracker không an toàn. |
| **Face Alignment (Affine 112×112)** | ✅ STABLE | Căn chỉnh chuẩn xác theo 5 landmark của ArcFace. |
| **Face Embedding (ArcFace 512-D)** | ✅ STABLE | L2-normalized vector, trích xuất chuẩn xác. |
| **Vector DB (Postgres + pgvector)**| ✅ STABLE | Connection pool ổn định, Mean centroid search tối ưu tài nguyên. |
| **Attendance Tracking & DB Cooldown**| ✅ STABLE | Chế độ checkin/checkout/none, composite B-Tree index, parameterized query, streak reset khi mất focus/spoof. |
| **Identity Enrollment Pipeline** | ✅ STABLE | Tự động loại outlier, nạp thư mục `gallery/` tự động. |
| **PAD Realtime Runtime Integration**| ✅ STABLE | Tích hợp trong `app.py`, batching ONNX, temporal smoothing chống rung. |
| **PAD Preprocessing & Gamma** | ✅ STABLE | E1 dùng crop 1.55×, RGB, mean/std, reflect padding, gamma OFF. |
| **PAD Production Model** | ✅ STABLE | E1 MobileNetV3 ONNX 224×224, 3-class, threshold đã calibrate. |
| **PAD Legacy/Fallback Models** | ✅ STABLE | MiniFASNetV2SE 128×128 và các baseline 224×224 vẫn giữ để đối chiếu. |
| **PAD Frequency Branch (2D-DCT)** | ⏳ PLANNED | Mục tiêu của giai đoạn Prototype V3 (Ablation study). |

---

### 4.2 Trạng thái & Vấn đề Cần Khắc phục trong Nghiên cứu PAD (research notebooks/protocols)

Quy trình nghiên cứu và huấn luyện mô hình Spatial Baseline (MobileNetV3-Large trên CelebA-Spoof) hiện có các điểm kỹ thuật cốt lõi cần chuẩn hóa:

1. **Lỗi Rò rỉ Dữ liệu Validation (Data Leakage - CRITICAL):**
   - *Hiện trạng:* Trong các artifact nghiên cứu/retrain E1, cần kiểm tra biến `VAL_JSON` không trỏ thẳng vào `metas/intra_test/test_label.json`. Tập test chính thức không được dùng để chọn `best_checkpoint` mỗi epoch.
   - *Khắc phục:* Phải tách một tập Validation con (stratified ~10% từ `train_label.json`), giữ nguyên `test_label.json` làm Held-out Test Set độc lập.
2. **Tiêu chí Chọn Checkpoint Tối ưu:**
   - *Hiện trạng:* Code cũ đang lưu model theo Accuracy cao nhất.
   - *Khắc phục:* Với bài toán PAD có mất cân bằng dữ liệu lớn (Real 33% vs Spoof 67%), bắt buộc phải chọn checkpoint dựa trên **Validation ACER thấp nhất** ($\text{ACER} = (\text{APCER} + \text{BPCER}) / 2$).
3. **Chính sách Augmentation (Mild Policy):**
   - Không dùng các phép biến đổi phá hủy cấu trúc tần số cao như `GaussianBlur` nặng hay nén JPEG nát ảnh. Chỉ dùng Mild Augmentation: lật ngang ($p=0.5$), co giãn nhẹ (0.9–1.0), xoay nhẹ ($\pm 5^\circ$), biến thiên độ sáng/tương phản ($\pm 10\%$).
4. **Đồng bộ Lịch trình Học (Scheduler Mismatch):**
   - Đảm bảo `T_max` trong `CosineAnnealingLR` khớp chính xác với tổng số epoch thực tế huấn luyện (tránh lỗi $T_{\max}=10$ nhưng chạy 20 epochs).

---

## 5. Hướng dẫn Thao tác Thực tế & Lệnh Vận hành

### 5.1 Khởi tạo Môi trường & Cơ sở Dữ liệu
```bash
# 1. Khởi động PostgreSQL + pgvector qua Docker
docker-compose up -d

# 2. Cài đặt thư viện phụ thuộc
pip install -r requirements.txt

# 3. Khởi tạo / Reset Database sạch (3 bảng)
python scripts/reset_database.py --reset

# 4. Đăng ký nhân viên (Chuẩn bị ảnh trong gallery/<employee_id>/)
python scripts/run_enroll.py --gallery gallery
```

### 5.2 Khởi chạy Ứng dụng Realtime
```bash
# Chạy mặc định (PAD bật, Detector theo .env, FPS bật, Mode từ .env)
python app.py

# Chạy với các chế độ điểm danh:
python app.py --mode checkin           # Chế độ Check-in (Mặc định)
python app.py --mode checkout          # Chế độ Check-out
python app.py --mode none              # Chế độ quan sát / chỉ nhận diện (không ghi log)

# Chạy với các tùy chọn dòng lệnh khác:
python app.py --no-pad                 # Tắt module Anti-Spoofing
python app.py --pad-model best_model_quantized.onnx  # Đổi sang model PAD 128px INT8
python app.py --pad-threshold 0.6      # Nâng ngưỡng khắt khe cho PAD
python app.py --no-fps                 # Tắt hiển thị FPS overlay
python app.py --camera 1               # Đổi index camera ngoài

# Xem lịch sử điểm danh (hỗ trợ phân màu & giới hạn số lượng):
python scripts/view_attendance.py --limit 50

# Xóa lịch sử điểm danh gần nhất phục vụ kiểm thử (hỗ trợ confirm an toàn):
python scripts/delete_attendance_logs.py --minutes 5
python scripts/delete_attendance_logs.py --minutes 10 --yes  # Bỏ qua xác nhận y/n
```

### 5.3 Chạy Kiểm thử & Đánh giá Độc lập
```bash
# Chạy toàn bộ test suite
python3 -m unittest discover -s tests -p "test_*.py" -v

# Đo benchmark tốc độ và filter min-face của detector active
python scripts/test_detection.py --detector scrfd

# Diagnostic PAD deterministic (không tracker/temporal smoothing)
python antispoof/notebooks/diagnostics/diagnose_pad_runtime.py \
  --image evaluation/lfw_demo_test_images/Claudia_Pechstein/Claudia_Pechstein_0001.jpg

# Ghi log camera để kiểm tra bbox nhỏ, detector/tracker source và PAD score
PAD_DIAGNOSTIC_LOG=true python app.py --mode none 2>&1 | tee output/pad_camera_diagnostic.log

# Chạy demo PAD với detector active SCRFD
python scripts/demo_antispoof.py

# Demo YunNet edge/legacy (chỉ dùng khi cần đối chiếu)
python scripts/demo_antispoof_yunnet.py

# Dọn dẹp dữ liệu người dùng thử nghiệm
python scripts/cleanup_demo.py --dry-run

```

---

## 6. Lộ trình Kế tiếp (Next Steps / To-Do List)

1. **Hoàn thiện protocol/notebook Training PAD Spatial Baseline (E1 v5.2):**
   - Áp dụng Stratified split (~10% từ train làm validation), sửa validation loop chọn checkpoint bằng ACER và thiết lập Mild augmentation.
2. **Đánh giá trên Held-Out Test Set:**
   - Đánh giá mô hình sau khi train trên tập test chính thức của CelebA-Spoof (tính toán đầy đủ: APCER, BPCER, ACER, AUC).
3. **Theo dõi provenance và calibration của E1:**
   - Quét ngưỡng `logit_diff` trên tập validation để đạt operating point mong muốn (ví dụ $\text{BPCER} \le 5\%$).
   - Lưu hash của model ONNX và sidecar `.onnx.data` cùng config, preprocessing contract và log đánh giá; chỉ thay model mặc định sau khi có kết quả held-out tương ứng.
4. **Phát triển Nhánh Tần số V3 (2D-DCT Frequency Branch):**
   - Thiết kế khối trích xuất đặc trưng DCT 2D song song trên ảnh raw (không qua Adaptive Gamma) để thực hiện bài toán so sánh độ lệch hiệu năng (Ablation study).

---

## 7. Nhật ký Phiên làm việc (Session Changelog)

| Ngày | Người thực hiện | Nội dung cập nhật chính |
|---|---|---|
| **2026-09-15** | Agent & Dev | Khởi tạo tài liệu Master Context `current_codebase_state.md`, đồng bộ toàn diện kiến trúc prototype, tích hợp Adaptive Gamma, đặc tả chi tiết 8 module cốt lõi và các issue trong nghiên cứu PAD. |
| **2026-09-15** | Agent & Dev | Triển khai hoàn chỉnh khối Điểm danh (Attendance Tracking): tạo bảng `attendance_logs` kèm Composite Index `idx_attendance_lookup (employee_id, action, timestamp DESC)`, chuyển toàn bộ truy vấn sang parameterized query với `make_interval`. Xử lý các edge cases trong tracker (reset streak khi mất focus hoặc dính spoof, reset cờ log khi đổi user). Bổ sung cấu hình linh hoạt trong `.env` (`ATTENDANCE_MODE`, `RECOGNIZE_INTERVAL_SECONDS`, ...), banner HUD trực quan trên camera, hoàn thiện script vận hành. |
| **2026-09-15** | Agent & Dev | Refactor kiến trúc điểm danh: Chuyển đổi cờ `has_logged_attendance` thành `last_attendance_time` kết hợp `can_log_attendance(gap_minutes)`. Đồng bộ giá trị `last_ts` từ DB về tracker để quản lý Local Cooldown Timer chính xác, hỗ trợ đếm ngược ngay cả khi ứng dụng bị ngắt và khởi động lại, cho phép người dùng tự động được điểm danh lại sau khi quá hạn interval. |
| **2026-09-17** | Agent & Dev | **Hoàn thiện refactor và chuẩn hóa face_auth:**<br>1. **PAD Predictor & Preprocess:** Chuẩn hóa công thức LogSumExp tổng quát $z_{\text{real}} - \text{logsumexp}(z_{\text{spoof}})$ cho cả mô hình 2 lớp và 3 lớp (tương đương chính xác $P_{\text{softmax}}(\text{REAL}) \ge p$). Ràng buộc chuẩn hóa kênh màu `convert_rgb` (BGR cho MiniFASNet, RGB cho MobileNet).<br>2. **Tracker PAD Gating:** Thiết lập trạng thái `PAD_PENDING`, chặn nhận diện và điểm danh trước khi tích lũy đủ `pad_min_votes=5`. Reset trạng thái PAD khi toggle phím `p`. Chuẩn hóa API sử dụng `employee_id`.<br>3. **Schema 3 bảng sạch & Re-enrollment transaction:** Đổi bảng sang `employees`, tìm kiếm theo active centroid và model_version, re-enroll xóa embedding cũ cùng model_version rồi insert trong 1 transaction an toàn.<br>4. **Scripts & Notebooks:** Đồng bộ toàn bộ scripts vận hành và notebook LFW (`lfw_demo_enroll.ipynb`, `lfw_identity_benchmark.ipynb`) với seed `42` bảo toàn. |
| **2026-09-17** | Agent & Dev | **Củng cố độ tin cậy và xử lý 4 rủi ro runtime / bảo mật cốt lõi:**<br>1. **PAD Stale Timeout:** Bổ sung cấu hình `PAD_STALE_TIMEOUT_SECONDS` (3.0s). `pad_ready` tự động trả về `False` khi phán quyết quá hạn. `update_pad()` chủ động reset window cũ trước khi nhận vote mới, ngăn chặn giả mạo khi đối tượng quay lại.<br>2. **Attendance Cooldown Guard:** Chỉ cập nhật `last_attendance_time` khi ghi nhận điểm danh DB thành công (`attendance_success is True`), tránh việc lỗi DB tạm thời làm tiêu hao nhầm cooldown khiến người dùng bị khóa điểm danh.<br>3. **BBox Format Standardization:** Xóa hoàn toàn heuristic phỏng đoán format trong `crop()`, thống nhất 100% định dạng `(x1, y1, x2, y2)` từ các detector.<br>4. **Bảo toàn Ký tự `#` trong .env:** Chuyển `_clean_env_val()` sang regex `\s+#.*$` để chỉ cắt bỏ trailing comment khi `#` đi liền sau khoảng trắng, bảo toàn nguyên vẹn mật khẩu hoặc giá trị có chứa `#`. |
| **2026-09-24** | Agent & Dev | **Đồng bộ runtime E1 và tối ưu pipeline realtime:** chuyển app sang SCRFD active runtime; thêm detector cadence `0.1s` với optical-flow propagation và redetection safety guard; sửa lỗi trạng thái `detected` khiến detector chạy mọi frame; chuẩn hóa E1 ONNX/RGB/gamma OFF/threshold calibrated; đổi PAD crop expansion lên `1.55x`; áp dụng `DETECTOR_MIN_FACE_SIZE=60px` cho cả SCRFD và YunNet; bổ sung tests và backup runtime trước các thay đổi. |
