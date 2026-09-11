# Prototype V2 — Secure Face Authentication with Anti-Spoofing (PAD)

## 1. Mục tiêu V2

V2 chèn module **Presentation Attack Detection (PAD / Anti-Spoofing)** vào trước bước Alignment trong pipeline, đồng thời thực hiện hàng loạt tối ưu hóa hạ tầng so với V1:

- Thay thế SCRFD bằng **YunNet** (detector siêu nhẹ, tích hợp sẵn trong OpenCV, không cần InsightFace runtime).
- Tích hợp module `antispoof/` với thiết kế **temporal smoothing** chống rung REAL/SPOOF giữa các frame.
- Nghiên cứu và huấn luyện mô hình PAD riêng trên dataset **CelebA-Spoof** (561K ảnh).
- Hỗ trợ toggle PAD bật/tắt qua CLI và phím tắt runtime mà không cần khởi động lại app.

Pipeline V2:

```text
Camera / Image
    ↓
Face Detection — YunNet (cv2.FaceDetectorYN, INT8, ~122 KB)
    ↓
Face Tracking (IOU Matching)
    ↓ (mỗi PAD_INTERVAL_SECONDS, mặc định 0.2s)
┌───────────────────────────────────────┐
│  Anti-Spoofing / PAD Module           │
│  Input: expanded crop × 1.5           │
│  ↓ Adaptive Gamma Correction          │  ← NEW: bật/tắt qua PAD_GAMMA_ENABLED
│  Model: [đang tối ưu — xem §3.2]     │
│  Temporal Smoothing (rolling window)  │
│  Verdict: REAL / SPOOF                │
└──────────────┬────────────────────────┘
               │ REAL only
               ↓ (mỗi RECOGNIZE_INTERVAL_SECONDS, mặc định 1.0s)
Face Crop & Alignment (5-point affine, 112×112)
               ↓
Face Embedding — ArcFace (buffalo_s, 512-D)
               ↓
L2 Normalization
               ↓
PostgreSQL + pgvector
               ↓
Top-K Similarity Search (mean_only=True)
               ↓
Threshold Decision (cosine ≥ 0.35)
               ↓
Identity / UNKNOWN
```

---

## 2. Kiến trúc module (V2)

```text
face_auth/
│
├── app.py                       # Main runtime — 5 visual states, PAD + Recognition pipeline
├── config.py                    # Cấu hình tập trung (PAD, tracking, temporal smoothing)
├── requirements.txt
├── docker-compose.yml
├── .env
│
├── detection/
│   ├── __init__.py
│   ├── detector.py              # Wrapper SCRFD cũ (giữ tương thích ngược)
│   ├── yunnet_detector.py       # ★ NEW — YunNet FaceDetector (thay thế chính)
│   └── models/
│       └── detector_quantized.onnx   # YunNet INT8 (~122 KB)
│
├── alignment/
│   ├── __init__.py
│   └── aligner.py               # 5-point affine → 112×112 (giữ nguyên từ V1)
│
├── recognition/
│   ├── __init__.py
│   └── embedder.py              # ArcFace (buffalo_s) — 512-D L2-normalized (giữ nguyên)
│
├── antispoof/                   # ★ NEW — toàn bộ module PAD
│   ├── __init__.py              # Export AntiSpoofPredictor, crop, preprocess, ...
│   ├── predictor.py             # AntiSpoofPredictor — ONNX batch inference + logit decision
│   ├── preprocess.py            # crop() 1.5× + BORDER_REFLECT_101, preprocess(), preprocess_batch()
│   ├── loader.py                # load_model() — ONNX Runtime với GPU/CPU auto-detect
│   ├── system.py                # get_cpu_info(), get_gpu_info(), get_execution_provider_name()
│   ├── README.md
│   ├── compare_model.md         # Bảng so sánh: MiniFASNetV2SE vs MNV3 / YunNet vs SCRFD
│   ├── antiproof.ipynb          # Training notebook — MNV3-Large spatial baseline
│   ├── models/
│   │   ├── best_model_quantized.onnx      # MiniFASNetV2SE INT8, 128×128, 2-class (~600 KB) — production default
│   │   ├── mnv3_large_3class_best.onnx    # MobileNetV3-Large FP32, 224×224, 3-class — tự train
│   │   └── mnv4_best_224.onnx             # MobileNetV4 FP32, 224×224 — experimental
│   └── notebooks/
│       ├── eda/
│       │   ├── celeb-a-spoof-eda.ipynb    # EDA notebook chính
│       │   ├── celeba_spoof_eda_summary.json
│       │   └── celeba_spoof_sample_gallery.png
│       └── quick_test/
│
├── database/
│   ├── __init__.py
│   └── vector_db.py             # PostgreSQL + pgvector (giữ nguyên từ V1)
│
├── enrollment/
│   ├── __init__.py
│   └── enroll.py                # Centroid mean embedding + outlier filtering (giữ nguyên)
│
├── tracking/
│   ├── __init__.py
│   └── tracker.py               # ★ UPDATED — Track class bổ sung PAD temporal smoothing
│
├── utils/
│   ├── __init__.py
│   └── image.py
│
├── scripts/
│   ├── run_enroll.py
│   ├── cleanup_demo.py
│   ├── demo_antispoof.py        # ★ NEW — Demo PAD standalone với webcam
│   ├── demo_antispoof_yunnet.py # ★ NEW — Demo PAD + YunNet detector
│   ├── export_onnx.py           # ★ NEW — Xuất checkpoint PyTorch → ONNX
│   └── test_detection.py        # ★ NEW — Script kiểm tra face detection pipeline
│
├── evaluation/
│   ├── __init__.py
│   ├── lfw_identity_benchmark.ipynb
│   └── lfw_demo_enroll.ipynb
│
├── data/
│   ├── lfw/
│   └── lfw_metadata/
│
├── benchmark/                   # ★ NEW — cache kết quả benchmark
│
└── md/
    ├── face_recognition_prototype_v1.md
    └── face_recognition_prototype_v2.md   # ← file này
```

---

## 3. Model cụ thể trong V2

### 3.1 Face Detection: YunNet (thay thế SCRFD)

**Thay đổi so với V1:** Chuyển từ SCRFD (InsightFace, ~570 KB) sang **YunNet** (`cv2.FaceDetectorYN`, ~122 KB).

| Tiêu chí | YunNet | SCRFD-0.5GF |
|---|---|---|
| Params | ~75K | ~570K |
| WIDERFace Hard | **75.03** | 68.51 |
| WIDERFace Easy/Medium | 88.44 / 86.56 | **90.57 / 88.12** |
| Landmark | Có (5 điểm) | Có (5 điểm) |
| Dependency | Tích hợp sẵn `opencv-python` | Cần InsightFace + ONNX Runtime |
| File model | ~122 KB (INT8) | ~570 KB |
| Edge phù hợp | ✅ Rất phù hợp (Pi/Jetson) | ⚠️ Chậm hơn, thêm dependency |

Interface `detect()` giữ **nguyên hoàn toàn** so với SCRFD cũ (tương thích ngược với `aligner.py`, `tracker.py`, `app.py`):

```python
# Output format (giống SCRFD):
[
    {
        "bbox": (x1, y1, x2, y2),     # xyxy, int
        "score": float,                 # confidence
        "landmarks": [[x,y], ...]       # 5 điểm float, thứ tự ArcFace
    },
    ...
]
```

Thứ tự 5 điểm landmark (ArcFace convention):
- `0`: mắt bên trái ảnh (right eye of subject)
- `1`: mắt bên phải ảnh (left eye of subject)
- `2`: đỉnh mũi
- `3`: khóe miệng trái ảnh
- `4`: khóe miệng phải ảnh

Cấu hình mặc định trong `config.py`:

```python
DETECTOR_INPUT_SIZE    = (320, 320)  # khởi tạo — tự động resize theo frame thực tế
DETECTOR_CONF_THRESH   = 0.5
DETECTOR_NMS_THRESH    = 0.3
DETECTOR_TOP_K         = 5000
DETECTOR_MIN_FACE_SIZE = 60          # px — bỏ qua mặt nhỏ (người hậu cảnh xa)
DETECTOR_MARGIN        = 5           # px — bỏ qua mặt sát mép ảnh
```

### 3.1.1 Detector Selection Policy

> Vấn đề cốt lõi: YunNet (cv2.FaceDetectorYN) hoạt động kém với ảnh lớn nếu không resize trước. SCRFD tự resize nội bộ qua `det_size` nên không có vấn đề này.

| Detector | Use-case | Frame size | Điều kiện |
|----------|----------|------------|----------|
| **YunNet** | `app.py` — edge inference | Cần giới hạn qua `cap.set()` | CPU yếu (Pi, Jetson), ưu tiên FPS |
| **SCRFD** | Enroll, eval, demo — offline | Full resolution, không giới hạn | CPU mạnh, cần độ chính xác cao hơn |

**Cấu hình qua `.env`:**

```dotenv
APP_DETECTOR=yunnet   # "yunnet" (default) hoặc "scrfd"
```

- `yunnet`: App.py giới hạn camera qua `cap.set(FRAME_WIDTH, CAMERA_WIDTH)` và `cap.set(FRAME_HEIGHT, CAMERA_HEIGHT)`.
- `scrfd`: App.py **không** set WIDTH/HEIGHT — camera chạy full native resolution; SCRFD tự resize nội bộ qua `DETECTOR_DET_SIZE = (640, 640)`. Phù hợp máy tính mạnh.

Các file khác theo **ưu tiên SCRFD**:

```text
run_enroll.py          → SCRFD  (offline batch, ảnh lớn bất kỳ)
scripts/demo_antispoof.py      → SCRFD
scripts/test_detection.py     → SCRFD (default), có flag --detector yunnet
evaluation/*.ipynb             → SCRFD
app.py                         → theo APP_DETECTOR (.env)
```

### 3.2 Anti-Spoofing / PAD Module (đang tối ưu)

> **Lưu ý:** Module PAD đang trong giai đoạn nghiên cứu và tối ưu. Kiến trúc model cuối cùng chưa được chốt. Phần này mô tả trạng thái hiện tại của hạ tầng inference và các model đang thử nghiệm, **không** là thiết kế nghiên cứu chính thức.

#### Model hiện có trong `antispoof/models/`

| File | Kiến trúc | Size | Input | Classes | Trạng thái |
|---|---|---|---|---|---|
| `best_model_quantized.onnx` | MiniFASNetV2SE INT8 | ~600 KB | 128×128 | 2-class | Production default (download) |
| `mnv3_large_3class_best.onnx` | MobileNetV3-Large FP32 | ~16 MB | 224×224 | 3-class | Tự train — spatial baseline |
| `mnv4_best_224.onnx` | MobileNetV4 FP32 | ~5 MB | 224×224 | 2-class | Tự train — experimental |

**Model production hiện tại** là **MiniFASNetV2SE INT8** — model download, không phải model research tự train. Dùng để hoàn thiện inference pipeline.

**Model research** (mục tiêu ablation) là **MobileNetV3-Large spatial baseline** (1 branch RGB spatial), đang tối ưu training protocol.

#### Hướng nghiên cứu (chưa chốt)

```text
Spatial Baseline (hiện tại đang xây dựng):
    MobileNetV3-Large → 3-class Classifier
    Input: 224×224 RGB, expanded crop ×1.5

Proposed (dự kiến, chưa implement):
    MobileNetV3-Large + DCT Frequency Branch → Fusion → Classifier
```

Câu hỏi nghiên cứu:
> Frequency-aware features có cung cấp thông tin bổ sung cho spatial representation trong Face PAD, đặc biệt khi đánh giá cross-domain?

#### Thiết kế 3-class (model tự train)

Training trên CelebA-Spoof với 3 label:
- **Class 0**: Real Face (mặt thật sống)
- **Class 1**: Physical Spoof (in giấy, mask vật lý)
- **Class 2**: Digital Spoof (màn hình điện thoại/máy tính/tablet)

Logic quyết định tại inference (`predictor.py::process_with_logits`):

```python
# 3-class model:
real_logit  = raw_logits[0]
spoof_logit = max(raw_logits[1:])   # lấy max của Physical + Digital
logit_diff  = real_logit - spoof_logit
is_real     = logit_diff >= logit_threshold
```

#### Tiền xử lý PAD (đồng nhất training ↔ inference)

```text
bbox (x1, y1, x2, y2)
    ↓
Expanded square crop × 1.5 (preprocess.py::crop())
    + BORDER_REFLECT_101 cho vùng tràn mép ảnh
    ↓
Adaptive Gamma Correction (preprocess.py::adaptive_gamma())   ← NEW
    — bật/tắt qua PAD_GAMMA_ENABLED; target luma qua PAD_GAMMA_TARGET
    — gamma tính từ luma kênh V (HSV); clamp trong [0.4, 2.5]
    — skip tự động nếu |gamma − 1| < 0.05 (zero cost)
    ↓
Letterbox resize về model_img_size (128 hoặc 224)
    + BORDER_REFLECT_101
    ↓
Normalize [0, 255] → [0.0, 1.0]
    + mean/std nếu cần (CelebA mean/std cho 224px model)
    ↓
HWC → CHW, float32
    ↓
ONNX Runtime inference
```

**Lý do crop ×1.5:** Model có thể học context ngoài vùng mặt (viền màn hình, mép giấy, bezel thiết bị, tóc/tai), đây là những cue quan trọng để phân biệt spoof.

### 3.3 Face Alignment & Recognition (giữ nguyên từ V1)

Không thay đổi:
- **Alignment:** 5-point affine → 112×112 (`alignment/aligner.py`)
- **Embedding:** ArcFace `buffalo_s` → 512-D, L2-normalized (`recognition/embedder.py`)
- **Database:** PostgreSQL + pgvector, mean centroid strategy (`database/vector_db.py`)

---

## 4. Dataset PAD: CelebA-Spoof

### 4.1 Tổng quan

**CelebA-Spoof** là dataset Face Anti-Spoofing quy mô lớn, dùng làm primary training/evaluation dataset cho module PAD trong V2.

| Thống kê | Giá trị |
|---|---|
| Tổng ảnh | **561,575** |
| Train | 494,405 ảnh |
| Test | 67,170 ảnh |
| Train subjects | 8,192 |
| Test subjects | 1,004 |
| Subject overlap (train/test) | 3 (gần như disjoint) |

### 4.2 Phân bố lớp — EDA

#### Train set

| Class | Số ảnh | Tỉ lệ |
|---|---|---|
| **0: Real Face** | 162,462 | 32.9% |
| **1: Physical Spoof** | 226,715 | 45.9% |
| **2: Digital Spoof** | 105,228 | 21.3% |

**Mất cân bằng đáng kể** (Real 32.9% vs Spoof 67.1%). Cần xử lý class imbalance khi training (weighted loss hoặc weighted sampler).

#### Test set

| Class | Số ảnh | Tỉ lệ |
|---|---|---|
| **0: Real Face** | 19,923 | 29.7% |
| **1: Physical Spoof** | 35,495 | 52.8% |
| **2: Digital Spoof** | 11,752 | 17.5% |

### 4.3 Phân bố loại tấn công (Attack Type)

#### Train

| Attack Type | Số ảnh |
|---|---|
| 0: Live (Real) | 162,462 |
| 1: Poster | 35,547 |
| 2: A4 Paper | 31,221 |
| 3: Face Mask | 31,776 |
| 4: Upper Body Mask | 33,647 |
| 5: Region Mask | 30,167 |
| 6: PC Screen | 33,285 |
| 7: Tablet/Pad Screen | 31,072 |
| 8: Phone Screen | 33,085 |
| 9: 3D Mask | 31,527 |
| 10: (khác) | 40,616 |

#### Test

| Attack Type | Số ảnh |
|---|---|
| 0: Live (Real) | 19,923 |
| 1: Poster | 3,600 |
| 2: A4 Paper | 5,421 |
| 3: Face Mask | 6,083 |
| 4: Upper Body Mask | 4,287 |
| 5: Region Mask | 6,097 |
| 6: PC Screen | 3,530 |
| 7: Tablet/Pad Screen | 6,477 |
| 8: Phone Screen | 3,659 |
| 9: 3D Mask | 4,483 |
| 10: (khác) | 3,610 |

### 4.4 Nhận xét EDA

- Dataset đủ lớn và đa dạng: bao phủ cả **physical attacks** (in giấy, mặt nạ vật lý) lẫn **digital attacks** (màn hình PC/tablet/điện thoại).
- 3D Mask (~31K train) làm dataset này khó hơn nhiều so với các dataset chỉ có print attack.
- Train/test gần như **subject-disjoint** (3 subject overlap) — đánh giá intra-dataset có ý nghĩa thực sự.
- **Không được dùng `test_label.json` làm validation trong training loop** — phải tách stratified validation subset từ train set (lỗi đã phát hiện trong notebook gốc, đang sửa).

### 4.5 Protocol training (đang áp dụng)

```text
CelebA-Spoof train.json
    ↓
Stratified split: ~90% actual train / ~10% validation
    (stratified theo label, hoặc tách theo subject nếu có thể)
    ↓
Training với mild augmentation
    ↓
Best checkpoint selection: validation ACER
    ↓
Lock checkpoint
    ↓
CelebA-Spoof test.json (held-out) → Final evaluation
    Metrics: Accuracy / AUC / APCER / BPCER / ACER
```

---

## 5. Training Protocol — Spatial Baseline (MobileNetV3-Large)

> Mô tả trạng thái **đang tối ưu**. Các vấn đề đã phát hiện và hướng sửa được ghi lại đầy đủ trong `spatial_pad_baseline_issues_and_revision.md`.

### 5.1 Augmentation — Mild Policy

**Vấn đề với augmentation mạnh:** PAD cần học fine texture (screen artifacts, moire, printing artifacts, mặt nạ boundary). GaussianBlur/JPEG compression mạnh phá hủy chính những cue này.

**Mild augmentation policy (baseline chính thức):**

```text
HorizontalFlip:     p = 0.5
Brightness:         ±10%, p ≈ 0.3
Contrast:           ±10%, p ≈ 0.3
Scale perturbation: 0.90–1.00, p ≈ 0.2
Translation:        ±5%, p ≈ 0.2
```

Loại bỏ khỏi main experiment: `GaussianBlur`, `CoarseDropout`, `RandomErasing`, heavy JPEG compression, large rotation, strong color distortion.

### 5.2 Crop Strategy

Expanded square crop × 1.5 với `BORDER_REFLECT_101` — nhất quán giữa training và inference.

### 5.3 Scheduler

```python
# Fix: T_max phải khớp với số epoch thực tế (lỗi cũ: T_max=10 trong khi epochs=20)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
```

### 5.4 Checkpoint Selection

Chọn checkpoint theo **validation ACER**, không phải accuracy.

```text
ACER  = (APCER + BPCER) / 2
APCER = Attack Presentation Classification Error Rate  (spoof → real)
BPCER = Bona fide Presentation Classification Error Rate  (real → spoof)
```

### 5.5 Threshold Calibration

Sau khi lock checkpoint, quét ngưỡng `logit_diff` trên validation set → tìm operating point (ví dụ BPCER ≤ 5%) → export threshold → truyền vào `AntiSpoofPredictor(threshold=...)`.

Mặc định hiện tại (`threshold=0.5` → `logit_threshold=0`) tương đương argmax tại 0, **chưa calibrate theo operating point thực tế**.

---

## 6. Logic chi tiết từng Module (V2 Updates)

### 6.1 `config.py` — Bổ sung PAD Config

Ngoài toàn bộ cấu hình từ V1, V2 bổ sung:

```python
# Detector (YunNet)
DETECTOR_MODEL_PATH    = None               # None = detection/models/detector_quantized.onnx
DETECTOR_CONF_THRESH   = 0.5
DETECTOR_NMS_THRESH    = 0.3
DETECTOR_TOP_K         = 5000
DETECTOR_MIN_FACE_SIZE = 60
DETECTOR_MARGIN        = 5
DETECTOR_INPUT_SIZE    = (320, 320)

# PAD
PAD_ENABLED          = True
PAD_MODEL_FILENAME   = "mnv3_large_3class_best.onnx"
PAD_THRESHOLD        = 0.5

# Adaptive Gamma Correction cho PAD crop           ← NEW
PAD_GAMMA_ENABLED    = True     # bật/tắt qua .env PAD_GAMMA_ENABLED
PAD_GAMMA_TARGET     = 110.0   # target luma [0-255], điều chỉnh qua .env PAD_GAMMA_TARGET

# PAD Temporal Smoothing
PAD_SMOOTH_WINDOW    = 5      # số phiếu bầu trong rolling window
PAD_SPOOF_MIN_RATIO  = 0.6   # >=60% phiếu SPOOF → kết luận SPOOF

# PAD_INTERVAL_SECONDS — tự động tính, KHÔNG chỉnh thủ công
# = RECOGNIZE_INTERVAL_SECONDS / PAD_SMOOTH_WINDOW = 1.0 / 5 = 0.2s
PAD_INTERVAL_SECONDS = RECOGNIZE_INTERVAL_SECONDS / PAD_SMOOTH_WINDOW

# FPS overlay
SHOW_FPS       = True
FPS_AVG_WINDOW = 30
```

### 6.2 `detection/yunnet_detector.py` (★ NEW)

- **Model:** `cv2.FaceDetectorYN` + `detector_quantized.onnx` (~122 KB, INT8).
- **Auto-resize:** `setInputSize((img_w, img_h))` mỗi frame — xử lý đúng với mọi độ phân giải.
- **Filter chain:** Bỏ qua bbox vượt mép ảnh → bỏ mặt sát mép (margin) → bỏ mặt quá nhỏ (min_face_size).
- **Output format:** Convention SCRFD `(x1, y1, x2, y2)` + 5 landmarks float — **không cần thay đổi** code phía sau.
- **Alias:** `YunNetFaceDetector = FaceDetector` (tương thích ngược).

### 6.3 `antispoof/predictor.py` — AntiSpoofPredictor (★ NEW)

- **Batch inference:** Nhận `List[np.ndarray]` face crops, trả về `List[Dict]`.
- **Hỗ trợ cả 2-class và 3-class** trong cùng một class.
- **Auto-detect model size:** Đọc `session.get_inputs()[0].shape` → tự đặt `model_img_size`.
- **Auto mean/std:** CelebA mean/std cho model 224px; không normalize cho 128px.
- **Decision logic:** `logit_diff = real_logit - max(spoof_logits)` so với `logit_threshold` (inverse-sigmoid của `threshold`).
- **Output mỗi face:**

```python
{
    "is_real": bool,
    "status": "real" | "spoof",
    "detailed_status": "REAL" | "SPOOF (Print)" | "SPOOF (Screen)",
    "class_probs": {"real": float, "physical_spoof": float, "screen_spoof": float},
    "logit_diff": float,
    "real_logit": float,
    "spoof_logit": float,
    "confidence": float,        # abs(logit_diff)
}
```

### 6.4 `antispoof/preprocess.py` — Crop & Preprocess (★ NEW)

**`crop(img, bbox, expansion_factor=1.5)`:**
- Auto-detect format: `(x, y, w, h)` hoặc `(x1, y1, x2, y2)`.
- Mở rộng ×1.5 quanh tâm bbox.
- Fill vùng tràn mép bằng `BORDER_REFLECT_101` (không có viền đen gây sai lệch PAD).

**`preprocess(img, model_img_size, mean, std, apply_gamma=True)`:**
- `apply_gamma=True`: gọi `adaptive_gamma()` trước khi normalize — đúng thứ tự (uint8 → gamma → float32).
- Letterboxing giữ tỉ lệ + `BORDER_REFLECT_101` → `[0, 1]` → mean/std → CHW float32.

**`adaptive_gamma(img)`** — ★ NEW:
- `luma = mean(kênh V trong HSV)` — perceived brightness.
- `gamma = log(TARGET/255) / log(luma/255)` — giải động từ ảnh thực tế.
- Clamp gamma ∈ [0.4, 2.5]; skip nếu |gamma − 1| < 0.05.
- LUT 256-entry cache theo `round(gamma, 2)` — nhanh hơn pixelwise pow() ~10×.

**`preprocess_batch(face_crops, model_img_size, mean, std, apply_gamma=True)`:**
- Stack N crops → tensor `(N, 3, H, W)` cho batch ONNX inference.

### 6.5 `tracking/tracker.py` — PAD Temporal Smoothing (★ UPDATED)

Bổ sung trong class `Track`:

```python
_pad_window: deque[bool]        # rolling window (True=REAL, False=SPOOF)
_pad_spoof_min_ratio: float
last_pad_time: Optional[float]

def update_pad(self, is_real: bool):
    """Ghi verdict vào rolling window."""

def needs_pad(self, interval_seconds: float) -> bool:
    """True nếu đã qua PAD_INTERVAL_SECONDS từ lần PAD cuối."""

@property
def is_spoof(self) -> bool:
    if not self._pad_window:
        return False   # window rỗng → default REAL (tránh false positive frame đầu)
    return (self._pad_window.count(False) / len(self._pad_window)) >= self._pad_spoof_min_ratio
```

`FaceTracker` nhận thêm:
```python
FaceTracker(
    ...,
    pad_smooth_window   = 5,     # ★ NEW
    pad_spoof_min_ratio = 0.6,   # ★ NEW
    pad_interval_seconds = 0.2,  # ★ NEW (auto-computed)
)
```

### 6.6 `app.py` — Main Runtime (★ UPDATED)

**5 visual states (tăng từ 4 lên 5):**

| State | Màu | Label | Ưu tiên |
|---|---|---|---|
| 🔴 **SPOOF** | Đỏ tươi | `⚠ SPOOF` | Cao nhất |
| 🟥 **Unknown** | Đỏ đậm | `UNKNOWN (0.xx)` | |
| 🟧 **Re-verifying** | Cam | `Tên (0.xx)` | |
| 🟩 **Locked** | Xanh | `[OK] Tên (0.xx)` | |
| ⬜ **Pending** | Xám | `Nhan dien...` | Thấp nhất |

**PAD batch flow trong frame loop:**

```python
# Pass 1: Collect tracks cần PAD inference
face_crops, track_ids = [], []
for detection, track in tracked:
    if track.needs_pad(PAD_INTERVAL_SECONDS):
        fc = pad_crop(frame, bbox, expansion_factor)
        face_crops.append(fc); track_ids.append(track.track_id)

# Batch PAD inference (1 ONNX call cho tất cả)
if face_crops:
    pad_results = pad_predictor.predict_crops(face_crops)
    for tid, res in zip(track_ids, pad_results):
        tid_to_track[tid].update_pad(res["is_real"])

# Pass 2: Recognition (skip nếu SPOOF)
for detection, track in tracked:
    is_spoof = pad_enabled_rt and track.is_spoof
    if not is_spoof and track.needs_recognition(RECOGNIZE_INTERVAL_SECONDS):
        aligned = get_input_face(...)
        embedding = embedder.embed_aligned(aligned)
        name, score = decide_identity(db.search(embedding), threshold=MATCH_THRESHOLD)
        track.update_result(name, score)
```

**CLI:**
```bash
python app.py                          # PAD ON, FPS ON (default)
python app.py --no-pad                 # Tắt PAD
python app.py --pad-model <file.onnx>  # Chọn model PAD khác
python app.py --pad-threshold 0.6      # Chỉnh threshold
python app.py --no-fps                 # Tắt FPS overlay

# Runtime: 'p' toggle PAD | 'f' toggle FPS | 'q'/ESC quit
```

---

## 7. Thiết kế PAD Temporal Smoothing

### 7.1 Nguyên lý

```text
Mỗi PAD inference → 1 phiếu (True=REAL, False=SPOOF)
Rolling window giữ N phiếu gần nhất (N = PAD_SMOOTH_WINDOW = 5)

Verdict:
  SPOOF nếu count(False) / N  >=  PAD_SPOOF_MIN_RATIO (0.6)
  REAL   nếu không đủ ngưỡng
```

### 7.2 Đồng bộ PAD ↔ Recognition

```text
RECOGNIZE_INTERVAL = 1.0s
PAD_SMOOTH_WINDOW  = 5
PAD_INTERVAL       = 1.0 / 5 = 0.2s  (auto-computed)

Trong mỗi chu kỳ recognition 1.0s:
    PAD chạy đúng 5 lần (t=0.2, 0.4, 0.6, 0.8, 1.0s)
    → Khi recognition fire tại t=1.0s, window đã đủ 5 phiếu → verdict ổn định
```

Tuning rules:
- Tăng `RECOGNIZE_INTERVAL_SECONDS` (giảm tải Pi) → `PAD_INTERVAL` tự tăng.
- Giảm `PAD_SMOOTH_WINDOW` (phản ứng nhanh, dễ nhiễu hơn) → `PAD_INTERVAL` tự giảm.
- Override `PAD_INTERVAL_SECONDS` trong `.env` nếu muốn tùy chỉnh (không khuyến nghị).

### 7.3 Behavior khi window chưa đủ

Track mới tạo: `_pad_window` rỗng → `is_spoof = False` (default REAL).

Tránh tình trạng track mới bị đánh SPOOF ngay lập tức trước khi có đủ data.

### 7.4 Ví dụ minh họa

```
Track A — PAD_SMOOTH_WINDOW=5, PAD_SPOOF_MIN_RATIO=0.6

t=0.0s: PAD→REAL  | window:[T]           | spoof_ratio=0/1=0.00 → REAL
t=0.2s: PAD→SPOOF | window:[T,F]         | spoof_ratio=1/2=0.50 → REAL
t=0.4s: PAD→SPOOF | window:[T,F,F]       | spoof_ratio=2/3=0.67 → SPOOF ⚠
t=0.6s: PAD→SPOOF | window:[T,F,F,F]     | spoof_ratio=3/4=0.75 → SPOOF ⚠
t=0.8s: PAD→REAL  | window:[T,F,F,F,T]   | spoof_ratio=3/5=0.60 → SPOOF ⚠ (đủ ngưỡng)
t=1.0s: Recognition fires — is_spoof=True → SKIP recognition → hiển thị ⚠ SPOOF
t=1.0s: PAD→REAL  | window:[F,F,F,T,T]   | spoof_ratio=3/5=0.60 → SPOOF ⚠
t=1.2s: PAD→REAL  | window:[F,F,T,T,T]   | spoof_ratio=2/5=0.40 → REAL ✓
```

---

## 8. Dataset Recognition (giữ nguyên từ V1)

### LFW — Calibration & Benchmark (đã hoàn thành)

Kết quả cuối cùng được áp dụng vào production:
- `MATCH_THRESHOLD = 0.35` (calibrate từ ~0.31 tối ưu, nâng strict mode cho FAR thấp)
- `RECOMMENDED_GALLERY_SIZE = 3` ảnh/identity
- `mean_only=True` — mean centroid thắng individual-max

Chi tiết thực nghiệm: `evaluation/lfw_identity_benchmark.ipynb`.

---

## 9. Metrics đánh giá

### 9.1 PAD Metrics

| Metric | Ý nghĩa |
|---|---|
| **APCER** | Tỉ lệ spoof bị nhận nhầm là real |
| **BPCER** | Tỉ lệ real bị nhận nhầm là spoof |
| **ACER** | `(APCER + BPCER) / 2` — metric chính để chọn checkpoint |
| Accuracy | Phụ, không dùng để chọn checkpoint |
| AUC | Phụ, đánh giá toàn bộ ROC curve |

### 9.2 Recognition Metrics

FAR, FRR, EER, Accuracy (pair-based) và TPIR@FAR (1:N identification) — giữ nguyên từ V1.

### 9.3 Edge Metrics (Target)

| Metric | Target |
|---|---|
| Detection latency | < 30ms (CPU) |
| PAD latency | < 20ms (128px) / < 50ms (224px) |
| Embedding latency | < 50ms |
| DB search latency | < 10ms |
| Total FPS | > 15 FPS trên Raspberry Pi 4 |

---

## 10. Acceptance Criteria V2

### Functional

- [x] YunNet detect khuôn mặt và 5 điểm landmarks.
- [x] PAD module load và inference thành công (batch).
- [x] Temporal smoothing chống nhấp nháy REAL/SPOOF.
- [x] Recognition skip khi PAD verdict là SPOOF.
- [x] CLI toggle `--pad` / `--no-pad` và `--pad-model`.
- [x] Runtime keyboard toggle `p` (PAD) và `f` (FPS).
- [x] 5 visual states hiển thị đúng ưu tiên.
- [ ] Model PAD research (MNV3 spatial baseline) hoàn thiện training protocol.
- [ ] Threshold calibration cho model research.
- [ ] ONNX export + edge benchmark model research.

### Engineering

- [x] Tất cả model load đúng 1 lần (LOAD ONCE rule).
- [x] PAD chạy batch inference — 1 ONNX call/frame cho tất cả track.
- [x] `PAD_INTERVAL_SECONDS` tự động sync với `RECOGNIZE_INTERVAL`.
- [x] `antispoof/` là package độc lập.
- [x] Interface `FaceDetector.detect()` tương thích ngược hoàn toàn với SCRFD.
- [x] Adaptive gamma correction cho PAD crop (bật/tắt qua `PAD_GAMMA_ENABLED` trong `.env`).

### Research

- [ ] Spatial baseline MNV3 với clean val/test protocol.
- [ ] Đánh giá chính thức CelebA-Spoof held-out (ACER/APCER/BPCER).
- [ ] Cross-dataset check sơ bộ (LCC-FASD hoặc tương đương).
- [ ] Ablation DCT Frequency Branch vs Spatial-only (giai đoạn tiếp theo).

---

## 11. Những thứ KHÔNG làm trong V2

- Retrain ArcFace.
- Training YunNet detector.
- Depth/IR sensor.
- Qdrant/Milvus.
- Temporal liveness (optical flow, blink detection).
- Multi-frame fusion cho recognition.
- Re-design DB schema.

---

## 12. Script tiện ích mới trong V2

| Script | Mô tả |
|---|---|
| `scripts/demo_antispoof.py` | Demo PAD standalone với webcam |
| `scripts/demo_antispoof_yunnet.py` | Demo PAD + YunNet detector |
| `scripts/export_onnx.py` | Export PyTorch checkpoint → ONNX |
| `scripts/test_detection.py` | Kiểm tra face detection: FPS, landmark, filter |

---

## 13. Kiến trúc mở rộng sang V3 (dự kiến)

```text
V3 — Proposed: Spatial + Frequency Branch

Camera → YunNet Detector
    ↓
PAD Module (Proposed):
    ┌──────────────────────────────────────────┐
    │  RGB Image (224×224)                     │
    │      ↓                        ↓          │
    │  MobileNetV3-Large       DCT Branch      │
    │  (Spatial Features)   (Freq Features)    │
    │      ↓                        ↓          │
    │              Fusion Layer                │
    │                   ↓                      │
    │           3-class Classifier             │
    └────────────────────┬─────────────────────┘
                         │ REAL
                         ↓
              Alignment → ArcFace → pgvector → MATCH/UNKNOWN
```

Câu hỏi nghiên cứu V3:
> Frequency-aware features có cải thiện cross-domain PAD robustness mà vẫn phù hợp edge deployment không?

Đánh giá:
```text
In-domain:    CelebA-Spoof test
Cross-domain: OULU-NPU / SiW / LCC-FASD (preliminary)
Metrics: ACER / APCER / BPCER
Edge: Params / FLOPs / Latency / FPS
```
