# Spatial-Only PAD Baseline — Issues & Proposed Revision


# 1. Mục đích của baseline

Baseline này là mô hình **Spatial-only Face Presentation Attack Detection (PAD)** dùng làm mốc so sánh cho proposed model:

```text
Spatial baseline
MobileNetV3-Large → Classifier

Proposed model
MobileNetV3-Large + DCT Frequency Branch → Fusion → Classifier
```

Baseline phải được thiết kế đủ sạch để sau này có thể trả lời câu hỏi nghiên cứu:

> Frequency-aware features có cung cấp thông tin bổ sung cho spatial representation trong Face PAD hay không, đặc biệt khi đánh giá cross-domain?

Ở giai đoạn hiện tại, mục tiêu là xây dựng và kiểm tra **spatial baseline ổn định** trước. Cross-dataset evaluation trong notebook hiện tại chỉ là **bước kiểm tra sơ bộ**; cross-dataset evaluation chính thức sẽ được thiết kế và thực hiện kỹ hơn ở giai đoạn sau.

> **[NOTE — Tình trạng model hiện tại trong production]**
>
> Production hiện tại (`antispoof/predictor.py`, `antispoof/models/`) đang dùng **MiniFASNetV2SE (INT8, 128×128, 2-class)** cho real-time inference — đây là model riêng biệt, không phải MobileNetV3-Large đang được huấn luyện như spatial baseline trong notebook này.
>
> Ba file model hiện có trong `antispoof/models/`:
> - `best_model_quantized.onnx` — MiniFASNetV2SE INT8 (600 KB) — **download, production default**
> - `mnv3_large_3class_best.onnx` — MobileNetV3-Large FP32 3-class — **tự train, spatial baseline chính thức**
> - `mnv4_best_224.onnx` — MobileNetV4 FP32 224×224 — **tự train, experimental**
>
> Tài liệu này tập trung vào cải thiện **model tự train (MNV3 spatial baseline)**. MiniFASNetV2SE là model download, giữ nguyên, không liên quan đến research flow.

---

# 2. Pipeline hiện tại

Pipeline hiện tại:

```text
CelebA-Spoof
    ↓
3-class labeling
    Real / Physical Spoof / Digital Spoof
    ↓
Read image + bounding box
    ↓
Expanded square crop × 1.5
    ↓
Augmentation
    ↓
Resize 224×224
    ↓
Normalize with CelebA mean/std
    ↓
MobileNetV3-Large ImageNet pretrained
    ↓
3-class classifier
    ↓
Best checkpoint (lưu từ validation)
    ↓
CelebA-Spoof Held-out Test
    +
LCC-FASD preliminary external check
    ↓
Metrics
Accuracy / AUC / APCER / BPCER / ACER
    ↓
ONNX export
```

> **[VERIFIED]** `antispoof/preprocess.py::crop()` xác nhận crop factor 1.5× mặc định với `BORDER_REFLECT_101` (dòng 104–174), khớp với pipeline.
>
> **[VERIFIED]** `antispoof/predictor.py::process_with_logits()` (dòng 90–152) hỗ trợ cả 2-class lẫn 3-class logits. Với 3-class: `spoof_logit = max(raw_logits[1:])` — đúng với thiết kế.
>
> **[DISCREPANCY — critical]** Trong `antispoof/antiproof.ipynb` Cell 5, biến `VAL_JSON` đang trỏ thẳng vào `metas/intra_test/test_label.json`. Tập test chính thức đang bị dùng làm validation loader để chọn `best_model_state` mỗi epoch (Cell 16) và đánh giá lại ở Cell 20. Cần tách rõ: stratified subset từ train để làm validation (không overlap), còn `test_label.json` giữ làm held-out test set (có thể lấy subset nếu muốn test nhanh). Xem chi tiết mục 17.

---

# 3. Các vấn đề hiện tại

## 3.1. Augmentation đang quá mạnh đối với PAD

### Hiện tại

Notebook đang dùng:

```text
HorizontalFlip
ColorJitter
ImageCompression
GaussianBlur
Affine
CoarseDropout
```

Trong đó:

```text
ColorJitter:
brightness=0.2
contrast=0.2
saturation=0.2
hue=0.1

ImageCompression:
quality 50–95

GaussianBlur:
3–7

Affine:
scale 0.9–1.1
rotation ±15°
translation ±5%

CoarseDropout:
1–4 holes
16–32 px
```

### Vấn đề

Đây là augmentation khá mạnh đối với một bài toán mà model được kỳ vọng khai thác:

```text
fine texture
screen artifacts
printing artifacts
moire
aliasing
local high-frequency patterns
boundary/context cues
```

Blur có thể loại bỏ texture.

JPEG compression có thể tạo ra compression artifacts mới.

CoarseDropout có thể che mất các local cues.

Rotation/scale mạnh có thể thay đổi đáng kể spatial distribution của spoof cues.

Điều này đặc biệt quan trọng cho research sau này với DCT/Frequency Branch, vì augmentation không nên trở thành nguồn artifact mà frequency branch học thay cho spoof-related cues.

### Hướng sửa

Dùng một **mild augmentation policy** làm protocol chính.

Đề xuất:

```text
Horizontal Flip:
    p = 0.5

Brightness:
    ±10%
    p ≈ 0.3

Contrast:
    ±10%
    p ≈ 0.3

Scale / crop perturbation:
    nhẹ, khoảng 0.90–1.00
    p ≈ 0.2

Translation:
    ±5%
    p ≈ 0.2
```

Có thể triển khai brightness + contrast bằng `ColorJitter`, nhưng không cần saturation/hue mạnh trong baseline chính.

Không dùng trong main experiment:

```text
GaussianBlur
CoarseDropout
RandomErasing
Heavy JPEG compression
Strong perspective
Large rotation
Strong color distortion
```

Các augmentation này có thể được giữ lại cho **robustness ablation** sau này.

### Nguyên tắc

> Augmentation nên mô phỏng variation thực tế của camera/cropping nhưng hạn chế phá texture và low-level cues mà PAD cần học.

Không nên dùng wording tuyệt đối như “brightness chỉ tác động DC” hoặc “không thay đổi high-frequency”. Cách diễn đạt an toàn hơn:

> Mild appearance perturbations primarily change global intensity/contrast while largely preserving fine-grained spatial structure.

---

# 4. Crop strategy — giữ expanded crop × 1.5

## Hiện tại

Bounding box được mở rộng ×1.5, tạo square crop và dùng `BORDER_REFLECT_101`.

Đây là một quyết định hợp lý cho PAD vì model có thể nhìn thấy thêm context:

```text
face
+
hair / ear
+
paper boundary
+
screen bezel
+
surrounding artifacts
```

> **[VERIFIED]** `antispoof/preprocess.py::crop()` (dòng 104–174) và `antispoof/predictor.py` (dòng 35–36, 59) đang dùng `bbox_expansion_factor=1.5` mặc định với `BORDER_REFLECT_101` — khớp với training strategy.

## Hướng sửa

**Giữ crop ×1.5 cho baseline**, nhưng phải khóa nó thành một phần của protocol.

Không để baseline dùng 1.5× trong khi proposed model dùng crop khác.

Training và inference phải có cùng semantics:

```text
bbox
    ↓
1.5× expanded square crop
    ↓
mild augmentation
    ↓
resize 224×224
```

---

# 5. Random resized crop — cần định nghĩa lại

Nếu dùng `RandomResizedCrop(scale=0.85–1.0)`, có nguy cơ cắt mất context:

```text
screen bezel
paper edge
mask boundary
```

đây lại là những tín hiệu hữu ích cho PAD.

## Hướng sửa đề xuất

Ưu tiên **perturb bbox scale/position nhẹ** thay vì crop ảnh ngẫu nhiên mạnh.

Ví dụ:

```text
scale:
    0.95–1.05

translation:
    ±3–5%
```

Nếu vẫn dùng random resized crop thì nên giới hạn ở:

```text
scale ≈ 0.90–1.00
```

và kiểm tra trực quan rằng face + contextual region vẫn được giữ.

---

# 6. Scheduler đang không khớp với số epoch

## Hiện tại

Notebook đang dùng:

```python
epochs = 20

CosineAnnealingLR(
    optimizer,
    T_max=10
)
```

Trong khi training kéo dài 20 epochs.

## Vấn đề

Nếu mục tiêu là cosine decay xuyên suốt training thì `T_max` nên khớp với số epoch thực tế.

## Hướng sửa

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=epochs,
    eta_min=1e-6
)
```

Nếu:

```text
epochs = 20
```

thì:

```text
T_max = 20
```

Đây là **sửa bắt buộc** trước khi chạy baseline chính thức.

---

# 7. Chọn best checkpoint chỉ theo Validation Accuracy

## Hiện tại

Checkpoint đang được chọn dựa trên:

```text
best validation accuracy
```

## Vấn đề

PAD là bài toán security-oriented.

Accuracy không phản ánh đầy đủ:

```text
APCER
BPCER
ACER
```

Đặc biệt false acceptance của spoof là một lỗi quan trọng.

## Hướng sửa

Ở giai đoạn baseline:

```text
Training
    ↓
Validation
    ↓
Select checkpoint using validation ACER
    ↓
Lock checkpoint
    ↓
Threshold calibration
    ↓
Final evaluation
```

Có thể tiếp tục log:

```text
Accuracy
AUC
APCER
BPCER
ACER
```

nhưng metric dùng để chọn checkpoint nên phản ánh mục tiêu PAD tốt hơn accuracy thuần túy.

Nếu implementation hiện tại chưa đủ ổn định để chọn trực tiếp theo ACER, có thể giữ validation accuracy ở vòng debug đầu tiên, nhưng **baseline chính thức nên chuyển sang ACER hoặc một operating-point metric được định nghĩa rõ ràng**.

---

# 8. Threshold hiện tại chưa được calibration

## Hiện tại

Prediction đang về bản chất là:

```text
real_logit - spoof_logit >= 0
    → Real

real_logit - spoof_logit < 0
    → Spoof
```

Đây là argmax/threshold tại 0.

> **[VERIFIED]** `antispoof/predictor.py` (dòng 62–63) đã hỗ trợ calibrated threshold thông qua `logit_threshold` (tính từ sigmoid inverse của `threshold` parameter). Tuy nhiên giá trị mặc định `threshold=0.5` tương đương với `logit_threshold=0` — tức là threshold tại 0, chưa calibrate theo operating point thực tế. Notebook training cần export threshold đã calibrate và truyền vào `AntiSpoofPredictor(threshold=...)`.

## Vấn đề

Threshold này chưa phản ánh operating point của hệ thống authentication.

## Hướng sửa

Dùng validation/dev set để calibration:

```text
Validation
    ↓
Sweep threshold
    ↓
Choose threshold according to PAD operating point
    ↓
Lock threshold
    ↓
Evaluate test
```

Ví dụ có thể đặt constraint:

```text
APCER ≤ target
```

sau đó chọn threshold phù hợp.

Điểm quan trọng:

> Threshold không được tối ưu trên final test set.

Sau khi calibrate, truyền giá trị threshold đã lock vào `AntiSpoofPredictor(threshold=<calibrated_value>)` trong production.

> **[MINH CHỨNG THỰC NGHIỆM — Quick Test 10K (`antispoof/notebooks/quick_test/pbl6-quicktest-10k.ipynb`)]**
> Kết quả chạy thực tế trên 10,000 ảnh test đã chứng minh tầm quan trọng mang tính quyết định của việc căn chỉnh ngưỡng (Threshold Calibration):
> - **Ngưỡng mặc định `Threshold = 0.0` (chưa calibrate)**: APCER = **14.19%** (có tới 998 ca tấn công bị lọt lưới!), BPCER = 0.54%, **ACER = 7.36%**.
> - **Ngưỡng tối ưu `Calibrated Threshold = 2.729`**: APCER giảm mạnh còn **6.68%** (chỉ còn 470 ca lọt lưới), BPCER = 1.58%, **ACER giảm sâu xuống 4.13%**.
> → Chỉ bằng thao tác dịch chuyển ngưỡng sang điểm vận hành tối ưu (ngưỡng 2.729), mô hình đã chặn đứng thêm **528 ca giả mạo** và giảm gần một nửa sai số ACER mà không cần huấn luyện lại bất kỳ trọng số nào. (Xem chi tiết tại mục 27.3).

---

# 9. 3-class training nhưng binary PAD decision

## Hiện tại

Training:

```text
0 = Real
1 = Physical Spoof
2 = Digital Spoof
```

Evaluation/production:

```text
Real
vs
Spoof
```

Cách làm này có thể giữ.

> **[VERIFIED]** `antispoof/predictor.py::process_with_logits()` (dòng 120–137) xác nhận: với 3-class model, `spoof_logit = float(np.max(raw_logits[1:]))` — tức là lấy max của Physical Spoof và Digital Spoof logit. Đây là cách định nghĩa binary score đúng, nhất quán với thiết kế.

## Quy tắc

Khi đánh giá binary:

```text
Real = class 0

Spoof =
    class 1 OR class 2
```

Ví dụ:

```python
spoof_score = max(
    physical_spoof_logit,
    digital_spoof_logit
)
```

hoặc quy về probability tương ứng tùy cách định nghĩa score.

## Lưu ý

Nếu dùng `real_logit - spoof_logit`, phải giữ cùng định nghĩa score trong:

```text
validation calibration
test
cross-dataset preliminary test
```

---

# 10. Class weighting hiện tại chưa thực sự được áp dụng

Notebook có:

```python
class_weights = torch.tensor([1, 1, 1])
```

> **[VERIFIED — bổ sung chi tiết]** Trong `antiproof.ipynb`, `class_weights` này được truyền vào `nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)` — tức loss hiện tại có `label_smoothing=0.1` đi kèm dù chưa có class weighting thực sự. Cần giữ lại `label_smoothing=0.1` này (hoặc quyết định rõ có tiếp tục dùng không) khi chỉnh sửa phần loss, để tránh vô tình bỏ mất cấu hình đang có.

Nếu mục tiêu là weight class thì cấu hình `class_weights` này không tạo ra weighting.

## Hướng sửa

Đối với baseline chính, nên bắt đầu với:

```text
[1, 1, 1]
```

nếu class distribution vẫn đủ ổn định.

Không tự ý tăng weight cho Real hoặc Spoof chỉ để đạt metric đẹp.

Nếu imbalance trở thành vấn đề:

```text
class weighting
```

nên được xem như một experiment riêng và áp dụng **giống nhau cho baseline/proposed**.

---

# 11. Best model state cần lưu snapshot độc lập

Không nên chỉ:

```python
best_model_state = model.state_dict()
```

mà nên:

```python
import copy

best_model_state = copy.deepcopy(model.state_dict())
```

và đồng thời lưu checkpoint ra disk.

Mục tiêu là đảm bảo best checkpoint thực sự là snapshot của epoch tốt nhất.

---

# 12. RGB/BGR phải được khóa rõ ràng

Training dataset hiện tại đọc:

```text
cv2.imread()
→ BGR
→ cv2.cvtColor(..., BGR2RGB)
```

nên training input là RGB.

Trong `preprocess.py`, hàm preprocessing chỉ:

```text
HWC
→ CHW
→ /255
→ optional normalization
```

nó **không tự chuyển BGR sang RGB**.

Vì vậy caller phải đảm bảo input đúng màu.

> **[VERIFIED]** `antispoof/preprocess.py::preprocess()` (dòng 17–70) KHÔNG tự chuyển BGR→RGB. Docstring (dòng 32) ghi rõ: `img (np.ndarray): Ảnh crop khuôn mặt (RGB/BGR)` — caller phải đảm bảo đúng color order. Đây là điểm dễ gây bug nếu camera frame ở BGR mà không convert trước.

## Hướng sửa

Khóa rule:

```text
Training:
    RGB

Inference:
    RGB

Normalization:
    cùng mean/std
```

Nếu camera frame đang ở BGR thì phải convert trước khi đưa vào predictor:

```python
frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
result = predictor.predict_frame(frame_rgb, bbox)
```

Đây là một sanity check bắt buộc trước benchmark.

---

# 13. Mean/std phải đồng nhất

Baseline hiện dùng:

```text
mean = [0.5931, 0.4690, 0.4229]
std  = [0.2471, 0.2214, 0.2157]
```

> **[VERIFIED]** `antispoof/predictor.py` (dòng 83–85) xác nhận các giá trị này được áp dụng khi `model_img_size == 224`.
>
> **[DISCREPANCY — cần chú ý]** `predictor.py` dòng 83 kiểm tra `"mnv4" in str(self.model_path).lower()` nhưng file hiện tại là `mnv3_large_3class_best.onnx` (không chứa `mnv4`). Mean/std được apply vì điều kiện thứ 2 (`model_img_size == 224`) — đây đúng về kết quả nhưng logic detection dễ nhầm. Nên cân nhắc đổi điều kiện thành `"mnv3" in path or "mnv4" in path or model_img_size == 224` hoặc dùng config tường minh.

Nếu đây là normalization đã được chọn cho training baseline, phải dùng chính xác cùng preprocessing khi inference.

Không thay đổi:

```text
resize
color order
mean
std
crop strategy
```

giữa training và inference.

---

# 14. Training subset 100K — phù hợp cho bước sơ bộ, nhưng phải ghi rõ

Notebook hiện đang dùng:

```text
Train source:
494K+ images

Actual preliminary training:
100,000 images
```

và sampling có phân tầng theo 3 class với seed cố định.

Đây là lựa chọn **hợp lý cho training/debug sơ bộ** để giảm thời gian.

Tuy nhiên:

> 100K không nên được coi là training protocol cuối cùng của paper nếu chưa kiểm tra ảnh hưởng của sample size.

## Hướng xử lý

### Giai đoạn hiện tại

Có thể giữ:

```text
TRAIN_SAMPLE_SIZE = 100000
VAL_SAMPLE_SIZE = 15000       # Trích xuất từ train, KHÔNG overlap
SEED = 42
```

để:

```text
debug
kiểm tra pipeline
kiểm tra loss
kiểm tra preprocessing
kiểm tra checkpoint
```

> **[LƯU Ý QUAN TRỌNG]** Khi lấy subset 100K để train, **bắt buộc phải trích xuất đồng thời một validation subset độc lập (ví dụ 10K–15K ảnh) từ cùng tập train gốc** và đảm bảo `train_subset ∩ val_subset = ∅`. Không được dùng test set làm validation (xem mục 17).

### Baseline chính thức

Sau khi pipeline ổn định:

```text
100K
vs
Full training set
```

có thể được kiểm tra nếu tài nguyên cho phép.

Ít nhất phải ghi rõ trong log/report rằng kết quả hiện tại dùng 100K train samples.

---

# 15. Cross-dataset hiện tại chỉ là preliminary sanity check

## Hiện tại

Notebook đang có:

```text
Train:
CelebA-Spoof

Preliminary external test:
LCC-FASD
```

Điều này hữu ích để kiểm tra model có bị collapse hoàn toàn khi rời dataset training hay không.

## Nhưng KHÔNG nên coi đây là cross-domain protocol cuối

Cross-dataset evaluation chính thức sẽ được thực hiện sau với protocol chặt chẽ hơn.

Mục tiêu hiện tại chỉ là:

```text
“Model có generalize sơ bộ sang dataset khác không?”
```

không phải:

```text
“Đã chứng minh cross-domain robustness.”
```

## Report hiện tại

Có thể ghi:

```text
In-domain:
    Validation: CelebA-Spoof val subset (tách từ train_label.json, disjoint)
    Test:       CelebA-Spoof held-out test set (hoặc quick test subset)

Preliminary external test:
    LCC-FASD
```

Không dùng LCC-FASD preliminary result để claim mạnh về research hypothesis.

---

# 16. Cross-dataset evaluation chính thức ở giai đoạn sau

Sau khi baseline và proposed model ổn định, thực hiện:

```text
Train
CelebA-Spoof
     ↓
 ┌───┴───────────────┐
 ↓                   ↓
OULU-NPU             SiW
 ↓                   ↓
Cross-dataset Test   Cross-dataset Test
```

Protocol cuối cần khóa:

```text
training data
validation/calibration
test data
threshold
metrics
```

và không fine-tune trên dataset test.

Mục tiêu là kiểm tra:

```text
performance degradation
```

và đặc biệt so sánh:

```text
Spatial-only
vs
Spatial + Frequency
```

dưới domain shift.

---

# 17. Phân chia dữ liệu (Data Split): Tách bạch Validation Set & Test Set, Chống Data Leakage

## 17.1. Lỗ hổng nghiêm trọng trong code notebook hiện tại (`antiproof.ipynb`)

Khi đối chiếu với code trong notebook `antispoof/antiproof.ipynb` (Cell 5, Cell 16, Cell 20), phát hiện một **lỗ hổng phương pháp luận cốt lõi (critical methodological flaw)**:

```python
# antiproof.ipynb Cell 5 hiện tại:
DATA_ROOT = "/kaggle/input/datasets/attentionlayer241/celeba-spoof-for-face-antispoofing/CelebA_Spoof_/CelebA_Spoof"
TRAIN_JSON = os.path.join(DATA_ROOT, "metas/intra_test/train_label.json")
VAL_JSON = os.path.join(
    DATA_ROOT, "metas/intra_test/test_label.json"
)  # ⚠️ NGUY HIỂM: Trỏ thẳng vào TEST SET!

train_dataset_full = CelebASpoof3ClassDataset(
    DATA_ROOT, TRAIN_JSON, transform=train_transform
)
val_dataset = CelebASpoof3ClassDataset(
    DATA_ROOT, VAL_JSON, transform=val_transform
)
# val_loader nạp val_dataset -> thực chất là nạp toàn bộ CelebA-Spoof TEST SET!
```

### Các hệ quả nghiêm trọng:

1. **Không hề có Validation Set thực sự**: Notebook gọi là `val_loader` nhưng bên dưới lại nạp `test_label.json` (~60,000 ảnh) — tức là tập TEST chính thức của giao thức intra-test trong CelebA-Spoof.
2. **Data Leakage & Test-Set Snooping (Tràn dữ liệu kiểm thử)**:
   - Trong Cell 16 (vòng lặp train), sau mỗi epoch, hàm validate tính `val_acc` trên `val_loader` (tập test).
   - Biến `best_acc` và checkpoint `best_model_state = copy.deepcopy(model.state_dict())` được lựa chọn và lưu trữ **dựa trên chính hiệu năng của tập test** (`if val_acc > best_acc`).
3. **Mất hoàn toàn tính Unbiased (khách quan) khi Benchmark**:
   - Ở Cell 20, script evaluate lại chạy trên `val_loader` để tính Accuracy, AUC, ACER.
   - Kết quả công bố bị bias nặng nề (thổi phồng hiệu năng) vì checkpoint tốt nhất đã được "chọn lọc thiên vị" để tối ưu hóa tập test này.
4. **Vi phạm nguyên lý Biometrics PAD (tiêu chuẩn ISO/IEC 30107-3)**:
   - Nghiêm cấm tuyệt đối việc dùng tập test để: (a) chọn model checkpoint, (b) điều chỉnh early stopping, (c) căn chỉnh ngưỡng operating point (`threshold calibration`).

---

## 17.2. Chuẩn hóa kiến trúc 3 tập dữ liệu (Gold Standard 3-Split Protocol)

Quy trình chuẩn hóa bắt buộc phải phân định rạch ròi 3 tập dữ liệu độc lập:

```text
                        ┌──────────────────────────────────────────────────────────┐
                        │                   CelebA-Spoof Dataset                   │
                        └────────────┬─────────────────────────────────┬───────────┘
                                     │                                 │
                         metas/intra_test/train_label.json metas/intra_test/test_label.json
                                     │                                 │
                 ┌───────────────────┴───────────────────┐             │
                 ▼                                       ▼             ▼
          ┌──────────────┐                        ┌──────────────┐ ┌──────────────────────────┐
          │ Train Subset │                        │  Val Subset  │ │    Held-out Test Set     │
          │   (~100K)    │                        │  (10K–20K)   │ │      (~60K ảnh gốc)      │
          └──────┬───────┘                        └──────┬───────┘ └────────────┬─────────────┘
                 │                                       │                      │
                 │   Disjoint 100%: Train ∩ Val = ∅      │                      │ Tuỳ chọn chạy nhanh
                 │   (Cùng phân tầng theo 3 class)       │                      ▼
                 ▼                                       ▼             ┌──────────────────┐
         Train Model Weights                     Đánh giá mỗi epoch    │Quick Test Subset │
         (Mild Augmentation)                     Chọn Best Checkpoint  │   (5K–10K ảnh)   │
         Backpropagation                         Calibrate Threshold   └────────┬─────────┘
                                                 (KHÔNG Augmentation)           │
                                                         │                      │
                                                         └──────────────┬───────┘
                                                                        ▼
                                                             Final Unbiased Benchmark
                                                             (Lock model & Lock threshold)
                                                             APCER / BPCER / ACER / AUC
```

### Bảng đặc tả chi tiết 3 phân vùng

| Phân vùng dữ liệu | Nguồn file | Kích thước đề xuất | Transform | Mục đích sử dụng | Ràng buộc bảo toàn |
|---|---|---|---|---|---|
| **`train_subset`** | `train_label.json` | 100,000 ảnh (hoặc 80–90% tập train) | `train_transform` (Mild Augmentation) | Tính CrossEntropyLoss, cập nhật trọng số mạng bằng AdamW/SGD | Stratified theo 3 class (Real / Physical / Digital) |
| **`val_subset`** | `train_label.json` | 10,000 – 15,000 ảnh (hoặc 10–20% tập train) | `val_transform` (**Chỉ Resize + Normalize, KHÔNG Augment**) | Theo dõi loss/acc mỗi epoch, chọn `best_model_state`, căn chỉnh threshold (`APCER ≤ target`) | **`train_indices ∩ val_indices = ∅`**, Stratified theo 3 class |
| **`test_set` (Full)** | `test_label.json` | Toàn bộ (~60,000 ảnh) | `val_transform` (Resize + Normalize) | Đánh giá khách quan cuối cùng (Final Benchmark) để báo cáo đồ án / paper | **Held-out 100%**, không dùng trong training hay chọn model |
| **`quick_test_subset`** | `test_label.json` | 5,000 – 10,000 ảnh | `val_transform` (Resize + Normalize) | Thử nghiệm nhanh tốc độ, debug pipeline, so sánh sơ bộ các model trong `compare_models.ipynb` | Stratified theo 3 class từ `test_label.json`, cố định `SAMPLE_SEED = 42` |

---

## 17.3. Thuật toán phân tầng không chồng lấn (Disjoint Stratified Splitting)

Khi chia `train_label.json` thành `train_subset` và `val_subset`, phải tuân thủ các nguyên tắc kỹ thuật sau:

1. **Đảm bảo tính Disjoint 100%**:
   ```text
   set(train_indices) ∩ set(val_indices) == ∅
   ```
   Không được phép có bất kỳ mẫu nào xuất hiện đồng thời trong cả tập train và validation.

2. **Phân tầng (Stratification) theo đúng tỷ lệ 3 class**:
   Tỷ lệ giữa `Real` (0), `Physical Spoof` (1), và `Digital Spoof` (2) trong cả `train_subset` và `val_subset` phải phản ánh trung thực tỷ lệ của tập `train_label.json` gốc, tránh hiện tượng lệch class làm hỏng việc đánh giá ACER.

3. **Tách biệt Dataset Object để tránh rò rỉ Augmentation**:
   - `Subset` trong PyTorch chỉ lọc chỉ số (`indices`), nó **dùng chung thuộc tính `transform` của Dataset cha**.
   - Nếu khởi tạo `train_dataset_full` với `train_transform` rồi trích xuất `val_subset = Subset(train_dataset_full, val_indices)`, tập validation sẽ **vô tình bị áp dụng ColorJitter, Affine, Blur...** làm méo mó kết quả đánh giá validation loss và ACER!
   - **Cách làm đúng**: Tạo 2 instance Dataset từ `TRAIN_JSON`: một cái gán `train_transform` cho train, một cái gán `val_transform` cho val.

---

## 17.4. Quick Test Subset cho tập Test CelebA-Spoof

Tập test chính thức của CelebA-Spoof rất lớn (~60,000 ảnh). Một lần evaluate toàn bộ tập test có thể mất:
- Trên GPU Kaggle T4: ~15–20 phút.
- Trên CPU máy cá nhân: ~1.5–2 giờ.

Điều này làm chậm đáng kể chu trình phát triển (iteration cycle).

### Giải pháp Quick Test:
- Cung cấp tham số `TEST_SAMPLE_SIZE = 10000` (hoặc `5000` tuỳ nhu cầu). Nếu đặt `TEST_SAMPLE_SIZE = None`, hệ thống sẽ nạp toàn bộ Full Test Set.
- Lấy mẫu phân tầng (stratified) theo 3 class từ `test_label.json` với random seed cố định (`SAMPLE_SEED = 42`).
- **Nguyên tắc báo cáo**:
  - **Quick Test**: Dùng cho debug nội bộ, so sánh nhanh latency / metrics giữa các model trong `compare_models.ipynb`.
  - **Full Test Set**: **Bắt buộc** phải chạy khi xuất bảng kết quả cuối cùng cho đồ án / báo cáo chính thức.

---

## 17.5. Code triển khai chuẩn mực (Drop-in Replacement cho Cell 5 Notebook)

Đoạn code hoàn chỉnh dưới đây khắc phục triệt để lỗi data leakage, phân chia rõ ràng 3 tập dữ liệu và hỗ trợ Quick Test:

```python
# ==============================================================================
# Cell 3: Transforms & DataLoaders (CHUẨN HÓA 3 TẬP ĐỘC LẬP: TRAIN / VAL / TEST)
# ==============================================================================
import os, json
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# 1. Định nghĩa Chuẩn hóa Mean / Std của CelebA-Spoof
CELEBA_MEAN = [0.5931, 0.4690, 0.4229]
CELEBA_STD = [0.2471, 0.2214, 0.2157]

# 2. Data Transforms (Xem chi tiết mục 3.1: Mild Augmentation cho PAD)
train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.0, hue=0.0, p=0.3),
    A.Affine(scale=(0.95, 1.05), translate_percent=(-0.05, 0.05), rotate=(-5, 5), p=0.2),
    A.Normalize(mean=CELEBA_MEAN, std=CELEBA_STD),
    ToTensorV2(),
])

# Validation & Test: Chỉ Resize và Normalize, KHÔNG áp dụng bất kỳ augmentation nào
val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=CELEBA_MEAN, std=CELEBA_STD),
    ToTensorV2(),
])

DATA_ROOT = "/kaggle/input/datasets/attentionlayer241/celeba-spoof-for-face-antispoofing/CelebA_Spoof_/CelebA_Spoof"
TRAIN_JSON = os.path.join(DATA_ROOT, "metas/intra_test/train_label.json")
TEST_JSON  = os.path.join(DATA_ROOT, "metas/intra_test/test_label.json")

# --- CẤU HÌNH KÍCH THƯỚC DATASET ---
TRAIN_SAMPLE_SIZE = 100000   # Số ảnh train (hoặc None để dùng toàn bộ phần còn lại)
VAL_SAMPLE_SIZE   = 15000    # Số ảnh validation (tách từ train_label.json, KHÔNG overlap)
TEST_SAMPLE_SIZE  = 10000    # Số ảnh quick test từ test_label.json (hoặc None để dùng full ~60K ảnh)
SAMPLE_SEED       = 42

def _label_of(v):
    lr = int(v[40])
    if lr == 0:
        return 0
    elif lr in [1, 2, 3, 4, 5, 6, 7]:
        return 1
    else:
        return 2

# 3. Đọc Metadata của Tập Train để thực hiện Stratified Train / Val Split
print(f"Loading Train Metadata from: {TRAIN_JSON}")
with open(TRAIN_JSON, 'r') as f:
    train_meta = json.load(f)

train_keys = list(train_meta.keys())
n_train_total = len(train_keys)
train_labels = np.array([_label_of(train_meta[k]) for k in train_keys])

rng = np.random.default_rng(SAMPLE_SEED)
train_indices = []
val_indices = []

# Tách Stratified đảm bảo Disjoint 100%
for c in [0, 1, 2]:
    c_indices = np.where(train_labels == c)[0]
    rng.shuffle(c_indices)
    
    # 1. Trích xuất Val Subset trước từ class c
    n_val_c = int(round(VAL_SAMPLE_SIZE * len(c_indices) / n_train_total))
    val_c = c_indices[:n_val_c]
    val_indices.extend(val_c.tolist())
    
    # 2. Trích xuất Train Subset từ phần còn lại (không bao giờ trùng với val)
    rem_c = c_indices[n_val_c:]
    if TRAIN_SAMPLE_SIZE is not None:
        n_train_c = int(round(TRAIN_SAMPLE_SIZE * len(c_indices) / n_train_total))
        n_train_c = min(n_train_c, len(rem_c))
        train_c = rem_c[:n_train_c]
    else:
        train_c = rem_c
    train_indices.extend(train_c.tolist())

rng.shuffle(train_indices)
rng.shuffle(val_indices)

# KIỂM TRA BẢO TOÀN TÍNH KHÔNG CHỒNG LẤN
overlap = set(train_indices) & set(val_indices)
assert len(overlap) == 0, f"LỖI NGUY HIỂM: Phát hiện {len(overlap)} mẫu bị overlap giữa Train và Val!"

# 4. Khởi tạo Dataset (Tách riêng transform cho Train và Val)
train_base = CelebASpoof3ClassDataset(DATA_ROOT, TRAIN_JSON, transform=train_transform)
val_base   = CelebASpoof3ClassDataset(DATA_ROOT, TRAIN_JSON, transform=val_transform)

train_dataset = Subset(train_base, train_indices)
val_dataset   = Subset(val_base, val_indices)

print(f"--> Đã phân chia: Train = {len(train_dataset)} ảnh (Mild Aug), "
      f"Val = {len(val_dataset)} ảnh (No Aug, Non-overlapping, Seed={SAMPLE_SEED})")

# 5. Xử lý Tập TEST chính thức từ test_label.json
print(f"Loading Test Metadata from: {TEST_JSON}")
test_base = CelebASpoof3ClassDataset(DATA_ROOT, TEST_JSON, transform=val_transform)

if TEST_SAMPLE_SIZE is not None and TEST_SAMPLE_SIZE < len(test_base):
    test_labels = np.array([_label_of(test_base.data[k]) for k in test_base.keys])
    test_indices = []
    rng_test = np.random.default_rng(SAMPLE_SEED)
    for c in [0, 1, 2]:
        c_idx = np.where(test_labels == c)[0]
        n_test_c = int(round(TEST_SAMPLE_SIZE * len(c_idx) / len(test_base)))
        n_test_c = min(n_test_c, len(c_idx))
        test_indices.extend(rng_test.choice(c_idx, size=n_test_c, replace=False).tolist())
    rng_test.shuffle(test_indices)
    test_dataset = Subset(test_base, test_indices)
    print(f"--> Quick Test Subset: {len(test_dataset)}/{len(test_base)} ảnh (Stratified, Seed={SAMPLE_SEED})")
else:
    test_dataset = test_base
    print(f"--> Full Test Set: {len(test_dataset)} ảnh (Held-out)")

# 6. Tạo 3 DataLoaders riêng biệt
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

print(f"--> DataLoaders Sẵn Sàng: TrainLoader ({len(train_loader)} batches), "
      f"ValLoader ({len(val_loader)} batches), TestLoader ({len(test_loader)} batches)")
```

### Ràng buộc trong quá trình huấn luyện & đánh giá:
- **Trong vòng lặp training (Cell 16)**: Chỉ truyền `val_loader` vào hàm validate để theo dõi và lưu `best_model_state`.
- **Căn chỉnh ngưỡng (Cell 16 / Cell 17)**: Tìm ngưỡng tối ưu (threshold calibration) trên kết quả logits của `val_loader`.
- **Đánh giá kiểm thử cuối cùng (Cell 20)**: Nạp mô hình từ `best_model_state`, áp dụng threshold đã khóa từ validation, và chỉ chạy đánh giá trên `test_loader`.
- **Subject-disjoint**: Theo quy ước lý thuyết của giao thức `intra_test` (CelebA-Spoof), tập train và test phải tách biệt danh tính (`Train subjects ∩ Test subjects = ∅`). Tuy nhiên, kết quả kiểm tra thực nghiệm toàn diện từ notebook EDA (`antispoof/notebooks/eda/celeb-a-spoof-eda.ipynb`) trên mirror Kaggle (`attentionlayer241`) cho thấy **có 3 subject IDs bị trùng lặp** giữa Train (8,192 subjects) và Test (1,004 subjects) (`is_subject_disjoint: false`, `overlap_subjects = 3`). Dù tỷ lệ trùng là rất nhỏ (3 / 9,193 ~ 0.03%), pipeline nghiêm ngặt khi tạo `train_subset` và `val_subset` nên loại bỏ hoàn toàn các thư mục thuộc 3 subject này khỏi tập train để đảm bảo tính held-out tuyệt đối cho tập test. (Xem chi tiết tại mục 27.3).

---

# 18. Evaluation metrics

Baseline nên báo:

```text
Accuracy
AUC

APCER
BPCER
ACER
```

Trong đó:

```text
APCER
= spoof được classify nhầm thành real

BPCER
= real bị classify nhầm thành spoof

ACER
= (APCER + BPCER) / 2
```

Accuracy và AUC có thể dùng để tham khảo, nhưng PAD result chính nên tập trung vào APCER/BPCER/ACER.

---

# 19. Training objective

Baseline:

```text
MobileNetV3-Large
ImageNet pretrained
        ↓
Feature extraction
        ↓
3-class classifier
        ↓
Cross-Entropy loss
```

Trong giai đoạn baseline, không thêm:

```text
frequency branch
DCT
gated fusion
domain adaptation
temporal cues
depth/IR
```

Mục đích là giữ research question rõ ràng:

> Spatial-only representation là baseline.

---

---

# 19b. Lựa chọn backbone và rủi ro nhiễm frequency supervision

## Backbone hiện tại

MobileNetV3-Large được chọn làm backbone **spatial-only baseline** vì:

```text
ImageNet pretrained — không có auxiliary supervision nào liên quan đến frequency
Kiến trúc thuần spatial CNN
Không có Fourier/DCT/wavelet trong pre-training
```

Đây là lựa chọn **đúng** cho mục tiêu của dự án.

## Tại sao MiniFASNetV2SE KHÔNG phù hợp làm spatial-only baseline

MiniFASNetV2SE được train với **Fourier Spectrum Auxiliary Supervision** trong quá trình huấn luyện gốc — một branch phụ phân tích phổ tần số của ảnh để hướng features về frequency-related spoof cues.

Hệ quả:

```text
MiniFASNetV2SE features
≠ pure spatial features

Backbone đã học frequency-related patterns trong quá trình train.

Nếu dùng làm baseline,
Stage 3 (thêm DCT/frequency branch) không thể chứng minh rằng
frequency branch bổ sung thêm thông tin gì —
vì baseline đã "nhúm" frequency rồi.

→ Research question bị vô hiệu hóa.
```

> **Kết luận:** MiniFASNetV2SE không phải spatial-only, không được dùng làm baseline cho Stage 2. Giữ lại để dùng cho **production real-time inference (model download)** — vai trò này hoàn toàn độc lập với research flow.

## Nếu muốn đổi backbone nhẹ hơn (giảm size/latency của model tự train)

Có thể cân nhắc:

| Backbone | Params | FLOPs | Ghi chú |
|---|---|---|---|
| **MobileNetV3-Large** | ~5.4M | ~0.22G | Hiện tại, chuẩn tốt |
| **MobileNetV3-Small** | ~2.5M | ~0.06G | Nhẹ hơn ~2x, không nhiễm frequency |
| **ShuffleNetV2 ×1.0** | ~2.3M | ~0.15G | Tương đương, channel shuffle architecture |

**Không khuyến khích:**

```text
MobileFaceNet
    → Thiết kế cho face recognition với angular margin loss
    → Dễ nhiễm spoof artifact vì face recognition training objective

MiniFASNet (bất kỳ variant nào)
    → Fourier auxiliary supervision trong kiến trúc gốc
    → Dùng làm baseline sẽ phá vỡ spatial-only research question

MobileFaceNet + Fourier Loss (bất kỳ dạng nào)
    → Lý do tương tự
```

> **Quy tắc chọn backbone cho spatial baseline:** backbone phải được train với ImageNet-only, không có bất kỳ frequency/spectral auxiliary loss nào trong pre-training. Cần xác nhận bằng cách đọc paper gốc của backbone.

## Ý nghĩa cho augmentation

Rủi ro nhiễm frequency không chỉ từ backbone mà còn từ augmentation. Cụ thể:

```text
GaussianBlur     → thay đổi high-frequency content nhân tạo
JPEG Compression → tạo DCT block artifact không phải từ spoof thật
CoarseDropout    → tạo spatial pattern ngẫu nhiên
```

Đây là lý do các augmentation này phải được loại khỏi baseline chính (xem mục 3.1).

---

# 20. Edge deployment

Sau khi model PyTorch ổn định:

```text
PyTorch
   ↓
ONNX
   ↓
ONNX Runtime
```

INT8 quantization có thể thực hiện sau bước baseline FP32.

Không gộp:

```text
architecture experiment
+
quantization experiment
```

vào cùng một bước, vì sẽ khó xác định performance change đến từ đâu.

---

# 21. ONNX inference hiện tại

`loader.py` đã có:

```text
ORT_ENABLE_ALL
ORT_SEQUENTIAL
CUDA → CPU fallback
```

và lấy input node trực tiếp từ graph.

Đây là implementation tốt để deployment.

> **[VERIFIED]** `antispoof/loader.py` (dòng 34–54) xác nhận: `ORT_ENABLE_ALL`, `ORT_SEQUENTIAL`, và auto-detect CUDA provider với fallback về CPU — đúng như mô tả.

`predictor.py` cũng đã:

```text
load model once
batch preprocess
batch inference
postprocess logits
```

> **[VERIFIED]** `antispoof/predictor.py::predict_crops()` (dòng 154–185) xác nhận batch inference, validate shape đầu ra (`logits.ndim != 2`), và map từng logit về dict kết quả.

Nên giữ cấu trúc này.

---

# 22. INT8 model naming phải rõ ràng

Hiện inference default đang trỏ đến:

```text
best_model_quantized.onnx
```

> **[DISCREPANCY — quan trọng]** `best_model_quantized.onnx` hiện tại là **MiniFASNetV2SE INT8 (600 KB)**, không phải MobileNetV3-Large quantized. Đây là hai model khác nhau hoàn toàn về kiến trúc. Khi MobileNetV3-Large spatial baseline được quantize, **phải dùng tên file khác** để tránh nhầm lẫn và không ghi đè file production.

Phân loại rõ ràng:

```text
MiniFASNetV2SE:
    best_model_quantized.onnx           ← production hiện tại, KHÔNG ghi đè

MobileNetV3-Large spatial baseline:
    mnv3_large_3class_best.onnx         ← FP32 (đã có)
    mnv3_large_3class_best_int8.onnx    ← INT8 (sau khi quantize)

MobileNetV4 (experimental):
    mnv4_best_224.onnx                  ← FP32
```

Không dùng tên `quantized` cho file chưa quantize, và không ghi đè `best_model_quantized.onnx` bằng file của architecture khác.

---

# 23. Recommended baseline protocol

## Data

```text
Dataset:
CelebA-Spoof

Split protocol (chuẩn 3 tập độc lập, chống data leakage):
- Train subset: Stratified từ metas/intra_test/train_label.json (~100K ảnh)
- Validation subset: Stratified từ metas/intra_test/train_label.json (~15K ảnh), DISJOINT 100% (train_idx ∩ val_idx = ∅)
- Final Test set: metas/intra_test/test_label.json (~60K ảnh, held-out hoàn toàn)
- Quick Test subset (tuỳ chọn): Stratified từ test_label.json (10K ảnh, fixed seed) cho debug/so sánh nhanh

Input:
224×224

Crop:
1.5× expanded bbox

Classes:
Real / Physical Spoof / Digital Spoof
```

## Augmentation

```text
HorizontalFlip:
    p=0.5

Brightness:
    ±10%

Contrast:
    ±10%

Scale perturbation:
    0.90–1.00 hoặc bbox-scale perturbation nhẹ

Translation:
    ±5%
```

## Model

```text
MobileNetV3-Large
ImageNet pretrained
3-class classifier
```

## Loss

```text
CrossEntropyLoss
```

Start with:

```text
class weights = [1,1,1]
```

## Optimizer

```text
AdamW
```

## Learning rate

```text
backbone:
    lower LR

classifier/new layers:
    higher LR
```

Ví dụ:

```text
backbone = 1e-5
new head = 1e-4
```

## Scheduler

```text
CosineAnnealingLR
T_max = number of training epochs
```

## Training

Preliminary:

```text
100K sampled training images
seed = 42
```

Official baseline:

```text
prefer full training set when resources allow
```

## Checkpoint

```text
select using validation ACER
```

## Threshold

```text
calibrate on validation/dev
lock threshold
evaluate on test
```

## Final metrics

```text
APCER
BPCER
ACER
AUC
Accuracy
```

---

# 24. Recommended experimental stages

## Stage 0 — Pipeline validation

Mục tiêu:

```text
data loading works
crop works
RGB/BGR correct
training works
checkpoint works
ONNX export works
ONNX inference matches PyTorch
```

Không dùng kết quả Stage 0 để claim research.

---

## Stage 1 — Preliminary Spatial Baseline

```text
CelebA-Spoof
- Train subset: 100K samples (stratified từ train_label.json)
- Validation subset: 15K samples (stratified từ train_label.json, disjoint 100%)
- Held-out Test set: test_label.json (có thể dùng quick test subset 10K để test nhanh)
mild augmentation cho train, val_transform cho val/test
MobileNetV3-Large
```

Đánh giá:

```text
Validation subset:
    Giám sát epoch, chọn best checkpoint, calibrate threshold

CelebA-Spoof held-out test set (hoặc quick test subset):
    Đánh giá unbiased chính thức

LCC-FASD:
    Preliminary external sanity check
```

---

## Stage 2 — Official Spatial Baseline

Khi pipeline ổn định:

```text
fixed split
fixed preprocessing
full/reasonably large training set
mild augmentation
validated checkpoint
calibrated threshold
```

Đánh giá chính thức:

```text
in-domain
```

và chuẩn bị protocol cross-domain.

---

## Stage 3 — Proposed Frequency-aware Model

Giữ toàn bộ protocol Stage 2, chỉ thay architecture:

```text
MobileNetV3
+
DCT
+
Tiny Frequency CNN
+
Concatenation
```

Hai model train độc lập.

So sánh:

```text
Spatial-only
vs
Spatial + Frequency
```

---

## Stage 4 — Official Cross-Dataset Evaluation

```text
Train:
CelebA-Spoof

Test:
OULU-NPU
SiW
+ có thể LCC-FASD
```

Không fine-tune trên test datasets trong main cross-dataset experiment.

---

# 25. Research interpretation

Không nên đặt mục tiêu:

```text
Frequency model MUST improve accuracy.
```

Thay vào đó:

```text
H1:
Frequency features provide complementary information.

H2:
Any benefit may be more visible under domain shift.

H3:
The additional frequency branch should have modest computational overhead.
```

Expected pattern:

```text
                         Spatial     + Frequency

In-domain               strong       similar / slightly better

Cross-domain            degrade      ideally less degradation

Cost                    baseline     modest increase
```

Đây là hypothesis, không phải kết quả được giả định trước.

---

# 26. Priority checklist

## 🔴 Must fix before official baseline

- [x] **Đã hoàn thành khảo sát EDA toàn diện (`antispoof/notebooks/eda/celeb-a-spoof-eda.ipynb`)**: Thống kê quy mô 561,575 ảnh, phát hiện 3 subject overlap trong Kaggle mirror, kiểm tra 3,000 BB files 100% hợp lệ. (Xem mục 27.3.1).
- [x] **Đã hoàn thành thử nghiệm Quick Test 10K (`antispoof/notebooks/quick_test/pbl6-quicktest-10k.ipynb`)**: Đạt AUC 0.9734, ACER 4.13% (sau calibrate ngưỡng 2.729), tốc độ 17.53 ms/ảnh (~57 FPS), phân tích rõ 98.5% ca lọt lưới là Physical Spoof. (Xem mục 27.3.2).
- [ ] **Tách bạch 3 tập dữ liệu (Train / Val / Test) chống Data Leakage**: Hiện tại notebook Cell 5 đang trỏ `VAL_JSON` vào `test_label.json` (test set bị dùng làm validation để chọn checkpoint và threshold). Cần:
  - Trích xuất `train_subset` (stratified) từ `train_label.json` với `train_transform`.
  - Trích xuất `val_subset` (stratified) từ `train_label.json`, **bảo toàn `train_idx ∩ val_idx = ∅`**, dùng `val_transform` (không augment) để theo dõi epoch, lưu `best_model_state` và calibrate threshold.
  - Giữ `test_label.json` làm Held-out Test Set độc lập (hỗ trợ `quick_test_subset` có stratified sampling nếu cần test nhanh).
- [ ] Giảm augmentation về mild policy.
- [ ] Sửa `CosineAnnealingLR(T_max=epochs)` — **vẫn chưa sửa trong notebook (hiện T_max=10)**.
- [ ] Calibration threshold trên validation subset (không optimize threshold trên test set).
- [ ] Chọn checkpoint bằng ACER hoặc metric PAD phù hợp trên validation subset.
- [ ] Sửa `best_model_state` bằng `copy.deepcopy(model.state_dict())` (tránh checkpoint bị mutate các epoch sau).
- [ ] Kiểm tra RGB/BGR giữa training và inference.
- [ ] Đồng nhất crop / resize / normalization.
- [ ] Tách rõ FP32 ONNX và INT8 ONNX, không ghi đè `best_model_quantized.onnx` (MiniFASNetV2SE).
- [ ] **Lưu trữ song song cả model PyTorch (`.pth`) và ONNX (`.onnx`)**: Không chỉ xuất ONNX mà phải lưu cả file `.pth` để có thể resume training qua các session Kaggle và phục vụ fine-tuning / transfer learning cho Stage 3 (thêm frequency branch).
- [ ] Ghi rõ training hiện tại dùng 100K samples.
- [ ] Sửa logic detect mean/std trong `predictor.py` để rõ ràng hơn cho MNV3 (hiện dùng `mnv4` keyword + fallback `size==224`).

## 🟡 Nên làm sau khi pipeline ổn định

- [ ] Benchmark full training set vs 100K.
- [ ] Robustness augmentation ablation.
- [ ] Compare checkpoint selection by Accuracy vs ACER.
- [ ] Benchmark threshold operating points.
- [ ] Benchmark FP32 vs INT8.
- [ ] Benchmark latency trên Raspberry Pi.

## 🟢 Giai đoạn research sau

- [ ] DCT frequency branch.
- [ ] Spatial-only vs spatial+frequency ablation.
- [ ] Official cross-dataset evaluation.
- [ ] OULU-NPU / SiW.
- [ ] Parameters / FLOPs / latency.
- [ ] Robustness analysis theo attack type/domain.

---

# 27. Lưu ý khi train trên Kaggle

Khi sử dụng Kaggle Notebooks để training (do giới hạn tài nguyên máy local):

## 27.1. Quy chuẩn lưu trữ sau mỗi phiên huấn luyện

> **Sau khi phiên huấn luyện trên Kaggle hoàn tất, toàn bộ notebook (`.ipynb` giữ nguyên đầy đủ output cells) bắt buộc phải được tải về và lưu vào thư mục `antispoof/notebooks/` nhằm:**
>
> - Đảm bảo tính minh bạch và khả năng tái lập thực nghiệm (reproducibility), cho phép thẩm định chéo log và metrics giữa các thành viên mà không cần chạy lại.
> - Dễ dàng đối chiếu định lượng giữa các lần chạy (hyperparameters, kích thước tập dữ liệu, augmentation policy...).
> - Bảo toàn dữ liệu thực nghiệm, tránh mất mát khi phiên làm việc Kaggle bị hủy hoặc hết hạn lưu trữ.

Quy ước đặt tên file notebook lưu xuống:

```text
antispoof/
└── notebooks/
    ├── eda/
    │   ├── celeb-a-spoof-eda.ipynb            ← Notebook EDA phân tích toàn diện 561K ảnh (đã run hoàn tất)
    │   ├── celeba_spoof_eda_summary.json      ← JSON thống kê tổng hợp (train/test/attack types/overlap)
    │   └── celeba_spoof_sample_gallery.png    ← Visual gallery mẫu 3 classes & crop 1.5x
    ├── quick_test/
    │   ├── pbl6-quicktest-10k.ipynb           ← Notebook Quick Test 10K stratified subset (đã run hoàn tất)
    │   ├── quick_test_summary.json            ← JSON kết quả định lượng chi tiết (AUC, ACER, APCER)
    │   ├── threshold_sweep_results.csv        ← Bảng quét 1,000 mốc ngưỡng threshold
    │   ├── quick_test_evaluation_plots.png    ← Đồ thị ROC, DET, Confusion Matrix, Threshold Sweep
    │   └── worst_failure_cases.png            ← Visual các ca sai nghiêm trọng nhất (False Accepts/Rejects)
    ├── train_mnv3_100k_mild_aug_v1.ipynb      ← notebook đã run, có output đầy đủ
    ├── train_mnv3_full_mild_aug_v2.ipynb      ← notebook đã run, có output đầy đủ
    └── compare_models.ipynb                   ← notebook so sánh các model (xem mục 27.2)
```

File model tương ứng lưu vào:

### ⚠️ BẮT BUỘC: Lưu trữ song song cả Model PyTorch (`.pth`) và ONNX (`.onnx`)

> **[QUAN TRỌNG] Sau mỗi đợt huấn luyện trên Kaggle, bắt buộc phải tải về và lưu trữ cả hai định dạng model (`.pth` và `.onnx`):**

| Định dạng | Tên file quy ước | Vai trò & Mục đích sử dụng | Lý do không được bỏ sót |
|---|---|---|---|
| **PyTorch Checkpoint (`.pth`)** | `best_mnv3_large_3class.pth`<br>`checkpoint_last.pth` | **Nghiên cứu & Tiếp tục Huấn luyện (Research & Training):**<br>- Dùng để resume training qua các session Kaggle mới khi chạm giới hạn 12h.<br>- Dùng để fine-tuning hoặc transfer learning cho Stage 3 (ghép nối nhánh tần số DCT / Tiny Frequency CNN).<br>- Phân tích trọng số, trích xuất feature embeddings.<br>- Cho phép export lại sang ONNX với các dynamic axes, opset mới mà không cần train lại. | Nếu chỉ lưu `.onnx`, **toàn bộ khả năng huấn luyện tiếp, fine-tune và mở rộng sang Stage 3 sẽ bị mất hoàn toàn**, buộc phải train lại từ đầu rất tốn thời gian. |
| **ONNX Runtime (`.onnx`)** | `mnv3_large_3class_best.onnx` (FP32)<br>`mnv3_large_3class_best_int8.onnx` (INT8) | **Triển khai & Đánh giá Hiệu năng (Deployment & Benchmark):**<br>- Triển khai trực tiếp lên backend production (FastAPI, Docker, Raspberry Pi) qua `onnxruntime` mà không cần cài thư viện PyTorch cồng kềnh.<br>- Thực hiện Quantization sang INT8 để tối ưu tốc độ.<br>- Benchmark tốc độ xử lý (ms/ảnh, FPS) và so sánh trực tiếp với MiniFASNetV2SE. | File `.pth` không thể chạy trực tiếp trên các edge runtime nhẹ hoặc đòi hỏi dependencies rất nặng (PyTorch/CUDA). |

Cấu trúc lưu trữ model khuyến nghị trong `antispoof/models/`:

```text
antispoof/models/
├── checkpoints/                                 ← Thư mục chứa trọng số PyTorch .pth
│   ├── best_mnv3_large_3class.pth               ← Best checkpoint theo validation ACER (dùng để export/fine-tune)
│   └── checkpoint_last.pth                      ← Checkpoint epoch cuối (để resume training qua các session Kaggle)
│
├── best_model_quantized.onnx                    ← MiniFASNetV2SE INT8 (600 KB, download, production default)
├── mnv3_large_3class_best.onnx                 ← MNV3 FP32 baseline (tự train, export từ best .pth)
├── mnv3_large_3class_best_int8.onnx            ← MNV3 INT8 (sau quantize)
└── mnv4_best_224.onnx                          ← MNV4 FP32 experimental
```

## 27.2. Notebook so sánh model (compare_models.ipynb)

Phải có **một notebook riêng** (`antispoof/notebooks/compare_models.ipynb`) để test và so sánh tất cả các model hiện có trong `antispoof/models/`. Notebook này phải:

**a. Load và so sánh tất cả model:**

```python
models_to_compare = {
    "MiniFASNetV2SE (INT8, 128px)": "antispoof/models/best_model_quantized.onnx",
    "MNV3-Large 3class (FP32, 224px)": "antispoof/models/mnv3_large_3class_best.onnx",
    "MNV4 (FP32, 224px)": "antispoof/models/mnv4_best_224.onnx",
}
```

**b. Các mục so sánh cần có:**

```text
Metrics trên tập test chung (CelebA-Spoof / LCC-FASD):
    - Accuracy
    - AUC
    - APCER / BPCER / ACER (ghi rõ threshold đang dùng)

Model size:
    - Dung lượng file (MB/KB)
    - FP32 vs INT8

Tốc độ inference:
    - ms/image trên CPU
    - ms/image trên GPU (nếu có)

Preprocessing pipeline:
    - model_img_size
    - mean/std normalization
    - color order (phải là RGB)
    - crop strategy
```

**c. Bảng tổng hợp cuối notebook:**

```text
| Model                   | Size    | Accuracy | AUC   | ACER  | Latency CPU |
|-------------------------|---------|----------|-------|-------|-------------|
| MiniFASNetV2SE INT8     | ~600 KB | ...      | ...   | ...   | <10 ms      |
| MNV3-Large FP32 (224px) | ~16 MB  | ...      | ...   | ...   | ...         |
| MNV4 FP32 (224px)       | ~5 MB   | ...      | ...   | ...   | ...         |
```

**d. Điều kiện notebook compare hợp lệ:**

- Dùng cùng một tập test ảnh — lấy từ `test_label.json` chính thức của CelebA-Spoof (hoặc `quick_test_subset` chuẩn hoá 10K ảnh, seed=42 để chạy nhanh), tuyệt đối không dùng training hay validation set của bất kỳ model nào.
- Dùng đúng preprocessing pipeline của từng model (MiniFASNetV2SE dùng 128×128; MNV3 dùng 224×224 crop 1.5×, RGB, đúng mean/std).
- Report đầy đủ threshold đang dùng khi tính APCER/BPCER/ACER (khóa threshold từ validation, không tune threshold trên test).
- Không fine-tune hay thay đổi trọng số model trong notebook compare.
- Notebook compare phải có thể chạy độc lập (không phụ thuộc vào notebook training).

---

## 27.3. Kết quả thực nghiệm sơ bộ từ 2 Notebooks (EDA & Quick Test 10K)

Hai notebook thực nghiệm đã được triển khai và hoàn tất trên môi trường Kaggle, toàn bộ logs, output cells và artifacts liên quan đã được lưu trữ tập trung tại thư mục `antispoof/notebooks/`. Phần này tổng hợp các kết quả thực nghiệm và phân tích định lượng làm cơ sở kỹ thuật cho dự án.

> **[LƯU Ý PHƯƠNG PHÁP LUẬN — Giới hạn của kết quả sơ bộ]**
> - **Dữ liệu EDA (mục 27.3.1)**: Phản ánh cấu trúc khách quan của bộ dữ liệu gốc và hoàn toàn độc lập với các vấn đề kỹ thuật của pipeline.
> - **Chỉ số Quick Test 10K (mục 27.3.2)**: **Là kết quả đánh giá sơ bộ mang tính kiểm tra thông suốt (sanity check)** nhằm khảo sát độ nhạy của ngưỡng quyết định và phân loại các dạng tấn công, **chưa phải mốc benchmark chính thức cuối cùng của dự án** do chịu ảnh hưởng từ 2 lớp leakage của code tiền nhiệm:
>   1. **Checkpoint leakage**: Mô hình `mnv3_large_3class_best.onnx` vốn được lưu từ `best_acc` trong `antiproof.ipynb`, nơi `val_loader` bị trỏ nhầm vào `test_label.json` (mục 17.1).
>   2. **Threshold leakage**: Ngưỡng `2.729` được quét (sweep) trực tiếp trên tập 10,000 ảnh này để đo biên độ cải thiện của ACER, chưa tuân thủ quy trình chuẩn là khóa ngưỡng độc lập trên `val_subset`.
>
> **Kết luận phương pháp**: Các chỉ số AUC, ACER, APCER, BPCER trong đợt test 10K này phục vụ mục đích **minh họa thực nghiệm cho tầm quan trọng của việc calibrate ngưỡng và xác định các điểm mù theo loại tấn công**. Mốc benchmark chính thức sẽ được thiết lập sau khi: (a) tách 3 tập dữ liệu `train_subset` / `val_subset` / `test_set` hoàn toàn độc lập (mục 17), (b) huấn luyện lại checkpoint chuẩn, (c) khóa ngưỡng tối ưu từ `val_subset`, và (d) thực hiện đánh giá một lần duy nhất trên Held-out Test Set.

### 27.3.1. Kết quả từ Notebook EDA (`antispoof/notebooks/eda/celeb-a-spoof-eda.ipynb`)

Mục đích: Khảo sát toàn diện quy mô dữ liệu, phân bố nhãn, cấu trúc các dạng tấn công và chất lượng bounding box trên bộ dữ liệu CelebA-Spoof (giao thức `intra_test`).

#### 1. Thống kê tổng quan dữ liệu:
- **Tổng số lượng mẫu**: **561,575 ảnh** (Train: 494,405 ảnh; Test: 67,170 ảnh).
- **Phân bố 3 classes (Train vs Test)**:
  - **Class 0 (Real Face)**:
    - Train: 162,462 ảnh (**32.86%**)
    - Test: 19,923 ảnh (**29.66%**)
  - **Class 1 (Physical Spoof — Print, Poster, Masks)**:
    - Train: 226,715 ảnh (**45.86%**)
    - Test: 35,495 ảnh (**52.84%**)
  - **Class 2 (Digital Spoof — Phone, Tablet, PC Replay)**:
    - Train: 105,228 ảnh (**21.28%**)
    - Test: 11,752 ảnh (**17.50%**)
- **Phân bố 11 loại hình tấn công (Attack Types)**:
  - Dữ liệu train và test bao quát đủ 11 nhóm: `Live`, `Poster`, `A4 Paper`, `Face Mask`, `Upper Body Mask`, `Region Mask`, `PC Screen`, `Tablet/Pad Screen`, `Phone Screen`, `3D Mask`, và `Khác (10)`. Tỷ lệ giữa các hình thức tấn công tương đối đồng đều trong từng nhóm lớn.

#### 2. Phát hiện quan trọng — Rò rỉ định danh (Subject Overlap Bug):
- **Khảo sát Subjects**: Train chứa **8,192 subjects**, Test chứa **1,004 subjects**.
- **Kết quả kiểm tra**: `is_subject_disjoint = false`, phát hiện **3 subjects bị trùng lặp** giữa tập Train và Test (`overlap_subjects = 3`).
- **Tác động & Khuyến nghị**: Dù con số 3 subjects là rất nhỏ (chiếm ~0.03% tổng số subjects), điều này cho thấy bộ mirror metadata trên Kaggle (`attentionlayer241`) không hoàn toàn subject-disjoint 100% như mô tả lý thuyết của paper. Khi tạo split train/val chính thức, cần đưa bộ lọc loại bỏ hoàn toàn các folder subject này khỏi tập train để bảo đảm tính held-out khách quan tuyệt đối cho tập test.

#### 3. Kiểm định chất lượng Bounding Box & Tiền xử lý:
- Đã kiểm tra ngẫu nhiên 3,000 file `_BB.txt`: **100% file tồn tại và có tọa độ hợp lệ**, không gặp tình trạng rỗng hoặc file lỗi.
- Công thức chuẩn hóa tọa độ và mở rộng vùng cắt **Crop 1.5×** (Center-based square expansion với `BORDER_REFLECT_101`) hoạt động ổn định, bao trọn toàn bộ khuôn mặt và vùng bối cảnh viền ngoài (context cues) cần thiết cho việc nhận diện gian lận.

---

### 27.3.2. Kết quả từ Notebook Quick Test 10K (`antispoof/notebooks/quick_test/pbl6-quicktest-10k.ipynb`)

Mục đích: Đánh giá nhanh hiệu năng của mô hình baseline hiện tại (`mnv3_large_3class_best.onnx`) trên 1 tập con kiểm thử chuẩn hóa 10,000 ảnh (stratified sample, `seed=42`) từ `metas/intra_test/test_label.json`.

> **[LƯU Ý]** Checkpoint kiểm thử và ngưỡng 2.729 được sử dụng dưới điều kiện khảo sát sơ bộ (xem lưu ý phương pháp luận ở đầu mục 27.3). Các số liệu dưới đây phản ánh bản chất tương đối giữa các nhóm tấn công và tác động của ngưỡng, không phải số liệu công bố cuối cùng.

#### 1. Cấu hình kiểm thử:
- **Mô hình**: `mnv3_large_3class_best.onnx` (MobileNetV3-Large FP32, tự train 3-class).
- **Tập mẫu**: 10,000 ảnh (Real Face: 2,966; Physical Spoof: 5,284; Digital Spoof: 1,750).
- **Phần cứng kiểm thử**: CPU trên Kaggle (onnxruntime `CPUExecutionProvider`).
- **Tốc độ inference**: **17.53 ms/ảnh** (~**57.0 FPS** trên CPU) — Đảm bảo khả năng chạy realtime mượt mà ngay cả khi không có GPU.

#### 2. Bảng số liệu hiệu năng chi tiết:

| Tiêu chí / Metric | Điểm vận hành mặc định (`Threshold = 0.0`) | Điểm vận hành tối ưu (`Calibrated Threshold = 2.729`) | Ý nghĩa & Nhận xét |
|---|---|---|---|
| **ROC AUC** | **0.9734** | **0.9734** | Năng lực phân tách tổng thể của spatial baseline rất tốt. |
| **Độ chính xác (Accuracy)** | 89.86% | **94.83%** | Tăng +4.97% sau khi căn chỉnh ngưỡng. |
| **APCER (Tấn công lọt lưới)** | **14.19%** (998 ca lỗi / 7,034 spoofs) | **6.68%** (470 ca lỗi / 7,034 spoofs) | **Giảm hơn 50% số vụ tấn công lọt lưới (chặn thêm 528 ca giả mạo)**. |
| **BPCER (Từ chối nhầm người thật)** | **0.54%** (16 ca lỗi / 2,966 reals) | **1.58%** (47 ca lỗi / 2,966 reals) | Tăng nhẹ ~1%, hoàn toàn nằm trong mức chấp nhận được của hệ thống xác thực. |
| **ACER (Sai số trung bình)** | **7.36%** | **4.13%** | **Giảm gần 1 nửa sai số (từ 7.36% xuống 4.13%)**. |

#### 3. Phân tích chi tiết theo loại hình tấn công (Failure Breakdown):
Khi mổ xẻ 470 ca tấn công lọt lưới (tại threshold tối ưu 2.729):
- **Digital Spoof (Màn hình điện thoại, máy tính bảng, PC)**:
  - APCER = **0.40%** (chỉ có **7 / 1,750** ca bị nhận nhầm).
  - Tỷ lệ nhận diện đúng màn hình đạt **99.60%**.
- **Physical Spoof (In ảnh trên giấy A4, Poster, Mặt nạ cắt/vùng)**:
  - APCER = **8.76%** (có **463 / 5,284** ca bị nhận nhầm).
  - **Chiếm 98.5% tổng số ca tấn công lọt lưới của toàn hệ thống** (463 / 470 ca false acceptances)! Các ca trượt nặng nhất rơi vào Region Mask (`v[40] == 5`) và Face Mask (`v[40] == 3`).

---

### 27.3.3. Đánh giá kỹ thuật & Định hướng triển khai

1. **Ý nghĩa thực nghiệm của việc Căn chỉnh Ngưỡng (Threshold Calibration)**:
   - Việc chỉ sử dụng argmax mặc định (`logit_diff >= 0`) khiến mô hình bị lọt tới **14.19% tấn công**, trong khi dịch chuyển ngưỡng lên 2.729 đưa ACER xuống **4.13%**. Khoảng chênh lệch lớn này khẳng định: **điểm vận hành (operating point) đóng vai trò quyết định đến hiệu năng thực tế của hệ thống PAD**, chứng minh việc bổ sung bước threshold calibration trên tập validation là bắt buộc.
   - **Lưu ý về độ tin cậy của chỉ số**: Ngưỡng 2.729 ở lần chạy này được quét trực tiếp trên tập 10,000 ảnh test, đồng thời checkpoint có sự ảnh hưởng từ test set (mục 17.1). Do đó, mức ACER 4.13% có xu hướng lạc quan hơn thực tế. Cần thực hiện quy trình chuẩn (khóa ngưỡng trên `val_subset` không chồng lấn rồi đánh giá trên `test_set`) trước khi lấy số liệu làm kết quả chính thức của dự án.
2. **Cơ sở thực nghiệm chứng minh sự cần thiết của Nhánh Tần số (Stage 3 — Frequency Branch)**:
   - Kết quả phân rã lỗi cho thấy mô hình spatial nhận diện rất tốt các vụ tấn công qua màn hình (Digital Replay — tỷ lệ lỗi chỉ **0.40%**), nhưng lại gặp khó khăn rõ rệt trước các dạng tấn công in ấn và mặt nạ (Physical Spoof — tỷ lệ lỗi **8.76%**, chiếm tới **98.5% tổng số ca lọt lưới**).
   - Hiện tượng này hoàn toàn phù hợp với lý thuyết thị giác máy tính: ảnh in trên giấy phẳng bị triệt tiêu các đặc trưng không gian (spatial depth, screen bezel, moiré), khiến mạng CNN thuần spatial dễ bị nhầm lẫn với da mặt thật. Đây là cơ sở thực nghiệm thuyết phục nhất khẳng định **nhánh tần số (DCT / High-frequency analysis)** ở Stage 3 là hướng đi đúng đắn, nhằm bổ khuyết thông tin về artifact in ấn và cấu trúc dải tần cao mà nhánh spatial đang bỏ sót.
3. **Biện pháp xử lý rò rỉ định danh (Subject Overlap)**:
   - Quá trình EDA đã phát hiện 3 subject IDs bị trùng lặp giữa `train_label.json` và `test_label.json` trong bản mirror dữ liệu CelebA-Spoof. Khi triển khai code phân chia dữ liệu cho baseline chính thức, cần áp dụng bộ lọc (filter) loại bỏ dứt điểm các thư mục thuộc 3 subject này khỏi tập train/val để đảm bảo tính held-out tuyệt đối cho tập test.
4. **Tính khả thi về Hiệu năng Thực thi (Inference Latency)**:
   - Tốc độ **17.53 ms/ảnh (~57 FPS)** đo đạc trực tiếp trên CPU Kaggle xác nhận kiến trúc MobileNetV3-Large 224×224 hoàn toàn đáp ứng tốt yêu cầu xử lý thời gian thực, sẵn sàng cho việc đóng gói và triển khai trên các thiết bị biên (Raspberry Pi, CPU Server) mà không gây tắc nghẽn đường truyền.
5. **Thống nhất định hướng thực nghiệm**: Toàn bộ kết quả thử nghiệm 10K hiện tại đóng vai trò là mốc kiểm chứng sơ bộ về tính thông suốt của pipeline và cơ sở định hướng cho kiến trúc Stage 3. Nhóm phát triển sẽ tiến hành chuẩn hóa lại pipeline huấn luyện theo protocol 3 tập độc lập (mục 17 và 28) để thiết lập benchmark baseline chính thức.

---

# 28. Final recommended pipeline

```text
                                 CelebA-Spoof
                                      │
                   ┌──────────────────┴──────────────────┐
                   ▼                                     ▼
      metas/intra_test/train_label.json     metas/intra_test/test_label.json
                   │                                     │
           Subject-disjoint                      Subject-disjoint
                   │                                     │
         Stratified Split (seed=42)              Stratified Quick Sample (optional)
                   │                                     │
         ┌─────────┴─────────┐                           │
         ▼                   ▼                           ▼
    Train Subset        Val Subset              Held-out Test Set
      (~100K)             (~15K)                 (~60K hoặc 10K quick)
         │                   │                           │
  Expanded Crop 1.5×  Expanded Crop 1.5×          Expanded Crop 1.5×
         │                   │                           │
  Mild Augmentation     NO Augmentation             NO Augmentation
         │                   │                           │
      224×224             224×224                     224×224
         │                   │                           │
   RGB Normalize       RGB Normalize               RGB Normalize
         │                   │                           │
         ▼                   ▼                           │
  MobileNetV3-Large   MobileNetV3-Large                  │
 (Gradient Update)   (Validation Evaluation)             │
         │                   │                           │
         └─────────┬─────────┘                           │
                   ▼                                     │
         Best Model Checkpoint                           │
                   │                                     │
                   ▼                                     │
         Validation Calibration                          │
         (Sweep on Val Subset)                           │
                   │                                     │
                   ▼                                     │
            Locked Threshold                             │
                   │                                     │
                   └──────────────────┬──────────────────┘
                                      ▼
                        In-domain Held-out Evaluation
                                      +
                        LCC-FASD External Check (Sanity)
                                      │
                                      ▼
                         APCER / BPCER / ACER / AUC
```

---

# 29. Key principle

Baseline spatial không cần quá phức tạp.

Mục tiêu của baseline là tạo một **clean reference point**:

```text
MobileNetV3-Large
+
same data
+
same preprocessing
+
same augmentation
+
same training protocol
+
same evaluation protocol
```

Sau đó proposed model chỉ thêm:

```text
DCT
+
Tiny Frequency CNN
```

và thay đổi được quy về đúng một câu hỏi:

> **Does explicit frequency-domain representation provide complementary information beyond the spatial baseline?**