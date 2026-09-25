# Hướng dẫn chẩn đoán Face PAD: Dataset/Kaggle vs Camera Runtime

## 1. Bộ công cụ

Bộ chẩn đoán gồm 2 phần:

1. `pad_post_training_deep_evaluation_kaggle.ipynb`
   - chạy trên Kaggle;
   - dùng CelebA-Spoof và exact split manifest của training run;
   - kiểm tra model sâu sau training;
   - không cần camera.

2. `camera_pad_diagnostic.py`
   - chạy trên máy/repository có camera;
   - dùng chính ONNX + runtime config đã export;
   - kiểm tra threshold, gamma, RGB/BGR và crop sensitivity;
   - xuất `camera_pad_diagnostic.csv`.

Hai phần nối với nhau bằng CSV:

```text
Camera diagnostic
      ↓
camera_pad_diagnostic.csv
      ↓
Upload to Kaggle
      ↓
Compare Camera Real vs CelebA Real PAD-score distribution
```

Mục tiêu là phân biệt:

```text
Model/training problem
vs
Runtime/preprocessing bug
vs
Detector/crop mismatch
vs
True camera-domain shift
```

---

# 2. Thứ tự chẩn đoán bắt buộc

Không fine-tune hay train lại ngay.

Chạy theo thứ tự:

```text
A. Kaggle: model khỏe trên CelebA?
              ↓
B. Local: đúng model/threshold?
              ↓
C. Local: gamma mismatch?
              ↓
D. Local: RGB/BGR mismatch?
              ↓
E. Local: bbox/crop sensitivity?
              ↓
F. Kaggle: Camera Real score có shift khỏi CelebA Real?
```

---

# 3. A — Chạy notebook Kaggle trước

## Inputs

Add:

- CelebA-Spoof;
- output của notebook train v5.1.

Artifacts tối thiểu:

```text
*_best.pth
*_best_meta.json
*_runtime_config.json
celeba_spoof_subject_disjoint_*_split_seed42.npz
```

Nên có thêm:

```text
*_best.onnx
```

Notebook tự cache prediction, vì vậy model chỉ cần chạy inference một lần cho mỗi split.

---

# 4. Những kết quả Kaggle cần xem

## 4.1. Validation/Test metrics

Quan trọng:

```text
APCER
BPCER
ACER
AUC
EER
```

Nếu CelebA held-out Test cũng gần như luôn Spoof:

> Đây không phải vấn đề riêng của camera. Kiểm tra model/training/checkpoint/threshold.

Nếu CelebA tốt nhưng camera hầu như luôn Spoof:

> Runtime mismatch hoặc domain shift có xác suất cao hơn.

---

## 4.2. Threshold audit

Notebook tính lại minimum-ACER threshold trên **Validation**.

Stored threshold và recomputed threshold phải gần nhau.

Nếu khác rất lớn:

```text
stored threshold ≠ validation optimum
```

hãy kiểm tra artifact/config/checkpoint có bị trộn giữa hai run hay không.

Không bao giờ chọn threshold mới dựa trên Test.

---

## 4.3. Per-attack breakdown

Đặc biệt xem:

```text
Physical Spoof APCER
Digital Spoof APCER
```

Nếu Digital rất tốt nhưng Physical yếu, đó có thể là limitation của spatial baseline chứ không phải bug.

---

## 4.4. Subject-cluster bootstrap

Notebook bootstrap theo **subject**, không chỉ theo ảnh.

Nếu confidence interval cực rộng:

> Metric có variance lớn theo identity/domain; đừng quá tin một con số point estimate.

---

# 5. B — Chạy camera diagnostic local

Đặt script tại repo root hoặc nơi import được package `antispoof`.

Ví dụ:

```bash
PYTHONPATH=. python camera_pad_diagnostic.py \
  --model antispoof/models/mnv3_e1_official_v5_1_best.onnx \
  --runtime-config path/to/mnv3_e1_official_v5_1_runtime_config.json \
  --camera 0
```

Controls:

```text
SPACE → freeze frame + chọn face ROI + chạy toàn bộ diagnostic
q     → thoát
```

Script không cần detector để test core model. Bạn tự chọn ROI, nhờ đó tách được:

```text
classifier/preprocessing problem
```

khỏi:

```text
detector problem
```

---

# 6. Test 1 — Đúng model chưa?

Script in:

```text
Model path
Input size
Color order
Mean/std
Probability threshold
Logit threshold
```

MNV3 baseline phải có:

```text
input = 224
color = RGB
crop = 1.5×
mean = [0.5931, 0.4690, 0.4229]
std  = [0.2471, 0.2214, 0.2157]
research gamma = False
```

Nếu app đang load:

```text
best_model_quantized.onnx
```

thì đó là MiniFASNet production model cũ, không phải MNV3 research baseline.

---

# 7. Test 2 — Threshold semantics

Runtime config có hai loại threshold:

```text
calibrated_logit_threshold
predictor_threshold_probability
```

`AntiSpoofPredictor(threshold=...)` nhận **probability**, không phải raw logit.

Đúng:

```python
threshold = predictor_threshold_probability
```

Sai:

```python
threshold = calibrated_logit_threshold
```

Ví dụ nếu logit threshold = `2.7` mà truyền:

```python
threshold=2.7
```

predictor sẽ clamp probability gần `0.999999`, tương đương logit threshold khoảng `13.8`.

Hệ quả:

> Gần như mọi frame sẽ thành Spoof.

Script tự assert:

```text
predictor.logit_threshold
≈
runtime_config.calibrated_logit_threshold
```

Nếu sai, script dừng ngay.

---

# 8. Test 3 — Gamma A/B

Research model được train/evaluate:

```text
apply_gamma=False
```

Script chạy cùng frame:

```text
No gamma
vs
Gamma=True
```

### Pattern

```text
No gamma:
score = +3.2 → Real

Gamma:
score = -1.5 → Spoof
```

Kết luận:

> Gamma preprocessing đang gây distribution mismatch.

Cách xử lý:

```text
Production no-gamma
```

hoặc nếu bắt buộc dùng gamma:

```text
Frozen model
→ camera/validation DEV set với gamma
→ calibrate deployment threshold riêng
```

Không reuse threshold no-gamma.

---

# 9. Test 4 — RGB/BGR double conversion

OpenCV camera:

```text
BGR
```

Predictor MNV3:

```text
convert_rgb=True
```

nên caller phải đưa frame BGR trực tiếp.

Đúng:

```python
predictor.predict_frame(frame_bgr, bbox)
```

Sai phổ biến:

```python
frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
predictor.predict_frame(frame_rgb, bbox)
```

Predictor đổi thêm lần nữa và model nhận sai channel order.

Script tự chạy một phiên bản **cố ý sai double-swap**.

Nếu:

```text
Correct BGR/no-gamma → Real
Wrong double RGB     → Spoof
```

và app hiện có `BGR2RGB` trước predictor:

> Đã xác định lỗi color-order integration.

---

# 10. Test 5 — Threshold mặc định 0.5

Script so:

```text
calibrated threshold
vs
threshold=0.5
```

Nếu score giống nhau nhưng status khác:

> Không phải model output thay đổi; chỉ là operating point.

Không đổi threshold dựa trên vài frame camera.

Cần một **camera DEV set** riêng để calibration hợp lệ.

---

# 11. Test 6 — Crop / detector sensitivity

Script test:

```text
1.0×
1.25×
1.5×
1.75×
2.0×
```

cùng một frame và ROI.

### Bình thường

Score có thay đổi nhưng không đảo class liên tục.

### Đáng nghi

Ví dụ:

```text
1.0×  -3.0 Spoof
1.25× -1.2 Spoof
1.5×  +3.5 Real
1.75× +4.1 Real
2.0×  +2.8 Real
```

Kết luận:

> Model nhạy với crop context / detector bbox.

Tiếp theo:

- lưu cùng frame;
- lấy bbox YunNet;
- lấy bbox SCRFD;
- chạy script với từng bbox;
- so score.

Nếu manual ROI hoạt động tốt nhưng detector bbox không:

> Detector/crop mismatch, không phải classifier failure.

---

# 12. Test 7 — Correct pipeline vẫn Spoof

Giả sử:

```text
correct BGR
gamma=False
correct calibrated threshold
manual ROI
crop sweep
```

đều cho camera Real score thấp.

Trong khi Kaggle:

```text
CelebA Real score cao
AUC/ACER tốt
```

lúc đó **domain shift** là giả thuyết mạnh.

Không nên kết luận chỉ từ 1 frame.

Thu ít nhất:

```text
50–200 Real camera snapshots
```

với:

- nhiều người;
- nhiều ánh sáng;
- khoảng cách khác nhau;
- góc mặt khác nhau nếu phù hợp deployment.

Script ghi:

```text
camera_pad_diagnostic.csv
```

---

# 13. Camera-vs-CelebA distribution test

Upload CSV vào Kaggle.

Trong notebook:

```python
CAMERA_DIAGNOSTIC_CSV = "/kaggle/input/.../camera_pad_diagnostic.csv"
```

Chạy Cell 19.

Notebook plot:

```text
CelebA Real PAD scores
vs
Camera Real PAD scores
```

## Strong domain-shift pattern

Ví dụ:

```text
CelebA Real median: +4.5
Camera Real median: -0.8

Locked threshold: +1.2
80% Camera Real nằm dưới threshold
```

và pipeline parity đã đúng.

Kết luận:

> Model learned a representation that does not transfer sufficiently to the deployment camera domain.

---

# 14. Nếu là domain shift thì làm gì?

Không tune trực tiếp trên toàn bộ camera samples rồi báo chính các samples đó.

Tạo:

```text
Camera DEV
Camera TEST
```

subject-disjoint nếu có nhiều subjects.

### Bước nhẹ nhất

Không train lại.

```text
Frozen model
→ Camera DEV
→ calibrate deployment threshold
→ Camera TEST
```

Nếu chỉ threshold shift thì có thể giải quyết đủ tốt.

### Nếu representation shift mạnh

Nếu:

```text
AUC camera cũng thấp
```

chứ không chỉ threshold lệch:

> Calibration không đủ; representation không phân tách Real/Spoof tốt trong camera domain.

Khi đó mới cân nhắc:

- thêm real deployment-domain data vào training;
- domain-balanced fine-tuning;
- robust preprocessing;
- proposed frequency branch;
- cross-domain/domain-generalization experiment.

---

# 15. Phân biệt threshold shift và representation shift

Đây là điểm rất quan trọng.

## Threshold shift

Camera:

```text
Real/Spoof distributions vẫn tách tốt
```

nhưng cả hai bị dịch score.

Dấu hiệu:

```text
AUC vẫn cao
ACER tại source threshold xấu
ACER sau DEV calibration cải thiện mạnh
```

→ Calibration/deployment operating point problem.

## Representation shift

Camera:

```text
Real và Spoof distributions overlap mạnh
```

Dấu hiệu:

```text
AUC giảm rõ rệt
```

Dù chọn threshold nào cũng khó đạt APCER/BPCER tốt.

→ Model/domain generalization problem.

---

# 16. Decision tree

```text
Camera almost always Spoof
          │
          ▼
CelebA held-out model healthy?
    │                 │
   NO                YES
    │                 │
    ▼                 ▼
Model/training      Correct ONNX loaded?
problem              │
                 ┌────┴────┐
                NO         YES
                │           │
                ▼           ▼
           Fix model     Threshold semantics correct?
                         │
                    ┌────┴────┐
                   NO         YES
                   │           │
                   ▼           ▼
              Fix threshold   Gamma=False works?
                              │
                         ┌────┴────┐
                        YES        NO
                        │           │
                        ▼           ▼
                 Gamma mismatch   Correct BGR works?
                                  │
                             ┌────┴────┐
                            YES        NO
                            │           │
                            ▼           ▼
                      RGB/BGR bug     Crop-factor sensitive?
                                     │
                                ┌────┴────┐
                               YES        NO
                               │           │
                               ▼           ▼
                         Detector/crop   Compare Camera Real
                           mismatch      vs CelebA Real scores
                                           │
                                      large shift?
                                     ┌─────┴─────┐
                                    YES         NO
                                    │             │
                                    ▼             ▼
                               Domain shift   inspect app
                                            result handling
```

---

# 17. Những lỗi không nên "fix" bằng train lại ngay

Không train lại model chỉ vì:

```text
camera says Spoof
```

trước khi kiểm tra:

1. wrong ONNX;
2. logit/probability threshold confusion;
3. gamma;
4. RGB/BGR;
5. crop;
6. detector.

Đây đều là lỗi có thể làm một model tốt trông như model hỏng.

---

# 18. Files nên giữ sau diagnostic

Local:

```text
camera_pad_diagnostic_output/
├── camera_pad_diagnostic.csv
├── snapshot_001_frame.jpg
├── snapshot_001_crop15.jpg
├── snapshot_001_overlay.jpg
└── ...
```

Kaggle:

```text
pad_deep_evaluation_summary.json
validation_predictions_detailed.csv
test_predictions_detailed.csv
test_per_attack_breakdown.csv
test_subject_breakdown.csv
test_bootstrap_ci.csv
```

Những file này đủ để truy nguyên vấn đề mà không phải rerun inference liên tục.

---

# 19. Kết luận

Bộ test được thiết kế để trả lời theo thứ tự:

```text
Model có tốt trên benchmark không?
↓
Runtime có đúng contract không?
↓
Camera crop/preprocessing có làm score lệch không?
↓
Nếu mọi thứ đúng, domain shift có thực sự tồn tại không?
```

Chỉ sau khi đi hết chuỗi này mới nên quyết định:

```text
recalibrate threshold
fine-tune
thêm deployment data
hay thay đổi architecture
```
