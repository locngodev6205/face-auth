# E1 / E2 Held-out Test on Kaggle — Run & Result Archive Guide

## 1. Mục đích

Training và final held-out evaluation nên tách riêng.

Dùng:

```text
E1 training notebook
E2 training notebook
```

chỉ để train, chọn checkpoint bằng Validation và calibrate threshold trên Validation.

Dùng:

```text
antispoof_e1_heldout_test_v1.ipynb
antispoof_e2_heldout_test_v1.ipynb
```

để chạy final held-out Test.

Protocol:

```text
TRAIN
  ↓
VALIDATION
  ├─ checkpoint selection
  └─ threshold calibration
          ↓
      LOCK MODEL
      LOCK THRESHOLD
          ↓
HELD-OUT TEST NOTEBOOK
```

Hai Test notebook không tối ưu threshold trên Test và không resample Test.

---

## 2. Artifact dùng chung cho cả E1 Test và E2 Test

Attach cùng CelebA-Spoof dataset đã dùng trong training.

Đồng thời attach hai artifact data protocol của E1:

```text
celeba_scrfd_bbox_cache_v5_3_<mode>_seed42.json
celeba_spoof_<mode>_v5_3_edge_mnv3_small_seed42.npz
```

Hai file này xác định:

```text
bbox / bbox source
crop factor
exact Train / Validation / Test membership
```

Không chạy lại SCRFD trong Test notebook và không tạo split mới.

---

## 3. Chạy E1 Test

Notebook:

```text
antispoof_e1_heldout_test_v1.ipynb
```

Artifact model bắt buộc:

```text
mnv3s_e1_<mode>_v5_3_edge_best.onnx
```

Threshold metadata: attach ít nhất một trong hai:

```text
mnv3s_e1_<mode>_v5_3_edge_runtime_config.json
mnv3s_e1_<mode>_v5_3_edge_best_meta.json
```

Khuyến nghị thêm:

```text
mnv3s_e1_<mode>_v5_3_edge_run_config.json
```

để notebook cross-check split fingerprint.

Ở cell config:

```python
E1_RUN_MODE = "preliminary"
E1_ARTIFACT_DIR = ""
```

hoặc:

```python
E1_RUN_MODE = "official"
```

phải đúng với model E1 đã train.

Nếu chỉ mount một bộ artifact, để `E1_ARTIFACT_DIR=""` và notebook tự tìm exact filename dưới `/kaggle/input`.

Nếu mount nhiều run khác nhau, set path cụ thể để tránh lấy nhầm.

---

## 4. Chạy E2 Test

Notebook:

```text
antispoof_e2_heldout_test_v1.ipynb
```

E2 Test vẫn cần **E1 SCRFD cache + E1 split manifest**, vì E2 được train trên chính protocol dữ liệu đó.

Artifact E2 bắt buộc:

```text
e2_freq_only_<mode>_dct_v1_best.onnx
```

Threshold metadata — ít nhất một:

```text
e2_freq_only_<mode>_dct_v1_runtime_config.json
e2_freq_only_<mode>_dct_v1_best_meta.json
```

Khuyến nghị:

```text
e2_freq_only_<mode>_dct_v1_run_config.json
```

Config:

```python
E1_RUN_MODE = "preliminary"
E1_ARTIFACT_DIR = ""
E2_ARTIFACT_DIR = ""
```

`E1_RUN_MODE` ở đây đại diện cho **data protocol** E2 đã reuse.

Không được dùng:

```text
E2 preliminary model + E1 official manifest/cache
```

hoặc ngược lại.

---

## 5. Quy tắc threshold

Cả hai Test notebook đều lấy threshold đã khóa từ Validation.

Ưu tiên:

```text
runtime_config.json
best_meta.json
```

Nếu cả hai cùng có threshold thì giá trị phải trùng nhau; notebook sẽ fail nếu mâu thuẫn.

Có manual override:

```python
MANUAL_LOCKED_LOGIT_THRESHOLD = None
```

chỉ dùng khi metadata bị thiếu và bạn biết chính xác threshold đã calibrate trên Validation.

Không quét threshold trên Test.

---

## 6. E1 Test preprocessing

```text
raw image
→ cached SCRFD / CelebA fallback bbox
→ crop
→ RGB
→ 224×224
→ mean/std normalization
→ MobileNetV3-Small ONNX
```

Frozen contract:

```text
SCRFD crop factor      = 1.55×
CelebA fallback factor = 1.50×
mean                   = [0.5931, 0.4690, 0.4229]
std                    = [0.2471, 0.2214, 0.2157]
gamma                  = OFF
```

---

## 7. E2 Test preprocessing

```text
raw image
→ exact same E1 cached crop
→ RGB 224×224
→ luminance
→ 2D DCT
→ sign(C) × log(1+|C|)
→ per-sample z-score
→ E2 ONNX
```

E2 v1 không dùng handcrafted frequency mask.

E1 Test và E2 Test dùng cùng `test_keys`, nên có thể so sánh sample-for-sample sau này.

---

## 8. Metrics được lưu

Cả hai notebook report:

```text
Binary Accuracy
3-class Accuracy
APCER
BPCER
ACER
AUC
binary confusion matrix
```

Ngoài ra lưu:

```text
locked logit threshold
locked probability threshold
N
ONNX SHA256
Test split fingerprint
model inference time
```

Inference timing trong Test notebook chỉ là diagnostic của môi trường Kaggle hiện tại, không phải Raspberry Pi benchmark.

---

## 9. Per-sample predictions

E1:

```text
e1_test_predictions.csv
```

E2:

```text
e2_test_predictions.csv
```

Các cột gồm:

```text
image key
subject id
attack code
true class
bbox source
3 logits
PAD score d
p_real
binary prediction
3-class prediction
binary correctness
3-class correctness
```

Phải giữ các CSV này vì E3 sau này có thể làm paired comparison với E1/E2 mà không cần chạy lại hai model.

---

## 10. Attack breakdown

Mỗi Test notebook xuất:

```text
e1_attack_breakdown.csv
e2_attack_breakdown.csv
```

Theo từng CelebA-Spoof attack code:

```text
N
class
mean d
median d
std(d)
APCER hoặc BPCER
```

Đây là diagnostic breakdown của frozen Test result, không phải một vòng model selection mới.

---

## 11. Plots

Mỗi notebook xuất:

```text
*_binary_confusion_matrix.png
*_score_distribution.png
```

Score plot chứa locked Validation threshold.

---

## 12. Cấu trúc output E1

```text
e1_<mode>_heldout_test_results/
├── e1_heldout_test_summary.json
├── e1_test_protocol_snapshot.json
├── e1_test_predictions.csv
├── e1_attack_breakdown.csv
├── e1_binary_confusion_matrix.png
└── e1_score_distribution.png
```

Notebook đồng thời tạo:

```text
e1_<mode>_heldout_test_results.zip
```

---

## 13. Cấu trúc output E2

```text
e2_<mode>_heldout_test_results/
├── e2_heldout_test_summary.json
├── e2_test_protocol_snapshot.json
├── e2_test_predictions.csv
├── e2_attack_breakdown.csv
├── e2_binary_confusion_matrix.png
└── e2_score_distribution.png
```

và:

```text
e2_<mode>_heldout_test_results.zip
```

Sau mỗi Kaggle run, tải ZIP về local.

---

## 14. Cách chạy E1 Test trên Kaggle

```text
1. Open antispoof_e1_heldout_test_v1.ipynb
2. Attach CelebA-Spoof
3. Attach frozen E1 artifacts
4. Set E1_RUN_MODE
5. Run config + verification cells
6. Phải thấy:
      === E1 TEST PROTOCOL VERIFIED ===
7. Run inference cells
8. Save Kaggle Version
9. Download:
      e1_<mode>_heldout_test_results.zip
```

Nếu verification fail, không bypass assertion.

---

## 15. Cách chạy E2 Test trên Kaggle

```text
1. Open antispoof_e2_heldout_test_v1.ipynb
2. Attach CelebA-Spoof
3. Attach E1 cache + split manifest
4. Attach frozen E2 artifacts
5. Set E1_RUN_MODE
6. Run verification cells
7. Phải thấy:
      === E2 TEST PROTOCOL VERIFIED ===
8. Run inference
9. Save Kaggle Version
10. Download:
      e2_<mode>_heldout_test_results.zip
```

---

## 16. Lưu trữ lâu dài

Khuyến nghị:

```text
pad_experiments/
├── E1_v5_3/
│   ├── training/
│   │   ├── best.pth
│   │   ├── best.onnx
│   │   ├── best_meta.json
│   │   ├── run_config.json
│   │   └── training_history.json
│   │
│   ├── data_protocol/
│   │   ├── scrfd_cache.json
│   │   ├── split_manifest.npz
│   │   └── preprocess_audit.json
│   │
│   └── test/
│       └── e1_<mode>_heldout_test_results.zip
│
└── E2_dct_v1/
    ├── training/
    │   ├── best.pth
    │   ├── best.onnx
    │   ├── best_meta.json
    │   ├── run_config.json
    │   ├── frequency_branch_best.pth
    │   └── frequency_branch_spec.json
    │
    └── test/
        └── e2_<mode>_heldout_test_results.zip
```

Không overwrite preliminary bằng official.

---

## 17. Artifact nên giữ để chuẩn bị E3

Trước E3, phải giữ ít nhất:

```text
E1:
best.pth
best.onnx
best_meta.json
run_config.json
SCRFD cache
split manifest
preprocess audit
E1 Test ZIP

E2:
best.pth
best.onnx
best_meta.json
run_config.json
frequency_branch_best.pth
frequency_branch_spec.json
E2 Test ZIP
```

E3 phải reuse:

```text
same split manifest
same SCRFD cache
same crop policy
same Test membership
same threshold-calibration procedure
```

---

## 18. Không dùng Test result để sửa E1/E2

Sau khi hai Test notebook chạy xong:

```text
E1 spatial-only
vs
E2 frequency-only
```

chỉ dùng để mô tả behavior.

Không:

```text
đổi threshold
đổi crop
đổi augmentation
retrain vì Test xấu
```

rồi lại gọi cùng Test là untouched held-out Test.

Research comparison chính tiếp theo vẫn là:

```text
E1 Spatial-only
vs
E3 Spatial + Frequency Concat
```

E2 chỉ là diagnostic để hiểu frequency branch.

---

## 19. Preliminary và official

Nếu E1/E2 hiện tại được train từ preliminary 100K setup:

```text
result label = PRELIMINARY
```

Standalone Test notebook không biến preliminary training thành official experiment.

Official flow:

```text
E1 official training
→ freeze official manifest/cache/model/threshold
→ E1 official Test

E2 official using exact official E1 data artifacts
→ freeze E2
→ E2 official Test
```

Lưu riêng hai thế hệ artifact.
