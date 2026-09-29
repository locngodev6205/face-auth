# PAD Micro-Experiment Roadmap v1
## Frequency Representation Search trước Full Training

> Mục tiêu: tìm nhanh một frequency representation có **complementary signal thật sự** với spatial MobileNetV3-Small, đặc biệt dưới domain shift, trước khi tiêu tốn compute cho 100K/full training hoặc gated fusion.

---

## 1. Quyết định chiến lược

Giai đoạn hiện tại chuyển từ:

```text
idea → full train → test → idea khác → full train
```

sang:

```text
hypothesis
   ↓
fixed micro protocol
   ↓
cheap in-domain + cross-domain evidence
   ↓
kill / promote
   ↓
chỉ top candidate mới được tăng budget
```

Full training không còn là công cụ dò architecture. Nó chỉ là **confirmation stage**.

Current E3 đã cho thấy pipeline + concat cơ bản chạy đúng và frequency cue có signal, nhưng cross-domain gain chưa đủ rõ. Vì vậy câu hỏi bây giờ là:

> **Representation frequency nào giữ được spoof cue tốt hơn và bổ trợ spatial một cách ổn định?**

---

## 2. Năm micro experiments

Tạo **5 notebook riêng**, cùng một template và chỉ thay đúng frequency representation:

```text
micro_m0_global_dct_gap.ipynb
micro_m1_dct_pool4x4.ipynb
micro_m2_coord_dct_pool4x4.ipynb
micro_m3_band_aware_dct.ipynb
micro_m4_block_dct_8x8.ipynb
```

### M0 — Global DCT + GAP 1×1 — CONTROL

```text
RGB crop → luminance → global DCT 224×224
→ signed-log → per-sample z-score
→ exact E2 Tiny CNN → GAP 1×1 → 64-D
```

### M1 — Global DCT + Pool 4×4

```text
global DCT → same CNN trunk
→ AdaptiveAvgPool2d(4×4)
→ flatten → Linear → 64-D
```

Hypothesis: coarse frequency position hữu ích và GAP 1×1 đang xóa quá nhiều thông tin.

### M2 — Coord-DCT + Pool 4×4

```text
channel 0 = normalized DCT map
channel 1 = U coordinate in [0,1]
channel 2 = V coordinate in [0,1]
→ lightweight CNN → Pool 4×4 → 64-D
```

Hypothesis: local pattern trên DCT plane chỉ có nghĩa khi biết nó nằm ở frequency coordinate nào.

### M3 — Band-aware DCT

Dùng normalized radial frequency:

```text
r = sqrt(u^2 + v^2) / sqrt(2)
LOW  : r < 1/3
MID  : 1/3 <= r < 2/3
HIGH : r >= 2/3
```

Ba masked maps thành 3 channels → lightweight CNN → Pool 4×4 → 64-D.

Không giả định HIGH luôn tốt nhất; band identity chỉ được expose để network tự học.

### M4 — Block-DCT 8×8

```text
224×224 luminance
→ 28×28 blocks, mỗi block 8×8
→ DCT từng block
→ 64 coefficient identities = 64 channels
→ tensor [64,28,28]
→ signed-log
→ channel-wise normalization từ MICRO-TRAIN only
→ 1×1 bottleneck 64→32
→ lightweight CNN trên spatial grid
→ 64-D
```

M4 giữ đồng thời **frequency identity** và **spatial locality**.

---

## 3. Dataset budget — ưu tiên tốc độ

### Shared fixed micro split

Lấy từ frozen E1 manifest, không resample tùy notebook:

```text
MICRO_TRAIN_N = 5,000
MICRO_VAL_N   = 2,000
SEED          = 42
```

Sampling stratified theo 3 class. Lưu exact keys + fingerprint thành:

```text
micro_manifest_v1_seed42.npz
micro_manifest_v1_seed42.json
```

Không dùng CelebA held-out Test.

### Smoke mode chỉ để debug

```text
SMOKE_TRAIN_N = 2,000
SMOKE_VAL_N   = 1,000
SMOKE_EPOCHS  = 2
```

Smoke result không được dùng để chọn winner.

### Cross-domain development

Dùng **full LCC-FASD evaluation split** vì inference trên vài nghìn ảnh rẻ hơn nhiều so với training.

- reuse cùng SCRFD cache;
- cùng valid-face set cho M0–M4;
- không tune threshold trên LCC;
- gọi LCC là **external-dev/stress set**.

### Reserved — không dùng trong search

```text
CelebA held-out Test
CASIA-FASD
```

CASIA giữ lại cho final external confirmation.

---

## 4. Spatial branch — không retrain 5 lần

Dùng frozen E1 v5.3 MobileNetV3-Small checkpoint làm spatial branch.

Trong Stage-1 screening:

```text
spatial backbone + 256-D projection = FROZEN
```

Mọi notebook phải load **cùng SHA256 checkpoint E1**.

Lợi ích:
- giảm compute;
- giảm variance giữa candidates;
- isolate đúng câu hỏi frequency representation;
- không cần 5 spatial baselines khác nhau.

---

## 5. Training budget cho mỗi notebook

### Stage A — Frequency-only

```text
candidate representation
→ 64-D
→ Linear 64→128
→ Hardswish/ReLU
→ Dropout .2
→ Linear 128→3
```

Config:

```text
max epochs       = 5
early stop       = patience 2
earliest stop    = epoch 3
batch            = 128
AMP              = ON
optimizer        = AdamW
LR               = 3e-4
weight decay     = 1e-4
label smoothing  = 0.1
grad clip        = 5
seed             = 42
```

Checkpoint select bằng micro-Val ACER. Threshold calibrate trên micro-Val only.

### Stage B — Spatial + Frequency CONCAT

Warm-start **chỉ cho architecture discovery**:

```text
spatial   ← frozen E1
frequency ← best Stage-A checkpoint
fusion    ← random
```

Fusion:

```text
spatial 256-D + frequency 64-D
→ concat 320-D
→ Linear 320→128
→ Hardswish
→ Dropout .2
→ Linear 128→3
```

Training:

```text
B1: 2 epochs
    freeze spatial + frequency
    train fusion head only

B2: 2 epochs
    spatial remains frozen
    unfreeze frequency + fusion
```

LR:

```text
frequency = 1e-4
fusion    = 3e-4
```

Sau micro-search, winner phải được retrain clean / independent-init trước final claim.

---

## 6. In-domain + cross-domain evaluation

### In-domain — CelebA micro-Val

Báo:

```text
APCER
BPCER
ACER
AUC
3-class accuracy (supplementary)
```

Đây là development result, không phải final Test.

### Cross-domain — LCC external-dev

Primary:

```text
AUC
HTER
```

Supplementary:

```text
APCER
BPCER
ACER
Accuracy
Detector coverage
```

Binary PAD score:

```text
d = real_logit - logsumexp([physical_logit, digital_logit])
```

Mỗi model dùng threshold riêng khóa từ CelebA micro-Val. Không sweep threshold trên LCC.

---

## 7. Branch diagnostics bắt buộc

Với concat candidate, chạy cùng samples dưới 4 mode:

```text
FULL          = spatial + đúng frequency
FREQ_ZERO     = spatial + zero frequency vector
FREQ_SHUFFLE  = spatial + frequency vector của sample khác
SPATIAL_ZERO  = zero spatial vector + frequency
```

Interpretation:

```text
FULL ≈ FREQ_ZERO
→ fusion ignore frequency

FULL ≈ FREQ_SHUFFLE
→ frequency không mang sample-specific complementary information

FULL > FREQ_ZERO và FULL > FREQ_SHUFFLE
→ evidence tốt rằng frequency cue đang được dùng đúng sample

SPATIAL_ZERO mạnh source nhưng yếu cross-domain
→ nguy cơ frequency shortcut
```

Log thêm:

```text
||spatial_feat||2 mean/std
||frequency_feat||2 mean/std
fusion weight norm: spatial slice
fusion weight norm: frequency slice
```

---

## 8. Artifact tối thiểu

Trong screening **không export ONNX**.

```text
micro_<id>/
├── config.json
├── micro_manifest_fingerprint.json
├── frequency_best.pth
├── concat_best.pth
├── summary.json
├── metrics.csv
├── branch_ablation.csv
└── training_history.csv
```

Optional: một ROC plot. Không cần quantization, Pi benchmark hay nhiều figure.

`summary.json` phải chứa ít nhất candidate/seed/budget/hash E1, frequency-only metrics, concat metrics, LCC metrics, branch ablation và parameter count.

---

## 9. Promotion / kill criteria

Không rank bằng một metric duy nhất.

Đánh dấu `PROMISING` khi có ít nhất một cross-domain gain:

```text
LCC AUC >= M0 + 2.0 percentage points
OR
LCC HTER <= M0 - 2.0 percentage points
```

và:

```text
CelebA micro-Val ACER regression <= 1.5 pp
```

và branch không bị ignore:

```text
FULL > FREQ_ZERO hoặc FULL > FREQ_SHUFFLE
```

Đánh dấu `KILL` nếu:
- source đẹp nhưng LCC degrade rõ;
- `FULL ≈ FREQ_ZERO ≈ FREQ_SHUFFLE`;
- gain chỉ khoảng 0.2–0.5 pp và không có complementary pattern.

Nếu APCER giảm mạnh nhưng BPCER tăng, báo riêng `ΔAPCER / ΔBPCER / ΔHTER / ΔAUC`; không gọi đó là representation improvement nếu AUC/HTER không cải thiện.

Các ngưỡng 2.0 pp / 1.5 pp là **screening heuristics**, không phải significance claim.

---

## 10. Stage-2 — Top-2 confirmation

Chỉ hai candidate tốt nhất:

```text
TRAIN = 10,000
VAL   = 3,000
SEEDS = [42, 133]
```

```text
frequency-only max 8 epochs
concat max 6 epochs
spatial vẫn frozen
LCC full evaluation
```

Survive khi:
- gain cùng hướng ở cả hai seeds;
- mean cross-domain improvement còn tồn tại;
- không chỉ là threshold trade-off;
- branch diagnostics vẫn xác nhận frequency contribution.

Báo `mean ± std`.

---

## 11. Stage-3 — Winner integration test

Chỉ một candidate:

```text
TRAIN = 25,000
VAL   = 5,000
SEED  = 42
```

Lần này chạy clean:

```text
MobileNetV3-Small ImageNet init
frequency random init
fusion random init
joint training
backbone LR 1e-5
new layers LR 1e-4
max 10–12 epochs
```

Nếu advantage biến mất ở Stage-3 thì **không full train**.

---

## 12. Stage-4 — Full / official chỉ sau khi Stage-3 pass

```text
100K preliminary hoặc official full protocol
```

Sau freeze:

```text
CelebA held-out Test
→ CASIA-FASD external confirmation
```

Sau final confirmation mới quyết định có cần E4 gated/adaptive fusion hay không.

---

## 13. Compute budget tương đối

Old-style full run:

```text
100K × 24 epochs ≈ 2.4M sample-epochs
```

Vòng M0–M4:

```text
5 candidates × 5K × (5 frequency + 4 concat epochs)
≈ 225K sample-epochs
```

Tức khoảng **9.4% sample-epoch budget của một full 100K×24 run**; spatial branch còn bị frozen nên backward thực tế rẻ hơn nữa.

---

## 14. Shared caches để tăng tốc Kaggle

Reuse/tạo một lần:

```text
micro_manifest_v1_seed42.npz/json
E1 SCRFD bbox cache
optional CelebA 224 base-crop cache
lcc_scrfd_cache_micro_v1.json
optional LCC 224 crop cache
```

Không rerun SCRFD cho từng notebook.

---

## 15. Run order

```text
0. Verify E1 artifacts + micro manifest + LCC cache
1. M0 establish control
2. M1/M2/M3/M4 có thể chạy song song
3. Gom 5 summary.json
4. Kill 3, promote top 2
5. Top-2: 10K × 2 seeds
6. Chọn winner
7. Winner: 25K clean integration
8. Chỉ nếu pass → 100K/full
```

---

## 16. Bảng tổng hợp sau 5 runs

```text
Candidate
Freq-only Val ACER
Freq-only LCC AUC
Freq-only LCC HTER
Concat Val ACER
Concat Val AUC
Concat LCC AUC
Concat LCC HTER
Concat LCC APCER
Concat LCC BPCER
FULL - FREQ_ZERO ΔAUC
FULL - FREQ_SHUFFLE ΔAUC
Frequency params
Status
```

Status: `PROMISING / BORDERLINE / KILL`.

Không tạo composite score giả. Dùng Pareto-style reasoning: source quality + cross-domain robustness + genuine branch contribution + efficiency.

---

## 17. Quy tắc không được phá

1. Không thay micro keys giữa M0–M4.
2. Không nhìn CelebA held-out Test để chọn design.
3. Không dùng CASIA trong search.
4. Không tune threshold trên LCC.
5. Không đổi augmentation riêng cho candidate nếu hypothesis không liên quan augmentation.
6. Không thêm gate/attention vào M0–M4.
7. Không thay spatial branch giữa candidate.
8. Không chạy full chỉ vì thắng 0.3 pp.
9. Không gọi LCC là final external benchmark nữa.
10. Không kết luận frequency tốt chỉ từ source accuracy.

---

## 18. Sau 5 notebooks cần gửi gì để review?

```text
M0/summary.json
M1/summary.json
M2/summary.json
M3/summary.json
M4/summary.json
M0..M4/branch_ablation.csv
```

Nếu có thêm `comparison.csv` càng tốt.

Review cuối vòng sẽ trả lời:
1. Candidate nào tăng cross-domain separability thật?
2. Candidate nào chỉ đổi APCER/BPCER operating point?
3. Fusion có thực sự dùng frequency không?
4. Candidate nào đáng tăng budget lên 10K?

---

## 19. Decision tree

```text
5 micro candidates
       ↓
Có cross-domain gain rõ?
   ┌───┴────┐
  NO       YES
   │         │
   ▼         ▼
redesign   Top-2 10K × 2 seeds
             │
             ▼
        reproducible?
          │       │
         NO      YES
          │       │
          ▼       ▼
       redesign 25K clean integration
                    │
                    ▼
               gain survives?
                 │       │
                NO      YES
                 │       │
                 ▼       ▼
                stop   100K/full
                          │
                          ▼
                   held-out + CASIA
                          │
                          ▼
                   consider E4 gate
```

---

## 20. Kết luận vận hành

Trong giai đoạn hiện tại, **tốc độ học từ experiment quan trọng hơn độ chính xác tuyệt đối của từng run**.

Mỗi notebook phải trả lời đúng một câu hỏi về frequency representation dưới cùng một micro protocol. Chỉ tăng compute khi evidence rõ hơn.

Mục tiêu M0–M4 là tìm ra:

> **một representation mà frequency branch có signal in-domain, giữ được signal cross-domain, và thực sự bổ trợ spatial thay vì chỉ làm dịch threshold.**
