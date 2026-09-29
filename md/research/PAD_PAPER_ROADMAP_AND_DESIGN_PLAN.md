# Paper Roadmap & Experimental Story — Lightweight Spatial–Frequency Face PAD

> **Mục đích của tài liệu:** chuẩn bị cho bản báo cáo kế hoạch paper sau khi hoàn tất micro-search M0–M4.  
> Tài liệu này tách rõ:
>
> 1. **Evidence đã có** từ E1/E2/E3 và LCC-FASD;
> 2. **Architecture-development evidence** từ micro-search;
> 3. **Final paper experiments** chỉ được chạy sau khi kiến trúc được freeze.
>
> Các con số micro-search chưa có sẽ được điền vào các bảng `[TO FILL]` sau khi chạy xong.

---

# 1. One-sentence paper idea

> **Nghiên cứu cách biểu diễn frequency cue hiệu quả và nhẹ cho single-frame RGB Face Presentation Attack Detection, với mục tiêu bổ trợ spatial MobileNetV3-Small và cải thiện robustness dưới domain shift mà vẫn phù hợp edge deployment.**

Điểm nhấn không nên là:

> “Chúng tôi thêm DCT vào MobileNet.”

Mà nên tiến tới một câu chuyện sâu hơn:

> **Naïve global-frequency encoding chỉ cho cải thiện rất nhỏ và không cải thiện cross-domain separation ổn định; vì vậy chúng tôi thực hiện một controlled representation study để xác định cách giữ frequency identity / position / spatial locality tốt hơn, rồi xác nhận thiết kế thắng bằng clean retraining và unseen-domain evaluation.**

---

# 2. Research story dự kiến

## 2.1. Bối cảnh

Spatial CNN như MobileNetV3-Small có ưu điểm:

- nhẹ;
- dễ deploy;
- học texture/structure tốt;
- đạt kết quả in-domain mạnh.

Nhưng Face PAD có một vấn đề quan trọng:

```text
source dataset
      ↓
camera / illumination / background / attack medium thay đổi
      ↓
domain shift
      ↓
spatial model có thể dựa vào shortcut/domain-specific cue
```

Frequency-domain information là một candidate complementary cue vì presentation attacks có thể tạo:

- moiré;
- print halftone;
- display periodicity;
- aliasing;
- resampling/compression artifacts;
- texture reproduction artifacts.

Tuy nhiên:

> **frequency không mặc định robust hơn spatial** và bản thân frequency branch cũng có thể học shortcut.

Vì vậy paper phải được xây quanh **empirical verification**, không phải assumption.

---

# 3. Story từ thiết kế ban đầu đến kiến trúc mới

## Phase A — Spatial baseline

### E1 — MobileNetV3-Small spatial-only

Kiến trúc:

```text
224×224 RGB
→ MobileNetV3-Small ImageNet
→ GAP
→ 576→256
→ BN + Hardswish + Dropout
→ 256→128
→ Hardswish + Dropout
→ 3 logits
```

Vai trò:

> Thiết lập một lightweight spatial baseline đủ mạnh và có deployment relevance.

### Evidence hiện có

#### CelebA-Spoof held-out preliminary

| Model | APCER | BPCER | ACER | AUC |
|---|---:|---:|---:|---:|
| **E1 Spatial** | 18.91% | 0.40% | **9.66%** | **0.9814** |

#### LCC-FASD cross-domain stress test

| Model | APCER | BPCER | HTER | AUC |
|---|---:|---:|---:|---:|
| **E1 Spatial** | 40.44% | 22.62% | **31.53%** | **0.7533** |

Interpretation:

- E1 mạnh hơn rõ rệt in-domain so với cross-domain.
- LCC cho thấy generalization gap đáng kể.
- Đây là động lực hợp lý để nghiên cứu complementary cue.

---

## Phase B — Frequency-only diagnostic

### E2 — Global DCT + Tiny CNN

Pipeline:

```text
RGB crop
→ luminance
→ global 2D DCT 224×224
→ signed-log
→ per-sample z-score
→ Tiny CNN
→ GAP
→ 64-D
→ classifier
```

### Evidence hiện có

| Model | APCER | BPCER | ACER | AUC |
|---|---:|---:|---:|---:|
| **E2 Frequency-only** | 40.99% | 9.51% | **25.25%** | **0.8366** |

Interpretation:

- Frequency cue **có discriminative signal**.
- Nhưng frequency-only yếu hơn spatial baseline đáng kể.
- Vì vậy frequency nên được xem là **complementary evidence**, không phải replacement.

---

## Phase C — Initial dual-branch design

### E3 — Spatial + Global DCT + CONCAT

```text
Spatial 256-D
+
Frequency 64-D
↓
Concat 320-D
↓
MLP 128-D
↓
3 logits
```

### CelebA-Spoof held-out preliminary

| Model | ACER | AUC | ΔACER vs E1 | ΔAUC vs E1 |
|---|---:|---:|---:|---:|
| E1 Spatial | 9.66% | 0.9814 | — | — |
| **E3 Concat** | **9.13%** | **0.9857** | **−0.53 pp** | **+0.0043** |

In-domain có cải thiện, nhưng nhỏ.

### LCC-FASD external stress test

| Model | APCER | BPCER | HTER | AUC |
|---|---:|---:|---:|---:|
| E1 Spatial | 40.44% | 22.62% | 31.53% | **0.7533** |
| **E3 Concat** | **35.34%** | 26.48% | **30.91%** | 0.7443 |

Delta:

```text
HTER : -0.62 pp  (nhẹ hơn)
AUC  : -0.90 pp  (xấu hơn)
APCER: -5.10 pp
BPCER: +3.86 pp
```

### Key finding

> **Naïve spatial + global-DCT concatenation không tạo ra cải thiện cross-domain separation đủ thuyết phục.**

E3 chủ yếu thay đổi **operating-point trade-off**:

```text
fewer attacks accepted
but
more bona fide samples rejected
```

trong khi AUC giảm.

Đây là turning point của research story.

---

# 4. Research diagnosis sau E3

Không kết luận:

> “Frequency không hữu ích.”

Thay vào đó:

> **Representation/fusion hiện tại có thể chưa khai thác frequency cue đúng cách.**

Hai vấn đề chính:

## 4.1. Frequency-position loss

Trong global DCT:

```text
DCT[u,v]
```

tọa độ `(u,v)` mang semantic trực tiếp về frequency.

Nhưng pipeline cũ:

```text
DCT map
→ convolution weight sharing
→ GAP 1×1
```

có xu hướng tạo invariance theo vị trí.

Điều này có thể không phù hợp vì:

> cùng một local pattern xuất hiện ở low-frequency và high-frequency không có cùng ý nghĩa.

---

## 4.2. Spatial locality bị mất

Global DCT cho biết:

```text
ảnh có coefficient frequency nào
```

nhưng không trực tiếp cho biết:

```text
artifact nằm ở vùng spatial nào trên mặt
```

Ví dụ:

- vùng màn hình;
- vùng mắt;
- da;
- background;
- edge của print medium;

có thể mang frequency cue khác nhau.

---

## 4.3. Fusion có thể chưa phải root cause

Concat hiện tại rất đơn giản:

```text
256-D spatial + 64-D frequency
```

Nhưng trước khi tăng complexity bằng gate/attention:

> **phải chứng minh representation frequency đủ tốt trước.**

Do đó roadmap thay đổi từ:

```text
E3 concat
→ E4 gate
```

thành:

```text
E3 concat
→ representation search
→ confirm representation
→ fusion search nếu cần
→ final model
```

---

# 5. Controlled Micro-Search — Architecture Development Study

## 5.1. Mục tiêu

Micro-search không dùng để claim final accuracy.

Nó trả lời:

> **Frequency representation nào có complementary signal rõ nhất với spatial branch dưới cùng một budget?**

---

## 5.2. Fixed development protocol

```text
CelebA micro-Train = 5,000
CelebA micro-Val   = 2,000
Seed               = 42
```

Cross-domain development:

```text
LCC-FASD = external-dev / stress set
```

Reserved:

```text
CelebA held-out Test = không dùng để select architecture
CASIA-FASD           = không dùng để select architecture
```

### Evaluation

In-domain:

```text
APCER
BPCER
ACER
AUC
```

Cross-domain development:

```text
HTER
AUC
APCER
BPCER
```

Threshold:

```text
CelebA micro-Val
→ calibrate
→ LOCK
→ apply unchanged to LCC
```

---

# 6. Five frequency representations

| ID | Representation | Hypothesis |
|---|---|---|
| **M0** | Global DCT + GAP 1×1 | Original control |
| **M1** | Global DCT + Pool4×4 | Preserve coarse frequency position |
| **M2** | Coord-DCT + Pool4×4 | Explicitly encode frequency coordinates |
| **M3** | Band-aware DCT | Expose low/mid/high-band identity |
| **M4** | Block-DCT 8×8 | Preserve frequency identity + spatial locality |

---

## M0 — Original global-DCT control

```text
Global DCT
→ signed-log
→ z-score
→ Tiny CNN
→ GAP
→ 64-D
```

Purpose:

> Reproduce the original frequency design under the same micro budget.

---

## M1 — Position-preserving frequency pooling

```text
Global DCT
→ Tiny CNN
→ AdaptivePool 4×4
→ Linear
→ 64-D
```

Question:

> Does retaining coarse frequency location improve robustness?

---

## M2 — Coordinate-aware DCT

```text
DCT
+ U coordinate
+ V coordinate
→ CNN
→ Pool4×4
→ 64-D
```

Question:

> Does explicitly telling the CNN where a spectral pattern lies improve learning?

---

## M3 — Band-aware DCT

```text
LOW band
MID band
HIGH band
→ separate masked channels
→ CNN
→ Pool4×4
→ 64-D
```

Question:

> Is exposing band identity more useful than treating the full DCT plane uniformly?

---

## M4 — Block-DCT

```text
224×224
→ 8×8 blocks
→ per-block DCT
→ selected coefficient identities as channels
→ K×28×28
→ Tiny CNN
→ 64-D
```

Question:

> Is it better to encode frequency identity as channels while preserving spatial position on the face?

---

# 7. Micro-search result table — TO FILL AFTER RUN

## 7.1. Frequency-only

| Candidate | Val ACER | Val AUC | LCC HTER | LCC AUC | Params |
|---|---:|---:|---:|---:|---:|
| M0 Global DCT | [ ] | [ ] | [ ] | [ ] | [ ] |
| M1 Pool4×4 | [ ] | [ ] | [ ] | [ ] | [ ] |
| M2 Coord-DCT | [ ] | [ ] | [ ] | [ ] | [ ] |
| M3 Band-aware | [ ] | [ ] | [ ] | [ ] | [ ] |
| M4 Block-DCT | [ ] | [ ] | [ ] | [ ] | [ ] |

---

## 7.2. Spatial + Frequency CONCAT

| Candidate | Val ACER | Val AUC | LCC HTER | LCC AUC | ΔHTER vs M0 | ΔAUC vs M0 |
|---|---:|---:|---:|---:|---:|---:|
| M0 | [ ] | [ ] | [ ] | [ ] | — | — |
| M1 | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| M2 | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| M3 | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| M4 | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |

---

## 7.3. Comparison against shared spatial micro-E1

| Candidate | Val ΔACER vs Spatial | Val ΔAUC vs Spatial | LCC ΔHTER vs Spatial | LCC ΔAUC vs Spatial |
|---|---:|---:|---:|---:|
| M0 | [ ] | [ ] | [ ] | [ ] |
| M1 | [ ] | [ ] | [ ] | [ ] |
| M2 | [ ] | [ ] | [ ] | [ ] |
| M3 | [ ] | [ ] | [ ] | [ ] |
| M4 | [ ] | [ ] | [ ] | [ ] |

---

# 8. Branch diagnostic table — TO FILL

Winner selection không chỉ dựa vào final metric.

Required diagnostics:

```text
FULL
FREQ_ZERO
FREQ_SHUFFLE
SPATIAL_ZERO
```

### LCC AUC example table

| Candidate | FULL | FREQ_ZERO | FREQ_SHUFFLE | SPATIAL_ZERO |
|---|---:|---:|---:|---:|
| M0 | [ ] | [ ] | [ ] | [ ] |
| M1 | [ ] | [ ] | [ ] | [ ] |
| M2 | [ ] | [ ] | [ ] | [ ] |
| M3 | [ ] | [ ] | [ ] | [ ] |
| M4 | [ ] | [ ] | [ ] | [ ] |

Evidence mong muốn:

```text
FULL > FREQ_ZERO
FULL > FREQ_SHUFFLE
```

Interpretation:

- `FULL ≈ FREQ_ZERO` → fusion gần như ignore frequency.
- `FULL ≈ FREQ_SHUFFLE` → frequency feature không có sample-specific contribution rõ.
- `SPATIAL_ZERO` → measure standalone frequency signal after fusion training.

---

# 9. Winner-selection logic

## Candidate mạnh

Một candidate đáng promote nếu có pattern:

```text
in-domain:
không regression lớn

cross-domain:
AUC tăng rõ
và/hoặc
HTER giảm rõ

branch diagnostic:
FULL > FREQ_ZERO
FULL > FREQ_SHUFFLE

cost:
overhead vẫn nhỏ
```

### Screening heuristic hiện tại

```text
LCC AUC +~1.5–2.0 pp
OR
LCC HTER -~1.5–2.0 pp
```

và:

```text
CelebA micro-Val ACER regression <~1–1.5 pp
```

Đây là **screening heuristic**, không phải statistical significance threshold.

---

## Không gọi là winner nếu

### Case A

```text
HTER tốt hơn
AUC giảm
```

→ có thể chỉ là operating-point shift.

### Case B

```text
FULL ≈ FREQ_ZERO
```

→ frequency branch bị ignore.

### Case C

```text
FULL ≈ FREQ_SHUFFLE
```

→ không có evidence mạnh về sample-specific complementary cue.

### Case D

```text
in-domain tăng mạnh
cross-domain giảm
```

→ suspect source-domain shortcut.

---

# 10. Decision point after micro-search

Sau khi điền số liệu:

## Scenario 1 — Một candidate thắng rõ

Ví dụ:

```text
Mx
→ Val stable
→ LCC AUC ↑
→ LCC HTER ↓
→ branch diagnostics positive
```

Decision:

```text
Mx = new frequency representation
```

Sau đó:

```text
10K confirmation
→ 2 seeds
```

---

## Scenario 2 — Hai candidate gần nhau

Decision:

```text
top-2
→ 10K / 3K
→ 2 seeds
→ compare mean ± std
```

Không chọn theo chênh lệch một run nhỏ.

---

## Scenario 3 — Không candidate nào thắng

Đây vẫn là finding hữu ích:

> Global/handcrafted DCT representation family hiện tại chưa tạo complementary signal đủ mạnh.

Lúc đó:

```text
STOP full training
→ redesign
```

Candidate next-generation có thể là:

- DWT / wavelet;
- residual / high-pass branch;
- learnable frequency filter bank;
- frequency-aware auxiliary supervision;

nhưng **không được thêm vào paper roadmap hiện tại cho đến khi cần**.

---

# 11. Paper research questions dự kiến

## RQ1 — Spatial baseline

> **How well does a lightweight MobileNetV3-Small PAD model generalize beyond its source dataset?**

Evidence:

- CelebA held-out;
- cross-domain evaluation.

---

## RQ2 — Frequency representation

> **Which lightweight frequency representation provides the most useful complementary cue to spatial features?**

Evidence:

```text
M0–M4 controlled micro-search
```

---

## RQ3 — Complementarity

> **Does the selected frequency representation provide sample-specific information beyond the spatial branch?**

Evidence:

```text
FULL
vs FREQ_ZERO
vs FREQ_SHUFFLE
vs SPATIAL_ZERO
```

---

## RQ4 — Generalization

> **Does the selected representation improve robustness on an unseen domain after clean full-scale retraining?**

Evidence:

```text
CelebA source training
→ CASIA-FASD unseen external confirmation
```

---

## RQ5 — Efficiency

> **Can the added frequency branch improve robustness while retaining edge-oriented efficiency?**

Evidence:

```text
params
MACs/FLOPs
model size
latency
FPS
memory
```

---

# 12. Tentative contribution statements

Các contribution chỉ nên được finalize sau số liệu.

### Contribution candidate C1

> A lightweight spatial–frequency Face PAD architecture built around MobileNetV3-Small and an explicitly designed frequency representation.

### C2

> A controlled study showing that the way frequency cues are represented matters: naïve global spectral pooling is not necessarily sufficient.

### C3

Nếu micro-search winner cho kết quả mạnh:

> A frequency representation that preserves `[winner-specific principle]` and provides more robust complementary evidence than naïve global-DCT pooling.

Ví dụ:

```text
M1 winner → coarse frequency position
M2 winner → explicit frequency coordinates
M3 winner → frequency-band identity
M4 winner → frequency identity + spatial locality
```

### C4

> A rigorous development/evaluation protocol separating architecture-selection domains from final unseen-domain confirmation.

### C5

> Edge-efficiency analysis of the proposed PAD architecture.

Không claim tất cả 5 contribution nếu data không support.

---

# 13. Possible paper titles

Title chỉ chốt sau winner.

## Neutral title

**Lightweight Spatial–Frequency Representation Learning for Generalizable Face Presentation Attack Detection**

## Nếu M4 thắng

**Locality-Preserving Block-DCT Features for Lightweight Spatial–Frequency Face Anti-Spoofing**

## Nếu M2 thắng

**Coordinate-Aware Frequency Representation for Lightweight Generalizable Face Anti-Spoofing**

## Nếu M3 thắng

**Band-Aware Frequency Representation for Lightweight Cross-Domain Face Anti-Spoofing**

## Nếu contribution chính là study

**Rethinking Frequency Representation in Lightweight Face Anti-Spoofing: From Global Spectra to Generalizable Complementary Cues**

---

# 14. Paper methodology after architecture freeze

Micro-search chỉ chọn kiến trúc.

Sau khi winner được chọn:

```text
winner architecture
↓
clean independent retraining
```

Không reuse micro warm-start để claim final result.

## Final fair training

```text
Spatial E1:
independent training

Proposed:
independent training
```

Giữ:

- same source split;
- same preprocessing;
- same crop;
- same augmentation;
- same optimizer family;
- same epoch/scheduler policy;
- same checkpoint-selection procedure;
- same threshold-calibration procedure.

Nếu budget cho phép:

```text
≥ 3 seeds
mean ± std
```

đặc biệt nếu final gain nhỏ.

---

# 15. Final experiment roadmap

## Stage A — COMPLETED

```text
E1 spatial
E2 frequency diagnostic
E3 naïve concat
LCC stress test
```

Finding:

```text
frequency has signal
but naïve concat gives only limited improvement
```

---

## Stage B — CURRENT

```text
M0–M4 micro-search
```

Output:

```text
selected representation
or
NO WINNER
```

---

## Stage C — Top candidate confirmation

```text
10K Train
3K Val
2 seeds
```

Goal:

> xác nhận gain không phải micro-split noise.

---

## Stage D — Fusion study, only if justified

Nếu winner frequency branch có signal nhưng concat chưa khai thác tốt:

```text
Normalized Concat
Residual-logit Fusion
FiLM / modulation
Adaptive Gate
```

Không bắt buộc phải có Stage D nếu static concat đã tốt.

---

## Stage E — Final full retraining

Train:

```text
E1 final baseline
Proposed final architecture
```

independent initialization.

---

## Stage F — Final evaluation

### In-domain

```text
CelebA-Spoof held-out Test
```

### Final unseen external confirmation

```text
CASIA-FASD
```

CASIA phải không được dùng trong architecture search.

Optional later:

```text
Replay-Attack
MSU-MFSD
```

---

## Stage G — Efficiency

Evaluate:

```text
Params
MACs/FLOPs
model size
peak memory
CPU latency
Raspberry Pi latency/FPS
```

---

# 16. Dataset roles in final paper

| Dataset | Role | Used for architecture selection? | Final claim? |
|---|---|---:|---:|
| CelebA Train | source training | Yes | Yes |
| CelebA Val | checkpoint/threshold/development | Yes | No final-test claim |
| **LCC-FASD** | external development / stress | **Yes** | Development evidence only |
| CelebA held-out Test | final in-domain test | No | **Yes** |
| **CASIA-FASD** | untouched external confirmation | **No** | **Yes** |

Critical wording:

> LCC-FASD must not be presented as an untouched final cross-domain test after it has been used to choose M0–M4.

---

# 17. Main tables dự kiến trong paper

## Table 1 — Baselines

```text
E1 Spatial
E2 Frequency-only
E3 Initial Concat
```

Metrics:

```text
CelebA held-out
LCC development stress
```

Purpose:

> show why naïve frequency integration was insufficient.

---

## Table 2 — Frequency representation development study

```text
M0
M1
M2
M3
M4
```

Report:

```text
micro-Val ACER / AUC
LCC-dev HTER / AUC
Params
```

---

## Table 3 — Complementarity ablation

Winner:

```text
FULL
FREQ_ZERO
FREQ_SHUFFLE
SPATIAL_ZERO
```

---

## Table 4 — Final clean comparison

```text
E1 final
Proposed final
```

Across:

```text
CelebA held-out
CASIA unseen
```

Prefer:

```text
mean ± std
```

---

## Table 5 — Efficiency

```text
Params
MACs
model size
CPU latency
Pi latency
FPS
```

---

# 18. Main figures dự kiến

## Figure 1 — Overall architecture

```text
RGB
├─ Spatial MobileNetV3-Small
└─ Selected frequency representation
        ↓
      Fusion
        ↓
      PAD
```

---

## Figure 2 — Research evolution

A compact story figure:

```text
Spatial E1
   ↓
Add Global DCT
   ↓
small / inconsistent gain
   ↓
representation analysis
   ↓
M0–M4
   ↓
selected design
```

Đây là figure rất phù hợp cho cả presentation và paper Discussion/Method motivation.

---

## Figure 3 — Frequency representations

Show visually:

```text
M0 global spectrum
M1 spectral grid
M2 coordinate-aware spectrum
M3 band masks
M4 block-DCT tensor
```

---

## Figure 4 — Cross-domain ROC

Final:

```text
E1
vs
Proposed
```

on CASIA.

---

## Figure 5 — Feature/branch diagnostics

Optional:

- feature norm distribution;
- FULL vs FREQ_SHUFFLE ROC;
- t-SNE only if genuinely useful, not decorative.

---

# 19. Planned paper structure

## 1. Introduction

Story:

```text
Face PAD cần generalization
↓
lightweight spatial CNN dễ deploy nhưng domain-sensitive
↓
frequency cues là complementary candidate
↓
naïve frequency fusion không đảm bảo robust gain
↓
câu hỏi quan trọng là HOW to represent frequency
↓
our controlled lightweight study + final model
```

---

## 2. Related Work

### 2.1 Lightweight Face PAD

### 2.2 Frequency-aware Face PAD

### 2.3 Domain-generalizable Face PAD

### 2.4 Spatial–frequency fusion

Không claim novelty chỉ vì dual branch.

---

## 3. Method

### 3.1 Spatial branch

MobileNetV3-Small.

### 3.2 Initial global-frequency baseline

DCT + Tiny CNN.

### 3.3 Selected frequency representation

`[TO FILL AFTER MICROSEARCH]`

### 3.4 Fusion

`[TO FILL]`

### 3.5 Training objective

3-class:

```text
Real / Physical / Digital
```

binary PAD score derived from logits.

---

## 4. Experimental Protocol

### 4.1 Datasets

### 4.2 Preprocessing

### 4.3 Metrics

### 4.4 Development architecture search

Explicitly state:

```text
5K/2K micro split
LCC external-dev
M0–M4
```

### 4.5 Final evaluation protocol

CelebA held-out + CASIA unseen.

### 4.6 Efficiency protocol

---

## 5. Results

### 5.1 Initial baseline study E1/E2/E3

### 5.2 Frequency representation study M0–M4

### 5.3 Final proposed model

### 5.4 Cross-domain generalization

### 5.5 Branch ablation

### 5.6 Efficiency

---

## 6. Discussion

Discuss:

- why naïve global DCT did/did not work;
- what winner teaches about frequency representation;
- source-vs-target behavior;
- attack-error vs bona-fide-error trade-off;
- limitations.

---

## 7. Conclusion

---

# 20. Presentation plan for tomorrow

Recommended: **10–12 slides**.

## Slide 1 — Problem & paper objective

Title:

> Lightweight Spatial–Frequency Face PAD: From Naïve DCT Fusion to Representation-Aware Design

Show:

```text
Goal = Security × Generalization × Efficiency
```

---

## Slide 2 — Existing E1 baseline

Show E1 architecture.

Numbers:

```text
CelebA held-out:
ACER 9.66%
AUC 98.14%

LCC:
HTER 31.53%
AUC 75.33%
```

Message:

> Strong source performance, large cross-domain gap.

---

## Slide 3 — Initial frequency hypothesis

Show E2.

Numbers:

```text
E2 held-out:
ACER 25.25%
AUC 83.66%
```

Message:

> Frequency contains signal but is not strong enough standalone.

---

## Slide 4 — Initial dual-branch E3

Show:

```text
Spatial + Global DCT → CONCAT
```

Numbers:

```text
CelebA:
ACER 9.66 → 9.13
AUC 98.14 → 98.57

LCC:
HTER 31.53 → 30.91
AUC 75.33 → 74.43
```

Main statement:

> In-domain improvement exists, but cross-domain gain is small and AUC declines.

---

## Slide 5 — Why E3 is not enough

Show two hypotheses:

```text
Global DCT + GAP
→ loses spectral position

Global DCT
→ loses spatial locality
```

Optional third:

```text
Concat may not be the root cause yet
```

---

## Slide 6 — Research pivot

Show:

```text
DO NOT:
E3 → immediately add gate

DO:
E3 → representation search → confirm → fusion if needed
```

---

## Slide 7 — M0–M4 design matrix

| ID | Key idea |
|---|---|
| M0 | Global DCT baseline |
| M1 | Preserve spectral position |
| M2 | Add frequency coordinates |
| M3 | Expose band identity |
| M4 | Preserve frequency + spatial locality |

---

## Slide 8 — Micro-search protocol

```text
5K Train
2K Val
LCC external-dev
same spatial control
same fusion
same threshold rule
```

Stress:

> only representation changes.

---

## Slide 9 — REAL RESULTS

Use tomorrow's result table.

Recommended columns:

```text
Val ACER
Val AUC
LCC HTER
LCC AUC
ΔAUC vs M0
```

Highlight winner only after objective review.

---

## Slide 10 — Branch evidence

Show:

```text
FULL
FREQ_ZERO
FREQ_SHUFFLE
SPATIAL_ZERO
```

Main question:

> Is the frequency branch actually contributing sample-specific information?

---

## Slide 11 — Proposed final paper architecture

Fill after winner:

```text
MobileNetV3-Small
+
[SELECTED FREQUENCY BRANCH]
+
[CONCAT or later selected fusion]
```

State:

> architecture selected, but final claim still requires clean retraining.

---

## Slide 12 — Paper roadmap

```text
Micro-search
↓
10K confirmation
↓
optional fusion search
↓
clean full retrain
↓
CelebA held-out
↓
CASIA unseen
↓
edge benchmark
↓
paper
```

---

# 21. Recommended wording for tomorrow's report

## Opening

> “Our initial hypothesis was that an explicit frequency branch could complement a lightweight spatial PAD model, especially under domain shift. The first global-DCT concatenation model slightly improved the in-domain result, but the cross-domain gain was marginal and the ranking AUC actually decreased. Instead of immediately increasing fusion complexity, we treated this as a representation problem and designed a controlled micro-search over five lightweight frequency encodings.”

---

## Transition to micro-search

> “The goal of the micro-search is not to report final accuracy. It is to identify whether preserving spectral position, band identity, or spatial locality produces a stronger complementary cue under a fixed compute budget.”

---

## If a clear winner appears

> “Among the tested representations, Mx showed the most consistent development evidence across source validation, LCC external-development performance, and branch-ablation diagnostics. We therefore select Mx as the architecture candidate for larger-scale confirmation.”

---

## If two candidates are close

> “The micro-search narrows the design space to Mx and My, but the difference is not large enough to justify a final selection from a single seed. Both will therefore advance to the 10K two-seed confirmation stage.”

---

## If no candidate wins

> “The study does not support the current DCT representation family strongly enough to justify a full retraining. We therefore treat this as a negative design result and redesign the frequency representation before spending additional compute.”

---

# 22. What NOT to claim tomorrow

Do not say:

> “The selected micro model generalizes better.”

unless there is final unseen confirmation.

Prefer:

> “The selected candidate shows stronger development-domain evidence.”

Do not say:

> “LCC proves cross-domain generalization.”

Because LCC has been used for architecture selection.

Prefer:

> “LCC is used as an external-development stress set.”

Do not say:

> “M4/M2/M3 is novel.”

before literature comparison confirms the exact design gap.

Prefer:

> “This is the current proposed design direction.”

---

# 23. Final confirmation criteria before writing the paper

Paper should not be considered experiment-complete until:

- [ ] M0–M4 micro-search complete;
- [ ] winner/top-2 selected from documented criteria;
- [ ] 10K confirmation complete;
- [ ] final architecture frozen;
- [ ] clean independent retraining complete;
- [ ] E1 final retrained under same protocol;
- [ ] CelebA held-out evaluated once after freeze;
- [ ] CASIA unseen external confirmation complete;
- [ ] branch ablation complete;
- [ ] ≥2–3 seeds for main models if compute permits;
- [ ] params/MACs/model size measured;
- [ ] edge latency measured;
- [ ] literature direct-comparison protocol checked;
- [ ] limitations written explicitly.

---

# 24. Timeline proposal

## Day 1 — Micro-search

```text
Stage 0
Stage 1
M0
M1–M4
comparison
```

Deliverable:

```text
selected candidate / top-2
```

---

## Day 2 — Architecture review / paper-plan presentation

Deliver:

- initial E1/E2/E3 evidence;
- reason E3 was insufficient;
- M0–M4 real results;
- selected direction;
- final paper RQs;
- experiment roadmap.

---

## Day 3–4 — Confirmation

```text
10K / 3K
2 seeds
```

---

## Day 5+ — Fusion decision

Only if representation evidence justifies it.

---

## Final phase

```text
clean full training
→ final test
→ unseen external
→ efficiency
→ manuscript
```

---

# 25. Core narrative to preserve

The paper story should remain:

```text
1. Spatial baseline is strong in-domain but weakens under domain shift.

2. Frequency-only contains useful signal but is not sufficient.

3. Naïve global-DCT + concat gives only a small in-domain gain
   and does not improve cross-domain separation convincingly.

4. This motivates a representation-level investigation rather than
   immediately adding more complex fusion.

5. A controlled lightweight micro-search compares alternative
   ways of preserving frequency position / band identity / locality.

6. The strongest candidate is confirmed at a larger scale.

7. The frozen architecture is retrained independently and evaluated
   on held-out source data and an untouched external dataset.

8. Final contribution is judged jointly by:
   generalization + complementarity + efficiency.
```

This narrative is stronger than:

```text
“We tried several architectures and picked the best one.”
```

because every design change is tied to a specific hypothesis.

---

# 26. Tomorrow — fields to fill immediately after micro-search

```text
Selected candidate:
[ ]

Why selected:
[ ]

CelebA micro-Val ACER:
[ ]

CelebA micro-Val AUC:
[ ]

LCC HTER:
[ ]

LCC AUC:
[ ]

Δ LCC HTER vs M0:
[ ]

Δ LCC AUC vs M0:
[ ]

Δ LCC HTER vs Spatial:
[ ]

Δ LCC AUC vs Spatial:
[ ]

FULL vs FREQ_ZERO:
[ ]

FULL vs FREQ_SHUFFLE:
[ ]

Params:
[ ]

Key interpretation:
[ ]

Next experiment:
[ ]
```

---

# 27. Provisional abstract skeleton

> Lightweight face presentation attack detection is attractive for edge deployment, but spatial models can suffer substantial degradation under domain shift. We first investigate a MobileNetV3-Small spatial baseline and an explicit DCT-based frequency branch. Although naïve global spectral concatenation provides a small in-domain improvement, its cross-domain benefit is limited, motivating a systematic study of lightweight frequency representations. We compare global, position-preserving, coordinate-aware, band-aware, and block-wise DCT encodings under a controlled micro-training protocol. The selected representation, **[WINNER]**, is then retrained independently and evaluated on **[FINAL DATASETS]**. Results show **[FINAL FINDING]**, while adding only **[PARAM/FLOP OVERHEAD]** computational overhead. These findings indicate that **[WINNER-SPECIFIC PRINCIPLE]** is important for extracting complementary frequency cues in lightweight face anti-spoofing.

Do not finalize this abstract until final experiments are complete.

---

# 28. Final paper positioning

The safest positioning is:

> **A lightweight, empirically grounded study of how frequency cues should be represented and integrated into spatial Face PAD models under domain shift.**

The strongest possible positioning, only if final results support it, is:

> **A new lightweight frequency representation that improves cross-domain Face PAD robustness while preserving edge efficiency.**

Do not choose the stronger claim before CASIA/final unseen results exist.
