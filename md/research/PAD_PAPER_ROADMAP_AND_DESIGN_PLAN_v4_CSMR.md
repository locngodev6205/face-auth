# PAD Paper Roadmap & Experimental Story — CSMR-PAD

_Last updated: 2026-09-29_

> **Mục đích:** thay thế roadmap cũ dựa trên dual-branch DCT bằng roadmap hiện tại sau khi M0–M4 không tạo ra cross-domain complementarity đủ ổn định và CSMR-v1 đã cho positive signal trên hai backbone candidates.
>
> Tài liệu này tách rõ:
> 1. evidence lịch sử / motivation;
> 2. evidence development hiện tại;
> 3. decision gate chọn backbone;
> 4. scale sanity 10–20K;
> 5. final 100K training;
> 6. ablation, unseen confirmation và paper packaging.
>
> **Quan trọng:** CSMR-v1 đã được freeze về mechanism. Backbone cuối **chưa được freeze** cho đến khi hoàn tất fair MNV3-S `2.7× / 12 epochs` check.

---

# 1. One-sentence paper idea

> **CSMR-PAD dùng frequency chỉ như một training-time counterfactual stressor: với mỗi sample, model xác định spectral band LOW/MID/HIGH làm giảm PAD margin nguy hiểm nhất và học chống lại perturbation đó, trong khi inference vẫn là một lightweight spatial-only backbone với zero FFT overhead.**

Không định vị novelty là:

> “Dùng FFT cho Face PAD.”

Không định vị novelty là:

> “Frequency augmentation giúp generalization.”

Định vị hợp lý hơn:

> **sample-adaptive worst-band spectral intervention + PAD-margin-aware harmful-drop regularization + zero inference-time frequency branch.**

---

# 2. Research story hiện tại

## 2.1. Starting point

Project bắt đầu với một lightweight spatial-only PAD model:

```text
RGB
→ face crop
→ MobileNetV3-Small
→ 3 logits: Real / Physical / Digital
```

Spatial model đạt source-domain performance tốt nhưng suy giảm rõ dưới domain shift.

Điều này tạo câu hỏi:

> Có thể khai thác frequency cue để tăng robustness mà vẫn giữ deployment cost thấp hay không?

---

## 2.2. Explicit frequency branch: useful signal, weak robustness

Các hướng E2/E3 và micro-search M0–M4 đã kiểm tra:

```text
M0 Global DCT + GAP
M1 DCT + Pool4×4
M2 Coord-DCT
M3 Band-aware DCT
M4 Block-DCT
```

Finding tổng hợp:

- frequency cue có discriminative signal in-domain;
- explicit frequency-only / fusion có thể cải thiện source metric;
- cross-domain behavior không ổn định;
- một số configuration chủ yếu làm dịch APCER/BPCER operating point;
- explicit frequency branch không tạo ra một winner đủ mạnh để promote thành final inference architecture.

Do đó paper không nên kể:

```text
frequency branch search
→ tìm ra branch tốt nhất
→ final dual branch
```

Mà nên kể:

```text
frequency has signal
→ explicit use can learn domain-sensitive spectral shortcuts
→ do not trust frequency as an inference feature
→ use frequency as a training-time robustness intervention instead
```

---

# 3. Main method — CSMR-PAD v1

**CSMR = Counterfactual Spectral Margin Robustness.**

## 3.1. Inference

Inference không dùng FFT:

```text
RGB
→ frozen preprocessing contract
→ selected lightweight spatial backbone
→ 3-class PAD logits
```

Không có:

- DCT/FFT branch;
- late fusion;
- frequency CNN;
- extra test-time augmentation.

## 3.2. Training-time counterfactuals

Từ cùng base-augmented sample:

```text
clean image
├─ LOW-band attenuation
├─ MID-band attenuation
└─ HIGH-band attenuation
```

Normalized radial centers:

```text
LOW  = 0.15
MID  = 0.45
HIGH = 0.75
sigma = 0.10
```

Gain:

```text
g ~ Uniform(0.65, 0.90)
G_k = 1 - (1-g) M_k
```

Constraints:

- one `g` shared across LOW/MID/HIGH of the same sample;
- same gain across RGB/BGR channels;
- no amplification;
- DC preserved;
- phase unchanged;
- FFT on image float `[0,1]`.

## 3.3. PAD margin

```text
d = z_real - logsumexp(z_physical, z_digital)

sign = +1 for Real
sign = -1 for Physical/Digital

m = sign * d
```

Worst spectral counterfactual:

```text
k* = argmin_k m_k
```

## 3.4. Loss

Epoch 1:

```text
L = CE(clean)
```

Epoch 2+:

```text
L_cls = 0.5 CE(clean) + 0.5 CE(worst)
target = m_clean.detach()
harmful = (m_worst < target)

L_margin =
    mean(
        SmoothL1(m_worst, target)
        * harmful
    )

L_total = L_cls + 0.20 L_margin
```

Frozen CSMR-v1:

```text
lambda = 0.20
warmup = 1 epoch
centers = [0.15, 0.45, 0.75]
sigma = 0.10
gain = [0.65, 0.90]
worst selection = label-aligned margin argmin
margin penalty = harmful-only
```

Không tune lại các giá trị này theo LCC/Test.

---

# 4. Evidence đã có

## 4.1. MNV3-S seed42 pilot

Stage-1 reference vs CSMR seed42:

| Metric | Spatial reference | CSMR-v1 | Delta |
|---|---:|---:|---:|
| Val ACER | ~2.11% | ~2.22% | +0.11 pp |
| Val AUC | ~0.9973 | ~0.9971 | ~−0.0002 |
| LCC HTER | ~34.63% | ~32.10% | **−2.53 pp** |
| LCC AUC | ~0.6938 | ~0.7256 | **+3.18 pp** |

Interpretation:

```text
source ≈ preserved
cross-domain ranking ↑
```

Đây là development evidence, không phải significance claim.

---

## 4.2. MNV3-S paired seed100 replication

Frozen 5K/2K manifest; same seed100 initialization policy.

| Metric | Vanilla | CSMR | Delta CSMR−Vanilla |
|---|---:|---:|---:|
| Val ACER | 2.1062% | 2.1357% | +0.0295 pp |
| Val AUC | 0.997817 | 0.997508 | −0.000309 |
| LCC HTER | 34.0274% | 33.1171% | −0.9103 pp |
| LCC AUC | 0.690181 | 0.714238 | **+2.4057 pp** |

Pre-registered freeze gate:

```text
PASS — PRIMARY_AUC_GATE
```

Điều này đủ để freeze **CSMR mechanism**, không phải backbone.

---

## 4.3. Opened CelebA Test-2000 descriptor

Same fixed 2,000 keys, locked Val thresholds.

| Metric | MNV3 Vanilla | MNV3 + CSMR |
|---|---:|---:|
| Test ACER | 15.9992% | 17.2611% |
| Test AUC | 0.957275 | 0.958933 |

Interpretation:

- AUC preserved / slightly higher;
- threshold-dependent ACER worse;
- calibration / operating-point stability cần được report;
- Test-2000 đã **opened** và từ giờ chỉ là fixed descriptive benchmark, không còn pristine development-free test.

---

## 4.4. MiniFASNetV2 transfer

Recipe:

```text
official PAD-pretrained MiniFASNetV2
crop = 2.7×
input = 80×80
classifier reinitialized for project labels
seed = 100
max epochs = 12
```

Model size:

```text
~434,560 trainable params
```

Paired result:

| Metric | MiniFAS Vanilla | MiniFAS + CSMR | Delta |
|---|---:|---:|---:|
| Val ACER | 1.6909% | 2.1691% | +0.4782 pp |
| Val AUC | 0.998290 | 0.996634 | −0.001657 |
| LCC HTER | 31.8242% | 28.7548% | **−3.0694 pp** |
| LCC AUC | 0.819484 | 0.846264 | **+2.6779 pp** |

Transfer descriptor:

```text
TRANSFER_STRONG
```

Opened Test-2000:

| Metric | MiniFAS Vanilla | MiniFAS + CSMR |
|---|---:|---:|
| Test ACER | 19.6058% | 21.3520% |
| Test AUC | 0.937166 | 0.935649 |

Key finding:

> CSMR transfers to a PAD-specific lightweight backbone, but MiniFAS absolute comparison against MNV3 is not yet fully fair because crop, input resolution, pretraining and training budget differ.

---

# 5. Backbone decision gate — CURRENT

MiniFAS currently has much stronger LCC metrics, but comparison hiện tại:

```text
MNV3-S:
crop 1.55×
224×224
ImageNet pretraining
6 epochs

MiniFASNetV2:
crop 2.7×
80×80
PAD pretraining
12 epochs
```

Do đó trước khi chốt backbone phải chạy:

```text
MNV3-S Vanilla seed100
MNV3-S + CSMR seed100
crop = 2.7×
input = 224×224
max epochs = 12
patience = 3
earliest stop = 6
```

Mọi thứ khác giữ nguyên MNV3 recipe.

## Decision principle

Không rank chỉ bằng một metric.

Ưu tiên Pareto:

```text
1. LCC AUC / HTER
2. Val stability
3. fixed Test-2000 descriptor
4. params / latency / memory
5. deployment simplicity
```

---

# 6. CASE A — Chọn MiniFASNetV2

Chọn case này nếu sau fair MNV3 `2.7× / 12ep`:

- MiniFAS vẫn giữ cross-domain advantage rõ;
- MNV3 wider context không đóng đáng kể gap;
- MiniFAS latency/size phù hợp deployment;
- CSMR vẫn là positive transfer.

## Final inference architecture

```text
SCRFD bbox
→ 2.7× context crop
→ resize 80×80
→ MiniFASNetV2
→ 128-D embedding
→ 3 logits
```

Pretraining:

```text
official PAD-pretrained checkpoint
→ load compatible feature weights
→ discard/reinitialize original classifier
```

CSMR chỉ active khi training.

## Scale sanity

Default:

```text
Train = 20K
Val   = frozen development Val
Seed  = 100
Vanilla + CSMR
```

10K chỉ dùng nếu cần speed-first smoke confirmation.

Mục tiêu:

> xác nhận CSMR gain không chỉ tồn tại trong 5K regime.

Không dùng 20K để tune CSMR.

## Final budget

```text
Train = 100K
Val   = 15K
Seed  = 100
Vanilla + CSMR
```

Optimization budget được pre-register sau 20K trajectory.

Default candidate:

```text
max epochs = 12
patience = 3
earliest stop = 6
```

Nếu 20K còn cải thiện tại epoch 12, có thể pre-register budget lớn hơn **trước khi chạy 100K**.

## Role of M0–M4

Không rerun full M0–M4 trên MiniFAS.

Existing M0–M4 được giữ như historical motivation.

Nếu paper cần chứng minh diagnosis transfer qua backbone:

```text
MiniFAS Vanilla
MiniFAS + one representative explicit-frequency baseline
MiniFAS + Full CSMR
```

Chỉ một representative baseline, ví dụ M3 hoặc best prior explicit-frequency candidate.

Priority vẫn là CSMR ablation, không phải architecture search mới.

---

# 7. CASE B — Chọn MobileNetV3-Small

Chọn case này nếu MNV3 `2.7× / 12ep`:

- đóng phần lớn gap LCC với MiniFAS;
- giữ source/Test descriptor tốt hơn;
- hoặc tạo Pareto trade-off tốt hơn khi xét performance + deployment + pretraining fairness.

## Final inference architecture

```text
SCRFD bbox
→ 2.7× context crop
→ 224×224 RGB
→ CelebA normalization
→ MobileNetV3-Small ImageNet
→ project head 576→256→128→3
```

Trainable params current micro recipe:

```text
~1.11M
```

## Scale sanity

```text
Train = 20K
Val   = frozen development Val
Seed  = 100
Vanilla + CSMR
max epochs ≈ 12
```

Mục tiêu:

> xác nhận wider-crop MNV3 + CSMR giữ gain khi tăng data.

## Final budget

Để gần với E1 v5.3 historical full recipe:

```text
Train = 100K
Val   = 15K
```

MNV3 historical recipe đã dùng:

```text
max epochs = 24
patience = 4
earliest stop = 10
```

Nếu case B được chọn, đây là default final optimization budget hợp lý để giữ continuity với E1 v5.3.

Vanilla và CSMR phải dùng cùng schedule.

## Role of M0–M4

Không cần rerun M0–M4.

Chúng đã được phát triển trực tiếp quanh MNV3-S và có thể dùng làm historical motivation / negative design evidence.

---

# 8. Main ablations — quan trọng hơn full M0–M4 rerun

Sau khi backbone freeze, tối thiểu cần:

| ID | Training method | Worst selection | Margin-aware | Purpose |
|---|---|---:|---:|---|
| A0 | Vanilla | No | No | Spatial control |
| A1 | Random spectral augmentation | No | No | “augmentation alone?” |
| A2 | Generic spectral consistency | No | No | “consistency alone?” |
| A3 | Worst-band CE only | Yes | No | value of adaptive worst view |
| A4 | Full CSMR | Yes | Yes | proposed method |

Optional:

```text
A5 = random-band + margin
A6 = all-three-band average
```

Nhưng không mở ablation quá rộng nếu A0–A4 đã trả lời đủ mechanism.

## Scale cho ablation

Không bắt buộc 100K cho mọi ablation.

Khuyến nghị:

```text
20K medium-scale:
A0–A4

100K final:
Vanilla
Full CSMR
```

---

# 9. Final dataset roles

| Dataset / split | Role | Architecture selection? | Final claim? |
|---|---|---:|---:|
| CelebA micro Train/Val | development | Yes | No |
| LCC-FASD | external-development / stress | Yes | Development evidence |
| fixed CelebA Test-2000 | already-opened descriptive benchmark | No further tuning | Descriptor only |
| 20K Train | scale sanity | No method retuning | No |
| 100K Train + 15K Val | final training/checkpoint protocol | Method frozen | Yes |
| CASIA-FASD | untouched external confirmation | **No** | **Yes** |

Critical rule:

> CASIA phải không được dùng để chọn backbone, crop, CSMR hyperparameters, training budget hoặc threshold.

---

# 10. Final cross-domain confirmation

Sau khi:

```text
backbone frozen
CSMR frozen
100K training complete
Val threshold locked
```

mới mở CASIA.

Evaluate once:

```text
Final Vanilla
vs
Final CSMR
```

Report:

```text
HTER
AUC
APCER
BPCER
coverage
```

Nếu có thêm compute, Replay-Attack/MSU/OULU có thể bổ sung sau, nhưng không cần mở trước CASIA.

---

# 11. Research questions

## RQ1 — Failure mode of explicit frequency use

> Can explicit frequency representations be discriminative in-domain yet unstable under domain shift in lightweight Face PAD?

Evidence:

- E2/E3;
- M0–M4;
- LCC development behavior.

## RQ2 — Training-time frequency robustness

> Can frequency information be used more effectively as a training-time robustness intervention than as an inference-time feature?

Evidence:

- Vanilla;
- explicit-frequency control;
- CSMR.

## RQ3 — Mechanism

> Does selecting the most harmful spectral counterfactual and regularizing PAD-margin degradation provide more benefit than generic spectral augmentation or consistency?

Evidence:

- A1/A2/A3/A4.

## RQ4 — Backbone transfer

> Does frozen CSMR-v1 transfer across lightweight spatial backbones?

Evidence:

- MNV3-S;
- MiniFASNetV2.

## RQ5 — Generalization

> Does the frozen method improve unseen-domain performance after final 100K training?

Evidence:

- untouched CASIA.

## RQ6 — Efficiency

> Can robustness improve without adding inference-time frequency computation?

Evidence:

- parameter count;
- FLOPs/MACs;
- latency/FPS;
- memory;
- same inference graph Vanilla vs CSMR within a backbone.

---

# 12. Contribution statements — recommended

Keep contribution claims modest.

### C1 — Empirical finding

> Explicit spectral representations can carry useful PAD signal while remaining vulnerable to domain-specific spectral shortcuts.

### C2 — Method

> CSMR, a sample-adaptive worst-band spectral intervention that directly regularizes harmful degradation of the Real-vs-Attack decision margin.

### C3 — Practicality

> Training-time spectral robustness with no additional FFT branch or inference-time overhead.

### C4 — Transfer evidence

Only if final experiments support it:

> CSMR transfers across two lightweight spatial backbones with consistent development-domain gains.

Do not claim architecture-independence from only two backbones / one seed.

---

# 13. Paper title candidates

Primary:

**Counterfactual Spectral Margin Robustness for Lightweight Cross-Domain Face Anti-Spoofing**

Alternative:

**Training-Time Spectral Robustness for Lightweight Face Anti-Spoofing with Zero Inference Overhead**

Narrative-oriented:

**From Frequency Cues to Frequency Robustness: Lightweight Face Anti-Spoofing under Domain Shift**

---

# 14. Main tables

## Table 1 — Historical motivation

```text
Spatial baseline
Frequency-only
Naïve spatial+frequency fusion
Representative M0–M4 findings
```

Purpose:

> explicit frequency cue is not automatically robust.

## Table 2 — CSMR micro/freeze evidence

```text
MNV3 seed42
MNV3 paired seed100
MiniFAS paired seed100
```

## Table 3 — Backbone decision

```text
MNV3 2.7× / 12ep
MiniFAS 2.7× / 12ep
```

Clearly annotate:

```text
MNV3 = ImageNet pretrained, 224
MiniFAS = PAD pretrained, 80
```

Do not call this pure architecture isolation.

## Table 4 — CSMR mechanism ablation

```text
Vanilla
Random spectral aug
Generic consistency
Worst-band CE
Full CSMR
```

## Table 5 — Final 100K result

```text
Final Vanilla
Final CSMR
source validation
LCC development
CASIA untouched
```

## Table 6 — Efficiency

```text
Params
MACs/FLOPs
model size
CPU latency
Raspberry Pi latency/FPS
peak memory
```

---

# 15. Main figures

## Figure 1 — CSMR concept

```text
clean sample
├─ LOW perturbation
├─ MID perturbation
└─ HIGH perturbation
     ↓
label-aligned PAD margin
     ↓
worst spectral counterfactual
     ↓
harmful-only margin robustness
```

## Figure 2 — Training vs inference

Training:

```text
RGB + FFT counterfactual generator + CSMR loss
```

Inference:

```text
RGB → spatial backbone → PAD
```

Highlight:

> zero inference-time FFT overhead.

## Figure 3 — Research evolution

```text
Spatial
→ explicit frequency
→ unstable cross-domain complementarity
→ spectral-shortcut diagnosis
→ CSMR
```

## Figure 4 — Backbone decision

Two panels:

```text
Case A MiniFASNetV2
Case B MNV3-S
```

## Figure 5 — CASIA ROC

Final Vanilla vs Full CSMR.

---

# 16. Planned paper structure

## 1. Introduction

- Face PAD generalization problem.
- Lightweight deployment constraint.
- Explicit frequency cue is attractive but can itself encode shortcut.
- CSMR shifts frequency from inference feature to training-time robustness stressor.
- contributions.

## 2. Related Work

### 2.1 Lightweight Face PAD
### 2.2 Frequency-aware Face PAD
### 2.3 Cross-domain / domain-generalizable PAD
### 2.4 Frequency augmentation, shortcut suppression and consistency

## 3. Method

### 3.1 Spatial PAD backbone
Use a generic notation so paper works for either final backbone.

### 3.2 Spectral counterfactual generator
### 3.3 Label-aligned PAD margin
### 3.4 Worst-band selection
### 3.5 Harmful-only robustness loss
### 3.6 Training/inference complexity

## 4. Experimental Protocol

- datasets;
- 5K development;
- backbone decision;
- 20K sanity if used;
- final 100K/15K;
- LCC development;
- opened Test-2000 status;
- CASIA untouched;
- metrics;
- efficiency protocol.

## 5. Results

### 5.1 Motivation / explicit-frequency diagnosis
### 5.2 CSMR development evidence
### 5.3 Backbone selection
### 5.4 Mechanism ablation
### 5.5 Final 100K result
### 5.6 Unseen CASIA confirmation
### 5.7 Efficiency

## 6. Discussion

- why explicit frequency can shortcut;
- why worst-band selection helps;
- calibration / APCER-BPCER trade-off;
- pretraining confound;
- limitations.

## 7. Conclusion

---

# 17. Backbone-dependent wording in the final paper

## If MiniFASNetV2 wins

Method text:

> We instantiate the spatial PAD model with the lightweight MiniFASNetV2 architecture and initialize its feature extractor from the public PAD-pretrained checkpoint, while reinitializing the task classifier.

Must disclose:

- PAD-specific initialization;
- `2.7×` crop;
- `80×80`;
- pretraining is not comparable to ImageNet-only initialization.

## If MNV3-S wins

Method text:

> We instantiate the spatial PAD model with ImageNet-pretrained MobileNetV3-Small and a lightweight three-class PAD head.

Must disclose:

- `2.7×` crop in final recipe;
- `224×224`;
- CelebA normalization;
- final schedule continuity with E1 v5.3.

---

# 18. What NOT to claim

Do not claim:

> CSMR is the first method to use frequency perturbation in Face PAD.

Do not claim:

> frequency features are inherently domain invariant.

Do not claim:

> MiniFASNet architecture alone causes the LCC improvement.

Do not claim:

> Test-2000 is still untouched.

Do not claim:

> one-seed evidence establishes statistical significance.

Prefer:

> development evidence supports...

> consistent with improved cross-domain robustness...

> the final unseen claim is reserved for CASIA...

---

# 19. Experiment-complete checklist

Paper is not experiment-complete until:

- [x] explicit-frequency development study completed;
- [x] CSMR-v1 mechanism frozen;
- [x] seed100 MNV3 paired replication completed;
- [x] MiniFASNet paired transfer completed;
- [ ] MNV3 `2.7× / 12ep` fair check completed;
- [ ] final backbone selected;
- [ ] 10–20K scale sanity completed or explicitly skipped;
- [ ] final 100K/15K Vanilla completed;
- [ ] final 100K/15K CSMR completed;
- [ ] CSMR mechanism ablation completed;
- [ ] efficiency benchmark completed;
- [ ] CASIA untouched confirmation completed;
- [ ] direct-vs-contextual literature comparisons audited;
- [ ] limitations section finalized.

---

# 20. Final decision tree

```text
MNV3 2.7× / 12ep fair check
        ↓
Does MiniFAS still have clear cross-domain Pareto advantage?
        │
   ┌────┴────┐
  YES        NO / GAP CLOSES
   │             │
   ▼             ▼
CASE A          CASE B
MiniFASNetV2    MNV3-S
   │             │
   └──────┬──────┘
          ▼
freeze final backbone
          ↓
20K scale sanity
          ↓
100K / 15K final paired training
          ↓
mechanism ablation + efficiency
          ↓
CASIA untouched confirmation
          ↓
paper
```

---

# 21. Core narrative to preserve

```text
1. Spatial PAD works well in-domain but degrades under domain shift.

2. Explicit frequency representations contain signal, but this signal
   is not automatically robust and can behave like a shortcut.

3. Therefore, frequency should not necessarily be trusted as an
   inference feature.

4. CSMR uses controlled spectral perturbations to identify the
   most harmful counterfactual for each sample.

5. The model is trained to resist harmful decision-margin degradation.

6. The inference graph remains purely spatial and lightweight.

7. Development evidence shows CSMR gains on MNV3-S and MiniFASNetV2.

8. The final backbone is selected only after the 2.7× fair comparison.

9. Final evidence comes from 100K training, mechanism ablation,
   efficiency analysis and untouched CASIA confirmation.
```

This is the paper story. Do not reopen architecture search after the final backbone gate unless the fair comparison reveals a clear implementation failure.
