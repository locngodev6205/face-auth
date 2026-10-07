# MAST-PAD Paper Roadmap & Experimental Design Plan
## Course-paper final story centered on the full-scale R7-SC configuration

_Last updated: 2026-10-07_

> **Title candidate:** **MAST-PAD: Margin-Aware Spectral Training for Lightweight Cross-Domain Face Anti-Spoofing**
>
> **Primary method:** MAST-PAD
>
> **Primary checkpoint:** `R7_SC_FULL_crop15`
>
> **Primary metric:** AUC only
>
> **Primary targets:** LCC Combined and reconstructed CASIA Combined

---

# 1. One-sentence paper idea

> **MAST-PAD uses the current Face PAD decision margin to identify a sample-specific harmful spectral counterfactual during source training and applies worst-view spectral supervision only when that intervention degrades the clean margin, while deployment remains the same 0.434M-parameter spatial MiniFASNetV2.**

---

# 2. Final paper scope

This remains a **course paper**, not a final publication-grade benchmark study.

Primary protocol:

```text
full eligible CelebA-Spoof Train (~484.6K usable)
→ full usable CelebA Test-as-Val for checkpoint selection / source calibration
→ freeze model
→ LCC-FASD Combined evaluation
→ CASIA-FASD reconstructed Combined evaluation
```

No target-domain adaptation is used.

The earlier matched100K study is retained as supporting development evidence only.

---

# 3. Problem statement

Lightweight RGB Face PAD models can fit source-domain spoof appearance very well but degrade under camera, media, illumination, and compression shifts.

Frequency-domain artifacts are useful because print/replay attacks can alter:

- fine texture;
- resampling patterns;
- display-grid / Moiré structure;
- camera-display transfer statistics;
- spectral energy distribution.

But frequency cues can themselves become domain-specific shortcuts.

Research question:

> **Instead of trusting frequency as a permanent inference feature, can controlled spectral interventions expose fragile PAD decisions during training and improve robustness without enlarging the deployed model?**

---

# 4. Method progression

## 4.1 Clean baseline

MiniFASNetV2 multi-task PAD trained only on the base/clean image.

## 4.2 WBST — Worst-Band Spectral Training

```text
LOW / MID / HIGH candidates
→ minimum aligned margin
→ always train on selected worst view
```

## 4.3 MAST-PAD

```text
if m_worst < m_clean:
    apply selected-worst spectral CE
else:
    skip spectral CE for this sample
```

Paper-facing progression:

```text
Clean → WBST → MAST-PAD
```

Because WBST and MAST-PAD also use different final spectral sigma values, this progression is a method comparison rather than a perfectly isolated one-factor causal ablation.

---

# 5. Shared deployed model

```text
SCRFD face localization
→ crop factor 1.5
→ 80×80 BGR [0,1]
→ MiniFASNetV2 trunk
→ 128-D embedding
→ binary PAD head
```

Training-only heads:

```text
spoof type: 11 classes
lighting: 5 classes
attributes: 40 binary labels, Real-only
```

Deployment complexity:

```text
params = 434,434
training model params = 441,658
MACs = 40,810,892
reported GFLOPs = 0.081621784
```

---

# 6. MAST-PAD method

## 6.1 Label-aligned margin

```text
d = z_real - z_attack
m = +d for Real
m = -d for Attack
```

## 6.2 Spectral candidates

```text
centers = [0.0833333333, 0.25, 0.4166666667]
sigma = 0.10
gain ~ Uniform(0.65, 0.90)
```

```text
M_k(r) = exp(-(r-c_k)^2 / (2 sigma^2))
G_k(r) = 1 - (1-g) M_k(r)
```

No amplification; DC and phase are preserved.

## 6.3 Worst-view selection

```text
k* = argmin_k m_k
```

Selection is BN-safe using temporary eval/no-grad candidate forwards.

## 6.4 Harmful gate

```text
harmful = (m_worst < m_clean)
```

## 6.5 Objective

Epoch 1: clean PAD CE + auxiliary losses.

Epoch 2+:

```text
L_PAD = 0.75 CE(clean) + 0.25 CE_harmful(selected_worst)
L_aux = 0.1 CE(spoof) + 0.1 CE(light) + BCE(attributes on Real)
L_total = L_PAD + L_aux
```

---

# 7. Primary full-scale experimental protocol

## 7.1 Source training / selection

```text
usable CelebA Train = 484,561
full Test-as-Val scored = 66,379 / 67,170 candidates
```

Test-as-Val is used for checkpoint selection and source threshold calibration. It is not presented as an untouched source test.

## 7.2 Optimization

```text
SGD
lr = 0.005
momentum = 0.9
weight_decay = 5e-4
batch = 256
AMP = on
max_epoch = 12
milestones = [4, 8]
gamma = 0.2
patience_start = 9
patience = 3
```

Best checkpoints:

```text
Clean     epoch 9
WBST      epoch 6
MAST-PAD  epoch 7
```

## 7.3 Target evaluation

### LCC-FASD

Use all available project LCC train/dev/eval predictions pooled as **LCC Combined**. No LCC refitting.

### CASIA-FASD

Use the extracted-image Kaggle copy, reconstruct video identity, average valid-frame scores per reconstructed video, and form a unique train/test union as **CASIA Combined**. No target-domain training or calibration.

Caveat: this is not claimed as the official CASIA subject-disjoint full-video protocol.

---

# 8. Main results — AUC only

## 8.1 Full-scale cross-domain table

| Method | LCC Combined AUC ↑ | CASIA Combined AUC ↑ | Params ↓ |
|---|---:|---:|---:|
| Clean | 0.846011 | 0.920398 | 0.434M |
| WBST | **0.860831** | 0.922264 | 0.434M |
| **MAST-PAD** | 0.860272 | **0.935199** | **0.434M** |

Paper-facing rounding:

```text
Clean      0.846 / 0.920
WBST       0.861 / 0.922
MAST-PAD   0.860 / 0.935
```

## 8.2 Interpretation

LCC:

- both spectral methods materially improve over Clean;
- WBST is higher than MAST-PAD by only `0.000559` AUC;
- do not claim MAST-PAD wins LCC pooled.

CASIA:

- MAST-PAD improves over Clean by `+0.014801`;
- MAST-PAD improves over WBST by `+0.012935`;
- this is the strongest independent target-domain result in the full-scale experiment.

Recommended sentence:

> **Full-scale training shows that spectral worst-view supervision improves pooled LCC robustness, while margin-aware harmful gating yields the strongest reconstructed CASIA cross-domain AUC without changing the 0.434M inference graph.**

---

# 9. Supporting 100K evidence

| Method | LCC Combined AUC |
|---|---:|
| Clean | 0.848273 |
| WBST | 0.851978 |
| MAST-PAD | **0.867844** |

Use this as developmental confirmation only. It should not replace the full-scale headline table.

---

# 10. Main literature comparison

Use **AUC only**.

## 10.1 LCC contextual table

| Method | LCC AUC ↑ | Params ↓ |
|---|---:|---:|
| Graph for Transformer Feature | 0.833 | ~10.84M |
| AENet | **0.868** | 11.22M |
| **MAST-PAD** | **0.860** | **0.434M** |

Interpretation:

> **MAST-PAD numerically exceeds the 0.833 Graph Transformer reference and is only 0.8 AUC points below the reported 0.868 AENet result, while using roughly 25× fewer parameters. Protocols differ.**

Sources:

Graph Transformer:
- https://doi.org/10.14428/esann/2023.ES2023-14
- https://www.esann.org/sites/default/files/proceedings/2023/ES2023-14.pdf

AENet:
- original ECCV paper: https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/1485_ECCV_2020_paper.php
- LCC benchmark value: https://github.com/kprokofi/light-weight-face-anti-spoofing

## 10.2 CASIA cross-domain contextual table

| Method | Venue | Target protocol | CASIA AUC ↑ |
|---|---|---|---:|
| MADDG | CVPR 2019 | O&M&I → C | 0.8451 |
| AMEL | ACM MM 2022 | O&M&I → C | 0.9439 |
| PatchNet | CVPR 2022 | O&M&I → C | 0.9458 |
| GAC-FAS | CVPR 2024 | O&M&I → C | 0.9516 |
| SA-FAS | CVPR 2023 | O&M&I → C | 0.9537 |
| **MAST-PAD** | ours | CelebA-full → reconstructed C Combined | **0.9352** |

Main trade-off message:

- MAST-PAD exceeds MADDG by about `+9.01 pp`;
- it is within `0.87 pp` of AMEL;
- within `1.06 pp` of PatchNet;
- within `1.64 pp` of GAC-FAS;
- within `1.85 pp` of SA-FAS;
- MAST-PAD deployment remains only `0.434M` parameters.

Sources:

MADDG:
- https://openaccess.thecvf.com/content_CVPR_2019/html/Shao_Multi-Adversarial_Discriminative_Deep_Domain_Generalization_for_Face_Presentation_Attack_Detection_CVPR_2019_paper.html
- https://openaccess.thecvf.com/content_CVPR_2019/papers/Shao_Multi-Adversarial_Discriminative_Deep_Domain_Generalization_for_Face_Presentation_Attack_Detection_CVPR_2019_paper.pdf

AMEL:
- https://arxiv.org/abs/2207.09868
- https://arxiv.org/pdf/2207.09868
- https://doi.org/10.1145/3503161.3547769

PatchNet:
- https://openaccess.thecvf.com/content/CVPR2022/html/Wang_PatchNet_A_Simple_Face_Anti-Spoofing_Framework_via_Fine-Grained_Patch_Recognition_CVPR_2022_paper.html
- https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_PatchNet_A_Simple_Face_Anti-Spoofing_Framework_via_Fine-Grained_Patch_Recognition_CVPR_2022_paper.pdf

SA-FAS:
- https://openaccess.thecvf.com/content/CVPR2023/html/Sun_Rethinking_Domain_Generalization_for_Face_Anti-Spoofing_Separability_and_Alignment_CVPR_2023_paper.html
- https://openaccess.thecvf.com/content/CVPR2023/papers/Sun_Rethinking_Domain_Generalization_for_Face_Anti-Spoofing_Separability_and_Alignment_CVPR_2023_paper.pdf

GAC-FAS:
- https://openaccess.thecvf.com/content/CVPR2024/html/Le_Gradient_Alignment_for_Cross-Domain_Face_Anti-Spoofing_CVPR_2024_paper.html
- https://openaccess.thecvf.com/content/CVPR2024/papers/Le_Gradient_Alignment_for_Cross-Domain_Face_Anti-Spoofing_CVPR_2024_paper.pdf

---

# 11. Full-scale diagnostics

MAST-PAD (`R7_SC_FULL_crop15`):

```text
best epoch = 7
stop epoch = 11
harmful fraction = 0.832135
margin drop = 0.391421
spectral flip rate = 0.003077
LOW = 0.426452
MID = 0.273035
HIGH = 0.300513
```

These support the optimization story that margin degradation is much more common than outright label flipping.

---

# 12. Main contributions

## C1 — lightweight deployment baseline

> **A 0.434M-parameter MiniFASNetV2 binary PAD graph suitable for edge-oriented inference.**

## C2 — MAST-PAD

> **Margin-aware selection of harmful LOW/MID/HIGH spectral counterfactuals with harmful-only worst-view supervision.**

## C3 — two-target full-scale cross-domain evidence

> **Full-scale MAST-PAD reaches 0.860 LCC Combined AUC and 0.935 reconstructed CASIA Combined AUC without target adaptation.**

## C4 — unchanged inference graph

> **All frequency-domain computation is training-only.**

---

# 13. Recommended paper structure

## 1. Introduction
- lightweight Face PAD and domain shift;
- frequency cues and shortcut risk;
- frequency-as-training-stressor;
- contributions.

## 2. Related Work
- lightweight Face PAD;
- frequency-aware Face PAD;
- domain generalization;
- spectral shortcut / robustness training.

## 3. Method
- MiniFASNetV2 backbone;
- spectral counterfactual generator;
- aligned margin;
- worst-band selection;
- harmful gate;
- loss;
- inference complexity.

## 4. Experimental Setup
- full CelebA source training;
- Test-as-Val selection caveat;
- LCC Combined construction;
- CASIA reconstructed Combined construction;
- AUC as headline metric;
- literature protocol caveats.

## 5. Results
### 5.1 Full-scale Clean vs WBST vs MAST-PAD
### 5.2 LCC contextual comparison
### 5.3 CASIA cross-domain contextual comparison
### 5.4 100K supporting evidence
### 5.5 Training diagnostics and efficiency

## 6. Discussion
- why MAST-PAD and WBST can rank differently across targets;
- why multi-target evidence is more informative than one LCC number;
- protocol heterogeneity;
- one-seed limitation;
- reconstructed CASIA limitation.

## 7. Conclusion
Keep claims bounded to evidence.

---

# 14. Recommended figures

1. Method overview.
2. Training-only vs inference graph.
3. Full-scale two-target AUC grouped chart.
4. LCC AUC vs parameters contextual chart.
5. CASIA literature AUC contextual chart.
6. Full-scale MAST-PAD training diagnostics.
7. Optional 100K supporting chart.

---

# 15. Reporting precision

```text
AUC: 3 decimals in main paper tables
params: 2–3 significant decimals
full precision retained in artifacts
```

Use:

```text
LCC Combined 0.860272 → 0.860
CASIA Combined 0.935199 → 0.935
params 0.434434M → 0.434M
```

---

# 16. Limitations

1. one source-training seed;
2. full CelebA Test is used as Test-as-Val for selection/calibration;
3. LCC Combined pools multiple LCC populations;
4. reconstructed CASIA uses an extracted-image copy and is not a verified reproduction of the official subject-disjoint video benchmark;
5. literature source-domain protocols differ;
6. MAST-PAD increases training compute through multiple candidate forwards;
7. architecture evidence remains MiniFASNetV2-specific;
8. parameter/FLOP conventions differ across papers.

---

# 17. Claims to avoid

Do not claim:

- SOTA;
- statistical superiority;
- MAST-PAD beats WBST on every target;
- all literature comparisons use identical protocols;
- CASIA Combined is the official CASIA benchmark;
- 12 epochs proves universal sample efficiency.

---

# 18. Final paper decision

```text
PROPOSED METHOD
MAST-PAD

PRIMARY CHECKPOINT
R7_SC_FULL_crop15

PRIMARY METRIC
AUC

PRIMARY TARGET RESULTS
LCC Combined   = 0.860
CASIA Combined = 0.935

DEPLOYMENT PARAMS
0.434M

SUPPORTING DEVELOPMENT STUDY
Train100K / Val15K
```
