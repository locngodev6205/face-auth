# PAD Experiment Roadmap v3 — CSMR-PAD Final-Scale Evidence and Remaining Validation

_Last updated: 2026-09-30_

> This document supersedes the previous roadmap that still contained backbone-selection and intermediate scale-sanity stages.
>
> **Current status:** MiniFASNetV2 is the final experimental backbone; the main paired experiments have been completed at **100K Train / 15K Val** for both random initialization and PAD-pretrained initialization. Intermediate scale-check results are archived for provenance only and are intentionally omitted from the main research story because the 100K evidence supersedes them.

---

# 1. Current research question

The project is no longer searching for a frequency branch or a final backbone.

The active paper question is now:

> **Can Counterfactual Spectral Margin Robustness (CSMR) improve cross-domain Face PAD robustness while preserving a spatial-only, lightweight inference graph?**

The final experimental model family is:

```text
Face image
→ SCRFD localization
→ 2.7× context crop
→ 80×80 BGR [0,1]
→ MiniFASNetV2
→ 3 logits: Real / Physical Spoof / Digital Spoof
```

CSMR is active **only during training**.

Inference remains:

```text
RGB crop → MiniFASNetV2 → PAD logits
```

No FFT, frequency branch, fusion module, or extra inference parameters are used at deployment.

---

# 2. Research story to preserve

The paper narrative should remain compact:

```text
1. Spatial PAD is strong on the source domain but degrades under domain shift.

2. Explicit frequency representations contain useful PAD signal,
   but their cross-domain complementarity is unstable.

3. This supports the hypothesis that frequency cues can become
   domain-sensitive spectral shortcuts when used directly as inference features.

4. CSMR therefore uses frequency as a controlled training-time stressor,
   not as an inference feature.

5. For each sample, CSMR searches LOW / MID / HIGH spectral interventions
   for the one that most harms the label-aligned PAD margin.

6. Training penalizes harmful margin degradation while keeping
   inference spatial-only.

7. Final 100K paired experiments show positive CSMR effects in the
   scratch regime and a strong LCC cross-domain effect in the
   PAD-pretrained regime.

8. A separate CASIA-FASD Kaggle image-copy evaluation provides an
   additional untouched cross-dataset confirmation, with important
   protocol limitations documented below.
```

Do not reopen architecture search unless a reproducibility failure is discovered.

---

# 3. Historical frequency diagnosis — retain only as motivation

The old explicit-frequency line of work remains useful for **RQ1 / motivation**, not as the final architecture.

Studied variants:

```text
M0 Global DCT
M1 Grid-DCT / pooled DCT
M2 frequency-coordinate representation
M3 LOW/MID/HIGH band-aware DCT
M4 Block-DCT
```

Main finding:

- explicit frequency representations can be discriminative on the source data;
- their LCC behavior did not reliably improve the spatial baseline;
- the overall pattern is consistent with domain-sensitive spectral shortcut learning;
- no explicit frequency branch is carried into final inference.

For the paper, avoid overclaiming that this “proves” spectral shortcuts. Use:

> **The observed cross-domain degradation is consistent with frequency representations encoding domain-sensitive shortcut cues.**

Important internal note:

> M2 provenance should be rechecked before publishing its exact number. Do not rely on M2 as the sole negative example.

Intermediate architecture-search and scale-check results are not part of the main results section.

---

# 4. Final method — CSMR-PAD

**CSMR = Counterfactual Spectral Margin Robustness.**

## 4.1 PAD score

For logits:

```text
z_real, z_physical, z_digital
```

define:

```text
d = z_real - logsumexp(z_physical, z_digital)
```

Higher `d` means more Real.

Label-aligned PAD margin:

```text
Real   → m = +d
Attack → m = -d
```

Higher `m` means a safer correct binary PAD decision.

## 4.2 Spectral counterfactuals

For each base-augmented sample, generate:

```text
LOW-band attenuation
MID-band attenuation
HIGH-band attenuation
```

with radial Gaussian masks:

```text
centers = [0.15, 0.45, 0.75]
sigma   = 0.10
```

Gain field:

```text
G_k = 1 - (1-g) M_k
```

Constraints:

- phase unchanged;
- DC preserved;
- no amplification;
- one sampled gain per sample, shared across LOW/MID/HIGH;
- same radial gain across channels;
- FFT operates on image float `[0,1]`.

Worst spectral view:

```text
k* = argmin_k m_k
```

## 4.3 Loss

Warmup epoch:

```text
L = CE(clean)
```

CSMR-active epochs:

```text
L_cls = 0.5 CE(clean) + 0.5 CE(worst)

target = m_clean.detach()

harmful = (m_worst < target)

L_margin =
    mean(
        SmoothL1(m_worst, target)
        * harmful
    )

L_total = L_cls + lambda * L_margin
```

The contribution is not simply “frequency augmentation”.

The method-specific combination is:

> **sample-adaptive worst-band intervention + label-aligned PAD-margin degradation regularization + zero inference-time frequency branch.**

---

# 5. Development tuning protocol — frozen before final 100K

The final regime-specific recipe came from the controlled `00b-v2` development search.

Development data:

```text
CelebA-Spoof Train = 5,000
CelebA-Spoof Tune-Val = 2,000
LCC-FASD common-valid ≈ 3,762
training seed = 100
```

Roles must be described precisely:

- optimizer/LR selection used **CelebA Tune-Val only**;
- CSMR-specific `lambda` and gain were selected only after a source-domain regression guard;
- among eligible CSMR candidates, **LCC AUC** was the primary external-development ranking signal, followed by LCC HTER and Tune-Val ACER;
- therefore LCC-FASD is **external-development**, not an untouched final test;
- CelebA Test5K was not used for tuning;
- CASIA was not used during method tuning.

Frozen selection artifact:

```text
frozen_hparams_v2 SHA256
bfc8652b237d7a8156bf455667bcbc00794baa9b0f0283c4a449406da6caeb0a
```

## 5.1 Final scratch recipe

```text
backbone LR = 1e-3
head LR     = 1e-3
lambda      = 0.20
gain        = Uniform(0.55, 0.85)
sigma       = 0.10
warmup      = 1 epoch
```

## 5.2 Final PAD-pretrained recipe

```text
backbone LR = 2e-4
head LR     = 1e-3
lambda      = 0.40
gain        = Uniform(0.65, 0.90)
sigma       = 0.10
warmup      = 1 epoch
```

No further LR / lambda / gain tuning should be performed from LCC, CelebA Test, or CASIA outcomes.

---

# 6. Final 100K protocol

Main final training protocol:

```text
Train = 100,000 CelebA-Spoof samples
Val   = 15,000 CelebA-Spoof samples
Test  = fixed 5,000 official CelebA Test samples
LCC   = same common-valid external-development set
seed  = 100
```

Two paired initialization regimes are deliberately retained.

## Regime A — Scratch

```text
MiniFASNetV2 random initialization
Vanilla vs Full CSMR
same initial-state policy
same Train/Val manifests
same optimization budget
```

Scientific role:

> **Primary methodological evidence**, because the CSMR effect is not inherited from an external PAD-pretrained representation.

## Regime B — PAD-pretrained

```text
Official MiniFASNetV2 PAD-pretrained compatible feature weights
fresh project 3-class classifier
Vanilla vs Full CSMR
```

Scientific role:

> **Complementary transfer/practical evidence** showing how CSMR behaves on an already strong PAD representation.

Do not use absolute Scratch-vs-Pretrained differences as evidence for CSMR.

Only use paired comparisons:

```text
Scratch CSMR     vs Scratch Vanilla
Pretrained CSMR  vs Pretrained Vanilla
```

---

# 7. Main result — 100K scratch

| Split | Metric | Vanilla | CSMR | Δ CSMR−Vanilla |
|---|---:|---:|---:|---:|
| CelebA Val15K | ACER ↓ | 1.9278% | **1.7409%** | **−0.1869 pp** |
| CelebA Val15K | AUC ↑ | 0.997166 | **0.997670** | **+0.000504** |
| CelebA Test5K | ACER ↓ | 20.2137% | **19.0630%** | **−1.1508 pp** |
| CelebA Test5K | AUC ↑ | 0.888879 | **0.926548** | **+3.7669 pp** |
| LCC external-dev | HTER ↓ | 37.4299% | **36.2687%** | **−1.1611 pp** |
| LCC external-dev | AUC ↑ | 0.687602 | **0.696461** | **+0.8859 pp** |

Additional LCC error decomposition:

```text
APCER: 53.7800% → 51.2007%   (−2.5793 pp)
BPCER: 21.0797% → 21.3368%   (+0.2571 pp)
```

Interpretation:

> At 100K from random initialization, CSMR improves Val ACER/AUC, fixed CelebA Test ACER/AUC, and both LCC AUC and locked-threshold HTER. The external-development gain is modest but directionally coherent.

This is currently the cleanest evidence that CSMR itself can help rather than merely exploiting a pretrained PAD representation.

Do not call a one-seed result statistically significant.

---

# 8. Main result — 100K PAD-pretrained

| Split | Metric | Vanilla | CSMR | Δ CSMR−Vanilla |
|---|---:|---:|---:|---:|
| CelebA Val15K | ACER ↓ | **0.9653%** | 1.1061% | +0.1408 pp |
| CelebA Val15K | AUC ↑ | **0.998464** | 0.997759 | −0.000705 |
| CelebA Test5K | ACER ↓ | 17.3968% | **16.9309%** | **−0.4659 pp** |
| CelebA Test5K | AUC ↑ | **0.928305** | 0.881835 | **−4.6471 pp** |
| LCC external-dev | HTER ↓ | 34.3071% | **29.3791%** | **−4.9280 pp** |
| LCC external-dev | AUC ↑ | 0.741400 | **0.795833** | **+5.4433 pp** |

LCC error decomposition:

```text
APCER: 47.0205% → 43.8482%   (−3.1723 pp)
BPCER: 21.5938% → 14.9100%   (−6.6838 pp)
```

Interpretation:

- LCC external-development improvement is strong in both threshold-free ranking and locked-threshold error;
- source Val regresses only slightly;
- CelebA Test ACER improves slightly, but Test AUC drops substantially.

Therefore the correct claim is **not** “CSMR improves every distribution”.

Use:

> **The pretrained regime shows a strong LCC cross-domain gain, accompanied by a distribution-dependent ranking trade-off on the CelebA Test split.**

The fixed CelebA Test split should be described as:

> **same-dataset held-out official Test split**

not automatically as IID/in-domain, because its behavior differs strongly from CelebA Val.

---

# 9. External confirmation — CASIA-FASD Kaggle image copy

CASIA was not used for CSMR tuning, optimizer selection, checkpoint selection, or source-threshold calibration.

The four frozen seed-100 100K models were evaluated with their own CelebA-Val locked thresholds.

Primary paired results on the reconstructed video-level evaluation:

| Initialization | Method | AUC ↑ | APCER ↓ | BPCER ↓ | HTER ↓ |
|---|---|---:|---:|---:|---:|
| Scratch | Vanilla | 0.645059 | 16.2879% | 56.8182% | 36.5530% |
| Scratch | CSMR | **0.701963** | **13.2576%** | **54.5455%** | **33.9015%** |
| PAD-pretrained | Vanilla | 0.687758 | **23.8636%** | 46.5909% | 35.2273% |
| PAD-pretrained | CSMR | **0.710744** | 34.0909% | **27.2727%** | **30.6818%** |

Paired CSMR deltas:

```text
Scratch:
AUC   +5.6904 pp
HTER  −2.6515 pp
APCER −3.0303 pp
BPCER −2.2727 pp

PAD-pretrained:
AUC   +2.2986 pp
HTER  −4.5455 pp
APCER +10.2273 pp   ← security-sensitive regression
BPCER −19.3182 pp
```

Coverage:

```text
352 / 360 reconstructed video IDs scorable
2,405 / 2,405 planned JPEG frames successfully localized
```

## Critical protocol limitation

The available Kaggle dataset is an **extracted JPEG image copy**, not the original official CASIA-FASD video protocol.

The executed evaluation found:

```text
test_img contains 30 subjects numbered 1–30
360 reconstructed video IDs
train/test packaging overlaps in subjects/video/frame identifiers
source videos are unavailable
```

Therefore:

- this result is useful as an **untouched cross-dataset confirmation within this project**;
- it is **not** an official subject-disjoint CASIA-FASD benchmark result;
- it must not be directly compared to published CASIA numbers that follow a different protocol;
- for a publication-grade standard benchmark claim, obtain the original protocol or add another standard external dataset such as OULU-NPU / Replay-Attack / MSU-MFSD.

Interpretation:

> Both initialization regimes improve CASIA-copy AUC and HTER under frozen CelebA thresholds, which strengthens the cross-domain signal beyond LCC. However, pretrained CSMR increases APCER markedly while reducing BPCER, so the operating-point/security trade-off must be reported explicitly.

---

# 10. Efficiency / deployment evidence

MiniFASNetV2 deployment model:

```text
parameters = 434,560
input      = 3×80×80 BGR
ONNX size  ≈ 1.74 MB
```

Vanilla and CSMR use the **same ordinary MiniFASNetV2 inference graph**.

ONNX parity checks passed for final Scratch and PAD-pretrained models.

Core deployment claim:

> **CSMR adds training-time computation but no frequency branch, FFT, or model-parameter overhead at inference.**

For the final paper efficiency table, report the exact measured values from the saved experiment artifacts:

- Params;
- MACs/FLOPs;
- ONNX size;
- CPU latency / FPS;
- peak RAM;
- Raspberry Pi latency / FPS if available.

Do not generalize an offline CPU benchmark to real camera end-to-end throughput without measuring the full deployment pipeline.

---

# 11. What is now frozen

Freeze:

```text
Backbone: MiniFASNetV2
Input: 80×80 BGR [0,1]
Crop: 2.7×
PAD score: real - logsumexp(physical, digital)
CSMR centers: [0.15, 0.45, 0.75]
sigma: 0.10
DC preserved
phase unchanged
worst-band: argmin label-aligned PAD margin
harmful-only SmoothL1 margin regularization
regime-specific lambda / gain / LR from frozen_hparams_v2
```

Do not:

1. retune CSMR using final LCC, CelebA Test, or CASIA results;
2. reopen intermediate recipe search;
3. replace the frozen 03 recipe because a historical pilot recipe looks better on one metric;
4. add a new inference branch solely to improve novelty;
5. select a new threshold on LCC or CASIA;
6. hide Test AUC or APCER/BPCER trade-offs.

---

# 12. Remaining experiments with highest paper value

## Priority 1 — CSMR mechanism ablation

Minimum matrix:

| ID | Variant | Purpose |
|---|---|---|
| A | Vanilla | spatial baseline |
| B | Random spectral augmentation | augmentation-only control |
| C | Worst-band CE, `lambda=0` | value of adaptive worst-view selection |
| D | Generic KL/logit consistency | generic consistency control |
| E | Full CSMR | proposed method |

The ablation should isolate:

```text
spectral augmentation
vs
sample-adaptive worst-band selection
vs
generic consistency
vs
PAD-margin-aware harmful-drop regularization
```

Prefer the final frozen backbone and the same protocol. If a reduced training budget is used for ablation because of compute constraints, pre-register it and keep it clearly separate from the final 100K main table.

## Priority 2 — Additional training seeds

Current main evidence is still one training seed.

Recommended paired seeds:

```text
100  completed
132  pending
150  pending
```

At minimum, repeat the **Scratch 100K Vanilla/CSMR pair** because scratch is the cleanest methodological regime.

If compute allows, repeat the PAD-pretrained pair too.

Report:

```text
mean ± std
```

for paired deltas, especially:

```text
LCC AUC
LCC HTER
CelebA Test AUC/ACER
external confirmation metrics
```

## Priority 3 — Standard external benchmark protocol

Because the current CASIA Kaggle copy is not the official subject-disjoint protocol, one of the following would materially strengthen the paper:

```text
official CASIA-FASD protocol
or
OULU-NPU
or
Replay-Attack
or
MSU-MFSD
```

Use frozen models and source-Val thresholds. No target adaptation.

## Priority 4 — Complete efficiency table

Add:

```text
MACs/FLOPs
peak RAM
Raspberry Pi latency/FPS
```

---

# 13. Paper-facing claim discipline

Supported:

> CSMR uses sample-adaptive spectral counterfactuals and PAD-margin-aware regularization during training while preserving a spatial-only MiniFASNetV2 inference graph.

Supported:

> At 100K scale, scratch training shows coherent paired improvements across CelebA Val, fixed CelebA Test, and LCC external-development metrics.

Supported:

> With PAD-pretrained initialization, CSMR substantially improves LCC AUC and HTER, although the effect is distribution-dependent and CelebA Test AUC decreases.

Supported with CASIA protocol qualifier:

> Frozen CSMR models improve AUC and HTER on an untouched CASIA-FASD Kaggle image-copy evaluation in both initialization regimes.

Not yet supported:

> CSMR is statistically significant across training seeds.

Not supported:

> CSMR improves every metric on every domain.

Not supported:

> Current CASIA numbers are directly comparable with the official CASIA-FASD benchmark literature.

Not supported:

> CSMR is architecture-independent.

Not supported:

> CSMR is SOTA.

---

# 14. Paper-ready result hierarchy

Main paper evidence should prioritize:

```text
1. M0–M4 / explicit-frequency diagnosis
   → motivation, concise

2. Final 100K Scratch Vanilla vs CSMR
   → primary methodological evidence

3. Final 100K PAD-pretrained Vanilla vs CSMR
   → complementary transfer/practical evidence

4. CASIA-FASD Kaggle image-copy frozen evaluation
   → external confirmation with explicit protocol limitation

5. CSMR mechanism ablation
   → required for contribution isolation

6. Multi-seed confirmation
   → uncertainty / reproducibility

7. Efficiency
   → zero inference-overhead deployment claim
```

Do not place superseded intermediate scale-sanity results in the main paper tables when the corresponding 100K results are available.

---

# 15. Updated run order

```text
COMPLETED
---------
1. explicit-frequency diagnostic / M0–M4
2. 5K/2K regime-specific tuning and freeze
3. MiniFASNetV2 final backbone freeze
4. 100K Scratch Vanilla vs CSMR
5. 100K PAD-pretrained Vanilla vs CSMR
6. frozen CASIA-FASD Kaggle image-copy evaluation
7. ONNX export and parity validation

NEXT
----
8. mechanism ablation A–E
9. +2 paired training seeds, prioritizing Scratch
10. standard external benchmark protocol if obtainable
11. complete edge efficiency measurements
12. final paper tables / figures / writing
```

---

# 16. Stop rule

The method itself is now **frozen**.

The remaining work is no longer “find a better CSMR”.

It is:

> **demonstrate the frozen method more rigorously through mechanism ablation, training-seed replication, standard external evaluation, and deployment measurement.**
