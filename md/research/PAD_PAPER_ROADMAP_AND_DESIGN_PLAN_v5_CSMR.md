# PAD Paper Roadmap & Experimental Story v5 — CSMR-PAD

_Last updated: 2026-09-30_

> **Purpose:** paper-facing research plan after completion of the final 100K MiniFASNetV2 experiments and the first frozen CASIA cross-dataset confirmation.
>
> The previous backbone gate and intermediate scale-sanity narrative is obsolete. Those runs remain archived for provenance, but the paper should focus on final 100K evidence, mechanism ablation, reproducibility, external validation, and efficiency.

---

# 1. One-sentence paper idea

> **CSMR-PAD treats frequency not as an inference feature but as a training-time counterfactual stressor: for each sample, it identifies the LOW/MID/HIGH spectral intervention that most degrades the label-aligned PAD margin and regularizes that harmful degradation, while deployment remains a lightweight spatial-only MiniFASNetV2 with zero FFT overhead.**

Do not define novelty as:

> “We use FFT for Face PAD.”

Do not define novelty as:

> “Frequency augmentation improves generalization.”

The method contribution is more specific:

> **sample-adaptive worst-band spectral intervention + Real-vs-Attack margin-aware harmful-drop regularization + zero inference-time spectral branch.**

---

# 2. Final paper story

## 2.1 Problem

Single-frame RGB Face PAD models can achieve excellent source-validation performance while degrading sharply under distribution shift.

The project uses three project labels:

```text
0 Real
1 Physical Spoof
2 Digital Spoof
```

but evaluates the security decision as:

```text
Real
vs
Attack = {Physical, Digital}
```

Core score:

```text
d = z_real - logsumexp(z_physical, z_digital)
```

## 2.2 Why frequency is interesting — and dangerous

Frequency cues are physically plausible for PAD:

- replay media may create display-grid / Moiré / resampling artifacts;
- print attacks may alter fine texture and high-frequency statistics;
- capture pipelines can leave spectral signatures.

However, the project's explicit-frequency experiments did not produce reliable cross-domain complementarity.

The correct paper interpretation is:

> **Frequency can be discriminative without being domain invariant.**

The observed M0–M4 behavior is consistent with:

> **domain-sensitive spectral shortcut learning.**

Avoid writing that M0–M4 “prove” a shortcut mechanism unless a dedicated causal diagnostic is added.

## 2.3 Pivot

The conceptual pivot is:

```text
OLD:
frequency as a feature the model should trust at inference

NEW:
frequency as a controlled stressor used to teach a spatial model
what it should not become fragile to
```

This is the bridge into CSMR.

---

# 3. Final method

**CSMR = Counterfactual Spectral Margin Robustness.**

## 3.1 Spectral intervention

From one base-augmented image, create three counterfactuals:

```text
LOW
MID
HIGH
```

using smooth radial Gaussian attenuation:

```text
centers = [0.15, 0.45, 0.75]
sigma   = 0.10

G_k = 1 - (1-g) M_k
```

Constraints:

- phase unchanged;
- DC preserved;
- no amplification;
- same gain across channels;
- one gain sampled per sample and shared across LOW/MID/HIGH;
- FFT on float image `[0,1]`.

## 3.2 Label-aligned PAD margin

```text
d = z_real - logsumexp(z_physical, z_digital)

sign = +1 for Real
sign = -1 for Attack

m(z,y) = sign * d
```

Higher `m` means a safer correct binary PAD decision.

Worst counterfactual:

```text
k* = argmin_k m_k
```

## 3.3 Harmful-drop regularization

After one clean warmup epoch:

```text
L_cls = 0.5 CE(clean) + 0.5 CE(worst)

target = m_clean.detach()
harmful = (m_worst < target)

L_margin =
    mean(
        SmoothL1(m_worst, target)
        * harmful
    )

L_total = L_cls + lambda L_margin
```

The penalty is asymmetric:

> if the spectral intervention does not reduce the correct PAD margin, CSMR does not penalize it.

## 3.4 Inference

Final deployment:

```text
SCRFD
→ 2.7× crop
→ 80×80 BGR [0,1]
→ MiniFASNetV2
→ 3 logits
```

No FFT / DCT branch is present at inference.

---

# 4. Final backbone and why two initialization regimes are retained

The final experimental backbone is **MiniFASNetV2**:

```text
434,560 parameters
80×80 input
2.7× context crop
ordinary spatial inference graph
```

Two 100K regimes are retained intentionally.

## Scratch

Randomly initialized MiniFASNetV2.

Role:

> **Primary methodological evidence.**

Reason:

- avoids attributing the CSMR gain to a pre-existing PAD representation;
- provides the cleanest paired test of the training method.

## PAD-pretrained

Compatible feature weights from the public MiniFASNetV2 PAD checkpoint; project classifier reinitialized.

Role:

> **Complementary transfer / practical evidence.**

Reason:

- tests whether the method remains useful on a strong PAD-specific initialization;
- gives a deployment-relevant model.

Important limitation:

> public checkpoint provenance and possible source overlap are not sufficiently controlled for it to be the sole scientific evidence.

Therefore the paper should not replace Scratch evidence with Pretrained evidence.

---

# 5. Development and tuning protocol

The final hyperparameters were frozen before the 100K runs.

Development resources:

```text
CelebA-Spoof Train      5,000
CelebA-Spoof Tune-Val   2,000
LCC-FASD common-valid   ~3,762
seed                    100
```

Selection hierarchy:

### Optimizer / LR

Selected on:

```text
CelebA Tune-Val only
```

### CSMR-specific hyperparameters

Candidates first had to satisfy a source-domain guard:

```text
CSMR Tune-Val ACER
<=
selected Vanilla Tune-Val ACER + 1 percentage point
```

Among eligible candidates:

```text
1. highest LCC AUC
2. lowest LCC HTER
3. lowest Tune-Val ACER
4. deterministic conservative tie-break
```

Therefore:

> **LCC-FASD is external-development, not an untouched final test.**

CelebA Test5K was not used for this selection.

CASIA was not used for method tuning.

Frozen selection SHA256:

```text
bfc8652b237d7a8156bf455667bcbc00794baa9b0f0283c4a449406da6caeb0a
```

## Scratch frozen recipe

```text
backbone LR = 1e-3
head LR     = 1e-3
lambda      = 0.20
gain        = [0.55, 0.85]
sigma       = 0.10
warmup      = 1
```

## PAD-pretrained frozen recipe

```text
backbone LR = 2e-4
head LR     = 1e-3
lambda      = 0.40
gain        = [0.65, 0.90]
sigma       = 0.10
warmup      = 1
```

Do not perform further hyperparameter search from final results.

---

# 6. Dataset roles in the final paper

| Dataset / split | Role | Used for tuning? | How it may be described |
|---|---|---:|---|
| CelebA Train5K / Tune-Val2K | development search | Yes | hyperparameter development |
| CelebA Train100K | final source training | No new method tuning | final training |
| CelebA Val15K | checkpoint + source threshold | Yes, within frozen protocol | final source validation |
| CelebA official Test5K | held-out same-dataset descriptor | No | same-dataset held-out Test; not assumed IID |
| LCC-FASD | external-development stress domain | Yes, during CSMR HPO | development cross-domain evidence |
| CASIA-FASD Kaggle image copy | untouched cross-dataset confirmation | No | project-specific external confirmation, **not official CASIA benchmark** |

This distinction must remain visible in Methods and Limitations.

---

# 7. Main result I — final 100K Scratch

| Split | Metric | Vanilla | CSMR | Δ CSMR−Vanilla |
|---|---:|---:|---:|---:|
| CelebA Val15K | ACER ↓ | 1.9278% | **1.7409%** | **−0.1869 pp** |
| CelebA Val15K | AUC ↑ | 0.997166 | **0.997670** | **+0.000504** |
| CelebA Test5K | ACER ↓ | 20.2137% | **19.0630%** | **−1.1508 pp** |
| CelebA Test5K | AUC ↑ | 0.888879 | **0.926548** | **+3.7669 pp** |
| LCC external-dev | HTER ↓ | 37.4299% | **36.2687%** | **−1.1611 pp** |
| LCC external-dev | AUC ↑ | 0.687602 | **0.696461** | **+0.8859 pp** |

LCC error directions:

```text
APCER −2.5793 pp
BPCER +0.2571 pp
```

Paper interpretation:

> **When trained from random initialization at 100K scale, CSMR yields coherent paired improvements on source validation, the fixed CelebA Test split, and LCC external-development AUC/HTER.**

This is the strongest current evidence for a method effect independent of PAD pretraining.

Keep wording limited to one seed until replication is complete.

---

# 8. Main result II — final 100K PAD-pretrained

| Split | Metric | Vanilla | CSMR | Δ CSMR−Vanilla |
|---|---:|---:|---:|---:|
| CelebA Val15K | ACER ↓ | **0.9653%** | 1.1061% | +0.1408 pp |
| CelebA Val15K | AUC ↑ | **0.998464** | 0.997759 | −0.000705 |
| CelebA Test5K | ACER ↓ | 17.3968% | **16.9309%** | **−0.4659 pp** |
| CelebA Test5K | AUC ↑ | **0.928305** | 0.881835 | **−4.6471 pp** |
| LCC external-dev | HTER ↓ | 34.3071% | **29.3791%** | **−4.9280 pp** |
| LCC external-dev | AUC ↑ | 0.741400 | **0.795833** | **+5.4433 pp** |

LCC:

```text
APCER −3.1723 pp
BPCER −6.6838 pp
```

Paper interpretation:

> **PAD-pretrained CSMR produces a strong LCC external-development improvement in both AUC and frozen-threshold HTER, while showing a distribution-dependent trade-off: source Val changes little, Test ACER improves slightly, but Test AUC decreases substantially.**

Do not hide the Test AUC result.

This pattern is useful for Discussion:

- ranking robustness and operating-point behavior are not interchangeable;
- CSMR can change the score distribution differently across domains;
- robustness should not be summarized using a single metric.

---

# 9. Main result III — frozen CASIA cross-dataset confirmation

The four final 100K seed-100 models were frozen before the CASIA run.

Each model retained its own CelebA-Val threshold.

No CASIA threshold calibration or target adaptation was performed.

## 9.1 Primary reconstructed video-level results

| Initialization | Method | AUC ↑ | APCER ↓ | BPCER ↓ | HTER ↓ |
|---|---|---:|---:|---:|---:|
| Scratch | Vanilla | 0.645059 | 16.2879% | 56.8182% | 36.5530% |
| Scratch | CSMR | **0.701963** | **13.2576%** | **54.5455%** | **33.9015%** |
| PAD-pretrained | Vanilla | 0.687758 | **23.8636%** | 46.5909% | 35.2273% |
| PAD-pretrained | CSMR | **0.710744** | 34.0909% | **27.2727%** | **30.6818%** |

Paired effects:

```text
Scratch:
ΔAUC   = +5.6904 pp
ΔHTER  = −2.6515 pp
ΔAPCER = −3.0303 pp
ΔBPCER = −2.2727 pp

PAD-pretrained:
ΔAUC   = +2.2986 pp
ΔHTER  = −4.5455 pp
ΔAPCER = +10.2273 pp
ΔBPCER = −19.3182 pp
```

## 9.2 What the CASIA result supports

The important positive pattern is:

```text
Scratch:      AUC ↑ and HTER ↓
PAD-pretrained: AUC ↑ and HTER ↓
```

This makes a purely LCC-specific explanation less plausible.

For Scratch, both APCER and BPCER improve.

For PAD-pretrained, HTER improves because BPCER falls strongly, while APCER worsens substantially.

Therefore:

> **The pretrained CASIA result is a real overall AUC/HTER gain but includes a security-sensitive operating-point trade-off that must be disclosed.**

## 9.3 CASIA protocol limitation

The Kaggle source is not the original CASIA-FASD video benchmark.

The evaluation discovered:

```text
test_img/test_img/color
30 subjects numbered 1–30
360 reconstructed video IDs
JPEG extracted frames
train/test copy overlap in subjects/video/frame identifiers
source videos unavailable
```

Evaluation coverage:

```text
352 / 360 reconstructed video IDs scorable
2,405 / 2,405 planned frames localized
```

Consequences:

- call this **CASIA-FASD Kaggle image-copy cross-dataset evaluation**;
- it is untouched with respect to project tuning;
- it is not the official subject-disjoint CASIA benchmark;
- do not compare these values directly with papers using the standard CASIA protocol;
- a standard external benchmark remains desirable before a strong literature-comparison claim.

---

# 10. What the final evidence currently says

The current evidence is stronger than the earlier pilot stage because:

```text
1. Full-scale 100K Scratch:
   paired CSMR gains are positive across Val, Test, and LCC.

2. Full-scale 100K PAD-pretrained:
   strong paired LCC AUC + HTER improvement.

3. New external CASIA image-copy domain:
   AUC + HTER improve in BOTH initialization regimes.

4. Inference:
   identical lightweight spatial graph; no FFT deployment cost.
```

A defensible current statement is:

> **The results provide consistent one-seed evidence that CSMR can improve cross-domain robustness across two initialization regimes, while the magnitude and error trade-offs remain domain dependent.**

Do not write:

> “CSMR universally improves robustness.”

Do not write:

> “CSMR is statistically proven.”

---

# 11. Research questions — final version

## RQ1 — Reliability of explicit frequency representations

> **Are explicit frequency representations reliably complementary under domain shift in lightweight Face PAD?**

Evidence:

- E2/E3;
- M0–M4;
- LCC development behavior.

Expected conclusion wording:

> Explicit spectral representations are discriminative but did not provide stable cross-domain complementarity in our development study.

## RQ2 — Training-time spectral robustness

> **Can frequency be used more effectively as a sample-adaptive training-time robustness intervention while retaining spatial-only inference?**

Evidence:

- 100K Scratch Vanilla vs CSMR;
- 100K PAD-pretrained Vanilla vs CSMR;
- LCC paired effects;
- CASIA-copy paired effects.

## RQ3 — Mechanism

> **Which component drives the CSMR effect: spectral augmentation, worst-band selection, generic consistency, or PAD-margin-aware regularization?**

Evidence still needed:

- ablation A–E.

## RQ4 — Initialization transfer

> **Does CSMR remain useful both from random initialization and from a PAD-pretrained representation?**

Evidence:

- final 100K Scratch;
- final 100K PAD-pretrained.

Do not call this architecture-independence.

## RQ5 — External generalization

> **Does the frozen method improve performance on an unseen target domain without target calibration?**

Current evidence:

- CASIA-FASD Kaggle image-copy evaluation.

Stronger future evidence:

- official CASIA protocol or another standard benchmark.

## RQ6 — Efficiency

> **Can cross-domain robustness improve without adding inference-time spectral computation?**

Evidence:

- identical MiniFASNetV2 inference graph;
- 434,560 parameters;
- ~1.74 MB ONNX;
- parity-validated ONNX;
- CPU / edge measurements to complete.

---

# 12. Required ablation

Main ablation matrix:

| ID | Training method | Worst-band selection | PAD-margin harmful-drop regularization | Purpose |
|---|---|---:|---:|---|
| A | Vanilla | No | No | spatial control |
| B | Random spectral augmentation | No | No | augmentation-only |
| C | Worst-band CE (`lambda=0`) | Yes | No | adaptive worst-view value |
| D | Generic KL/logit consistency | controlled spectral view | No | generic consistency control |
| E | Full CSMR | Yes | Yes | proposed method |

This section is essential because current final results establish an effect but do not yet isolate **why** the effect occurs.

Do not add many optional variants unless A–E leave a specific mechanism ambiguity.

---

# 13. Multi-seed plan

Current final experiments use one training seed.

Recommended:

```text
seed 100 — completed
seed 132 — pending
seed 150 — pending
```

Priority:

```text
1. Scratch 100K paired Vanilla/CSMR
2. PAD-pretrained 100K paired Vanilla/CSMR if compute allows
```

Report paired deltas as:

```text
mean ± std
```

Do not mix:

- subject-level bootstrap uncertainty;
- training-seed uncertainty.

They answer different questions.

---

# 14. Efficiency story

Deployment model:

```text
MiniFASNetV2
434,560 parameters
80×80
~1.74 MB ONNX
```

CSMR does not alter the inference model.

The paper should report:

```text
Params
MACs/FLOPs
ONNX size
CPU latency
FPS
peak RAM
Raspberry Pi latency/FPS
```

Central wording:

> **CSMR increases training-time computation but introduces no inference-time FFT, frequency branch, or parameter overhead.**

Avoid using “zero overhead” without qualifying it as:

> **zero additional model / spectral computation at inference relative to the same Vanilla backbone.**

Preprocessing and face detection still incur deployment cost.

---

# 15. Main paper tables

## Table 1 — Motivation: explicit frequency is not automatically robust

Keep compact:

```text
Spatial baseline
representative explicit-frequency variants
source metric
LCC AUC / HTER
```

Purpose:

> motivate the shift from frequency-as-feature to frequency-as-stressor.

Do not overload this table with every experimental branch.

## Table 2 — Final 100K Scratch result

Include:

```text
Val15K
Test5K
LCC
Vanilla vs CSMR
APCER / BPCER where useful
```

This is the **primary method table**.

## Table 3 — Final 100K PAD-pretrained result

Same structure.

Explicitly show:

```text
LCC gain
CelebA Test AUC regression
```

Do not selectively omit the latter.

## Table 4 — Frozen CASIA image-copy external confirmation

Show both initialization regimes and:

```text
AUC
HTER
APCER
BPCER
```

Caption must state:

> Kaggle extracted-frame copy; project-specific cross-dataset protocol; not directly comparable with official CASIA benchmark results.

## Table 5 — Mechanism ablation

```text
Vanilla
Random spectral augmentation
Worst-band CE
Generic consistency
Full CSMR
```

## Table 6 — Efficiency

```text
Params
MACs/FLOPs
ONNX size
CPU latency
edge latency
RAM
```

## Optional supplementary table — additional seeds

Report per-seed and mean ± std.

---

# 16. Main paper figures

## Figure 1 — Research motivation

```text
spatial PAD
→ domain shift
→ explicit frequency exploration
→ unstable cross-domain complementarity
→ spectral-shortcut hypothesis
→ CSMR
```

## Figure 2 — CSMR method

```text
clean sample
├─ LOW attenuation
├─ MID attenuation
└─ HIGH attenuation
        ↓
shared model
        ↓
label-aligned PAD margin
        ↓
worst counterfactual
        ↓
harmful-drop regularization
```

## Figure 3 — Training vs inference

Training:

```text
RGB + spectral counterfactual generator + CSMR objective
```

Inference:

```text
RGB → MiniFASNetV2 → PAD
```

Highlight:

> no inference-time FFT/frequency branch.

## Figure 4 — Cross-domain paired deltas

Recommended visualization:

```text
Scratch:
LCC ΔAUC / ΔHTER
CASIA-copy ΔAUC / ΔHTER

PAD-pretrained:
LCC ΔAUC / ΔHTER
CASIA-copy ΔAUC / ΔHTER
```

This directly visualizes the method effect rather than absolute backbone differences.

## Figure 5 — ROC curves

At minimum:

- LCC Vanilla vs CSMR;
- CASIA-copy Vanilla vs CSMR.

Use separate Scratch / PAD-pretrained panels if space permits.

---

# 17. Proposed paper structure

## 1. Introduction

- Face PAD domain shift;
- lightweight edge constraint;
- attraction and risk of frequency cues;
- frequency-as-feature vs frequency-as-stressor;
- CSMR contributions.

## 2. Related Work

### 2.1 Lightweight Face PAD
### 2.2 Frequency-aware Face PAD
### 2.3 Domain-generalizable / cross-dataset Face PAD
### 2.4 Spectral augmentation, shortcut learning, and robustness regularization

## 3. Method

### 3.1 MiniFASNetV2 spatial PAD backbone
### 3.2 Spectral counterfactual generator
### 3.3 Label-aligned Real-vs-Attack margin
### 3.4 Sample-adaptive worst-band selection
### 3.5 Harmful-drop margin regularization
### 3.6 Training and inference complexity

## 4. Experimental Protocol

- dataset roles;
- 5K/2K hyperparameter development;
- regime-specific freeze;
- final 100K/15K protocol;
- CelebA Test5K role;
- LCC external-development role;
- CASIA image-copy external protocol and limitation;
- thresholds;
- metrics;
- efficiency.

Do **not** include superseded intermediate scale results in the main protocol narrative unless needed purely as an implementation provenance note.

## 5. Results

### 5.1 Explicit-frequency diagnostic
### 5.2 Final 100K Scratch
### 5.3 Final 100K PAD-pretrained
### 5.4 External CASIA image-copy confirmation
### 5.5 Mechanism ablation
### 5.6 Multi-seed robustness
### 5.7 Efficiency

## 6. Discussion

Must discuss:

- why frequency can become domain-specific;
- why worst-band intervention may reduce shortcut dependence;
- Scratch vs PAD-pretrained effect-size difference;
- CelebA Test ranking trade-off in pretrained CSMR;
- CASIA pretrained APCER/BPCER trade-off;
- LCC being external-development;
- CASIA-copy not being the official benchmark;
- one-seed limitation until replication completes;
- external checkpoint provenance.

## 7. Conclusion

Keep the conclusion bounded to the evidence.

---

# 18. Contribution statements — recommended wording

## C1 — Empirical diagnosis

> **We show that explicit frequency representations can carry useful PAD information while failing to provide stable cross-domain complementarity in a lightweight setting.**

Avoid claiming causal proof of shortcut learning.

## C2 — Method

> **We introduce CSMR, which selects a sample-specific worst spectral counterfactual according to a label-aligned Real-vs-Attack margin and regularizes only harmful margin degradation.**

## C3 — Cross-domain evidence

Current one-seed wording:

> **At 100K scale, CSMR improves paired cross-domain metrics on LCC and produces consistent AUC/HTER gains on a separate CASIA-FASD image-copy evaluation across random and PAD-pretrained initialization regimes.**

Add the CASIA protocol qualifier in the experiment section/caption.

After successful multi-seed replication, this can be strengthened.

## C4 — Deployment

> **CSMR is training-time only and preserves the original lightweight MiniFASNetV2 inference graph without an FFT or frequency branch.**

---

# 19. What NOT to claim

Do not claim:

> CSMR is the first frequency perturbation method for Face PAD.

Do not claim:

> Frequency features are inherently non-robust.

Do not claim:

> CSMR removes overfitting.

Prefer:

> CSMR is consistent with reduced dependence on source-specific spectral cues.

Do not claim:

> CSMR improves every source and target metric.

Do not claim:

> One seed establishes statistical significance.

Do not claim:

> The current CASIA image-copy results are official CASIA benchmark results.

Do not claim:

> CSMR is architecture-independent.

Do not claim:

> Current evidence establishes SOTA.

---

# 20. Current limitations

1. **Single training seed** for the final 100K experiments.
2. **LCC used in CSMR development**, so it is not untouched final evidence.
3. **CASIA Kaggle copy differs from the official video protocol** and has packaging overlap between train/test directories.
4. **PAD-pretrained checkpoint provenance is not fully controlled**.
5. **Pretrained CSMR shows domain-dependent ranking/calibration trade-offs**, including:
   - CelebA Test AUC decrease;
   - CASIA-copy APCER increase despite HTER/AUC improvement.
6. **Mechanism ablation is still pending**.
7. **Full edge efficiency characterization** still needs MACs/FLOPs, RAM and target-device timing.

These limitations should be stated proactively rather than left for reviewers to discover.

---

# 21. Experiment-complete checklist

Completed:

- [x] explicit-frequency diagnostic / M0–M4;
- [x] CSMR mechanism definition;
- [x] regime-specific 5K/2K hyperparameter freeze;
- [x] MiniFASNetV2 final model contract;
- [x] final 100K Scratch Vanilla;
- [x] final 100K Scratch CSMR;
- [x] final 100K PAD-pretrained Vanilla;
- [x] final 100K PAD-pretrained CSMR;
- [x] fixed CelebA Test5K descriptive evaluation;
- [x] LCC external-development evaluation;
- [x] parity-checked ONNX export;
- [x] first untouched CASIA image-copy cross-dataset evaluation.

Still needed for the strongest paper:

- [ ] mechanism ablation A–E;
- [ ] +2 paired training seeds, at least for Scratch;
- [ ] standard external benchmark protocol if obtainable;
- [ ] MACs/FLOPs + peak RAM + Raspberry Pi timing;
- [ ] literature comparison audit;
- [ ] final statistical/reporting pass.

---

# 22. Final experiment priority

```text
NOW
↓
mechanism ablation A–E
↓
additional paired 100K seeds
↓
official/standard external protocol if obtainable
↓
complete edge measurements
↓
paper tables + figures
↓
writing
```

No new architecture or hyperparameter search should be inserted into this path without a concrete failure that requires it.

---

# 23. Core narrative to preserve

```text
Spatial PAD is strong on source data
        ↓
domain shift exposes fragility
        ↓
explicit frequency features contain useful signal
but do not yield stable cross-domain complementarity
        ↓
frequency may encode source-specific spectral shortcuts
        ↓
CSMR changes the role of frequency:
from inference feature
to training-time counterfactual stressor
        ↓
sample-adaptive worst-band selection
+ harmful PAD-margin degradation regularization
        ↓
same lightweight spatial inference model
        ↓
100K Scratch: coherent paired gains
        ↓
100K PAD-pretrained: strong LCC gain with explicit trade-offs
        ↓
CASIA image-copy: AUC + HTER gain in both regimes
        ↓
remaining work:
ablation + multi-seed + standard external benchmark + efficiency
```

This is the current paper story.

Do not reintroduce superseded intermediate-scale results into the main narrative when the final 100K evidence is already available.
