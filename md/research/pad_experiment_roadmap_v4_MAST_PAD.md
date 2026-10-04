# MAST-PAD Experiment Roadmap
## Final course-paper configuration, evidence hierarchy, and experiment provenance

_Last updated: 2026-10-05_

> **Official paper-facing method name:** **MAST-PAD — Margin-Aware Spectral Training for Face Presentation Attack Detection**
>
> **Primary implementation / provenance ID:** `R7_SC_100K_crop15`
>
> `R7`, `R7-SC`, `P3`, `P3-SF`, and `C` are development names only. In the paper, use **MAST-PAD** for the final proposed method, **Worst-Band Spectral Training (WBST)** for the P3-style intermediate ablation, and **Clean baseline** for C.
>
> The course paper intentionally uses the matched **Train100K / Val15K** experiment as the **primary controlled protocol**. The completed full-scale experiment is retained as a secondary scale-up study, not as the headline configuration.

---

# 1. Final research question

The project is no longer searching for a backbone, crop factor, frequency branch, or new loss family.

The active paper question is:

> **Can a lightweight Face PAD model improve cross-domain robustness by using frequency only as a margin-aware training-time stressor, while preserving the same spatial inference graph?**

The final answer is instantiated by **MAST-PAD**, built on MiniFASNetV2.

The final deployment path is:

```text
face image
→ frozen SCRFD localization
→ crop factor 1.5
→ 80×80 BGR float image
→ MiniFASNetV2
→ binary PAD score: Real vs Attack
```

No FFT, spectral mask, band selector, or additional spectral branch is used at inference.

---

# 2. Final paper story

The main narrative is intentionally compact:

```text
1. A clean lightweight PAD model is strong on source validation
   but loses performance under domain shift.

2. Earlier explicit-frequency and spectral-training experiments show
   that frequency contains useful PAD information, but naive use is unstable.

3. P3/WBST improves robustness by training against the currently worst
   LOW/MID/HIGH spectral view for every sample.

4. R7 introduces the key final idea:
   spectral supervision should be applied only when the selected spectral
   intervention actually reduces the sample's label-aligned PAD margin.

5. This harmful-only design becomes MAST-PAD.

6. In the matched 100K experiment, MAST-PAD produces the best pooled
   LCC-FASD result among C / P3-SF / R7-SC while keeping the exact same
   inference architecture.

7. Full-scale training is reported separately as a scale-up check.
```

The method claim is therefore not:

> “frequency augmentation improves Face PAD.”

It is:

> **MAST-PAD uses the current PAD decision margin to select a damaging spectral counterfactual and gates spectral supervision to samples for which that counterfactual is genuinely harmful.**

---

# 3. Official naming map

| Development name | Paper-facing name | Role |
|---|---|---|
| `C` | **Clean baseline** | matched lightweight control |
| `P3_SF` | **WBST — Worst-Band Spectral Training** | intermediate always-on worst-view training |
| `R7_SC` | **MAST-PAD** | final proposed method |
| `R7_SC_100K_crop15` | **MAST-PAD (primary)** | main course-paper checkpoint |
| `R7_SC_FULL_crop15` | **MAST-PAD (full-scale)** | secondary scale-up evidence |

Do not call the final method CSMR. Historical CSMR pilots used a different regularization formulation and are not identical to the final R7 loss.

---

# 4. Final primary protocol — matched 100K

## 4.1 Data and selection

```text
Source training: CelebA-Spoof Train100K
Source validation: CelebA-Spoof Val15K
Held-out source evaluation: full usable CelebA-Spoof Test
Target-domain evaluation: full LCC-FASD train/dev/eval + pooled Combined
DATA_SEED = 43
TRAIN_SEED = 100
```

All C / WBST / MAST-PAD runs use identical Train100K and Val15K manifests and the same initial state.

Selection protocol:

```text
train
→ evaluate Val15K
→ select best checkpoint
→ calibrate source min-ACER threshold on Val15K
→ freeze checkpoint + threshold
→ evaluate full CelebA Test
→ evaluate LCC-FASD with no target threshold tuning
```

This is the primary protocol to describe in the course paper.

## 4.2 Shared optimization

```text
optimizer       = SGD
initial LR      = 0.005
momentum        = 0.9
weight decay    = 5e-4
batch size      = 256
AMP             = enabled
max epochs      = 20
LR milestones   = [6, 14]
gamma           = 0.2
patience start  = epoch 15
patience        = 3
```

C is checkpoint-eligible from epoch 1.

WBST and MAST-PAD use epoch 1 as clean-only warmup; epoch 1 is excluded from their best-checkpoint selection.

---

# 5. Final MAST-PAD configuration

## 5.1 Spectral geometry

Final MAST-PAD uses the R7-SC geometry found during development:

```text
crop factor = 1.5

LOW  center = 0.0833333333
MID  center = 0.25
HIGH center = 0.4166666667

sigma = 0.10
gain ~ Uniform(0.65, 0.90)
```

The centers are the original `[0.15, 0.45, 0.75]` scaled by `1.5 / 2.7`, while `sigma` remains `0.10`.

For a normalized radial frequency coordinate `r`:

```text
M_k(r) = exp(-(r-c_k)^2 / (2 sigma^2))
G_k(r) = 1 - (1-g) M_k(r)
```

Constraints:

- attenuation only, no amplification;
- phase unchanged;
- DC preserved;
- the same sampled gain is shared by LOW/MID/HIGH for a sample;
- the same radial field is applied across channels;
- spectral intervention is training-only.

## 5.2 Margin and harmful gate

For binary PAD logits:

```text
z_real, z_attack
d = z_real - z_attack
```

Label-aligned margin:

```text
Real   → m = +d
Attack → m = -d
```

Generate LOW/MID/HIGH views and choose:

```text
k* = argmin_k m_k
```

Let `m_clean` and `m_worst` be measured in the BN-safe selection pass.

The harmful mask is:

```text
harmful = (m_worst < m_clean)
```

A sample can be harmful even if it remains correctly classified. The gate measures degradation of decision safety, not only label flips.

## 5.3 Final loss

After the clean-only warmup:

```text
L_PAD =
    0.75 * CE(clean)
  + 0.25 * CE_harmful(selected_worst)
```

where `CE_harmful` is averaged over harmful samples only.

If no sample in a batch is harmful, the harmful term is a differentiable zero; the clean coefficient remains `0.75`.

Auxiliary supervision is clean-view only:

```text
L_aux =
    0.1 * CE(spoof_type_11)
  + 0.1 * CE(lighting_5)
  + 1.0 * BCE(attributes_40, real_samples_only)
```

Final training objective:

```text
L_total = L_PAD + L_aux
```

The core difference from WBST is simple:

```text
WBST:
  every sample receives selected-worst spectral CE

MAST-PAD:
  selected-worst spectral CE is applied only to samples whose
  spectral intervention reduces the clean aligned PAD margin
```

---

# 6. Primary 100K results

## 6.1 Matched C / WBST / MAST-PAD

| Method | Val15K AUC | Val15K ACER | CelebA Test AUC | LCC Eval AUC | LCC Combined AUC | Combined EER | Combined HTER | TPR@FPR1% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Clean baseline | 0.999523 | 0.8271% | 0.982413 | 0.798517 | 0.848273 | 23.5328% | 24.0514% | 0.199776 |
| WBST | 0.999587 | 0.8463% | 0.982573 | 0.812841 | 0.851978 | 23.8822% | 23.8322% | 0.214838 |
| **MAST-PAD** | **0.999606** | **0.8266%** | **0.984804** | **0.825330** | **0.867844** | **21.5867%** | **21.5738%** | **0.270479** |

Primary paper-facing LCC number:

```text
MAST-PAD LCC Combined AUC = 0.867844 → 0.868
Combined EER             = 21.5867%  → 21.59%
Combined HTER            = 21.5738%  → 21.57%
Deployment parameters    = 0.434M
```

## 6.2 Paired gains over the clean baseline

Using unrounded metrics:

```text
Δ Combined AUC  = +0.019571  (+1.9571 pp)
Δ Combined EER  = -1.9461 pp
Δ Combined HTER = -2.4776 pp
Δ TPR@FPR1%     = +0.070703
```

The strongest current claim is descriptive, not statistical:

> **Under the matched 100K protocol, MAST-PAD yields the strongest cross-domain LCC result of the three frozen strategies while preserving the same lightweight inference graph.**

One training seed does not establish statistical superiority.

---

# 7. Training diagnostics for the primary MAST-PAD run

The selected MAST-PAD checkpoint is epoch 11; training stops at epoch 17 under the phase-aware patience rule.

At the last executed epoch:

```text
clean CE            = 0.014845
selected worst CE   = 0.024699
harmful-only CE     = 0.022382
harmful samples     = 84,779
harmful fraction    = 0.847790
clean margin        = 7.614003
worst margin        = 7.196639
mean margin drop    = 0.417364
spectral flip rate  = 0.004370
sampled gain mean   = 0.774548

selected LOW  = 0.427060
selected MID  = 0.254670
selected HIGH = 0.318270
```

Interpret carefully:

- the high harmful fraction shows that margin degradation is common under the selected intervention;
- the very low label-flip rate shows that the gate is more sensitive than a simple correctness-flip criterion;
- band fractions show that no single band dominates all samples;
- these diagnostics describe optimization behavior and do not by themselves prove the causal mechanism.

---

# 8. Full-scale experiment — secondary scale-up evidence

The full-scale experiment uses approximately 484K usable CelebA-Spoof training samples and a different training/selection contract:

```text
max epochs      = 12
milestones      = [4, 8]
patience start  = 9
patience        = 3
selection set   = full CelebA Test-as-Val
```

Therefore it is not directly interchangeable with the 100K protocol.

Completed full-scale results:

| Method | Test-as-Val AUC | Test ACER | LCC Eval AUC | LCC Combined AUC | Combined EER | Combined HTER |
|---|---:|---:|---:|---:|---:|---:|
| Clean baseline | 0.990736 | 4.6248% | 0.811622 | 0.846011 | 23.5793% | 28.0493% |
| WBST | 0.992242 | 4.3548% | 0.819687 | **0.860831** | **21.7906%** | **26.7891%** |
| MAST-PAD | **0.992280** | **4.3459%** | **0.840060** | 0.860272 | 22.0559% | 27.1228% |

Interpretation:

- MAST-PAD remains around `0.860` pooled AUC at full scale;
- MAST-PAD has the best physical LCC Evaluation AUC (`0.840060`);
- WBST is marginally higher on pooled Combined AUC (`+0.000559`);
- full-scale is supporting evidence, not the primary course-paper configuration;
- the full-scale Test ACER is calibrated on the same Test-as-Val population and must not be presented as untouched-test evidence.

---

# 9. Literature comparison selected for the course paper

The main contextual comparison set is intentionally limited to:

1. Graph for Transformer Feature;
2. AENet benchmark result;
3. ViT-B/16;
4. ViT-B/16 + LoRA;
5. LBP-GBM from the original LCC-FASD paper.

These protocols are heterogeneous. The comparison is contextual, not a claim of identical evaluation.

| Method | LCC setting | AUC | EER | HTER / ACER | Approx. params | Source role |
|---|---|---:|---:|---:|---:|---|
| Graph for Transformer Feature | LCC intra-dataset | 0.833 | 12.23% | — | ~10.84M | peer-reviewed comparator |
| AENet | LCC result reported by lightweight-FAS benchmark | 0.868 | 20.91% | ACER 22.61% | 11.22M | strong reference |
| ViT-B/16 | SiW-M → LCC | — | 30.98% | HTER 34.62% | ~86.6M | cross-domain reference |
| ViT-B/16 + LoRA | SiW-M → LCC | — | 28.95% | HTER 29.33% | ~86.6M backbone | cross-domain reference |
| LBP-GBM | official LCC protocol | — | 14.60% | — | classical | historical dataset baseline |
| **MAST-PAD (ours)** | **CelebA 100K → LCC pooled** | **0.868** | **21.59%** | **HTER 21.57%** | **0.434M** | proposed method |

Paper-facing interpretation:

- after rounding to the same AUC precision, MAST-PAD reports `0.868`;
- this is numerically above the Graph Transformer result `0.833`;
- it is numerically equal to the reported AENet AUC `0.868`;
- it uses about `0.434M` deployment parameters, far below the parameter counts reported for Graph Transformer, AENet, and ViT-B/16;
- protocol differences must be stated explicitly.

Safe wording:

> **MAST-PAD reaches a pooled LCC AUC of 0.868 with a 0.434M-parameter inference model. This is numerically higher than the 0.833 AUC reported by Graph for Transformer Feature and matches the 0.868 AUC reported for AENet, although the compared studies use different LCC training/evaluation protocols.**

Do not call this SOTA.

---

# 10. Experiment history and what each stage contributed

The experiment archive should remain available for provenance, but the paper should not narrate all runs equally.

## Stage A — early binary / explicit spectral exploration

`01` mini ablation A–F and historical CSMR pilots established that simply adding a spectral mechanism was not reliably beneficial.

Role now:

> development history only.

## Stage B — worst-view training

P1–P6 and later P3-style experiments showed that selected-worst spectral CE was more promising than random/generic consistency alternatives.

Role now:

> motivation for WBST.

## Stage C — R-family discovery

R0–R4 varied worst-view CE strength.

R6–R9 compared:

```text
all-band CE
harmful-gated worst CE
worst/random mixing
worst CE + KL
```

R7 (`harmful_gated_worst`) emerged as the most promising selective strategy.

## Stage D — crop and geometry

Crop experiments showed that the final cross-domain behavior depended strongly on face context.

The final choice became:

```text
crop = 1.5
```

Geometry experiments then selected:

```text
R7_SC:
scaled band centers
sigma retained at 0.10
```

This is the final MAST-PAD geometry.

## Stage E — matched 100K confirmation

The decisive course-paper experiment is:

```text
C_100K_crop15
P3_SF_100K_crop15
R7_SC_100K_crop15
```

Result:

```text
MAST-PAD > WBST > Clean
```

by LCC Combined AUC.

## Stage F — full-scale support

The completed full-scale experiment tests whether the same frozen design remains viable when source training is expanded.

It is useful as supporting evidence but does not replace the primary 100K controlled protocol.

---

# 11. Efficiency and deployment evidence

Current measured model complexity:

```text
deployment parameters = 434,434
training model params  = 441,658
MACs                    = 40,810,892
reported GFLOPs         = 0.081621784
input                   = [1, 3, 80, 80]
```

The GFLOP profiler covers supported Conv/Linear operations and is not guaranteed to use the same counting convention as external papers.

Central deployment claim:

> **MAST-PAD adds training-time spectral computation but no additional spectral path, FFT, or model parameters to the deployed PAD graph relative to the matched clean baseline.**

Do not say “zero total deployment overhead” because face detection, cropping, preprocessing, and runtime framework overhead still exist.

---

# 12. Remaining work

For a course paper, the core experiment is already complete.

Highest-value remaining items:

1. **paper writing and figure/table preparation**;
2. optional multi-seed confirmation if compute/time permits;
3. optional Raspberry Pi / CPU latency measurement;
4. optional independent standard cross-domain dataset for a future publication-grade extension.

Do not reopen architecture or hyperparameter search unless a reproducibility failure is found.

---

# 13. Claim discipline

Supported:

> MAST-PAD is a margin-aware, harmful-selective spectral training strategy for lightweight Face PAD.

Supported:

> MAST-PAD uses frequency only during training and preserves the clean MiniFASNetV2 inference graph.

Supported:

> Under the matched 100K protocol, MAST-PAD improves LCC Combined AUC from 0.848 to 0.868 over the clean baseline.

Supported:

> The 0.434M-parameter MAST-PAD model is numerically competitive with several substantially larger literature references on LCC-FASD.

Use with protocol qualifier:

> MAST-PAD numerically exceeds the Graph Transformer AUC and matches the reported AENet AUC on LCC-FASD.

Not supported:

> MAST-PAD is SOTA.

Not supported:

> MAST-PAD is statistically superior.

Not supported:

> MAST-PAD universally improves every dataset or every metric.

Not supported:

> All literature comparison rows use an identical protocol.

---

# 14. Stop rule

The final course-paper method is frozen:

```text
MAST-PAD
= R7_SC_100K_crop15
= MiniFASNetV2
+ crop1.5
+ scaled spectral centers
+ harmful-only selected-worst spectral CE
+ clean-only auxiliary heads
+ no spectral inference branch
```

The remaining task is to **write and present the method clearly**, not to search for another R8/R9-style variant.

---

# 15. Key source links for the comparison table

- Graph for Transformer Feature — ESANN 2023:
  - DOI: https://doi.org/10.14428/esann/2023.ES2023-14
  - PDF: https://www.esann.org/sites/default/files/proceedings/2023/ES2023-14.pdf
- Lightweight FAS benchmark containing the AENet LCC result:
  - https://github.com/kprokofi/light-weight-face-anti-spoofing
- ViT / ViT+LoRA cross-domain reference:
  - https://github.com/ozogxyz/spoof
- Original LCC-FASD paper:
  - DOI: https://doi.org/10.1109/CSITechnol.2019.8895208
  - PDF: https://pureportal.spbu.ru/files/51584200/CSIT2019_Grishkin_Timoshenko.pdf
