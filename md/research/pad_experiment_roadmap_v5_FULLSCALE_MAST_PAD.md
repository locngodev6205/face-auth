# MAST-PAD Experiment Roadmap
## Full-scale course-paper configuration, two-target cross-domain evidence, and experiment provenance

_Last updated: 2026-10-07_

> **Official paper-facing method:** **MAST-PAD — Margin-Aware Spectral Training for Face Presentation Attack Detection**
>
> **Primary implementation / provenance ID:** `R7_SC_FULL_crop15`
>
> Development names `R7`, `R7-SC`, `P3`, `P3-SF`, and `C` remain provenance only. In the paper use **MAST-PAD**, **WBST — Worst-Band Spectral Training**, and **Clean baseline**.
>
> **Primary paper evidence is now the full-scale experiment.** The earlier Train100K/Val15K study is retained as supporting development evidence, not as the headline result.

---

# 1. Final research question

> **Can a very small Face PAD model improve cross-domain robustness by using frequency only as a margin-aware training-time stressor, while preserving the same spatial inference graph?**

Final deployment path:

```text
face image
→ frozen SCRFD localization
→ crop factor 1.5
→ 80×80 BGR float image
→ MiniFASNetV2
→ binary PAD score: Real / Attack
```

No FFT, spectral mask, candidate selector, or extra spectral branch is used at inference.

---

# 2. Final evidence hierarchy

```text
PRIMARY
Full eligible CelebA-Spoof Train
→ full usable CelebA Test-as-Val checkpoint selection / source calibration
→ frozen evaluation on LCC-FASD
→ frozen evaluation on CASIA-FASD extracted-image copy

SUPPORTING
Matched Train100K / Val15K study
→ development evidence that motivated the frozen MAST-PAD design
```

The main reported cross-domain metric is **AUC only** to avoid mixing threshold policies across datasets and literature.

Paper-facing target values are always reported on the project-defined **Combined** populations:

```text
LCC Combined AUC
CASIA reconstructed Combined AUC
```

Do not mix LCC Evaluation AUC into the headline table once the paper uses Combined as the reporting convention.

---

# 3. Official naming map

| Development name | Paper-facing name | Role |
|---|---|---|
| `C_FULL_crop15` | **Clean baseline** | full-scale lightweight control |
| `P3_SF_FULL_crop15` | **WBST** | full-scale always-on worst-view training |
| `R7_SC_FULL_crop15` | **MAST-PAD** | **primary proposed checkpoint** |
| `R7_SC_100K_crop15` | MAST-PAD-100K | supporting development checkpoint |

Do not call the final method CSMR.

---

# 4. Primary full-scale protocol

## 4.1 Source data

```text
CelebA-Spoof official Train candidates ≈ 494K
usable source training samples          = 484,561
full usable source Test candidates      = 66,379 scored / 67,170 candidates
```

The full Test population is used as **Test-as-Val** for checkpoint selection and source-threshold calibration. It is therefore not an untouched final source test.

## 4.2 Cross-domain targets

### LCC-FASD

```text
Training split
Development split
Evaluation split
→ pooled LCC Combined
```

No LCC threshold fitting or model adaptation.

### CASIA-FASD extracted-image copy

```text
Kaggle train/test JPEG extraction
→ reconstructed video IDs
→ mean valid-frame model score per reconstructed video
→ unique Combined population
```

No CASIA training, adaptation, threshold fitting, or checkpoint selection.

Important limitation: the available CASIA resource is an **extracted-image copy**, not a verified official subject-disjoint full-video benchmark. Use it as an independent cross-domain validation set, not as a claim of reproducing the official CASIA protocol.

## 4.3 Shared optimization

```text
optimizer        = SGD
initial LR       = 0.005
momentum         = 0.9
weight decay     = 5e-4
batch size       = 256
AMP              = enabled
max epochs       = 12
LR milestones    = [4, 8]
gamma            = 0.2
patience start   = epoch 9
patience         = 3
DATA_SEED        = 43
TRAIN_SEED       = 100
```

All full-scale C / WBST / MAST-PAD runs share the same initial state, source manifests, crop, base augmentation, optimizer family, and deployed architecture.

---

# 5. Final MAST-PAD configuration

## 5.1 Spectral geometry

```text
crop factor = 1.5
LOW  center = 0.0833333333
MID  center = 0.25
HIGH center = 0.4166666667
sigma       = 0.10
gain        ~ Uniform(0.65, 0.90)
```

For normalized radial frequency coordinate `r`:

```text
M_k(r) = exp(-(r-c_k)^2 / (2 sigma^2))
G_k(r) = 1 - (1-g) M_k(r)
```

Constraints: attenuation only; phase unchanged; DC preserved; same sampled gain shared across LOW/MID/HIGH for one sample; same radial field across channels.

## 5.2 Margin and harmful gate

```text
d = z_real - z_attack
Real   → m = +d
Attack → m = -d
k* = argmin_k m_k
harmful = (m_worst < m_clean)
```

A candidate can be harmful before it flips the predicted class.

## 5.3 Final objective

After epoch-1 clean warmup:

```text
L_PAD = 0.75 CE(clean) + 0.25 CE_harmful(selected_worst)
L_aux = 0.1 CE(spoof_type) + 0.1 CE(lighting) + BCE(attributes, real only)
L_total = L_PAD + L_aux
```

WBST applies selected-worst CE to every sample; MAST-PAD applies it only when the selected view reduces the clean aligned margin.

---

# 6. Primary full-scale results

## 6.1 One-metric paper-facing table

Use **AUC only** in the main paper/slide result table.

| Method | CelebA Test-as-Val AUC | LCC Combined AUC | CASIA Combined AUC | Deployment params |
|---|---:|---:|---:|---:|
| Clean baseline | 0.990736 | 0.846011 | 0.920398 | 0.434M |
| WBST | 0.992242 | **0.860831** | 0.922264 | 0.434M |
| **MAST-PAD** | **0.992280** | 0.860272 | **0.935199** | **0.434M** |

Paper-facing rounding:

```text
Clean      LCC 0.846 | CASIA 0.920
WBST       LCC 0.861 | CASIA 0.922
MAST-PAD   LCC 0.860 | CASIA 0.935
```

## 6.2 Main descriptive findings

```text
LCC Combined:
WBST - Clean      = +0.014820  (+1.48 pp)
MAST-PAD - Clean  = +0.014261  (+1.43 pp)
MAST-PAD - WBST   = -0.000559  (-0.056 pp; effectively near-tied descriptively)

CASIA Combined:
WBST - Clean      = +0.001866  (+0.19 pp)
MAST-PAD - Clean  = +0.014801  (+1.48 pp)
MAST-PAD - WBST   = +0.012935  (+1.29 pp)
```

Strongest safe interpretation:

> **At full scale, both spectral strategies improve pooled LCC AUC over the clean baseline. MAST-PAD is essentially tied with WBST on LCC Combined while providing the strongest CASIA Combined AUC, supporting a more stable cross-domain trade-off across two independent target datasets.**

Do not claim statistical superiority: all source-training runs use one seed.

---

# 7. Full-scale MAST-PAD training diagnostics

Primary checkpoint:

```text
best epoch  = 7
stop epoch  = 11
```

Last executed epoch diagnostics:

```text
clean CE            = 0.012028
selected worst CE   = 0.019878
harmful-only CE     = 0.017835
harmful samples     = 403,220
harmful fraction    = 0.832135
mean margin drop    = 0.391421
spectral flip rate  = 0.003077
sampled gain mean   = 0.775127
LOW                 = 0.426452
MID                 = 0.273035
HIGH                = 0.300513
```

Interpretation:

- margin degradation remains common at full scale;
- prediction flips remain rare;
- all three spectral regions are selected non-trivially;
- these diagnostics describe optimization behavior, not causal proof.

---

# 8. Supporting 100K evidence

The earlier matched100K experiment remains valuable because it showed the developmental ordering:

| Method | LCC Combined AUC |
|---|---:|
| Clean | 0.848273 |
| WBST | 0.851978 |
| MAST-PAD | **0.867844** |

Use this only as supporting evidence that motivated freezing MAST-PAD before full-scale training. It is no longer the headline protocol.

---

# 9. Literature comparison strategy

The main literature comparison uses **AUC only**.

## 9.1 LCC-FASD contextual references

| Method | Published / reported LCC AUC | Approx. params | Source role |
|---|---:|---:|---|
| Graph for Transformer Feature | 0.833 | ~10.84M | peer-reviewed ESANN comparator |
| AENet | 0.868 | 11.22M | ECCV architecture; LCC value from established lightweight-FAS benchmark |
| **MAST-PAD** | **0.860** | **0.434M** | CelebA-full → LCC Combined |

Interpretation:

- MAST-PAD is numerically above Graph Transformer by about **+2.7 pp AUC**;
- MAST-PAD is about **0.8 pp below AENet** while using about **25.8× fewer deployment parameters**;
- protocols differ, therefore this is contextual efficiency positioning, not a strict leaderboard.

### Direct sources

Graph for Transformer Feature — ESANN 2023:
- DOI: https://doi.org/10.14428/esann/2023.ES2023-14
- PDF: https://www.esann.org/sites/default/files/proceedings/2023/ES2023-14.pdf

AENet / CelebA-Spoof — ECCV 2020:
- Official page: https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/1485_ECCV_2020_paper.php
- LCC benchmark result: https://github.com/kprokofi/light-weight-face-anti-spoofing

## 9.2 CASIA-FASD cross-domain published references

Use the standard literature protocol `O&M&I → C` only as **context**, because MAST-PAD is trained on CelebA rather than OULU+MSU+Replay-Attack.

| Method | Venue | Target setting | CASIA AUC | Notes |
|---|---|---|---:|---|
| MADDG | CVPR 2019 | O&M&I → C | 0.8451 | established DG baseline |
| AMEL | ACM MM 2022 | O&M&I → C | 0.9439 | adaptive mixture of experts |
| PatchNet | CVPR 2022 | O&M&I → C | 0.9458 | ResNet-18 implementation in paper |
| GAC-FAS | CVPR 2024 | O&M&I → C | 0.9516 | ResNet-18-based DG method |
| SA-FAS | CVPR 2023 | O&M&I → C | 0.9537 | ResNet-18; 100-epoch training described |
| **MAST-PAD** | ours | CelebA-full → reconstructed CASIA Combined | **0.9352** | **0.434M params; 12-epoch source schedule** |

Safe interpretation:

> **MAST-PAD exceeds the classical MADDG cross-domain reference and remains within roughly 0.9–1.9 AUC points of several substantially heavier modern DG-FAS methods, while retaining a 0.434M-parameter inference graph.**

### Direct sources

MADDG — CVPR 2019:
- Official page: https://openaccess.thecvf.com/content_CVPR_2019/html/Shao_Multi-Adversarial_Discriminative_Deep_Domain_Generalization_for_Face_Presentation_Attack_Detection_CVPR_2019_paper.html
- Official PDF: https://openaccess.thecvf.com/content_CVPR_2019/papers/Shao_Multi-Adversarial_Discriminative_Deep_Domain_Generalization_for_Face_Presentation_Attack_Detection_CVPR_2019_paper.pdf
- `84.51` AUC is also reproduced in PatchNet Table 8 below.

AMEL — ACM Multimedia 2022:
- arXiv: https://arxiv.org/abs/2207.09868
- PDF: https://arxiv.org/pdf/2207.09868
- DOI: https://doi.org/10.1145/3503161.3547769

PatchNet — CVPR 2022:
- Official page: https://openaccess.thecvf.com/content/CVPR2022/html/Wang_PatchNet_A_Simple_Face_Anti-Spoofing_Framework_via_Fine-Grained_Patch_Recognition_CVPR_2022_paper.html
- Official PDF: https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_PatchNet_A_Simple_Face_Anti-Spoofing_Framework_via_Fine-Grained_Patch_Recognition_CVPR_2022_paper.pdf
- Table 8 reports `O&M&I → C`: AUC `94.58`.

SA-FAS — CVPR 2023:
- Official page: https://openaccess.thecvf.com/content/CVPR2023/html/Sun_Rethinking_Domain_Generalization_for_Face_Anti-Spoofing_Separability_and_Alignment_CVPR_2023_paper.html
- Official PDF: https://openaccess.thecvf.com/content/CVPR2023/papers/Sun_Rethinking_Domain_Generalization_for_Face_Anti-Spoofing_Separability_and_Alignment_CVPR_2023_paper.pdf
- `O&M&I → C`: AUC `95.37`.

GAC-FAS — CVPR 2024:
- Official page: https://openaccess.thecvf.com/content/CVPR2024/html/Le_Gradient_Alignment_for_Cross-Domain_Face_Anti-Spoofing_CVPR_2024_paper.html
- Official PDF: https://openaccess.thecvf.com/content/CVPR2024/papers/Le_Gradient_Alignment_for_Cross-Domain_Face_Anti-Spoofing_CVPR_2024_paper.pdf
- `O&M&I → C`: AUC `95.16`.

---

# 10. Efficiency and deployment evidence

```text
deployment parameters = 434,434
training model params  = 441,658
MACs                    = 40,810,892
reported GFLOPs         = 0.081621784
input                    = [1,3,80,80]
```

Central claim:

> **MAST-PAD adds training-time spectral computation but no additional spectral branch or model parameters to the deployed PAD graph relative to the clean baseline.**

---

# 11. Paper story

Use this final narrative:

```text
1. Lightweight PAD is attractive for edge authentication but degrades under domain shift.
2. Frequency contains spoof cues but may also encode domain-specific shortcuts.
3. WBST trains against the currently worst LOW/MID/HIGH spectral view.
4. MAST-PAD adds a margin-aware harmful gate so spectral CE is applied only when the intervention actually reduces clean decision margin.
5. Full-scale CelebA training is the primary experiment.
6. LCC and CASIA are frozen cross-domain targets; headline metric is Combined AUC.
7. Both spectral strategies improve LCC over Clean; MAST-PAD is essentially tied with WBST there.
8. MAST-PAD gives the clearest gain on independent CASIA Combined and remains extremely small at 0.434M deployment parameters.
9. Literature comparisons are contextual because source-domain protocols differ.
```

---

# 12. Claim discipline

Supported:

> MAST-PAD is a margin-aware, harmful-selective spectral training strategy for lightweight Face PAD.

Supported:

> Full-scale MAST-PAD uses frequency only during training and preserves the 0.434M spatial MiniFASNetV2 inference graph.

Supported:

> On LCC Combined, MAST-PAD improves AUC from 0.846 to 0.860 over the full-scale Clean baseline and is nearly tied with WBST.

Supported:

> On reconstructed CASIA Combined, MAST-PAD reaches 0.935 AUC, above the full-scale Clean and WBST checkpoints.

Supported with protocol qualifier:

> MAST-PAD is numerically competitive with selected published LCC and CASIA cross-domain references while being substantially smaller than many of them.

Not supported:

- MAST-PAD is SOTA.
- MAST-PAD statistically outperforms WBST.
- all literature rows use identical protocols.
- the reconstructed CASIA result is the official CASIA subject-disjoint benchmark.
- 12 epochs alone proves sample efficiency.

---

# 13. Remaining work

1. update paper, deck, figures, and captions to the full-scale/two-target story;
2. keep 100K experiments in provenance / supporting ablation;
3. optional multi-seed confirmation;
4. optional evaluation on an official full-video CASIA release or another independent target;
5. optional Raspberry Pi / ONNX / INT8 latency study.

Do not reopen backbone or loss search for the course paper unless a reproducibility failure is found.

---

# 14. Final stop rule

```text
PRIMARY METHOD
MAST-PAD = R7_SC_FULL_crop15

PRIMARY SOURCE TRAINING
full eligible CelebA-Spoof Train

PRIMARY CROSS-DOMAIN REPORTING
LCC Combined AUC   = 0.860
CASIA Combined AUC = 0.935

DEPLOYMENT
0.434M parameters
80×80 MiniFASNetV2
no spectral inference branch
```
