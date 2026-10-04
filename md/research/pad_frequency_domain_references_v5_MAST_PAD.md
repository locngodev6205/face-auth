# MAST-PAD Literature and Benchmark Reference Map
## Frequency-domain motivation, lightweight PAD context, and selected LCC-FASD comparison sources

_Last updated: 2026-10-05_

> **Official method:** **MAST-PAD — Margin-Aware Spectral Training for Face Presentation Attack Detection**
>
> The project no longer uses the historical CSMR formulation as its final method. The final paper method corresponds to development run `R7_SC_100K_crop15`: sample-adaptive worst spectral view selection followed by harmful-only spectral classification supervision.
>
> This document separates two roles:
>
> 1. **method-related literature** used to motivate and differentiate MAST-PAD;
> 2. **contextual LCC-FASD benchmarks** used in the course-paper comparison table.

---

# 1. Final literature positioning

The final research question is:

> **Can spectral information be used as a training-time stressor, rather than an inference feature, to improve the cross-domain robustness of a very small Face PAD model?**

The project trajectory is:

```text
explicit frequency cues
→ useful but unstable across domains
→ selected worst-band training
→ margin-aware harmful gating
→ MAST-PAD
```

The final novelty candidate is not FFT itself.

MAST-PAD combines:

```text
LOW / MID / HIGH smooth spectral attenuation
→ evaluate the current label-aligned PAD margin
→ select the sample-specific worst band
→ apply spectral CE only when the worst view reduces the clean margin
→ retain a spatial-only inference graph
```

---

# 2. What is not a novelty claim

Do not claim:

- first use of FFT in Face PAD;
- first use of frequency augmentation;
- first use of high/low-frequency decomposition;
- first use of frequency masking;
- first use of spectral consistency;
- first use of a frequency-aware Face PAD model.

All of these ideas have precedent.

The differentiating point should be:

> **margin-aware worst-band selection + harmful-only spectral supervision + no spectral inference branch.**

---

# 3. Core frequency-aware related work

## 3.1 Cao & Ma — frequency shortcut view

**J. Cao and C. Ma, “Towards Generalized Face Anti-Spoofing from a Frequency Shortcut View,” WACV 2025, pp. 1005–1015.**  
DOI: `10.1109/WACV61041.2025.00107`

Role:

- strongest conceptual motivation that frequency can encode domain-sensitive shortcuts;
- supports the idea that “frequency-rich” does not automatically mean “domain-general”.

Relationship to MAST-PAD:

```text
Cao & Ma:
frequency shortcut diagnosis / suppression

MAST-PAD:
probe several controlled spectral interventions
→ identify the one that most hurts the current PAD margin
→ train only on samples where that damage is real
```

Do not claim MAST-PAD is the first work to identify frequency shortcuts.

---

## 3.2 FSDA — frequency disentanglement and augmentation

**Y. Yu, Z. Du, H. Luo, C. Xiao, and J. Hu, “Fourier-Based Frequency Space Disentanglement and Augmentation for Generalizable Face Anti-Spoofing,” IEEE Journal of Biomedical and Health Informatics, 2025.**  
DOI: `10.1109/JBHI.2024.3417404`

Relevance:

- Fourier amplitude manipulation for generalizable Face PAD;
- frequency-space augmentation/disentanglement;
- consistency/distillation style objectives.

Key distinction:

```text
FSDA family:
augment / disentangle / regularize frequency information

MAST-PAD:
evaluate a fixed LOW/MID/HIGH counterfactual set with the current model
→ choose the lowest aligned-margin view
→ gate spectral classification by actual margin degradation
```

This is an important related work because reviewers may ask whether MAST-PAD is only frequency augmentation.

---

## 3.3 Chen et al. — high/low-frequency fusion

**B. Chen, W. Yang, and S. Wang, “Face Anti-Spoofing by Fusing High and Low Frequency Features for Advanced Generalization Capability,” IEEE MIPR, 2020.**  
DOI: `10.1109/MIPR49039.2020.00048`

Role:

- establishes that different spectral regions can carry complementary spoof cues;
- historical motivation for separating LOW/MID/HIGH behavior.

Contrast:

```text
Chen et al.:
frequency features participate in inference

MAST-PAD:
frequency is discarded after training
```

---

## 3.4 Bi-FPNFAS

**K. Roy et al., “Bi-Directional Feature Pyramid Network for Pixel-Wise Face Anti-Spoofing by Leveraging Fourier Spectra,” Sensors, vol. 21, no. 8, 2799, 2021.**  
DOI: `10.3390/s21082799`

Role:

- useful precedent for Fourier-derived supervision in Face PAD;
- supports the broader idea that spectral information can guide learning.

Difference:

> MAST-PAD uses spectral counterfactuals to stress a shared spatial PAD model rather than adding a learned Fourier feature path at deployment.

---

## 3.5 DWT-LBP-DCT

**W. Zhang and S. Xiang, “Face anti-spoofing detection based on DWT-LBP-DCT features,” Signal Processing: Image Communication, vol. 89, 115990, 2020.**  
DOI: `10.1016/j.image.2020.115990`

Role:

- classical / handcrafted frequency-texture precedent;
- useful when motivating why spectral statistics have historically mattered in PAD.

Not a direct method comparator.

---

## 3.6 Deep Frequent Spatial Temporal Learning

**Y. Huang, W. Zhang, and J. Wang, “Deep Frequent Spatial Temporal Learning for Face Anti-Spoofing,” arXiv:2002.03723, 2020.**

Role:

- demonstrates learned spectrum-image representations;
- historical motivation only because its temporal/multi-frame scope differs from the final single-frame MAST-PAD protocol.

---

# 4. Recent contextual frequency-aware work

## 4.1 Oculus

**V. W. de Dravo et al., “Oculus: Hierarchical Face Spoof Detection via Frequency-Enhanced Vision Transformers with Group-Aware Classification and Post-Fusion Attention,” ICCV Workshops, 2025.**

Role:

- modern spatial-frequency fusion reference;
- useful contrast against the training-only MAST-PAD design.

Do not use as a direct numerical baseline unless protocol equivalence is established.

---

## 4.2 Niu & Lin

**Y. Niu and X. Lin, “Similarity-aware contrastive learning for face anti-spoofing via frequency enhancement and reconstruction,” Neural Networks, vol. 199, 108734, 2026.**  
DOI: `10.1016/j.neunet.2026.108734`

Role:

- adaptive frequency enhancement and frequency-aware training objective;
- shows that recent Face PAD work increasingly treats frequency as a learned robustness signal rather than only a fixed branch.

---

## 4.3 DEFuseNet

**R. P. Singh, R. Dash, and R. K. Mohapatra, “DEFuseNet: A domain-enhanced fusion network for generalizable face anti-spoofing,” Neurocomputing, vol. 696, 134092, 2026.**  
DOI: `10.1016/j.neucom.2026.134092`

Role:

- contextual domain perturbation / fusion reference;
- not a direct spectral counterpart to MAST-PAD.

---

## 4.4 Sun et al.

**R. Sun, F. Wang, X. Yu, X. Gao, and X. Zhang, “Robust multimodal face anti-spoofing via frequency-domain feature refinement and aggregation,” Pattern Recognition Letters, vol. 197, pp. 31–36, 2025.**  
DOI: `10.1016/j.patrec.2025.07.003`

Contrast:

```text
frequency refinement / aggregation at inference
vs
MAST-PAD training-time spectral stress only
```

---

# 5. Lightweight deployment context

## Ali et al.

**F. A. Ali, S. Mali, R. Mahakud, and G. Yadav, “AI-enabled smart surveillance system for secure monitoring and authentication,” Scientific Reports, 2026.**  
DOI: `10.1038/s41598-026-52387-w`

Prior project review identified:

- MiniFAS-family lightweight PAD;
- Fourier-frequency component;
- edge / Raspberry Pi context.

Why it remains useful:

```text
Ali et al.:
lightweight MiniFAS-family + explicit Fourier component

MAST-PAD:
MiniFASNetV2 + spectral training
→ no frequency branch in deployed model
```

Use as conceptual deployment context, not a direct benchmark unless dataset/protocol compatibility is verified.

---

# 6. Selected LCC-FASD comparison references for the course paper

The following baselines are intentionally heterogeneous.

They are useful for **context**, not for a strict apples-to-apples leaderboard.

## 6.1 Graph for Transformer Feature — ESANN 2023

**Q.-H. Trinh, H. Nguyen, V. Nguyen, X.-M. Nguyen, and H.-D. Nguyen, “Graph for Transformer Feature: A New Approach for Face Anti-Spoofing,” ESANN 2023.**  
DOI: `10.14428/esann/2023.ES2023-14`

Direct PDF:  
https://www.esann.org/sites/default/files/proceedings/2023/ES2023-14.pdf

Reported LCC-FASD Table 1:

```text
AUC   = 0.833
EER   = 12.23%
APCER = 8.9%
```

Implementation details reported by the paper:

```text
~10.84M parameters
batch size = 64
50 epochs
single Tesla V100 16GB
```

Important audit note:

> The prose immediately above Table 1 contains inconsistent qualitative numbers for LCC. Use **Table 1** (`AUC 0.833`, `EER 12.23%`, `APCER 8.9%`) as the numerical citation.

Protocol:

- the paper states that models are trained independently on different datasets;
- treat the LCC row as an **LCC intra-dataset** reference.

Use in paper:

> strong peer-reviewed comparator with substantially larger parameter count.

---

## 6.2 AENet — LCC result from the lightweight-FAS benchmark

Benchmark repository:  
https://github.com/kprokofi/light-weight-face-anti-spoofing

Reported LCC row:

```text
AUC   = 0.868
EER   = 20.91%
APCER = 12.52%
BPCER = 32.70%
ACER  = 22.61%
Params = 11.22M
GFLOPs = 3.64
```

Original AENet / CelebA-Spoof paper:  
https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/1485_ECCV_2020_paper.php

Important wording:

> The `0.868` LCC number should be cited as a **benchmark-repository result for AENet**, not as a number reported in the original ECCV CelebA-Spoof paper unless separately verified there.

Use in paper:

- strong numerical reference;
- MAST-PAD rounds to the same AUC (`0.868`) with a much smaller deployment model.

Do not imply identical preprocessing, threshold selection, or train/test protocol.

---

## 6.3 ViT-B/16 and ViT-B/16 + LoRA

Source / thesis implementation:  
https://github.com/ozogxyz/spoof

Model:

```text
ViT-B/16
~86.6M backbone parameters
224×224 input
```

Cross-dataset `SiW-M → LCC` results:

| Variant | EER | HTER |
|---|---:|---:|
| ViT-B/16 | 30.98% | 34.62% |
| ViT-B/16 + LoRA | 28.95% | 29.33% |

Use:

- very useful cross-domain contextual reference;
- MAST-PAD has numerically lower EER/HTER under its CelebA→LCC setup;
- source datasets differ, so this is **not** a direct superiority claim.

Also reported in the same repository:

```text
LCC intra ViT:
EER 16.05%, HTER 20.28%

LCC intra ViT + LoRA:
EER 6.61%, HTER 6.62%
```

Do not mix these intra-LCC values with the cross-domain table.

---

## 6.4 LBP-GBM — original LCC-FASD paper

**D. Timoshenko, K. Simonchik, V. Shutov, P. Zhelezneva, and V. Grishkin, “Large Crowdcollected Facial Anti-Spoofing Dataset,” CSIT 2019, pp. 123–126.**  
DOI: `10.1109/CSITechnol.2019.8895208`

Direct PDF:  
https://pureportal.spbu.ru/files/51584200/CSIT2019_Grishkin_Timoshenko.pdf

Reported:

```text
LBP-GBM EER = 14.6%
```

The paper also reports neural baselines such as Xception, ResNeXt-50, and SENet-154.

Use:

- historical baseline from the paper that introduced LCC-FASD;
- protocol is the official LCC train/calibration/evaluation setup;
- do not compare its EER as if it were the same as CelebA→LCC cross-domain evaluation.

---

# 7. Primary MAST-PAD literature-comparison row

The primary course-paper result is:

```text
MAST-PAD
development ID: R7_SC_100K_crop15
training: CelebA-Spoof Train100K
selection + threshold: CelebA-Spoof Val15K
target: full LCC-FASD train/dev/eval pooled
AUC = 0.867844 → 0.868
EER = 21.5867% → 21.59%
HTER = 21.5738% → 21.57%
deployment params = 434,434 → 0.434M
```

Recommended contextual table:

| Method | Protocol / target role | AUC ↑ | EER ↓ | HTER / ACER ↓ | Params |
|---|---|---:|---:|---:|---:|
| Graph for Transformer Feature | LCC intra | 0.833 | 12.23% | — | ~10.84M |
| AENet | LCC result from benchmark repo | 0.868 | 20.91% | ACER 22.61% | 11.22M |
| ViT-B/16 | SiW-M → LCC | — | 30.98% | HTER 34.62% | ~86.6M |
| ViT-B/16 + LoRA | SiW-M → LCC | — | 28.95% | HTER 29.33% | ~86.6M backbone |
| LBP-GBM | official LCC protocol | — | 14.60% | — | classical |
| **MAST-PAD (ours)** | **CelebA100K → pooled LCC** | **0.868** | **21.59%** | **HTER 21.57%** | **0.434M** |

Use the table to discuss **efficiency and contextual competitiveness**, not SOTA.

---

# 8. Safe comparison language

Recommended:

> **MAST-PAD obtains a pooled LCC-FASD AUC of 0.868 with a 0.434M-parameter deployment model. Numerically, this exceeds the 0.833 AUC reported by Graph for Transformer Feature and matches the 0.868 AUC reported for AENet in the lightweight-FAS benchmark. Because the studies use different training and evaluation protocols, these values are treated as contextual references rather than a strict common-protocol leaderboard.**

For ViT:

> **Under its CelebA→LCC cross-domain protocol, MAST-PAD reports lower EER/HTER than the SiW-M→LCC ViT and ViT+LoRA thesis baselines; the source domains differ, so the comparison is directional rather than controlled.**

For LBP-GBM:

> **The original LCC-FASD paper reports a 14.6% EER for LBP-GBM under the dataset’s own train/calibration/evaluation protocol, providing a historical reference rather than a directly matched cross-domain baseline.**

---

# 9. Claims to avoid

Avoid:

> MAST-PAD beats AENet under the same protocol.

Avoid:

> MAST-PAD is better than ViT+LoRA.

Avoid:

> MAST-PAD achieves SOTA on LCC-FASD.

Avoid:

> 12 epochs versus 50 epochs proves MAST-PAD is more sample efficient.

The epoch counts, training-set sizes, initializations, and optimization recipes differ.

Safe:

> MAST-PAD is substantially smaller in parameter count and remains numerically competitive with selected published/reported LCC references.

---

# 10. Literature-to-method map

| MAST-PAD element | Related literature theme | Project-specific role |
|---|---|---|
| LOW/MID/HIGH attenuation | spectral decomposition / masking | fixed smooth counterfactual family |
| sample-adaptive worst band | adaptive frequency handling | chosen by current label-aligned PAD margin |
| harmful gate | robustness / selective hard-example training | activates only when worst view reduces clean margin |
| harmful-only spectral CE | adversarial/worst-view training | classification term restricted to harmful samples |
| no FFT at inference | auxiliary spectral supervision | deployed network stays spatial-only |

---

# 11. Recommended Related Work structure

## 2.1 Lightweight Face PAD

Discuss:

- MiniFASNet family;
- MobileNet-family efficient PAD;
- edge deployment constraints.

## 2.2 Frequency-aware Face PAD

Discuss:

- DCT/Fourier features;
- high/low-frequency branches;
- Fourier-supervised methods;
- recent frequency-enhanced fusion.

## 2.3 Generalizable Face PAD and spectral shortcut

Center:

- Cao & Ma;
- FSDA;
- domain perturbation / consistency literature.

## 2.4 Positioning of MAST-PAD

Close with:

```text
Prior work:
frequency representation / fusion / augmentation / masking

MAST-PAD:
sample-adaptive spectral counterfactual selection
based on the current PAD margin
+ harmful-only spectral supervision
+ unchanged spatial inference graph
```

---

# 12. IEEE-style references

```text
[1] B. Chen, W. Yang, and S. Wang, “Face Anti-Spoofing by Fusing High and Low Frequency Features for Advanced Generalization Capability,” in Proc. IEEE MIPR, 2020, pp. 199–204, doi: 10.1109/MIPR49039.2020.00048.

[2] Y. Huang, W. Zhang, and J. Wang, “Deep Frequent Spatial Temporal Learning for Face Anti-Spoofing,” arXiv:2002.03723, 2020.

[3] K. Roy et al., “Bi-Directional Feature Pyramid Network for Pixel-Wise Face Anti-Spoofing by Leveraging Fourier Spectra,” Sensors, vol. 21, no. 8, 2799, 2021, doi: 10.3390/s21082799.

[4] W. Zhang and S. Xiang, “Face anti-spoofing detection based on DWT-LBP-DCT features,” Signal Processing: Image Communication, vol. 89, 115990, 2020, doi: 10.1016/j.image.2020.115990.

[5] J. Cao and C. Ma, “Towards Generalized Face Anti-Spoofing from a Frequency Shortcut View,” in Proc. WACV, 2025, pp. 1005–1015, doi: 10.1109/WACV61041.2025.00107.

[6] Y. Yu, Z. Du, H. Luo, C. Xiao, and J. Hu, “Fourier-Based Frequency Space Disentanglement and Augmentation for Generalizable Face Anti-Spoofing,” IEEE J. Biomed. Health Inform., 2025, doi: 10.1109/JBHI.2024.3417404.

[7] V. W. de Dravo et al., “Oculus: Hierarchical Face Spoof Detection via Frequency-Enhanced Vision Transformers with Group-Aware Classification and Post-Fusion Attention,” ICCV Workshops, 2025.

[8] F. A. Ali, S. Mali, R. Mahakud, and G. Yadav, “AI-enabled smart surveillance system for secure monitoring and authentication,” Scientific Reports, 2026, doi: 10.1038/s41598-026-52387-w.

[9] Y. Niu and X. Lin, “Similarity-aware contrastive learning for face anti-spoofing via frequency enhancement and reconstruction,” Neural Networks, vol. 199, 108734, 2026, doi: 10.1016/j.neunet.2026.108734.

[10] R. P. Singh, R. Dash, and R. K. Mohapatra, “DEFuseNet: A domain-enhanced fusion network for generalizable face anti-spoofing,” Neurocomputing, vol. 696, 134092, 2026, doi: 10.1016/j.neucom.2026.134092.

[11] R. Sun, F. Wang, X. Yu, X. Gao, and X. Zhang, “Robust multimodal face anti-spoofing via frequency-domain feature refinement and aggregation,” Pattern Recognition Letters, vol. 197, pp. 31–36, 2025, doi: 10.1016/j.patrec.2025.07.003.

[12] Q.-H. Trinh, H. Nguyen, V. Nguyen, X.-M. Nguyen, and H.-D. Nguyen, “Graph for Transformer Feature: A New Approach for Face Anti-Spoofing,” ESANN, 2023, doi: 10.14428/esann/2023.ES2023-14.

[13] Y. Zhang et al., “CelebA-Spoof: Large-Scale Face Anti-Spoofing Dataset with Rich Annotations,” in Proc. ECCV, 2020.

[14] D. Timoshenko, K. Simonchik, V. Shutov, P. Zhelezneva, and V. Grishkin, “Large Crowdcollected Facial Anti-Spoofing Dataset,” in Proc. CSIT, 2019, pp. 123–126, doi: 10.1109/CSITechnol.2019.8895208.
```

Before formal submission, perform one final bibliographic audit for venue formatting, page ranges, author ordering, and publication year metadata.
