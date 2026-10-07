# MAST-PAD Literature and Benchmark Reference Map
## Full-scale frequency-domain motivation, lightweight PAD context, and two-target comparison sources

_Last updated: 2026-10-07_

> **Official method:** **MAST-PAD — Margin-Aware Spectral Training for Face Presentation Attack Detection**
>
> **Primary implementation:** `R7_SC_FULL_crop15`
>
> **Headline metric:** AUC only.
>
> **Primary target results:** LCC Combined AUC `0.860272 → 0.860`; reconstructed CASIA Combined AUC `0.935199 → 0.935`.

---

# 1. Final literature positioning

Research question:

> **Can spectral information be used as a training-time stressor, rather than an inference feature, to improve cross-domain robustness of a very small Face PAD model?**

MAST-PAD combines:

```text
LOW / MID / HIGH smooth spectral attenuation
→ current label-aligned PAD margin
→ sample-specific worst band
→ harmful-only spectral CE
→ unchanged spatial-only inference graph
```

The novelty claim is **not** FFT, frequency augmentation, masking, or frequency fusion by itself.

The differentiating point is:

> **margin-aware worst-band selection + harmful-only spectral supervision + no spectral inference branch.**

---

# 2. Core frequency-aware related work

## 2.1 Cao & Ma — frequency shortcut view

J. Cao and C. Ma, “Towards Generalized Face Anti-Spoofing from a Frequency Shortcut View,” WACV 2025.

- DOI: https://doi.org/10.1109/WACV61041.2025.00107
- Role: strongest conceptual motivation that frequency can encode domain-sensitive shortcuts.
- Difference: MAST-PAD uses the current decision margin to select and gate controlled spectral counterfactuals.

## 2.2 FSDA

Y. Yu, Z. Du, H. Luo, C. Xiao, and J. Hu, “Fourier-Based Frequency Space Disentanglement and Augmentation for Generalizable Face Anti-Spoofing,” IEEE JBHI, 2025.

- DOI: https://doi.org/10.1109/JBHI.2024.3417404
- Role: Fourier amplitude manipulation, frequency-space augmentation/disentanglement, generalizable PAD.
- Difference: MAST-PAD uses a fixed candidate family and margin-aware harmful gating rather than learning/deploying a frequency representation.

## 2.3 High/low-frequency fusion

B. Chen, W. Yang, and S. Wang, “Face Anti-Spoofing by Fusing High and Low Frequency Features for Advanced Generalization Capability,” IEEE MIPR 2020.

- DOI: https://doi.org/10.1109/MIPR49039.2020.00048
- Role: precedent that different spectral regions contain complementary PAD cues.
- Difference: frequency participates in their inference representation; MAST-PAD discards spectral operations after training.

## 2.4 Bi-FPNFAS

K. Roy et al., “Bi-Directional Feature Pyramid Network for Pixel-Wise Face Anti-Spoofing by Leveraging Fourier Spectra,” Sensors 2021.

- DOI: https://doi.org/10.3390/s21082799
- Role: Fourier-derived supervision / representation precedent.

## 2.5 DWT-LBP-DCT

W. Zhang and S. Xiang, “Face anti-spoofing detection based on DWT-LBP-DCT features,” Signal Processing: Image Communication 2020.

- DOI: https://doi.org/10.1016/j.image.2020.115990
- Role: handcrafted frequency-texture precedent.

## 2.6 Deep Frequent Spatial Temporal Learning

Y. Huang, W. Zhang, and J. Wang, “Deep Frequent Spatial Temporal Learning for Face Anti-Spoofing,” 2020.

- arXiv: https://arxiv.org/abs/2002.03723
- Role: spectrum-image / temporal precedent; not a direct comparator to single-frame MAST-PAD.

---

# 3. Recent frequency-aware context

## Oculus

V. W. de Dravo et al., “Oculus: Hierarchical Face Spoof Detection via Frequency-Enhanced Vision Transformers with Group-Aware Classification and Post-Fusion Attention,” ICCV Workshops 2025.

Role: modern spatial-frequency fusion reference; not used for direct numerical comparison unless protocol equivalence is established.

## Niu & Lin

Y. Niu and X. Lin, “Similarity-aware contrastive learning for face anti-spoofing via frequency enhancement and reconstruction,” Neural Networks, 2026.

- DOI: https://doi.org/10.1016/j.neunet.2026.108734

## DEFuseNet

R. P. Singh, R. Dash, and R. K. Mohapatra, “DEFuseNet: A domain-enhanced fusion network for generalizable face anti-spoofing,” Neurocomputing, 2026.

- DOI: https://doi.org/10.1016/j.neucom.2026.134092
- Use as DG context, not as an LCC/CASIA numerical comparator unless its exact target protocol matches.

## Frequency-domain refinement / aggregation

R. Sun et al., “Robust multimodal face anti-spoofing via frequency-domain feature refinement and aggregation,” Pattern Recognition Letters, 2025.

- DOI: https://doi.org/10.1016/j.patrec.2025.07.003

---

# 4. Lightweight deployment context

## MiniFAS-family edge context

F. A. Ali et al., “AI-enabled smart surveillance system for secure monitoring and authentication,” Scientific Reports, 2026.

- DOI: https://doi.org/10.1038/s41598-026-52387-w

Use as deployment/context literature, not as a direct cross-domain benchmark.

MAST-PAD deployment complexity:

```text
parameters = 434,434
training-model params = 441,658
MACs = 40,810,892
reported GFLOPs = 0.081621784
input = 80×80
```

---

# 5. Full-scale MAST-PAD target results

| Target | Combined AUC | Reporting role |
|---|---:|---|
| LCC-FASD | **0.860272 → 0.860** | primary cross-domain target |
| CASIA-FASD extracted-image copy | **0.935199 → 0.935** | independent cross-domain validation |

Use AUC only in literature comparison tables.

---

# 6. LCC-FASD comparison references

## 6.1 Graph for Transformer Feature — ESANN 2023

Q.-H. Trinh et al., “Graph for Transformer Feature: A New Approach for Face Anti-Spoofing,” ESANN 2023.

- DOI: https://doi.org/10.14428/esann/2023.ES2023-14
- Direct PDF: https://www.esann.org/sites/default/files/proceedings/2023/ES2023-14.pdf

Table 1 reports:

```text
LCC AUC = 0.833
EER     = 12.23%
APCER   = 8.9%
```

Implementation section reports approximately:

```text
params = 10.84M
batch = 64
training = 50 epochs
GPU = Tesla V100 16GB
```

Use **Table 1 AUC 0.833**; prose around the table contains inconsistent values.

## 6.2 AENet

Original method context:

Y. Zhang et al., “CelebA-Spoof: Large-Scale Face Anti-Spoofing Dataset with Rich Annotations,” ECCV 2020.

- Official page: https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/1485_ECCV_2020_paper.php

Established lightweight-FAS benchmark reporting the LCC row:

- https://github.com/kprokofi/light-weight-face-anti-spoofing

Reported LCC values:

```text
AUC    = 0.868
EER    = 20.91%
ACER   = 22.61%
params = 11.22M
GFLOPs = 3.64
```

Important wording:

> The `0.868` LCC number is a benchmark-reported AENet result, not a headline LCC result from the original ECCV paper.

## 6.3 Recommended LCC table

| Method | LCC AUC ↑ | Params ↓ | Provenance |
|---|---:|---:|---|
| Graph for Transformer Feature | 0.833 | ~10.84M | ESANN 2023 paper |
| AENet | **0.868** | 11.22M | ECCV model + established LCC benchmark |
| **MAST-PAD** | **0.860** | **0.434M** | full CelebA → LCC Combined |

Safe wording:

> **MAST-PAD reaches 0.860 pooled LCC AUC with a 0.434M-parameter deployment model. It is numerically above the 0.833 Graph Transformer reference and within 0.8 AUC points of the 0.868 AENet benchmark result, while using roughly 25× fewer parameters. Protocols differ, so these are contextual references rather than a common-protocol leaderboard.**

---

# 7. CASIA-FASD cross-domain comparison references

The literature convention below uses CASIA (`C`) as the held-out target in `O&M&I → C`. MAST-PAD uses a different source domain (`CelebA → C`), so the comparison is contextual.

## 7.1 MADDG — CVPR 2019

R. Shao et al., “Multi-Adversarial Discriminative Deep Domain Generalization for Face Presentation Attack Detection,” CVPR 2019.

- Official page: https://openaccess.thecvf.com/content_CVPR_2019/html/Shao_Multi-Adversarial_Discriminative_Deep_Domain_Generalization_for_Face_Presentation_Attack_Detection_CVPR_2019_paper.html
- Official PDF: https://openaccess.thecvf.com/content_CVPR_2019/papers/Shao_Multi-Adversarial_Discriminative_Deep_Domain_Generalization_for_Face_Presentation_Attack_Detection_CVPR_2019_paper.pdf

Cross-domain `O&M&I → C` value widely reproduced in later peer-reviewed tables:

```text
AUC = 84.51% = 0.8451
```

PatchNet Table 8 is a convenient verification source for this value.

## 7.2 AMEL — ACM Multimedia 2022

Q. Zhou et al., “Adaptive Mixture of Experts Learning for Generalizable Face Anti-Spoofing,” ACM MM 2022.

- arXiv page: https://arxiv.org/abs/2207.09868
- PDF: https://arxiv.org/pdf/2207.09868
- DOI: https://doi.org/10.1145/3503161.3547769

```text
O&M&I → C AUC = 94.39% = 0.9439
```

## 7.3 PatchNet — CVPR 2022

C.-Y. Wang et al., “PatchNet: A Simple Face Anti-Spoofing Framework via Fine-Grained Patch Recognition,” CVPR 2022.

- Official page: https://openaccess.thecvf.com/content/CVPR2022/html/Wang_PatchNet_A_Simple_Face_Anti-Spoofing_Framework_via_Fine-Grained_Patch_Recognition_CVPR_2022_paper.html
- Official PDF: https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_PatchNet_A_Simple_Face_Anti-Spoofing_Framework_via_Fine-Grained_Patch_Recognition_CVPR_2022_paper.pdf

Table 8:

```text
O&M&I → C AUC = 94.58% = 0.9458
```

The paper uses a ResNet-18 backbone in its main cross-domain implementation.

## 7.4 SA-FAS — CVPR 2023

Y. Sun et al., “Rethinking Domain Generalization for Face Anti-Spoofing: Separability and Alignment,” CVPR 2023.

- Official page: https://openaccess.thecvf.com/content/CVPR2023/html/Sun_Rethinking_Domain_Generalization_for_Face_Anti-Spoofing_Separability_and_Alignment_CVPR_2023_paper.html
- Official PDF: https://openaccess.thecvf.com/content/CVPR2023/papers/Sun_Rethinking_Domain_Generalization_for_Face_Anti-Spoofing_Separability_and_Alignment_CVPR_2023_paper.pdf

```text
O&M&I → C AUC = 95.37% = 0.9537
```

Implementation uses ResNet-18 for fair comparison; the paper describes 100-epoch training for most setups.

## 7.5 GAC-FAS — CVPR 2024

B. M. Le and S. S. Woo, “Gradient Alignment for Cross-Domain Face Anti-Spoofing,” CVPR 2024.

- Official page: https://openaccess.thecvf.com/content/CVPR2024/html/Le_Gradient_Alignment_for_Cross-Domain_Face_Anti-Spoofing_CVPR_2024_paper.html
- Official PDF: https://openaccess.thecvf.com/content/CVPR2024/papers/Le_Gradient_Alignment_for_Cross-Domain_Face_Anti-Spoofing_CVPR_2024_paper.pdf

```text
O&M&I → C AUC = 95.16% = 0.9516
```

The paper uses an ImageNet-pretrained ResNet-18 backbone in the standard DG setup.

## 7.6 Recommended CASIA table

| Method | Venue | Target protocol | CASIA AUC ↑ |
|---|---|---|---:|
| MADDG | CVPR 2019 | O&M&I → C | 0.8451 |
| AMEL | ACM MM 2022 | O&M&I → C | 0.9439 |
| PatchNet | CVPR 2022 | O&M&I → C | 0.9458 |
| GAC-FAS | CVPR 2024 | O&M&I → C | 0.9516 |
| SA-FAS | CVPR 2023 | O&M&I → C | 0.9537 |
| **MAST-PAD** | ours | CelebA-full → reconstructed CASIA Combined | **0.9352** |

Safe interpretation:

> **MAST-PAD exceeds the established MADDG baseline and remains within 0.87–1.85 AUC points of AMEL, PatchNet, GAC-FAS, and SA-FAS, while retaining only 0.434M deployment parameters. Source-domain protocols differ, and the current CASIA resource is an extracted-image reconstruction rather than a verified official full-video split.**

---

# 8. Literature-to-method map

| MAST-PAD element | Related literature theme | Project-specific role |
|---|---|---|
| LOW/MID/HIGH attenuation | spectral decomposition / masking | controlled counterfactual family |
| sample-adaptive worst band | adaptive frequency handling | chosen by current label-aligned margin |
| harmful gate | robustness / hard-example selection | activates only when margin is degraded |
| harmful-only spectral CE | worst-view training | CE restricted to harmful samples |
| no FFT at inference | auxiliary spectral supervision | deployed network remains spatial-only |

---

# 9. Main comparison rules

1. **Use AUC only** in headline literature tables.
2. Use project **Combined** AUC for both LCC and CASIA.
3. Do not mix project HTER/ACER with published HTER/ACER in the main table.
4. Always state that source-domain protocols differ.
5. Always state that reconstructed CASIA is not a verified reproduction of the official subject-disjoint full-video benchmark.
6. Do not claim SOTA.

---

# 10. IEEE-style core references

```text
[1] Q.-H. Trinh et al., “Graph for Transformer Feature: A New Approach for Face Anti-Spoofing,” ESANN, 2023, doi: 10.14428/esann/2023.ES2023-14.

[2] Y. Zhang et al., “CelebA-Spoof: Large-Scale Face Anti-Spoofing Dataset with Rich Annotations,” ECCV, 2020.

[3] R. Shao et al., “Multi-Adversarial Discriminative Deep Domain Generalization for Face Presentation Attack Detection,” CVPR, 2019.

[4] Q. Zhou et al., “Adaptive Mixture of Experts Learning for Generalizable Face Anti-Spoofing,” ACM Multimedia, 2022, doi: 10.1145/3503161.3547769.

[5] C.-Y. Wang et al., “PatchNet: A Simple Face Anti-Spoofing Framework via Fine-Grained Patch Recognition,” CVPR, 2022.

[6] Y. Sun et al., “Rethinking Domain Generalization for Face Anti-Spoofing: Separability and Alignment,” CVPR, 2023.

[7] B. M. Le and S. S. Woo, “Gradient Alignment for Cross-Domain Face Anti-Spoofing,” CVPR, 2024.

[8] J. Cao and C. Ma, “Towards Generalized Face Anti-Spoofing from a Frequency Shortcut View,” WACV, 2025, doi: 10.1109/WACV61041.2025.00107.

[9] Y. Yu et al., “Fourier-Based Frequency Space Disentanglement and Augmentation for Generalizable Face Anti-Spoofing,” IEEE JBHI, 2025, doi: 10.1109/JBHI.2024.3417404.
```
