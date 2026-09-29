# Frequency-Domain References for CSMR-PAD

_Last updated: 2026-09-29_

> **Purpose:** literature map for the current research direction.
>
> The project is no longer positioned as a final dual-branch DCT system. Historical DCT/Fourier work remains important as motivation, but the current method uses frequency **only during training** through CSMR.
>
> Bibliographic metadata below is carried forward from the previous reference file. Before submission, run one final DOI / venue / page audit.

---

# 1. Current literature positioning

The central question is no longer:

> Which explicit frequency branch should be fused with a spatial backbone?

It is now:

> Can frequency be used as a controlled training-time perturbation to reduce PAD decision sensitivity to domain-dependent spectral shortcuts?

The project history supports this shift:

```text
explicit frequency cue
→ discriminative in-domain
→ unstable cross-domain complementarity
→ frequency-shortcut concern
→ training-time spectral robustness
→ CSMR
```

---

# 2. What is and is not novel

## Not sufficient as novelty

These ideas already have substantial precedent:

- Fourier/DCT features for Face PAD;
- high/low-frequency decomposition;
- frequency augmentation;
- spectral disentanglement;
- frequency masking;
- spatial+frequency fusion;
- frequency auxiliary supervision.

Therefore avoid:

> “We are the first to use FFT for Face PAD.”

and avoid:

> “Frequency augmentation itself is the contribution.”

## Current novelty candidate

CSMR combines:

```text
controlled LOW/MID/HIGH amplitude interventions
→ current-model PAD margin evaluation
→ sample-adaptive worst-band selection
→ clean + worst-view classification
→ harmful-only aligned-margin robustness penalty
→ no frequency branch at inference
```

The novelty claim should be formulated around:

1. **sample-adaptive counterfactual selection**;
2. **PAD-specific label-aligned decision margin**;
3. **harmful-drop-only robustness regularization**;
4. **zero inference-time spectral overhead**.

---

# 3. Most important related works

## 3.1. Cao & Ma — Frequency Shortcut View

**J. Cao and C. Ma, “Towards Generalized Face Anti-Spoofing from a Frequency Shortcut View,” WACV 2025, pp. 1005–1015.**  
DOI: `10.1109/WACV61041.2025.00107`

Key ideas carried from the previous literature review:

- frequency shortcuts can themselves hurt generalization;
- frequency-aware autoencoding;
- dynamic frequency masking;
- style-inhibited modulation.

### Relationship to CSMR

Closest conceptual motivation:

```text
Cao & Ma:
identify / suppress frequency shortcut behavior

CSMR:
probe LOW/MID/HIGH interventions online
→ select the one that most harms PAD margin
→ train against that sample-specific counterfactual
```

Do not claim that CSMR is the first work to recognize frequency shortcut.

The differentiation must be in the **selection target and robustness objective**.

---

## 3.2. FSDA — Fourier-Based Frequency Space Disentanglement and Augmentation

**Y. Yu, Z. Du, H. Luo, C. Xiao, and J. Hu, “Fourier-Based Frequency Space Disentanglement and Augmentation for Generalizable Face Anti-Spoofing,” IEEE JBHI, 2025.**  
DOI: `10.1109/JBHI.2024.3417404`

Previous reference notes:

- Fourier amplitude is used for low-level appearance/texture manipulation;
- frequency space is disentangled/augmented;
- consistency/distillation is used for generalization.

### Relationship to CSMR

FSDA is important because a reviewer may ask:

> Is CSMR simply frequency augmentation plus consistency?

Ablation must separate:

```text
Random spectral augmentation
Generic consistency
Worst-band CE only
Full CSMR
```

The intended distinction:

```text
FSDA-style family:
augment / mix / disentangle spectral information
+ consistency or distillation

CSMR:
evaluate current PAD decision under several controlled spectral counterfactuals
→ select worst label-aligned margin
→ penalize harmful PAD-margin degradation
```

---

## 3.3. Chen et al. — High/Low Frequency Fusion

**B. Chen, W. Yang, and S. Wang, “Face Anti-Spoofing by Fusing High and Low Frequency Features for Advanced Generalization Capability,” IEEE MIPR, 2020.**  
DOI: `10.1109/MIPR49039.2020.00048`

Historical role:

- supports the idea that frequency components can contain useful spoof evidence;
- motivates explicit frequency decomposition;
- provides precedent for spatial/frequency multi-stream designs.

Current role:

> motivation / historical baseline, not the closest method to CSMR.

---

## 3.4. FreqSpatialTemporalNet

**Y. Huang, W. Zhang, and J. Wang, “Deep Frequent Spatial Temporal Learning for Face Anti-Spoofing,” arXiv:2002.03723, 2020.**

Historical role:

- demonstrates that spectrum images can be learned by a CNN;
- supports early E2/E3 motivation.

Not directly comparable:

- temporal/multi-frame scope differs from current single-frame RGB CSMR.

---

## 3.5. Bi-FPNFAS

**K. Roy et al., “Bi-Directional Feature Pyramid Network for Pixel-Wise Face Anti-Spoofing by Leveraging Fourier Spectra,” Sensors, 2021.**  
DOI: `10.3390/s21082799`

Key relevance:

- Fourier spectra used through auxiliary/self-supervised training;
- useful precedent that frequency can supervise representation without necessarily being the only inference cue.

This paper is more relevant to the current story than before because CSMR also treats frequency as a **training signal**, although the mechanism is different.

---

## 3.6. Zhang & Xiang — DWT-LBP-DCT

**W. Zhang and S. Xiang, “Face anti-spoofing detection based on DWT-LBP-DCT features,” Signal Processing: Image Communication, 2020.**  
DOI: `10.1016/j.image.2020.115990`

Historical role:

- DCT / multiresolution handcrafted precedent;
- useful for motivation of M4/block-frequency representations.

Current paper should not make DCT itself central.

---

# 4. Recent frequency-aware / fusion references

## 4.1. Oculus — ICCVW 2025

**V. W. de Dravo et al., “Oculus: Hierarchical Face Spoof Detection via Frequency-Enhanced Vision Transformers with Group-Aware Classification and Post-Fusion Attention,” ICCVW 2025.**

Prior review identified:

```text
Spatial feature
+
Frequency-domain feature
→ concat
→ fusion head
→ post-fusion attention
```

Current use:

- contextual reference for explicit frequency fusion;
- useful contrast against CSMR’s no-frequency-inference design.

Do not present as same protocol / direct baseline.

---

## 4.2. Niu & Lin — Neural Networks 2026

**Y. Niu and X. Lin, “Similarity-aware contrastive learning for face anti-spoofing via frequency enhancement and reconstruction,” Neural Networks, 2026.**  
DOI: `10.1016/j.neunet.2026.108734`

Reported components in the previous review:

- Frequency Adaptive Enhancement Module;
- high-frequency reconstruction;
- similarity-aware contrastive learning.

Relationship:

> further evidence that frequency integration can be adaptive and training-objective-driven rather than a fixed extra branch.

---

## 4.3. DEFuseNet — Neurocomputing 2026

**R. P. Singh, R. Dash, and R. K. Mohapatra, “DEFuseNet: A domain-enhanced fusion network for generalizable face anti-spoofing,” Neurocomputing, 2026.**  
DOI: `10.1016/j.neucom.2026.134092`

Prior notes:

- RGB + LBP texture cues;
- modulated fusion;
- domain perturbation;
- modality consistency.

Current role:

- contextual evidence for domain-aware perturbation and adaptive cue use;
- not a spectral counterpart to CSMR.

---

## 4.4. Sun et al. — PRL 2025

**R. Sun et al., “Robust multimodal face anti-spoofing via frequency-domain feature refinement and aggregation,” Pattern Recognition Letters, 2025.**  
DOI: `10.1016/j.patrec.2025.07.003`

Historical role:

- frequency refinement;
- RGB-guided feature interaction;
- progressive fusion.

Current contrast:

```text
feature refinement / fusion at inference
vs
CSMR training-time-only spectral robustness
```

---

# 5. Closest edge / lightweight conceptual work

## Ali et al. — Scientific Reports 2026

**F. A. Ali, S. Mali, R. Mahakud, and G. Yadav, “AI-enabled smart surveillance system for secure monitoring and authentication,” Scientific Reports, 2026.**  
DOI: `10.1038/s41598-026-52387-w`

Prior review notes:

- lightweight spatial branch;
- MiniFASNet;
- Fourier frequency branch;
- edge deployment;
- Raspberry Pi 3B+.

Current importance depends on final backbone.

### If final backbone = MiniFASNetV2

This becomes a particularly relevant architecture/deployment reference because:

```text
Ali et al.:
MiniFASNet + explicit Fourier cue + edge

Project:
MiniFASNetV2 + CSMR training
→ spatial-only MiniFASNetV2 inference
```

Useful contrast:

> same lightweight/edge family, different use of frequency.

### If final backbone = MNV3-S

Ali et al. remains the closest conceptual lightweight spatial+frequency reference, but backbone differs.

Do not direct-compare their private-dataset numbers with this project.

---

# 6. Literature mapping to the historical M0–M4 study

M0–M4 remain useful in the paper as **diagnostic development evidence**, not the proposed final architecture.

| ID | Representation | Literature/design principle | Current role |
|---|---|---|---|
| M0 | Global DCT + GAP | basic explicit spectrum | control |
| M1 | DCT + Pool4×4 | retain coarse spectral location | diagnosis |
| M2 | Coord-DCT | frequency coordinates have semantics | diagnosis |
| M3 | Band-aware DCT | band reliability differs | strongest conceptual bridge to CSMR bands |
| M4 | Block-DCT | preserve frequency identity + spatial locality | diagnosis |

Key transition:

```text
M0–M4 ask:
HOW should frequency be represented?

Observed answer:
none is robust enough as the final inference cue

CSMR asks:
HOW can frequency be used to make the spatial decision more robust?
```

That distinction should be explicit in Introduction/Discussion.

---

# 7. Literature mapping to CSMR components

| CSMR component | Related literature theme | What remains project-specific |
|---|---|---|
| LOW/MID/HIGH perturbation | band decomposition / frequency masking | controlled fixed smooth counterfactual family |
| sample-adaptive worst band | adaptive frequency treatment | selection by current label-aligned PAD margin |
| clean + worst CE | robust/adversarial-style training | worst spectral counterfactual is task-specific |
| harmful-only margin loss | consistency / robustness regularization | penalizes only degradation of binary PAD decision margin |
| zero FFT at inference | auxiliary frequency supervision precedent | frequency branch fully absent at deployment |

This table is useful for novelty defense.

---

# 8. Reviewer attack matrix

## Attack 1

> “This is just spectral augmentation.”

Response evidence required:

```text
Random spectral augmentation
vs
Full CSMR
```

If Full CSMR does not clearly beat random augmentation, novelty weakens.

## Attack 2

> “This is just consistency regularization.”

Response:

```text
Generic KL/logit consistency
vs
harmful PAD-margin regularization
```

## Attack 3

> “Worst-band selection is unnecessary.”

Response:

```text
random band
or average all bands
vs
worst-band CE
vs
Full CSMR
```

## Attack 4

> “The gain comes from a stronger PAD-pretrained backbone.”

Response:

- paired Vanilla vs CSMR within the same backbone;
- MNV3 transfer evidence;
- MiniFAS transfer evidence.

Do not use absolute MiniFAS vs MNV3 numbers as the only CSMR evidence.

## Attack 5

> “Frequency robustness increases compute.”

Response:

- FFT/counterfactual generation is training-only;
- inference graph is identical to Vanilla within a backbone;
- report deployment latency/FPS for Vanilla and CSMR checkpoints to verify no extra inference path.

---

# 9. Backbone-dependent literature framing

## CASE A — Final MiniFASNetV2

Paper positioning:

> CSMR is demonstrated on a lightweight PAD-specific backbone with public PAD pretraining and a canonical `2.7× / 80×80` recipe.

Literature emphasis:

1. Ali et al. for lightweight MiniFAS + frequency + edge context;
2. Cao & Ma for frequency shortcut;
3. FSDA for frequency augmentation/generalization;
4. MiniFASNet lineage / official implementation for architecture provenance.

Need explicitly disclose:

- final classifier reinitialized;
- official checkpoint is PAD-pretrained;
- absolute comparison with ImageNet MNV3 includes a pretraining confound.

## CASE B — Final MNV3-S

Paper positioning:

> CSMR improves a generic ImageNet-pretrained lightweight mobile backbone without requiring PAD-specific inference architecture.

Literature emphasis:

1. Cao & Ma;
2. FSDA;
3. Bi-FPNFAS / auxiliary spectral supervision;
4. lightweight mobile/edge context.

Potential advantage in paper narrative:

> stronger isolation from PAD-specific pretraining.

---

# 10. Direct vs contextual comparison rules

Only call results **directly comparable** if:

- same dataset;
- same split/protocol;
- same modality;
- same threshold rule;
- same train/test role;
- no target-domain fine-tuning.

Otherwise say:

> contextual reference — not directly comparable.

Especially do not directly compare:

- private Ali dataset metrics;
- OULU Protocol IV numbers;
- challenge leaderboard ACER;
- OCIM leave-one-domain-out;
- this project’s CelebA→LCC development stress test.

---

# 11. Claims that literature supports

Safe:

> Frequency cues are relevant to Face PAD.

Safe:

> Frequency information can also encode domain-specific shortcuts.

Safe:

> Frequency augmentation / masking / disentanglement has precedent in generalizable FAS.

Safe:

> Training objectives can use frequency as supervision or perturbation.

Needs project evidence:

> Worst-band counterfactual selection improves robustness.

Needs ablation:

> Harmful-only PAD-margin regularization is better than generic consistency.

Needs final unseen confirmation:

> CSMR improves unseen-domain generalization.

---

# 12. Claims to avoid

Avoid:

> Frequency is inherently domain invariant.

Avoid:

> High frequency always indicates spoof.

Avoid:

> CSMR is the first frequency augmentation method for FAS.

Avoid:

> MiniFASNet is categorically better than MNV3.

Avoid:

> LCC is an untouched final benchmark.

Avoid:

> Test-2000 remains unseen.

---

# 13. Recommended Related Work structure

## 2.1 Lightweight Face PAD

Discuss:

- MobileNet-family baseline;
- MiniFASNet-family PAD-specific models;
- edge constraints.

## 2.2 Frequency-aware Face PAD

Discuss:

- DCT/Fourier feature branches;
- high/low-frequency decomposition;
- auxiliary Fourier supervision;
- frequency-enhanced fusion.

## 2.3 Generalizable Face PAD and spectral shortcut

Center:

- frequency shortcut view;
- FSDA;
- domain perturbation / consistency.

## 2.4 Positioning of CSMR

Close Related Work with:

```text
Prior work:
frequency representation / fusion / augmentation / masking

CSMR:
sample-adaptive spectral counterfactual selection
based on PAD decision margin
+ harmful-only margin robustness
+ spatial-only inference
```

Do not oversell “first”.

---

# 14. Suggested novelty paragraph skeleton

> Existing frequency-aware Face PAD methods have demonstrated that spectral cues can be informative, but recent work also shows that such cues may encode domain-specific shortcuts. Rather than adding frequency as another inference representation, we use controlled spectral perturbations only during training. For each sample, CSMR evaluates a small set of smooth band-limited amplitude interventions, selects the intervention that most reduces the label-aligned Real-versus-Attack margin, and regularizes only harmful margin degradation. The resulting deployment model remains unchanged and requires no FFT or frequency branch at inference.

Finalize only after final experiments.

---

# 15. IEEE references carried forward

```text
[1] B. Chen, W. Yang, and S. Wang, "Face Anti-Spoofing by Fusing High and Low Frequency Features for Advanced Generalization Capability," in Proc. IEEE MIPR, 2020, pp. 199–204, doi: 10.1109/MIPR49039.2020.00048.

[2] Y. Huang, W. Zhang, and J. Wang, "Deep Frequent Spatial Temporal Learning for Face Anti-Spoofing," arXiv:2002.03723, 2020.

[3] K. Roy et al., "Bi-Directional Feature Pyramid Network for Pixel-Wise Face Anti-Spoofing by Leveraging Fourier Spectra," Sensors, vol. 21, no. 8, 2799, 2021, doi: 10.3390/s21082799.

[4] H. Kim, J. Lee, Y. Jeong, H. Jang, and Y. Yoo, "Advancing Cross-Domain Generalizability in Face Anti-Spoofing: Insights, Design, and Metrics," CVPR Workshops, 2024, arXiv:2406.12258.

[5] W. Zhang and S. Xiang, "Face anti-spoofing detection based on DWT-LBP-DCT features," Signal Processing: Image Communication, vol. 89, 115990, 2020, doi: 10.1016/j.image.2020.115990.

[6] J. Cao and C. Ma, "Towards Generalized Face Anti-Spoofing from a Frequency Shortcut View," WACV, 2025, pp. 1005–1015, doi: 10.1109/WACV61041.2025.00107.

[7] Y. Yu, Z. Du, H. Luo, C. Xiao, and J. Hu, "Fourier-Based Frequency Space Disentanglement and Augmentation for Generalizable Face Anti-Spoofing," IEEE Journal of Biomedical and Health Informatics, vol. 29, no. 8, pp. 5413–5423, 2025, doi: 10.1109/JBHI.2024.3417404.

[8] V. W. de Dravo et al., "Oculus: Hierarchical Face Spoof Detection via Frequency-Enhanced Vision Transformers with Group-Aware Classification and Post-Fusion Attention," ICCV Workshops, 2025.

[9] F. A. Ali, S. Mali, R. Mahakud, and G. Yadav, "AI-enabled smart surveillance system for secure monitoring and authentication," Scientific Reports, vol. 16, 21686, 2026, doi: 10.1038/s41598-026-52387-w.

[10] Y. Niu and X. Lin, "Similarity-aware contrastive learning for face anti-spoofing via frequency enhancement and reconstruction," Neural Networks, vol. 199, 108734, 2026, doi: 10.1016/j.neunet.2026.108734.

[11] R. P. Singh, R. Dash, and R. K. Mohapatra, "DEFuseNet: A domain-enhanced fusion network for generalizable face anti-spoofing," Neurocomputing, vol. 696, 134092, 2026, doi: 10.1016/j.neucom.2026.134092.

[12] R. Sun, F. Wang, X. Yu, X. Gao, and X. Zhang, "Robust multimodal face anti-spoofing via frequency-domain feature refinement and aggregation," Pattern Recognition Letters, vol. 197, pp. 31–36, 2025, doi: 10.1016/j.patrec.2025.07.003.
```

---

# 16. Final literature-to-experiment map

```text
Chen / Huang / Roy / Zhang
→ frequency cue is worth studying

M0–M4 project study
→ explicit representation is not reliably robust

Cao & Ma
→ frequency shortcut is a credible failure mode

FSDA
→ frequency augmentation/generalization is established territory

CSMR
→ worst PAD-margin counterfactual
   + harmful-only margin robustness

MiniFAS / MNV3 paired evidence
→ method transfer

CASIA untouched
→ final unseen-domain claim
```

This is the literature narrative to preserve.
