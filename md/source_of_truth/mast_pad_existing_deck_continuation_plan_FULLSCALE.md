# MAST-PAD Continuation Deck Plan
## Full-scale / two-target research section for the existing Secure Face Authentication deck

_Last updated: 2026-10-07_

> **Deck reporting convention:** headline metric is **AUC only**.
>
> **Project values:** always use **LCC Combined** and **CASIA reconstructed Combined** in the main result slides.
>
> **Primary checkpoint:** `R7_SC_FULL_crop15`.

---

# 0. Integration rule

Keep slides 1–14 as system/background context. Replace the old PAD research continuation with the full-scale MAST-PAD section. Move “Thank you” to the end.

Recommended final sequence:

```text
Slides 1–14: existing system/background
Slides 15–26: full-scale MAST-PAD research section
Slide 27: Thank you / Q&A
```

---

# 1. Minimal patches before the MAST-PAD section

## Slide 7

Do not present MobileNetV3 as the final method. Replace with:

```text
Early model options / historical exploration

Final deployed PAD path:
SCRFD crop1.5
→ 80×80 BGR
→ MiniFASNetV2
→ Real / Attack
```

## Slide 9

Mark LogSumExp as historical or move to backup:

```text
Historical rule from the older 3-class formulation.
Final MAST-PAD uses a binary Real/Attack aligned margin.
```

---

# 2. New continuation section

## Slide 14 — MAST-PAD divider

Subtitle:

```text
Margin-Aware Spectral Training for Face Presentation Attack Detection
```

Badges:

```text
0.434M params
LCC AUC 0.860
CASIA AUC 0.935
```

Small footer:

```text
Spectral operations are training-only.
```

---

# Slide 15 — Why Stage 3 still needs research

## Title
**Why lightweight PAD still fails under domain shift**

Content:

```text
Source-domain PAD can look excellent.
Deployment changes camera, lighting, media, compression and spoof texture.
A practical edge model must generalize without becoming large.
```

Visual:

```text
CelebA-Spoof source
        ↓
  frozen PAD model
   ↙          ↘
LCC-FASD     CASIA-FASD
```

Speaker message:

> “The research problem is not source accuracy. It is preserving useful PAD decisions when acquisition and spoof media change.”

---

# Slide 16 — MAST-PAD positioning

| Direction | Typical idea | Limitation | MAST-PAD choice |
|---|---|---|---|
| Lightweight CNN | compact spatial model | fragile texture cues | keep 0.434M inference |
| Frequency-aware PAD | FFT/DCT branch / fusion | added deployment path | frequency only in training |
| Large DG models | stronger representation | high parameter cost | retain MiniFASNetV2 |
| Robust training | hard / shifted views | may supervise harmless perturbations | margin-aware harmful gate |

Bottom takeaway:

```text
MAST-PAD = lightweight spatial inference + training-only spectral stress
```

---

# Slide 17 — Why frequency during training?

Attack media may introduce:

```text
Moiré patterns
pixel-grid artifacts
resampling traces
micro-texture changes
print/display spectral shifts
```

But spectral cues may become shortcuts.

Key sentence:

```text
Use frequency to stress the decision, not as a permanent deployed feature.
```

Figure:

```text
fig_frequency_counterfactual_examples.png
```

---

# Slide 18 — MAST-PAD in one diagram

```text
Clean crop
├── LOW attenuation
├── MID attenuation
└── HIGH attenuation
      ↓
shared MiniFASNetV2
      ↓
label-aligned margins
      ↓
select lowest-margin view
      ↓
harmful if m_worst < m_clean
      ↓
apply spectral CE only if harmful
```

Figure:

```text
fig_mast_pad_method_flow.png
```

Footer:

```text
Inference remains: crop1.5 → 80×80 → MiniFASNetV2 → Real / Attack
```

---

# Slide 19 — Margin-aware harmful gating

```text
d = z_real - z_attack

Real   → m = +d
Attack → m = -d

k* = argmin_k m_k
h = 1[m_worst < m_clean]

L = 0.75 CE_clean + 0.25 CE_harmful + L_aux
```

Speaker message:

> “MAST-PAD reacts to margin degradation before the perturbation necessarily becomes a classification error.”

---

# Slide 20 — Primary full-scale protocol

## Title
**Full-scale source training, frozen cross-domain evaluation**

| Component | Setting |
|---|---|
| Source train | full eligible CelebA-Spoof Train, 484,561 usable |
| Source selection | full usable CelebA Test-as-Val |
| Backbone | MiniFASNetV2 |
| Input | crop1.5, 80×80 BGR |
| Optimizer | SGD, lr 0.005, momentum 0.9, wd 5e-4 |
| Batch | 256 |
| Schedule | max12, milestones [4,8], gamma 0.2 |
| Early stop | starts epoch 9, patience 3 |
| Target 1 | LCC-FASD → Combined AUC |
| Target 2 | CASIA-FASD extracted-image copy → reconstructed Combined AUC |
| Target adaptation | none |

Bottom note:

```text
All three methods share the same source setup and deployed 0.434M architecture.
```

Caveat in speaker notes:

> CelebA Test is used as Test-as-Val; CASIA is a reconstructed extracted-image evaluation, not claimed as the official subject-disjoint full-video benchmark.

---

# Slide 21 — Main full-scale result

## Title
**Cross-domain AUC on two target datasets**

Use only AUC:

| Method | LCC Combined AUC ↑ | CASIA Combined AUC ↑ | Params |
|---|---:|---:|---:|
| Clean | 0.846 | 0.920 | 0.434M |
| WBST | **0.861** | 0.922 | 0.434M |
| **MAST-PAD** | 0.860 | **0.935** | **0.434M** |

Key messages:

```text
LCC: WBST and MAST-PAD are effectively near-tied; both improve over Clean.
CASIA: MAST-PAD gives the strongest cross-domain AUC (+1.48 pp vs Clean).
```

Figure:

```text
fig_main_ablation_two_target_auc.png
```

Do **not** claim that MAST-PAD beats WBST on every target.

---

# Slide 22 — What does the harmful gate select?

Full-scale MAST-PAD diagnostics:

```text
best checkpoint = epoch 7
training stopped = epoch 11

harmful fraction = 0.832135
margin drop      = 0.391421
flip rate        = 0.003077

LOW  = 0.426452
MID  = 0.273035
HIGH = 0.300513
```

Interpretation:

```text
Margin degradation is common.
Prediction flips are rare.
No single frequency band dominates all samples.
```

Figure:

```text
fig_mast_pad_training_diagnostics.png
```

---

# Slide 23 — Published literature context

## Title
**Competitive AUC with a 0.434M deployment model**

Use **two compact tables on the same slide** or split into 23A/23B if space is insufficient.

### LCC-FASD context

| Method | LCC AUC ↑ | Params ↓ | Model size | Input |
|---|---:|---:|---:|---:|
| Graph for Transformer Feature | 0.833 | ~10.84M | — | —¹ |
| AENet | **0.868** | 11.22M | — | 128×128 RGB² |
| **MAST-PAD** | **0.860** | **0.434M** | **~1.74 MB FP32 weights³** | **80×80 BGR** |

Takeaway:

```text
MAST-PAD > Graph Transformer numerically.
MAST-PAD is only 0.8 pp below AENet with ~25.8× fewer parameters.
The deployed MAST-PAD weights are only ~1.74 MB in FP32 before container/export overhead.
```

Notes:

```text
¹ Graph Transformer reports patch size = 16 and n = 100 patches, but does not explicitly state
  a single input-image resolution in the paper; do not infer one on the slide.
² 128×128 RGB is the preprocessing used by the lightweight-FAS benchmark that reports the AENet LCC row.
³ Approximate weights-only size = 434,434 parameters × 4 bytes; not an ONNX/checkpoint file-size claim.
```

### CASIA-FASD cross-domain context

| Method | Venue | Target setting | CASIA AUC ↑ | Params / backbone ↓ | Model size | Input / patch |
|---|---|---|---:|---|---:|---:|
| MADDG | CVPR'19 | O&M&I → C | 0.845 | ~3.35M MADDG-based backbone⁴ | — | — |
| AMEL | ACM MM'22 | O&M&I → C | 0.944 | MADDG backbone + lightweight experts; total not reported | — | 256×256×3 |
| PatchNet | CVPR'22 | O&M&I → C | 0.946 | ResNet-18 encoder (~11.69M std. backbone⁵) | ~44.7 MB std. backbone⁵ | 160×160 patch |
| GAC-FAS | CVPR'24 | O&M&I → C | 0.952 | ResNet-18 (~11.69M std. backbone⁵) | ~44.7 MB std. backbone⁵ | 256×256 |
| SA-FAS | CVPR'23 | O&M&I → C | 0.954 | ResNet-18 (~11.69M std. backbone⁵) | ~44.7 MB std. backbone⁵ | 256×256 |
| **MAST-PAD** | ours | CelebA → reconstructed C Combined | **0.935** | **0.434M** | **~1.74 MB FP32 weights³** | **80×80** |

Takeaway:

```text
MAST-PAD exceeds MADDG and stays within ~0.9–1.9 pp of stronger modern DG references.
The accuracy gap is obtained with a much smaller 0.434M deployment model and 80×80 input.
```

Mandatory caveat:

```text
Different source-domain and evaluation protocols; contextual comparison only.
CASIA literature rows use the standard O&M&I → C protocol, whereas our result uses frozen CelebA-trained
weights on the reconstructed CASIA image-copy Combined population.
```

Parameter / size footnotes:

```text
⁴ 3.35M is reported for the MADDG-based architecture in the SSDG CVPR 2020 architecture comparison;
  treat it as an architecture/backbone reference, not a newly measured MADDG checkpoint size.
⁵ The cited FAS papers specify ResNet-18. 11,689,512 params and 44.7 MB are the standard torchvision
  ResNet-18 reference values; method-specific heads/modules can change the exact total model size.
```

Optional figures:

```text
fig_lcc_auc_vs_params_context.png
fig_casia_auc_literature_context.png
```

### Slide 23 source audit — include in speaker notes or appendix

#### Graph for Transformer Feature — ESANN 2023
Published LCC value used: `AUC = 0.833`; paper reports approximately `10.84M` parameters and 50 training epochs.

Architecture details that are safe to quote:

```text
patch size = 16
number of patches n = 100
best model ≈ 10.84M parameters
```

The paper does **not** explicitly state a single image-resolution value that should be placed in the comparison table, so input size is left as `—`.

- DOI: https://doi.org/10.14428/esann/2023.ES2023-14
- Direct PDF: https://www.esann.org/sites/default/files/proceedings/2023/ES2023-14.pdf
- Numerical location: Table 1 for LCC AUC; Sec. 2.5 for ~10.84M parameters; Sec. 3.1 for 50 epochs.

#### AENet
LCC benchmark value used: `AUC = 0.868`; `11.22M` parameters.

- Original AENet / CelebA-Spoof ECCV paper: https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/1485_ECCV_2020_paper.php
- Official CelebA-Spoof repository: https://github.com/ZhangYuanhan-AI/CelebA-Spoof
- LCC benchmark value: https://github.com/kprokofi/light-weight-face-anti-spoofing
- Benchmark preprocessing used for the reported lightweight comparison: `128×128 RGB`.
- Wording requirement: the `0.868` LCC number comes from the benchmark, not from the original ECCV headline table.

#### MADDG — CVPR 2019
CASIA cross-domain value used: `O&M&I → C AUC = 84.51%`.

- Official page: https://openaccess.thecvf.com/content_CVPR_2019/html/Shao_Multi-Adversarial_Discriminative_Deep_Domain_Generalization_for_Face_Presentation_Attack_Detection_CVPR_2019_paper.html
- Official PDF: https://openaccess.thecvf.com/content_CVPR_2019/papers/Shao_Multi-Adversarial_Discriminative_Deep_Domain_Generalization_for_Face_Presentation_Attack_Detection_CVPR_2019_paper.pdf
- Convenient re-verification: PatchNet CVPR 2022 Table 8 reproduces `84.51`.
- Architecture-size cross-check: SSDG CVPR 2020 reports the **MADDG-based** feature generator at `3.35M` parameters (`47.59G` FLOPs in that paper's counting convention): https://vipl.ict.ac.cn/en/resources/codes/2020/202205/P020220601487076753217.pdf
- Do not present `3.35M` as an exact serialized checkpoint size from the original MADDG paper; label it `MADDG-based architecture`.

#### AMEL — ACM Multimedia 2022
CASIA cross-domain value used: `O&M&I → C AUC = 94.39%`.

Paper-reported implementation details:

```text
backbone = same MADDG (M) backbone used by prior DG-FAS works
input = 256×256×3 RGB
DSE = lightweight residual expert blocks
exact total parameter count = not reported in the paper
```

- arXiv: https://arxiv.org/abs/2207.09868
- Direct PDF: https://arxiv.org/pdf/2207.09868
- DOI / ACM record: https://doi.org/10.1145/3503161.3547769
- Numerical location: Table 1; implementation details immediately before Table 1.

#### PatchNet — CVPR 2022
CASIA cross-domain value used: `O&M&I → C AUC = 94.58%`.

Paper-reported implementation details:

```text
patch feature encoder = ResNet-18
fixed patch crop size = 160×160
maximum training = 200 epochs
```

- Official page: https://openaccess.thecvf.com/content/CVPR2022/html/Wang_PatchNet_A_Simple_Face_Anti-Spoofing_Framework_via_Fine-Grained_Patch_Recognition_CVPR_2022_paper.html
- Official PDF: https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_PatchNet_A_Simple_Face_Anti-Spoofing_Framework_via_Fine-Grained_Patch_Recognition_CVPR_2022_paper.pdf
- Author-hosted PDF: https://www.cs.nthu.edu.tw/~lai/pdf/publications/2022/PatchNet_A_Simple_Face_Anti-Spoofing_Framework_via_Fine-Grained_Patch_Recognition.pdf
- Numerical location: Table 8 for the DG result; Sec. 4.2 for ResNet-18 / patch size / training setup.
- The paper does not publish a definitive total PatchNet parameter count; use `ResNet-18 encoder` rather than inventing an exact total.

#### SA-FAS — CVPR 2023
CASIA cross-domain value used: `O&M&I → C AUC = 95.37%`.

Paper-reported setup:

```text
backbone = ResNet-18
input = 256×256
training = 100 epochs for the standard settings (with a longer schedule for ICM→O)
```

- Official page: https://openaccess.thecvf.com/content/CVPR2023/html/Sun_Rethinking_Domain_Generalization_for_Face_Anti-Spoofing_Separability_and_Alignment_CVPR_2023_paper.html
- Official PDF: https://openaccess.thecvf.com/content/CVPR2023/papers/Sun_Rethinking_Domain_Generalization_for_Face_Anti-Spoofing_Separability_and_Alignment_CVPR_2023_paper.pdf
- ResNet-18 parameter count shown in the table is a **standard-backbone reference**, not a total-parameter figure explicitly reported by SA-FAS.

#### GAC-FAS — CVPR 2024
CASIA cross-domain value used: `O&M&I → C AUC = 95.16%`.

Paper-reported setup:

```text
backbone = ImageNet-pretrained ResNet-18
input = 256×256
```

- Official page: https://openaccess.thecvf.com/content/CVPR2024/html/Le_Gradient_Alignment_for_Cross-Domain_Face_Anti-Spoofing_CVPR_2024_paper.html
- Official PDF: https://openaccess.thecvf.com/content/CVPR2024/papers/Le_Gradient_Alignment_for_Cross-Domain_Face_Anti-Spoofing_CVPR_2024_paper.pdf
- Official implementation: https://github.com/Leminhbinh0209/CVPR24-FAS
- ResNet-18 parameter count shown in the table is a **standard-backbone reference**, not a total-parameter figure explicitly reported by GAC-FAS.

#### Standard ResNet-18 parameter / weight-size reference
Use only to contextualize methods whose papers explicitly state a ResNet-18 backbone.

- Torchvision official docs: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
- Standard ResNet-18: `11,689,512` parameters, `44.7 MB` published weight-file size.
- Do not imply this is the exact full size of PatchNet / SA-FAS / GAC-FAS after their task-specific heads or modules are added.

#### MAST-PAD — project values used on the slide

```text
LCC Combined AUC   = 0.860272 → 0.860
CASIA Combined AUC = 0.935199 → 0.935
deployment params  = 434,434 → 0.434M
input              = 80×80 BGR
approx FP32 weights-only size = 434,434 × 4 bytes ≈ 1.74 MB
```

Do not call `~1.74 MB` the ONNX/checkpoint file size; serialization, metadata and buffers change the actual artifact size.

---

# Slide 24 — Efficiency and deployment

```text
SCRFD bbox
→ crop1.5
→ 80×80 BGR
→ MiniFASNetV2
→ Real / Attack
```

Training-only:

```text
FFT
LOW/MID/HIGH masks
worst-view selection
harmful gate
spectral CE
```

Complexity:

```text
deployment params = 434,434
training params   = 441,658
MACs              = 40,810,892
reported GFLOPs   = 0.081621784
```

Main claim:

```text
MAST-PAD adds no spectral branch or model parameters at inference.
```

---

# Slide 25 — Supporting matched100K development evidence

## Title
**The same idea emerged before full-scale training**

| Method | 100K LCC Combined AUC |
|---|---:|
| Clean | 0.848 |
| WBST | 0.852 |
| MAST-PAD | **0.868** |

Message:

```text
The matched100K study motivated freezing MAST-PAD.
Full-scale results are the primary paper evidence.
```

Do not present this slide if presentation time is tight; move it to backup.

---

# Slide 26 — Takeaways and limitations

Takeaways:

```text
1. Frequency is a training-time stressor, not a deployed branch.
2. Both spectral strategies improve full-scale LCC AUC over Clean.
3. MAST-PAD gives the strongest CASIA Combined AUC: 0.935.
4. Deployment remains only 0.434M parameters.
5. Published comparisons suggest a favorable accuracy–compactness trade-off.
```

Limitations:

```text
one source-training seed
CelebA Test used as Test-as-Val
literature source-domain protocols differ
LCC Combined pools multiple splits
CASIA uses reconstructed extracted-image copies, not verified official full-video protocol
training cost increases due to spectral candidate forwards
MiniFASNetV2-specific evidence
```

Closing line:

```text
Frequency helps most when it exposes fragile decisions during training,
not when it is blindly added as a permanent inference branch.
```

---

# 3. Speaker narrative

```text
Slide 15: source accuracy is not the deployment problem; domain shift is.
Slide 16: robustness usually costs capacity or inference complexity.
Slide 17: frequency is informative but can become a shortcut.
Slide 18: MAST-PAD turns spectral perturbations into controlled stress tests.
Slide 19: the current PAD margin decides whether supervision is useful.
Slide 20: full-scale CelebA training is frozen before two target-domain evaluations.
Slide 21: LCC shows spectral robustness; CASIA shows the clearest MAST-PAD advantage.
Slide 22: the harmful gate catches margin degradation before label flips.
Slide 23: 0.434M MAST-PAD remains competitive with much heavier published references.
Slide 24: deployment stays spatial-only.
Slide 25: matched100K is supporting development evidence.
Slide 26: concise takeaways + limitations.
```

---

# 4. Final checklist

- [ ] full-scale is described as the primary experiment
- [ ] `R7_SC_FULL_crop15` is the primary checkpoint
- [ ] headline metric is AUC only
- [ ] all project target values use Combined
- [ ] LCC main values are Clean 0.846 / WBST 0.861 / MAST-PAD 0.860
- [ ] CASIA main values are Clean 0.920 / WBST 0.922 / MAST-PAD 0.935
- [ ] no slide claims MAST-PAD beats WBST on LCC Combined
- [ ] LCC comparison uses only Graph Transformer + AENet + MAST-PAD
- [ ] CASIA comparison uses published peer-reviewed references
- [ ] slide/source notes contain direct paper links
- [ ] protocol caveat is visible near literature tables
- [ ] CASIA extracted-image limitation is stated
- [ ] 100K appears only as supporting development evidence
- [ ] no FFT/frequency branch is shown at inference
