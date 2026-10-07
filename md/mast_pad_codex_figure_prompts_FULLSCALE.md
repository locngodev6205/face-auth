# Codex Figure Generation Prompts for the MAST-PAD Slide Section
## Full-scale / two-target figure pack

_Last updated: 2026-10-07_

Output folder:

```text
slide_figures_mast_pad_fullscale/
```

Primary artifacts:

```text
fullscale C / P3_SF / R7_SC experiment
CASIA train/test extracted-image evaluation report
```

Do not retrain models.

---

# Prompt 1 — Main two-target full-scale AUC chart

```text
Create:
  slide_figures_mast_pad_fullscale/fig_main_ablation_two_target_auc.png
  slide_figures_mast_pad_fullscale/fig_main_ablation_two_target_auc.pdf

Plot grouped bars using AUC only.

LCC Combined:
  Clean     = 0.846011
  WBST      = 0.860831
  MAST-PAD  = 0.860272

CASIA Combined:
  Clean     = 0.920398
  WBST      = 0.922264
  MAST-PAD  = 0.935199

Requirements:
- x groups: LCC Combined, CASIA Combined
- y-axis: AUC
- annotate all values to 3 decimals
- subtitle: "Full-scale CelebA source training; frozen cross-domain evaluation"
- callout on LCC: "WBST and MAST-PAD nearly tied"
- callout on CASIA: "MAST-PAD +1.48 pp vs Clean"
- use consistent Clean / WBST / MAST-PAD labels
- PNG 300 DPI + vector PDF
```

---

# Prompt 2 — LCC literature AUC vs parameters

```text
Create:
  slide_figures_mast_pad_fullscale/fig_lcc_auc_vs_params_context.png
  slide_figures_mast_pad_fullscale/fig_lcc_auc_vs_params_context.pdf

Data:
  Graph Transformer: params_M=10.84, AUC=0.833
  AENet:            params_M=11.22, AUC=0.868
  MAST-PAD:         params_M=0.434434, AUC=0.860272

Plot:
- x-axis Parameters (M), log scale
- y-axis LCC AUC
- label all points
- emphasize MAST-PAD
- note: "Contextual references; protocols differ."

Reference provenance to record in manifest:
Graph Transformer:
  https://www.esann.org/sites/default/files/proceedings/2023/ES2023-14.pdf
AENet model:
  https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/1485_ECCV_2020_paper.php
AENet LCC number:
  https://github.com/kprokofi/light-weight-face-anti-spoofing
```

---

# Prompt 3 — CASIA literature AUC context chart

```text
Create:
  slide_figures_mast_pad_fullscale/fig_casia_auc_literature_context.png
  slide_figures_mast_pad_fullscale/fig_casia_auc_literature_context.pdf

Use AUC only:
  MADDG      = 0.8451
  MAST-PAD   = 0.935199
  AMEL       = 0.9439
  PatchNet   = 0.9458
  GAC-FAS    = 0.9516
  SA-FAS     = 0.9537

Order by AUC ascending or use MAST-PAD as emphasized reference.

Add protocol labels:
  MADDG/AMEL/PatchNet/GAC-FAS/SA-FAS: O&M&I → C
  MAST-PAD: CelebA-full → reconstructed CASIA Combined

Footnote:
  "Different source-domain protocols; contextual comparison only."
  "MAST-PAD deployment: 0.434M params."

Sources to include in figure manifest:
MADDG:
  https://openaccess.thecvf.com/content_CVPR_2019/html/Shao_Multi-Adversarial_Discriminative_Deep_Domain_Generalization_for_Face_Presentation_Attack_Detection_CVPR_2019_paper.html
PatchNet:
  https://openaccess.thecvf.com/content/CVPR2022/html/Wang_PatchNet_A_Simple_Face_Anti-Spoofing_Framework_via_Fine-Grained_Patch_Recognition_CVPR_2022_paper.html
SA-FAS:
  https://openaccess.thecvf.com/content/CVPR2023/html/Sun_Rethinking_Domain_Generalization_for_Face_Anti-Spoofing_Separability_and_Alignment_CVPR_2023_paper.html
GAC-FAS:
  https://openaccess.thecvf.com/content/CVPR2024/html/Le_Gradient_Alignment_for_Cross-Domain_Face_Anti-Spoofing_CVPR_2024_paper.html
AMEL:
  https://arxiv.org/abs/2207.09868
```

---

# Prompt 4 — MAST-PAD method flow

```text
Create:
  slide_figures_mast_pad_fullscale/fig_mast_pad_method_flow.png
  slide_figures_mast_pad_fullscale/fig_mast_pad_method_flow.svg
  slide_figures_mast_pad_fullscale/fig_mast_pad_method_flow.pdf

Diagram:
Clean crop
├── LOW attenuation
├── MID attenuation
└── HIGH attenuation
      ↓
Shared MiniFASNetV2
      ↓
label-aligned margins
      ↓
k* = argmin_k m_k
      ↓
harmful if m_worst < m_clean
      ↓
L = 0.75 CE_clean + 0.25 CE_harmful + L_aux

Bottom strip:
Inference: crop1.5 → 80×80 → MiniFASNetV2 → Real / Attack

Note: "Spectral views are training-only."
```

---

# Prompt 5 — Full-scale MAST-PAD training diagnostics

```text
Create:
  slide_figures_mast_pad_fullscale/fig_mast_pad_training_diagnostics.png
  slide_figures_mast_pad_fullscale/fig_mast_pad_training_diagnostics.pdf

Read R7_SC_FULL_crop15 training history / spectral diagnostics.

Panel A:
- epoch vs harmful_fraction
- epoch vs margin_drop
- epoch vs spectral_flip_rate
- vertical line: best epoch 7
- optional stop epoch 11

Panel B final epoch band selection:
  LOW  = 0.426452  (42.65%)
  MID  = 0.273035  (27.30%)
  HIGH = 0.300513  (30.05%)

Fallback final values:
  harmful_fraction = 0.832135
  margin_drop = 0.391421
  spectral_flip_rate = 0.003077

Caption:
  "Margin degradation is common while outright spectral prediction flips remain rare."
```

---

# Prompt 6 — Full-scale training curves

```text
Create:
  slide_figures_mast_pad_fullscale/fig_training_curves_fullscale.png
  slide_figures_mast_pad_fullscale/fig_training_curves_fullscale.pdf

Read histories for:
  C_FULL_crop15
  P3_SF_FULL_crop15
  R7_SC_FULL_crop15

Plot source-selection AUC vs epoch.
Optional second panel: source-selection ACER vs epoch, but do not use ACER in headline result figures.

Mark best epochs:
  Clean = 9
  WBST = 6
  MAST-PAD = 7

If files are missing, write a missing-data note rather than fabricating curves.
```

---

# Prompt 7 — Frequency counterfactual examples

```text
Create:
  slide_figures_mast_pad_fullscale/fig_frequency_counterfactual_examples.png
  slide_figures_mast_pad_fullscale/fig_frequency_counterfactual_examples.pdf

Select one Real and one Attack source crop.
Show:
  Clean | LOW | MID | HIGH | FFT magnitude

Use:
  centers = [0.0833333333, 0.25, 0.4166666667]
  sigma = 0.10
  gain = 0.775
  crop1.5
  80×80 BGR

Do not expose subject IDs, filenames, or sensitive metadata.
Caption:
  "Frequency counterfactuals are training-only; deployment remains spatial-only."
```

---

# Prompt 8 — Optional supporting 100K chart

```text
Create only if backup slides / appendix need it:
  slide_figures_mast_pad_fullscale/fig_supporting_100k_lcc_auc.png

Data:
  Clean = 0.848273
  WBST = 0.851978
  MAST-PAD = 0.867844

Title:
  "Supporting matched100K development evidence"

Do not label this as the primary paper result.
```

---

# Prompt 9 — Generate all figures and source manifest

```text
Generate all requested full-scale MAST-PAD figures.

Output:
  slide_figures_mast_pad_fullscale/

Always create:
  figure_manifest.md

For every figure record:
- exact numeric values plotted
- artifact/report files used
- method-to-run mapping
- whether static fallback values were used
- all external paper URLs used for literature numbers
- protocol caveats
- generation timestamp
- relevant package versions

Never retrain models.
Never modify checkpoints.
```
