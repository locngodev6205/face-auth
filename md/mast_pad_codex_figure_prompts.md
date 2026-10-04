# Codex Figure Generation Prompts for the MAST-PAD Slide Section
## Separate prompt pack for creating visual assets from the 100K run

_Last updated: 2026-10-05_

Use these prompts after the slide structure is fixed. The target output folder should be:

```text
slide_figures_mast_pad_100k/
```

All figures should be generated from existing run artifacts where possible, especially:

```text
17_scale100k_C_P3SF_R7SC_crop15/
```

Do not retrain any model.

---

# Prompt 1 — Main ablation bar chart

```text
Create figure files:

    slide_figures_mast_pad_100k/fig_main_ablation_lcc_combined_auc.png
    slide_figures_mast_pad_100k/fig_main_ablation_lcc_combined_auc.pdf

Use data from:

    17_scale100k_C_P3SF_R7SC_crop15/runs/*/metrics/lcc_combined_metrics.json

or, if needed, the consolidated report.

Plot a bar chart:

    Clean     = 0.848273
    WBST      = 0.851978
    MAST-PAD  = 0.867844

Label mapping:
    C_100K_crop15          -> Clean
    P3_SF_100K_crop15      -> WBST
    R7_SC_100K_crop15      -> MAST-PAD

Requirements:
- y-axis: "LCC Combined AUC"
- annotate bars with 3 decimals:
    0.848, 0.852, 0.868
- add callout arrow from Clean to MAST-PAD:
    "+1.96 pp"
- subtitle:
    "Matched Train100K / Val15K protocol; source threshold frozen before LCC"
- style:
    white background
    large readable labels
    no tiny fonts
    Clean gray, WBST blue, MAST-PAD teal/green
- save PNG at 300 DPI and vector PDF
```

---

# Prompt 2 — Literature/context AUC vs parameters plot

```text
Create figure files:

    slide_figures_mast_pad_100k/fig_lcc_auc_vs_params_context.png
    slide_figures_mast_pad_100k/fig_lcc_auc_vs_params_context.pdf

Use static contextual comparison data:

    Graph for Transformer Feature:
        params_M = 10.84
        auc = 0.833

    AENet:
        params_M = 11.22
        auc = 0.868

    MAST-PAD:
        params_M = 0.434434
        auc = 0.867844

Plot:
- x-axis: Parameters (M), log scale
- y-axis: LCC AUC
- label every point directly
- highlight MAST-PAD with larger marker
- annotate MAST-PAD:
    "0.434M params, AUC 0.868"
- add footnote inside the figure:
    "Contextual references; protocols differ."
- save PNG at 300 DPI and vector PDF
```

---

# Prompt 3 — MAST-PAD method flow diagram

```text
Create figure files:

    slide_figures_mast_pad_100k/fig_mast_pad_method_flow.png
    slide_figures_mast_pad_100k/fig_mast_pad_method_flow.svg
    slide_figures_mast_pad_100k/fig_mast_pad_method_flow.pdf

Create a clean research-presentation diagram.

Style:
- white background
- teal/blue accent
- rounded boxes
- minimal text
- no external icons required
- readable when placed on a 16:9 slide

Diagram content:
1. left box:
    "Clean crop"

2. branch into three boxes:
    "LOW attenuation"
    "MID attenuation"
    "HIGH attenuation"

3. all three connect to one shared model box:
    "Shared MiniFASNetV2 PAD"

4. output box:
    "label-aligned margins"
    m_low, m_mid, m_high

5. selection box:
    "k* = argmin_k m_k"

6. comparison box:
    "harmful if m_worst < m_clean"

7. loss box:
    "L = 0.75 CE_clean + 0.25 CE_harmful + L_aux"

8. bottom strip:
    "Inference: crop1.5 → 80×80 → MiniFASNetV2 → Real / Attack"

9. note:
    "Spectral views are training-only"

Do not write:
- R7
- R7-SC
- P3
- CSMR
- 3-class
- LogSumExp
```

---

# Prompt 4 — Training diagnostics from R7/MAST-PAD 100K

```text
Create figure files:

    slide_figures_mast_pad_100k/fig_mast_pad_training_diagnostics.png
    slide_figures_mast_pad_100k/fig_mast_pad_training_diagnostics.pdf

Use:

    17_scale100k_C_P3SF_R7SC_crop15/runs/R7_SC_100K_crop15/training_history.csv

and/or:

    17_scale100k_C_P3SF_R7SC_crop15/runs/R7_SC_100K_crop15/spectral_diagnostics.csv

If both exist, inspect their columns and use the file with the most complete epoch-level spectral diagnostics.

Create a 2-panel figure.

Panel A:
- x-axis: epoch
- line 1: harmful_fraction
- line 2: margin_drop
- line 3: spectral_flip_rate
- vertical dashed line at selected best epoch = 11
- optional vertical dotted line at stop epoch = 17
- legend with readable labels

Panel B:
- band-selection distribution at the last executed epoch:
    LOW  = 0.427060
    MID  = 0.254670
    HIGH = 0.318270
- use a stacked bar or pie chart
- label percentages:
    LOW 42.71%
    MID 25.47%
    HIGH 31.83%

Static fallback values if columns cannot be parsed:
    harmful_fraction = 0.847790
    margin_drop = 0.417364
    spectral_flip_rate = 0.004370
    LOW = 0.427060
    MID = 0.254670
    HIGH = 0.318270

Caption:
    "MAST-PAD uses margin degradation, not only prediction flips, to select harmful spectral supervision."

Save PNG at 300 DPI and vector PDF.
```

---

# Prompt 5 — Validation curves for Clean / WBST / MAST-PAD

```text
Create figure files:

    slide_figures_mast_pad_100k/fig_training_curves_val_auc_acer.png
    slide_figures_mast_pad_100k/fig_training_curves_val_auc_acer.pdf

Read training history files for:

    17_scale100k_C_P3SF_R7SC_crop15/runs/C_100K_crop15/
    17_scale100k_C_P3SF_R7SC_crop15/runs/P3_SF_100K_crop15/
    17_scale100k_C_P3SF_R7SC_crop15/runs/R7_SC_100K_crop15/

Possible filenames:
    training_history.csv
    history.csv
    epoch_history.csv

Inspect the available columns.

Create a 2-panel figure:

Panel A:
- Val AUC vs epoch
- methods:
    Clean
    WBST
    MAST-PAD

Panel B:
- Val ACER (%) vs epoch
- methods:
    Clean
    WBST
    MAST-PAD

Mark best epochs:
    Clean = 15
    WBST = 15
    MAST-PAD = 11

Use consistent labels:
    C_100K_crop15 -> Clean
    P3_SF_100K_crop15 -> WBST
    R7_SC_100K_crop15 -> MAST-PAD

Save PNG at 300 DPI and PDF.

If histories cannot be found, do not fabricate curves. Instead create a small markdown note:
    slide_figures_mast_pad_100k/missing_training_curves_note.md
explaining which files were missing.
```

---

# Prompt 6 — Frequency counterfactual examples

```text
Create figure files:

    slide_figures_mast_pad_100k/fig_frequency_counterfactual_examples.png
    slide_figures_mast_pad_100k/fig_frequency_counterfactual_examples.pdf

Use the existing 100K resource/cache if available.

Select 2 anonymized face crops:
- one Real sample
- one Attack sample

Do not show identity names, subject IDs, file paths, or sensitive metadata in the figure.

For each sample, show one row:

    clean crop
    LOW attenuation
    MID attenuation
    HIGH attenuation
    log FFT magnitude of clean crop

Use MAST-PAD spectral generator:

    centers = [0.0833333333, 0.25, 0.4166666667]
    sigma = 0.10
    gain = 0.775

Use crop1.5 and 80×80 BGR inputs.

Presentation requirements:
- all images same size
- clear labels: Clean, LOW, MID, HIGH, FFT magnitude
- row labels: Real, Attack
- caption:
    "Frequency views are used only during training; inference uses the clean spatial model."

Save PNG at 300 DPI and PDF.
```

---

# Prompt 7 — Full-scale supporting evidence chart

```text
Create figure files:

    slide_figures_mast_pad_100k/fig_fullscale_scaleup_summary.png
    slide_figures_mast_pad_100k/fig_fullscale_scaleup_summary.pdf

Use static full-scale data:

    Clean:
        Test AUC = 0.990736
        Test ACER = 4.6248
        LCC Eval AUC = 0.811622
        LCC Combined AUC = 0.846011

    WBST:
        Test AUC = 0.992242
        Test ACER = 4.3548
        LCC Eval AUC = 0.819687
        LCC Combined AUC = 0.860831

    MAST-PAD:
        Test AUC = 0.992280
        Test ACER = 4.3459
        LCC Eval AUC = 0.840060
        LCC Combined AUC = 0.860272

Create grouped bars:
- group 1: LCC Eval AUC
- group 2: LCC Combined AUC
- methods: Clean, WBST, MAST-PAD
- annotate values to 3 decimals

Add note:
    "Secondary full-scale check; primary paper result uses matched Train100K / Val15K."

Save PNG at 300 DPI and PDF.
```

---

# Prompt 8 — Generate all figures and manifest

```text
Generate all MAST-PAD slide figures in one pass.

Output folder:

    slide_figures_mast_pad_100k/

Figures:
1. fig_main_ablation_lcc_combined_auc
2. fig_lcc_auc_vs_params_context
3. fig_mast_pad_method_flow
4. fig_mast_pad_training_diagnostics
5. fig_training_curves_val_auc_acer
6. fig_frequency_counterfactual_examples
7. fig_fullscale_scaleup_summary

For every figure:
- save PNG at 300 DPI
- save PDF
- save SVG when the figure is a diagram
- use readable labels for slide presentation
- use method labels:
    Clean
    WBST
    MAST-PAD
- do not expose internal names in figure labels unless needed in the manifest.

Also create:

    slide_figures_mast_pad_100k/figure_manifest.md

The manifest must include:
- figure filename;
- data source files used;
- exact numeric values plotted;
- whether any static fallback values were used;
- generation timestamp;
- package versions if available.

Do not run model training.
Do not modify checkpoints.
Do not change existing experiment outputs.
```
