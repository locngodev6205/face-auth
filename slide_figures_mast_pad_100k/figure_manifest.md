# MAST-PAD figure material

Generated: 2026-10-05T01:03:32.183815+07:00

Six figures from existing artifacts, each with 300-DPI PNG and vector PDF; method flow also has SVG.
No training, checkpoint editing, threshold fitting, or changes to existing experiment outputs.

## Reproduce

`python3 slide_figures_mast_pad_100k/generate_figures.py`

Dependencies: Python 3.11.7, Matplotlib 3.10.8, NumPy 2.4.4

Exact plotted values and complete curve arrays: [figure_data.json](figure_data.json).

## Interpretation

- Main experiment: MAST-PAD is the selected model; one training seed. No significance claim.
- LCC Combined pools training, development and evaluation splits. LCC was not used to choose checkpoints or thresholds.
- AUC is threshold-independent; the source-frozen threshold statement describes the overall evaluation protocol.
- Contextual literature points have different protocols and come from the supplied prompt.
- Full-scale is secondary and uses source Test-as-Val, with a separate training schedule.
- Curves and metrics are read from individual run artifacts; stale top-level summary snapshots are not used.

## fig_main_lcc_combined_auc

Files: [fig_main_lcc_combined_auc.png](fig_main_lcc_combined_auc.png), [fig_main_lcc_combined_auc.pdf](fig_main_lcc_combined_auc.pdf)

Caption: MAST-PAD improves pooled LCC AUC by 1.96 percentage points over Clean in the main experiment.

Sources:

- `antispoof/notebooks/final/binary_crossdomain_v2/output/17_scale100k_C_P3SF_R7SC_crop15/runs/C_100K_crop15/metrics/lcc_combined_metrics.json`
- `antispoof/notebooks/final/binary_crossdomain_v2/output/17_scale100k_C_P3SF_R7SC_crop15/runs/P3_SF_100K_crop15/metrics/lcc_combined_metrics.json`
- `antispoof/notebooks/final/binary_crossdomain_v2/output/17_scale100k_C_P3SF_R7SC_crop15/runs/R7_SC_100K_crop15/metrics/lcc_combined_metrics.json`

Static fallback: **No**. Supplied contextual references: **No**.



Exact plotted values:

```json
{
  "Clean": 0.8482729697132887,
  "WBST": 0.8519784359546514,
  "MAST-PAD": 0.8678437816372155,
  "gain_pp": 1.9570811923926845
}
```

## fig_lcc_auc_vs_params_context

Files: [fig_lcc_auc_vs_params_context.png](fig_lcc_auc_vs_params_context.png), [fig_lcc_auc_vs_params_context.pdf](fig_lcc_auc_vs_params_context.pdf)

Caption: MAST-PAD has 434,434 deployment parameters and 0.868 pooled LCC AUC. Published reference points are contextual comparisons.

Sources:

- `antispoof/notebooks/final/binary_crossdomain_v2/output/17_scale100k_C_P3SF_R7SC_crop15/runs/R7_SC_100K_crop15/metrics/lcc_combined_metrics.json`
- `antispoof/notebooks/final/binary_crossdomain_v2/output/18_fullscale_C_P3SF_R7SC_crop15/complexity_report.json`
- `md/mast_pad_codex_figure_prompts.md`

Static fallback: **No**. Supplied contextual references: **Yes**.

Reference numbers are supplied by the figure prompt, not independently reproduced measurements. The MAST-PAD AUC uses unrounded run JSON.

Exact plotted values:

```json
{
  "Graph for Transformer Feature": {
    "params_M": 10.84,
    "auc": 0.833
  },
  "AENet": {
    "params_M": 11.22,
    "auc": 0.868
  },
  "MAST-PAD": {
    "params_M": 0.434434,
    "auc": 0.8678437816372155
  }
}
```

## fig_mast_pad_method_flow

Files: [fig_mast_pad_method_flow.png](fig_mast_pad_method_flow.png), [fig_mast_pad_method_flow.pdf](fig_mast_pad_method_flow.pdf), [fig_mast_pad_method_flow.svg](fig_mast_pad_method_flow.svg)

Caption: MAST-PAD selects the minimum-margin spectral view and applies harmful-view supervision when its margin is below the clean margin. Deployment uses the clean binary spatial model.

Sources:

- `antispoof/notebooks/final/binary_crossdomain_v2/17_scale100k_C_P3SF_R7SC_crop15.ipynb`
- `antispoof/notebooks/final/binary_crossdomain_v2/output/17_scale100k_C_P3SF_R7SC_crop15/runs/R7_SC_100K_crop15/config.json`

Static fallback: **No**. Supplied contextual references: **No**.

Selection uses eval/no_grad (BN-safe). Auxiliary heads are training-only; harmful CE is averaged over harmful samples.

Exact plotted values:

```json
{
  "active_loss_weights": [
    0.75,
    0.25
  ],
  "crop_factor": 1.5,
  "input_size": [
    80,
    80
  ],
  "selector": "argmin label-aligned margin",
  "gate": "worst margin < clean margin",
  "warmup_epochs": 1
}
```

## fig_mast_pad_training_diagnostics

Files: [fig_mast_pad_training_diagnostics.png](fig_mast_pad_training_diagnostics.png), [fig_mast_pad_training_diagnostics.pdf](fig_mast_pad_training_diagnostics.pdf)

Caption: MAST-PAD uses margin degradation, not only prediction flips, to select harmful spectral supervision.

Sources:

- `antispoof/notebooks/final/binary_crossdomain_v2/output/17_scale100k_C_P3SF_R7SC_crop15/runs/R7_SC_100K_crop15/spectral_diagnostics.csv`
- `antispoof/notebooks/final/binary_crossdomain_v2/output/17_scale100k_C_P3SF_R7SC_crop15/runs/R7_SC_100K_crop15/completion.json`

Static fallback: **No**. Supplied contextual references: **No**.

Spectral CSV contains all active epochs 2–17; no static fallback used. Band selection counts all samples, not just harmful samples. Spectral flip rate counts clean-correct / selected-view-wrong samples divided by total samples (not every label change).

Exact plotted values:

```json
{
  "epochs": [
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17
  ],
  "curves": {
    "harmful_fraction": [
      0.70683,
      0.73211,
      0.75904,
      0.77199,
      0.78819,
      0.82168,
      0.83661,
      0.84387,
      0.84201,
      0.83981,
      0.84708,
      0.83871,
      0.84315,
      0.84138,
      0.84788,
      0.84779
    ],
    "margin_drop": [
      0.4661341854286194,
      0.45889928581237793,
      0.48091704047203065,
      0.4584498077011108,
      0.4828448973941803,
      0.4475145171451569,
      0.4324652995300293,
      0.4327171043300629,
      0.42127446150779724,
      0.41874944160461425,
      0.4361953163433075,
      0.4194805598735809,
      0.41985635062217713,
      0.41234956099510195,
      0.4170905549907684,
      0.41736410013198855
    ],
    "spectral_flip_rate": [
      0.01428,
      0.01183,
      0.01084,
      0.00991,
      0.00918,
      0.007,
      0.00608,
      0.00519,
      0.00554,
      0.00551,
      0.00511,
      0.00506,
      0.00536,
      0.00461,
      0.00437,
      0.00437
    ]
  },
  "last_epoch": 17,
  "last_band_fractions": {
    "LOW": 0.42706,
    "MID": 0.25467,
    "HIGH": 0.31827
  },
  "best_epoch": 11
}
```

## fig_training_curves_val_auc_acer

Files: [fig_training_curves_val_auc_acer.png](fig_training_curves_val_auc_acer.png), [fig_training_curves_val_auc_acer.pdf](fig_training_curves_val_auc_acer.pdf)

Caption: Source validation metrics are evaluated throughout training. Stars mark selected checkpoints, including the original warmup eligibility rules.

Sources:

- `antispoof/notebooks/final/binary_crossdomain_v2/output/17_scale100k_C_P3SF_R7SC_crop15/runs/C_100K_crop15/training_history.csv`
- `antispoof/notebooks/final/binary_crossdomain_v2/output/17_scale100k_C_P3SF_R7SC_crop15/runs/P3_SF_100K_crop15/training_history.csv`
- `antispoof/notebooks/final/binary_crossdomain_v2/output/17_scale100k_C_P3SF_R7SC_crop15/runs/R7_SC_100K_crop15/training_history.csv`
- `antispoof/notebooks/final/binary_crossdomain_v2/output/17_scale100k_C_P3SF_R7SC_crop15/runs/C_100K_crop15/completion.json`
- `antispoof/notebooks/final/binary_crossdomain_v2/output/17_scale100k_C_P3SF_R7SC_crop15/runs/P3_SF_100K_crop15/completion.json`
- `antispoof/notebooks/final/binary_crossdomain_v2/output/17_scale100k_C_P3SF_R7SC_crop15/runs/R7_SC_100K_crop15/completion.json`

Static fallback: **No**. Supplied contextual references: **No**.

AUC and ACER axes are scaled to the observed validation range; all recorded epochs are shown. Best selection is preserved, not chosen using LCC.

Exact plotted values:

```json
{
  "Clean": {
    "epochs": [
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18
    ],
    "val_auc": [
      0.9969235276383007,
      0.9979359783755926,
      0.9982186944134862,
      0.9987647719932059,
      0.9984309721195777,
      0.9988985586937428,
      0.9994943963820049,
      0.9994102193664658,
      0.9993366121119914,
      0.9994291326201359,
      0.9993825614907383,
      0.9994515958694578,
      0.9995121965014452,
      0.9994314792274311,
      0.9995233779849246,
      0.9995157665535696,
      0.9995104716448009,
      0.9994621255175773
    ],
    "val_acer_percent": [
      2.303602206661252,
      1.9479086875333138,
      1.7930045269865456,
      1.4275916412323852,
      1.6646310684066912,
      1.2722662947608323,
      0.9819809441046967,
      0.930941232609835,
      1.032178283749989,
      0.896797093640622,
      0.9113340249874767,
      0.9470786704708714,
      0.9328786878638445,
      0.9022885558409102,
      0.8270767835579285,
      0.8663754359876017,
      0.8656173013229891,
      0.8824286366636812
    ],
    "best_epoch": 15,
    "checkpoint_eligible": [
      "True",
      "True",
      "True",
      "True",
      "True",
      "True",
      "True",
      "True",
      "True",
      "True",
      "True",
      "True",
      "True",
      "True",
      "True",
      "True",
      "True",
      "True"
    ]
  },
  "WBST": {
    "epochs": [
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18
    ],
    "val_auc": [
      0.9969235276383007,
      0.9980129450892289,
      0.99865421068795,
      0.9989255948188195,
      0.9993230338800354,
      0.9988271175383102,
      0.9995289637125462,
      0.9995138411321991,
      0.9995306283997727,
      0.9995042441100558,
      0.9995332658259208,
      0.9994919896052918,
      0.9995456807824656,
      0.9994914380522951,
      0.9995865257889334,
      0.9995588879696785,
      0.9995651255326599,
      0.999566549542215
    ],
    "val_acer_percent": [
      2.303602206661252,
      1.987375814332901,
      1.4631678123458658,
      1.3323514754022607,
      1.247167624938186,
      1.4079001964250701,
      0.9268818025538146,
      0.9112497878025199,
      1.0167147433686081,
      1.006911139557429,
      0.9165727756328939,
      1.011307518353277,
      1.0158723715190388,
      0.9365169319950798,
      0.8462628052555019,
      0.8864880667197015,
      0.8966286192707081,
      0.910997076247649
    ],
    "best_epoch": 15,
    "checkpoint_eligible": [
      "False",
      "True",
      "True",
      "True",
      "True",
      "True",
      "True",
      "True",
      "True",
      "True",
      "True",
      "True",
      "True",
      "True",
      "True",
      "True",
      "True",
      "True"
    ]
  },
  "MAST-PAD": {
    "epochs": [
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17
    ],
    "val_auc": [
      0.9969235276383007,
      0.9982633401215134,
      0.9987948466738827,
      0.9987946862221019,
      0.9988672405117641,
      0.9987010224950189,
      0.9995590684779321,
      0.9995420305044508,
      0.9995497723028778,
      0.9995694978436885,
      0.9996057900308741,
      0.9995238593402673,
      0.999494266014933,
      0.9995909281846711,
      0.9995652859844406,
      0.9996164701025384,
      0.9996097511842145
    ],
    "val_acer_percent": [
      2.303602206661252,
      1.8393109109457315,
      1.4223528905869678,
      1.4917362519298336,
      1.512270068586718,
      1.4359632128992963,
      0.8462628052555019,
      0.8615578712669689,
      0.8805754185946285,
      0.8961231961609665,
      0.8265713604481868,
      0.8804911814096715,
      0.9207164428738711,
      0.8820916879238534,
      0.8319785854635178,
      0.8363749642593659,
      0.8567403065463364
    ],
    "best_epoch": 11,
    "checkpoint_eligible": [
      "False",
      "True",
      "True",
      "True",
      "True",
      "True",
      "True",
      "True",
      "True",
      "True",
      "True",
      "True",
      "True",
      "True",
      "True",
      "True",
      "True"
    ]
  }
}
```

## fig_fullscale_scaleup_summary

Files: [fig_fullscale_scaleup_summary.png](fig_fullscale_scaleup_summary.png), [fig_fullscale_scaleup_summary.pdf](fig_fullscale_scaleup_summary.pdf)

Caption: MAST-PAD is strongest on LCC Evaluation; WBST is slightly higher on pooled LCC Combined in the secondary full-scale experiment.

Sources:

- `antispoof/notebooks/final/binary_crossdomain_v2/output/18_fullscale_C_P3SF_R7SC_crop15/runs/C_FULL_crop15/metrics/lcc_evaluation_metrics.json`
- `antispoof/notebooks/final/binary_crossdomain_v2/output/18_fullscale_C_P3SF_R7SC_crop15/runs/C_FULL_crop15/metrics/lcc_combined_metrics.json`
- `antispoof/notebooks/final/binary_crossdomain_v2/output/18_fullscale_C_P3SF_R7SC_crop15/runs/P3_SF_FULL_crop15/metrics/lcc_evaluation_metrics.json`
- `antispoof/notebooks/final/binary_crossdomain_v2/output/18_fullscale_C_P3SF_R7SC_crop15/runs/P3_SF_FULL_crop15/metrics/lcc_combined_metrics.json`
- `antispoof/notebooks/final/binary_crossdomain_v2/output/18_fullscale_C_P3SF_R7SC_crop15/runs/R7_SC_FULL_crop15/metrics/lcc_evaluation_metrics.json`
- `antispoof/notebooks/final/binary_crossdomain_v2/output/18_fullscale_C_P3SF_R7SC_crop15/runs/R7_SC_FULL_crop15/metrics/lcc_combined_metrics.json`
- `antispoof/notebooks/final/binary_crossdomain_v2/output/18_fullscale_C_P3SF_R7SC_crop15/runs/R7_SC_FULL_crop15/config.json`

Static fallback: **No**. Supplied contextual references: **No**.

Uses exact per-run JSON, not rounded static values. Full-scale changes data size, schedule and selection protocol; no untouched source Test claim.

Exact plotted values:

```json
{
  "Clean": {
    "LCC Evaluation AUC": 0.811622227103009,
    "LCC Combined AUC": 0.8460112694327802
  },
  "WBST": {
    "LCC Evaluation AUC": 0.8196867999121459,
    "LCC Combined AUC": 0.8608309944893372
  },
  "MAST-PAD": {
    "LCC Evaluation AUC": 0.8400595211948167,
    "LCC Combined AUC": 0.860272080856274
  }
}
```

## Not generated: frequency counterfactual examples

Original dataset images are not available locally. See [missing_frequency_examples_note.md](missing_frequency_examples_note.md).

## Contact sheet

[preview_contact_sheet.png](preview_contact_sheet.png) is a navigation preview; use individual PNG/PDF/SVG files for slides and paper.
