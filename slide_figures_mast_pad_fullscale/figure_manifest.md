# MAST-PAD full-scale figure manifest

Generated: 2026-10-07T17:54:18.812011+07:00

Seven figures; PNG 300 DPI and vector PDF, plus SVG for the method diagram. Figure 7 is missing source images; supporting 100K is explicitly appendix-only.

## Reproduction

`python3 slide_figures_mast_pad_fullscale/generate_figures.py`

Packages: Python 3.11.7, Matplotlib 3.10.8, NumPy 2.4.4

No model training, inference, recalibration, checkpoint editing or changes to existing experiment outputs.

## Run mapping

- Clean: `C_FULL_crop15`
- WBST: `P3_SF_FULL_crop15`
- MAST-PAD: `R7_SC_FULL_crop15`

## Common protocol caveats

- Full-scale selection/calibration uses source CelebA Test-as-Val; it is not untouched source Test.
- LCC Combined pools three image splits; CASIA Combined is a unique union of reconstructed frame IDs, with test-copy precedence.
- CASIA AUC in headline figures is video-level (358 scored / 360 candidates; minimum four valid frames). LCC AUC is image-level. Do not average the two target AUCs.
- CASIA copy is not verified to be the official subject-disjoint benchmark; its train/test copies overlap.
- Reference source domains, preprocessing, evaluation units and result selection/statistics differ. Context only; no controlled SOTA or significance claim.
- Existing CASIA results are already present; this script does not run target evaluation.

Exact plotted data and curve arrays: [figure_data.json](figure_data.json).

## fig_main_ablation_two_target_auc

Role: main

Files: [fig_main_ablation_two_target_auc.png](fig_main_ablation_two_target_auc.png), [fig_main_ablation_two_target_auc.pdf](fig_main_ablation_two_target_auc.pdf)

Caption: Full-scale WBST and MAST-PAD are nearly tied on LCC Combined; MAST-PAD improves CASIA Combined reconstructed-video AUC by 1.48 percentage points over Clean.

Sources:

- `antispoof/notebooks/final/binary_crossdomain_v2/output/18_fullscale_C_P3SF_R7SC_crop15/runs/C_FULL_crop15/metrics/lcc_combined_metrics.json`
- `antispoof/notebooks/final/binary_crossdomain_v2/output/18_fullscale_C_P3SF_R7SC_crop15/runs/P3_SF_FULL_crop15/metrics/lcc_combined_metrics.json`
- `antispoof/notebooks/final/binary_crossdomain_v2/output/18_fullscale_C_P3SF_R7SC_crop15/runs/R7_SC_FULL_crop15/metrics/lcc_combined_metrics.json`
- `antispoof/notebooks/final/binary_crossdomain_v2/output/casia_train_test_C_P3SF_R7SC_100k_fullscale/runs/C_FULL_crop15/combined/video_metrics.json`
- `antispoof/notebooks/final/binary_crossdomain_v2/output/casia_train_test_C_P3SF_R7SC_100k_fullscale/runs/P3_SF_FULL_crop15/combined/video_metrics.json`
- `antispoof/notebooks/final/binary_crossdomain_v2/output/casia_train_test_C_P3SF_R7SC_100k_fullscale/runs/R7_SC_FULL_crop15/combined/video_metrics.json`
- `antispoof/notebooks/final/binary_crossdomain_v2/output/casia_train_test_C_P3SF_R7SC_100k_fullscale/evaluation_protocol.json`

Static fallback used: **No**. Supplied literature values used: **No**.

Protocol caveats: LCC is image-level, CASIA is reconstructed video-level (mean score; ≥4 valid frames), so there is no cross-target pooled AUC. Source checkpoint/threshold selection uses CelebA Test-as-Val. Filename follows the prompt; this figure is labeled Main, not ablation.

Reference keys: none

Exact numeric values:

```json
{
  "LCC Combined": {
    "Clean": 0.8460112694327802,
    "WBST": 0.8608309944893372,
    "MAST-PAD": 0.860272080856274
  },
  "CASIA Combined": {
    "Clean": 0.9203980099502487,
    "WBST": 0.9222636815920399,
    "MAST-PAD": 0.9351990049751244
  },
  "CASIA_gain_pp": 1.4800995024875752
}
```

## fig_lcc_auc_vs_params_context

Role: main

Files: [fig_lcc_auc_vs_params_context.png](fig_lcc_auc_vs_params_context.png), [fig_lcc_auc_vs_params_context.pdf](fig_lcc_auc_vs_params_context.pdf)

Caption: MAST-PAD achieves pooled LCC AUC 0.860 with 434,434 clean deployment parameters; literature points provide context.

Sources:

- `antispoof/notebooks/final/binary_crossdomain_v2/output/18_fullscale_C_P3SF_R7SC_crop15/runs/R7_SC_FULL_crop15/metrics/lcc_combined_metrics.json`
- `antispoof/notebooks/final/binary_crossdomain_v2/output/18_fullscale_C_P3SF_R7SC_crop15/complexity_report.json`
- `md/mast_pad_codex_figure_prompts_FULLSCALE.md`

Static fallback used: **No**. Supplied literature values used: **Yes**.

Protocol caveats: Different training/evaluation/preprocessing protocols and parameter-count conventions; no controlled ranking claim.

Reference keys: Graph Transformer, AENet

Exact numeric values:

```json
{
  "Graph Transformer": {
    "params_M": 10.84,
    "auc": 0.833
  },
  "AENet": {
    "params_M": 11.22,
    "auc": 0.868
  },
  "MAST-PAD": {
    "params_M": 0.434434,
    "auc": 0.860272080856274
  }
}
```

## fig_casia_auc_literature_context

Role: main

Files: [fig_casia_auc_literature_context.png](fig_casia_auc_literature_context.png), [fig_casia_auc_literature_context.pdf](fig_casia_auc_literature_context.pdf)

Caption: Contextual CASIA AUC points, separating O&M&I → C literature protocols from CelebA-full → reconstructed CASIA Combined.

Sources:

- `antispoof/notebooks/final/binary_crossdomain_v2/output/casia_train_test_C_P3SF_R7SC_100k_fullscale/runs/R7_SC_FULL_crop15/combined/video_metrics.json`
- `antispoof/notebooks/final/binary_crossdomain_v2/output/casia_train_test_C_P3SF_R7SC_100k_fullscale/evaluation_protocol.json`
- `md/mast_pad_codex_figure_prompts_FULLSCALE.md`
- `antispoof/notebooks/final/binary_crossdomain_v2/output/18_fullscale_C_P3SF_R7SC_crop15/complexity_report.json`

Static fallback used: **No**. Supplied literature values used: **Yes**.

Protocol caveats: MAST-PAD uses a nonofficial extracted-image combined copy, 358 videos and one trained checkpoint. Literature numbers include selected-best results and a mean (GAC-FAS); no matched benchmark or significance claim.

Reference keys: MADDG, AMEL, PatchNet, GAC-FAS, SA-FAS

Exact numeric values:

```json
{
  "MADDG": 0.8451,
  "MAST-PAD": 0.9351990049751244,
  "AMEL": 0.9439,
  "PatchNet": 0.9458,
  "GAC-FAS": 0.9516,
  "SA-FAS": 0.9537
}
```

## fig_mast_pad_method_flow

Role: main

Files: [fig_mast_pad_method_flow.png](fig_mast_pad_method_flow.png), [fig_mast_pad_method_flow.pdf](fig_mast_pad_method_flow.pdf), [fig_mast_pad_method_flow.svg](fig_mast_pad_method_flow.svg)

Caption: Select the minimum-margin spectral counterfactual and supervise it only when it degrades the clean margin; deployment is the spatial binary PAD model.

Sources:

- `antispoof/notebooks/final/binary_crossdomain_v2/18_fullscale_C_P3SF_R7SC_crop15.ipynb`
- `antispoof/notebooks/final/binary_crossdomain_v2/output/18_fullscale_C_P3SF_R7SC_crop15/runs/R7_SC_FULL_crop15/config.json`

Static fallback used: **No**. Supplied literature values used: **No**.

Protocol caveats: Selection: eval/no_grad, BN-safe; harmful CE averages harmful samples; clean auxiliary losses are training-only.

Reference keys: none

Exact numeric values:

```json
{
  "active_loss_weights": [
    0.75,
    0.25
  ],
  "warmup_epochs": 1,
  "crop_factor": 1.5,
  "input_size": [
    80,
    80
  ],
  "selector": "argmin label-aligned margin",
  "gate": "worst margin < clean margin"
}
```

## fig_mast_pad_training_diagnostics

Role: main

Files: [fig_mast_pad_training_diagnostics.png](fig_mast_pad_training_diagnostics.png), [fig_mast_pad_training_diagnostics.pdf](fig_mast_pad_training_diagnostics.pdf)

Caption: Margin degradation is common while outright spectral prediction flips remain rare.

Sources:

- `antispoof/notebooks/final/binary_crossdomain_v2/output/18_fullscale_C_P3SF_R7SC_crop15/runs/R7_SC_FULL_crop15/spectral_diagnostics.csv`
- `antispoof/notebooks/final/binary_crossdomain_v2/output/18_fullscale_C_P3SF_R7SC_crop15/runs/R7_SC_FULL_crop15/completion.json`

Static fallback used: **No**. Supplied literature values used: **No**.

Protocol caveats: No static fallback. Band selection counts all samples. Flip rate is clean-correct / selected-view-wrong samples divided by total, not all prediction changes.

Reference keys: none

Exact numeric values:

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
    11
  ],
  "curves": {
    "harmful_fraction": [
      0.7672903927472495,
      0.7820418894628334,
      0.785740082260025,
      0.818866974436655,
      0.825037508177505,
      0.8328652120166501,
      0.8357606163104335,
      0.8342210784607098,
      0.8381731092679766,
      0.832134653841312
    ],
    "margin_drop": [
      0.4783284193040188,
      0.4599904396181176,
      0.4537041323672805,
      0.41903507647052257,
      0.4028503442077362,
      0.4077466705531436,
      0.40850767599815546,
      0.40638327913231154,
      0.39692125740460954,
      0.39142119763549976
    ],
    "spectral_flip_rate": [
      0.009583932673079344,
      0.00780087543157621,
      0.006866008613982553,
      0.0045443194974420145,
      0.004129511041953438,
      0.004102682634384526,
      0.004007751346063757,
      0.0034133989322293785,
      0.00323179125022443,
      0.003077011975788394
    ]
  },
  "best_epoch": 7,
  "stop_epoch": 11,
  "final_band_fractions": {
    "LOW": 0.42645198437348447,
    "MID": 0.2730347675524856,
    "HIGH": 0.3005132480740299
  }
}
```

## fig_training_curves_fullscale

Role: main

Files: [fig_training_curves_fullscale.png](fig_training_curves_fullscale.png), [fig_training_curves_fullscale.pdf](fig_training_curves_fullscale.pdf)

Caption: Full-scale source-selection curves, with stars marking the original chosen checkpoints.

Sources:

- `antispoof/notebooks/final/binary_crossdomain_v2/output/18_fullscale_C_P3SF_R7SC_crop15/runs/C_FULL_crop15/training_history.csv`
- `antispoof/notebooks/final/binary_crossdomain_v2/output/18_fullscale_C_P3SF_R7SC_crop15/runs/P3_SF_FULL_crop15/training_history.csv`
- `antispoof/notebooks/final/binary_crossdomain_v2/output/18_fullscale_C_P3SF_R7SC_crop15/runs/R7_SC_FULL_crop15/training_history.csv`
- `antispoof/notebooks/final/binary_crossdomain_v2/output/18_fullscale_C_P3SF_R7SC_crop15/runs/C_FULL_crop15/completion.json`
- `antispoof/notebooks/final/binary_crossdomain_v2/output/18_fullscale_C_P3SF_R7SC_crop15/runs/P3_SF_FULL_crop15/completion.json`
- `antispoof/notebooks/final/binary_crossdomain_v2/output/18_fullscale_C_P3SF_R7SC_crop15/runs/R7_SC_FULL_crop15/completion.json`

Static fallback used: **No**. Supplied literature values used: **No**.

Protocol caveats: Source selection and threshold calibration reuse official CelebA Test-as-Val. ACER here is a training diagnostic, not a headline target comparison; axes span observed values.

Reference keys: none

Exact numeric values:

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
      12
    ],
    "source_selection_auc": [
      0.9811297149543046,
      0.9818472303050618,
      0.9825050751430269,
      0.9796127982870859,
      0.986228587004768,
      0.9900302612424271,
      0.9895388653593659,
      0.9889882248413149,
      0.9907355300359331,
      0.989721706551124,
      0.9895721839390366,
      0.9872941170333415
    ],
    "source_selection_acer_percent": [
      7.01419875512509,
      7.259173407873962,
      6.875174923760838,
      7.051794998638272,
      5.886353334026859,
      4.864878930137814,
      5.186720197179582,
      5.220599182254011,
      4.624830931382525,
      4.908226309754444,
      4.847259981222847,
      5.3335550711840405
    ],
    "best_epoch": 9
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
      11
    ],
    "source_selection_auc": [
      0.9811297149543046,
      0.9819077314689691,
      0.9787617845853578,
      0.9845772655757528,
      0.9884355447636208,
      0.9922418127454586,
      0.9910260008842674,
      0.9898879109103897,
      0.9913393094974234,
      0.9906301228445875,
      0.9912928531016177
    ],
    "source_selection_acer_percent": [
      7.01419875512509,
      7.103542506760688,
      7.1665947707011135,
      6.336621326427857,
      5.457119051277159,
      4.354846482903047,
      4.698947630515475,
      4.882893777655811,
      4.526996247816335,
      4.801298891583482,
      4.573387652613067
    ],
    "best_epoch": 6
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
      11
    ],
    "source_selection_auc": [
      0.9811297149543046,
      0.9830567773793105,
      0.9803823291729996,
      0.9863642298966953,
      0.9883138714212248,
      0.9917695212968989,
      0.9922802748683524,
      0.9904149160220923,
      0.9919206348629133,
      0.9912689422546064,
      0.9916530012382601
    ],
    "source_selection_acer_percent": [
      7.01419875512509,
      6.824941040546206,
      7.075997953448818,
      5.786251974027494,
      5.264948321703785,
      4.685341781709952,
      4.345889087610217,
      4.875234849692084,
      4.4743065594619145,
      4.637140044509479,
      4.495123379883996
    ],
    "best_epoch": 7
  }
}
```

## fig_supporting_100k_lcc_auc

Role: supporting

Files: [fig_supporting_100k_lcc_auc.png](fig_supporting_100k_lcc_auc.png), [fig_supporting_100k_lcc_auc.pdf](fig_supporting_100k_lcc_auc.pdf)

Caption: Supporting matched100K development results, reserved for backup slides or appendix.

Sources:

- `antispoof/notebooks/final/binary_crossdomain_v2/output/17_scale100k_C_P3SF_R7SC_crop15/runs/C_100K_crop15/metrics/lcc_combined_metrics.json`
- `antispoof/notebooks/final/binary_crossdomain_v2/output/17_scale100k_C_P3SF_R7SC_crop15/runs/P3_SF_100K_crop15/metrics/lcc_combined_metrics.json`
- `antispoof/notebooks/final/binary_crossdomain_v2/output/17_scale100k_C_P3SF_R7SC_crop15/runs/R7_SC_100K_crop15/metrics/lcc_combined_metrics.json`

Static fallback used: **No**. Supplied literature values used: **No**.

Protocol caveats: 

Reference keys: none

Exact numeric values:

```json
{
  "Clean": 0.8482729697132887,
  "WBST": 0.8519784359546514,
  "MAST-PAD": 0.8678437816372155
}
```

## External references and verification

### Graph Transformer

AUC 0.833 verified in Table 1; 10.84M verified in architecture section. Abstract/result prose differ; use the requested table entry.

- [Source 1](https://www.esann.org/sites/default/files/proceedings/2023/ES2023-14.pdf)

### AENet

LCC AUC 0.868 and 11.22M verified in supplied repository benchmark; paper URL identifies the model.

- [Source 1](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/1485_ECCV_2020_paper.php)
- [Source 2](https://github.com/kprokofi/light-weight-face-anti-spoofing)

### MADDG

AUC 84.51% cross-checked in AMEL and PatchNet baseline tables, O&M&I to C. Original CVF page could not be fetched in this session.

- [Source 1](https://openaccess.thecvf.com/content_CVPR_2019/html/Shao_Multi-Adversarial_Discriminative_Deep_Domain_Generalization_for_Face_Presentation_Attack_Detection_CVPR_2019_paper.html)
- [Source 2](https://arxiv.org/pdf/2207.09868)
- [Source 3](https://arxiv.org/html/2203.14325v1)

### AMEL

AUC 94.39% verified in AMEL O&M&I to C table.

- [Source 1](https://arxiv.org/abs/2207.09868)
- [Source 2](https://arxiv.org/pdf/2207.09868)

### PatchNet

AUC 94.58% verified in original arXiv paper Table 8; CVF page could not be fetched.

- [Source 1](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_PatchNet_A_Simple_Face_Anti-Spoofing_Framework_via_Fine-Grained_Patch_Recognition_CVPR_2022_paper.html)
- [Source 2](https://arxiv.org/html/2203.14325v1)

### GAC-FAS

AUC 95.16% verified in Table 1, reported with 0.09 standard deviation. Requested point estimate only is plotted; this is not a matched seed comparison.

- [Source 1](https://openaccess.thecvf.com/content/CVPR2024/html/Le_Gradient_Alignment_for_Cross-Domain_Face_Anti-Spoofing_CVPR_2024_paper.html)
- [Source 2](https://arxiv.org/html/2402.18817v1)

### SA-FAS

AUC 95.37% verified in original arXiv paper Table 1. This selected-best reporting differs from its Table 2 mean/std protocol. CVF page could not be fetched.

- [Source 1](https://openaccess.thecvf.com/content/CVPR2023/html/Sun_Rethinking_Domain_Generalization_for_Face_Anti-Spoofing_Separability_and_Alignment_CVPR_2023_paper.html)
- [Source 2](https://arxiv.org/pdf/2303.13662)

## Missing figure

[Frequency counterfactual examples — source data needed](missing_frequency_examples_note.md)
