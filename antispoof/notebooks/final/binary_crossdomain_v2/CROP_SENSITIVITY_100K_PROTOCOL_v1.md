# Frozen 100K crop sensitivity, protocol v1

## Question and hypothesis

Measure whether reducing face-crop context changes cross-domain separability on LCC for frozen C/P3/R7 checkpoints trained with 2.7x crops. Tighter crops may reduce nuisance/background shift; no outcome is assumed. This is inference-time sensitivity, not evidence of the performance of models trained with each alternative crop.

## Paired controls

Support C100k_copy_pretrained, P3_100k_worst_ce25 and R7_100k_harmful_gated_worst. Default MODEL_FILTER is ["P3"]. Load existing best.pth only and run eval/no_grad with all parameters frozen. No training, optimizer, scheduler, BatchNorm update, detector, GT fallback, alignment, spectral perturbation, TTA or full CelebA Test is performed.

Evaluate exactly [1.0, 1.5, 2.0, 2.7]. Reuse the binary-v2 MiniFASNetV2 wrapper and original parameterized CropImage-compatible helper. Its image-boundary behavior includes limiting context scale to fit the image and translating edge crops inward; do not replace it with a new center/clamp implementation. At 2.7 it must produce exactly the existing preprocessing. Geometry diagnostics use those same final pixel bounds. Boundary clipping records whether the requested centered context would exceed the image bounds or require the original helper's scale limit.

Keep checkpoint, frozen SCRFD raw bbox, eligible candidates, images, 80x80 resize, BGR, float32 divided by 255, PAD score z_real-z_attack, and metric implementations fixed. No jitter/augmentation/normalization. Read the existing frozen scale15k_val.csv, never resample. Read the completed 00d LCC manifests/caches. Failed localization candidates remain unscored, with counts/coverage reported; image read errors stop rather than silently changing candidates.

## Calibration and evaluation

For every model/crop pair infer Val15K, compute its minimum-ACER threshold with the existing three-class source implementation and freeze that threshold before LCC. Every LCC threshold-based metric uses only that pair's source threshold. No LCC threshold fitting or automatic crop selection.

Infer physical LCC training/development/evaluation, raw counts 8299/2948/7580. Combined raw count is 18827. Concatenate prediction tables (including failed candidates) and recompute pooled metrics over scorable samples; never average split metrics. Primary outcomes are LCC evaluation/combined AUC. Secondary outcomes are EER, TPR@FPR1%, locked-source APCER/BPCER/HTER/ACER/accuracy. Also report source metrics, threshold, candidate/scored counts and coverage.

The 2.7x result is the control. Optional EXISTING_27X_METRICS supplies known source/LCC metric dictionaries, JSON files, or a run directory. Compare available metrics using absolute numerical tolerance 0.005 and exact candidate/scored counts. A discrepancy stops clearly. Without supplied references, report that historical numerical reproduction is unchecked; exact helper equivalence still holds. Do not invent reference values.

## Efficiency and artifacts

Each physical image is decoded once per enabled model/population; all four preprocessed crops are generated in that read, and model forwards run separately by crop within each batch. Only a bounded batch of crop tensors is held. No persistent full-image/tensor cache or large preflight framework. Print progress for source/LCC batches and artifact export.

Use /kaggle/working/crop_sensitivity_100k_v1/<C|P3|R7>/crop_<s>/ for source/LCC predictions, metrics, freeze and crop diagnostics. For each model save comparison CSV/JSON, deltas versus 2.7, report and four plots. Multi-model tables/reports describe shared versus method-specific effects. Save protocol at root and per-model ZIPs; multiple models also produce the combined ZIP.

Diagnostics are sample counts, fraction of requested crops limited by image boundaries and actual mean crop-width/image-width and crop-height/image-height, reported per population and pooled for LCC. They are descriptive, never selection criteria.

## Interpretation and validation

An AUC change describes separability; HTER-only improvement may reflect calibration/operating-point effects. Joint AUC and EER/HTER improvement supports a context-shift hypothesis more strongly. Isolated small improvements warrant caution. Do not declare an optimal crop, make significance claims from a single checkpoint, or treat this as a matched train+eval study. Consistent improvements can motivate a later matched training experiment.

Before Kaggle, validate notebook parse/compile and cleared outputs, exact crop factors, P3-only filtering, original crop equivalence, eval-only calls, per-crop source calibration and pooled LCC computation with small offline checks. Full numerical reproduction and performance require actual Kaggle inference.
