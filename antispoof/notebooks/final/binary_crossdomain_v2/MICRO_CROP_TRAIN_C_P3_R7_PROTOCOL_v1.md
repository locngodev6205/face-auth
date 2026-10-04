# Matched Train10K crop ablation: C / P3 / R7, v1

## Question and hypothesis

Does tighter source-training context improve transfer to LCC when training, source calibration and target evaluation use the same crop? Excessive expansion may encourage reliance on source-specific surrounding cues unavailable in already tightly cropped target imagery. This is a hypothesis, not a measured conclusion. Prior inference-only observations do not establish matched-training performance.

## Design and fixed controls

Six possible runs: C_crop15, C_crop20, P3_crop15, P3_crop20, R7_crop15, R7_crop20. The only treatment within each mechanism is crop factor 1.5 or 2.0. Do not retrain 2.7x. MODEL_FILTER is one of C/P3/R7, default P3. CROP_FILTER=None selects both crops; [1.5] or [2.0] selects one. Each family/run is independent and has no writable dependency on another family.

Read the existing mini_train10k.csv and mini_val3k.csv, seed 43; no resampling. Train seed 100. Reuse notebooks 06/07's MiniFASNetV2 binary/11/5/40 heads, official PAD-pretrained feature initialization, spectral/auxiliary logic, dataset order, augmentation streams and metrics. Record identical initial-state SHA across new runs. Bbox jitter is per sample/epoch with its own seeded RNG; base augmentation and spectral gains retain the micro streams, preventing crop or spectral selection from changing the underlying augmentation draw policy.

Keep frozen valid raw SCRFD boxes and eligibility, jitter p=.20, scale .95–1.05 and center shift ±.05 face size. Apply the run's crop to jittered training boxes and unjittered evaluation boxes. Use the exact current CropImage-compatible geometry including boundary scale limit and inward translation, 80x80 BGR float32 divided by 255. Keep HorizontalFlip .5, ISONoise .2, brightness/contrast limits .2 with probability .3, MotionBlur limit 5 with probability .2. No normalization, mixup, cutmix, weighted sampler, smoothing, alignment, detector or GT fallback.

## Training and mechanism losses

SGD lr .005, momentum .9, weight decay 5e-4; batch 256; maximum 20 epochs; MultiStepLR [6,14], gamma .2, stepped after each epoch. Preserve earliest stopping epoch 8 and patience 4 with minimum Val3K ACER/higher AUC tie-break. No phase-aware stopping change. C is eligible from epoch 1. P3/R7 have one clean warm-up epoch excluded from best selection and patience, then spectral-active selection.

```
L_aux = .1 L_spoof_type + .1 L_lighting + L_attributes
L_C = CE_clean + L_aux
L_P3 = .75 CE_clean + .25 CE_worst + L_aux
L_R7 = .75 CE_clean + .25 CE_harmful + L_aux
```

Auxiliary loss is clean only; attributes are Real only. P3/R7 warm-up uses CE_clean+L_aux. C does not generate spectral views. Spectral bands LOW/MID/HIGH have centers [.15,.45,.75], sigma .10, gain Uniform(.65,.90), preserved DC/phase, no amplification. Selection reuses tested eval/no_grad forwards with mode restored, selecting the lowest label-aligned margin. The existing micro helper also gathers clean selection margin without BN updates; only R7 uses it to gate the loss. R7 detaches harmful=(m_worst<m_clean), normalizes selected-view CE over harmful samples and uses differentiable zero for no harmful samples. No KL, SmoothL1, margin regression, random/all-band supervision.

## Matched evaluation and diagnostics

Only Val3K is inferred within the epoch loop, at that run's training crop. Reload best, recalibrate the source minimum-ACER threshold and freeze it before full LCC inference, always at the same crop. No CelebA full Test is loaded or evaluated. LCC train/dev/eval raw counts are 8299/2948/7580; combined is 18827. Concatenate physical prediction tables and recompute pooled metrics. Never tune on LCC or average split metrics.

Primary endpoint is LCC Combined AUC, with official-evaluation AUC and pooled EER/HTER/TPR@FPR1% alongside Val3K ACER/AUC as source guard. Save AUC/EER/TPR, APCER/BPCER/ACER/HTER/accuracy, locked threshold, candidate/scored counts and coverage. Failed localization candidates stay unscored; image errors stop rather than silently alter eligibility.

Collect actual crop geometry without extra decoding or RNG draws: boundary-limited fraction plus mean/median crop-width/image-width and crop-height/image-height. Training diagnostics describe each epoch's jittered boxes; Val and LCC describe deterministic boxes. Pooled geometry concatenates physical scored samples. These diagnostics do not influence checkpoint selection.

## Resume, artifacts and historical comparison

Each run has config, best.pth, last_state.pth, history, spectral diagnostics (also produced for C), crop diagnostics, freeze, Val3K/LCC metrics and predictions. Resume stores model/optimizer/scheduler/scaler/epoch, committed best checkpoint payload, ACER/AUC/patience, histories/geometry summaries and RNG states. Only one correctly matched restore_rng definition exists; run/config/crop checks reject cross-run resumes. Restore the committed best payload before continuing.

Output root is /kaggle/working/micro_crop_train_10k_v1/<family>/<run>/. Every completed run produces complete standalone artifacts and a run ZIP; each family execution creates micro_crop_<family>_10k_v1.zip with protocol and family comparison/report. Export summaries after each run so another unfinished crop does not block completed artifacts.

HISTORICAL_CROP27_CSV and HISTORICAL_P3_R7_CSV are optional read-only result paths, loaded after new-run evaluation. Accept historical C/R0_clean, P3/R1_p3_ce25, R7/R7_harmful_gated_worst. Append a historical 2.7 row and deltas only when measured values are available; missing controls remain pending. Training does not depend on these paths. Do not infer missing metrics or tune a crop from target results.

## Interpretation and validation

Ask explicitly whether reduced training context improves cross-domain separability rather than only changing calibration. Higher LCC AUC is stronger descriptive evidence of transfer improvement; HTER-only improvement may reflect operating-point effects. Look for consistent effects across C/P3/R7 after merging independent family tables; larger spectral-only effects may indicate mechanism interaction. Single-seed 10K screening and historical comparisons do not prove the hypothesis or establish a final method/crop winner.

Validate parse/compile/cleared outputs, exact filters/runs/crops, original helpers/streams, matched crop paths, three loss formulas, detached zero-safe harmful gate, BN-safe selection, unique resume restoration, no Test/detector/GT/target fitting, and pooled metrics using tiny offline checks. Do not launch expensive training during notebook creation.
