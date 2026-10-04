# Scale100K R6/R7 confirmation, protocol v1

## Question and scope

Confirm the selected all-band CE and harmful-gated worst mechanisms at the existing C/P3 100K scale. Train exactly `R6_100k_all3_ce25` and `R7_100k_harmful_gated_worst`. Existing `C100k_copy_pretrained` and `P3_100k_worst_ce25` are read-only references; no C/P3/Q3 or additional mechanism is trained. LCC remains an external-development domain. CASIA is reserved for later confirmation and is not accessed here.

## Frozen controls

Read `scale100k_train.csv` and `scale15k_val.csv` directly from `SCALE_MANIFEST_ROOT`, which must point to notebook 04's existing frozen manifest directory. Do not resample, rebuild, or derive splits. Data seed is 43 and train seed is 100. Use the completed 00d resource bundle for CelebA localization/auxiliary labels and LCC localization/manifests. Copy the two frozen manifest files into the output for portability without changing their content.

Reuse notebook 04's MiniFASNetV2 wrapper, 128-D representation, binary PAD head and 11/5/40 auxiliary heads. Load its compatible official PAD-pretrained feature weights, skipping the historical classifier and initializing project heads with seed 100. Record the initial state SHA and require the two runs to start from the same state. Preserve notebook 04's data-order generator and worker/epoch augmentation seed policy, with an independent bbox RNG. Spectral gains use a separately seeded generator for each run and cannot consume the base augmentation RNG.

Keep frozen raw SCRFD boxes, jitter probability 0.20 with scale 0.95–1.05 and shift ±0.05 face size, 2.7× crop, 80×80 BGR float input divided by 255, and the existing HorizontalFlip/ISONoise/brightness-contrast/MotionBlur recipe. No normalization, mixup, cutmix, label smoothing or weighted sampler is added. Evaluation has no jitter or augmentation, and no detector runs.

Use SGD with lr 0.005, momentum 0.9 and weight decay 5e-4; batch 256; at most 20 epochs; MultiStepLR milestones [6,14], gamma 0.2, stepped after each epoch. LR is 0.005 for epochs 1–6, 0.001 for 7–14 and 0.0002 for 15–20. Val15K minimum ACER selects the checkpoint, breaking ties with higher AUC. Earliest early stopping is epoch 8, with four active epochs without improvement. Both runs warm up for one clean epoch, excluded from best selection and patience.

## Mechanisms and loss

Use LOW/MID/HIGH centers [0.15,0.45,0.75], sigma 0.10 and one gain per sample from Uniform(0.65,0.90), shared across bands/channels. Preserve DC and phase; do not amplify or tune spectral parameters.

```
L_aux = 0.1 L_spoof_type + 0.1 L_lighting + L_attributes
L_R6 = 0.75 CE_clean + 0.25 mean(CE_LOW, CE_MID, CE_HIGH) + L_aux
L_R7 = 0.75 CE_clean + 0.25 CE_harmful + L_aux
```

Auxiliary supervision is clean-view only and attributes are Real-only. Warm-up loss is `CE_clean + L_aux`.

R6 supervises all three views unconditionally and uses the tested separate train-mode band forwards to preserve its BatchNorm updates. R7 temporarily uses `eval()` and `no_grad()` to evaluate clean and three bands. Its label-aligned margin is `z_real-z_attack` for Real and its negative for Attack. Select the smallest band margin and detach the gate `m_worst < m_clean`. Restore the previous training mode, forward the selected view differentiably and average its per-sample CE over harmful samples. Zero harmful samples gives a differentiable zero. No SmoothL1, margin regression, KL or random-band loss is used.

## Evaluation, diagnostics and resume

Evaluate Val15K only inside the epoch loop. Reload best, recompute the Val15K min-ACER threshold and write `evaluation_freeze.json` before CelebA full Test or LCC inference. Full Test is required in this scale-confirmation round. Use one model-specific locked threshold for Test and LCC training/development/evaluation/combined. Retain failed localization candidates as unscored; report candidate/scored counts and coverage. Raw LCC counts are 8299/2948/7580/18827. Concatenate the three physical LCC prediction tables and recompute pooled metrics; do not average metrics.

Report AUC, EER, TPR@FPR1%, APCER, BPCER, ACER/HTER, accuracy and threshold for Val15K, Test and each LCC population. R6 diagnostics include clean/all3/per-band CE, clean/per-band aligned margins, gain and total loss. R7 diagnostics include harmful CE/count/fraction, worst-band rates, clean/worst margin, margin drop, flips, gain and total loss.

Each run resumes independently from an epoch-level state containing model, optimizer, scheduler, scaler, epoch, best checkpoint payload, patience, histories and RNG states. Run/config/initial-state checks prevent cross-run resume. A stored best checkpoint payload recovers the committed best state if interruption occurred while writing a newer epoch.

## Artifacts and comparison

Use `/kaggle/working/binary_crossdomain_scale100k_R6_R7_v1/`, with independent `runs/<run_id>/` checkpoints, config, initialization audit, histories, diagnostics, frozen threshold, metrics and `predictions/`. Save this protocol, `comparison_summary.csv/json` and `scale100k_R6_R7_report.md` at root. `RUN_FILTER` can select either run for separate Kaggle sessions; each run produces its artifacts and run-specific ZIP without waiting for the other.

`EXISTING_SCALE100K_ROOT` is optional and is read only after new-run evaluation. Its C/P3 Val/Test/official-LCC results can enter a four-way table without loading or training their models. Optional `EXISTING_FULL_LCC_ROOT` supplies notebook 05's separate pooled-evaluation outputs; unavailable split metrics remain pending rather than inferred. Optional `MICRO_R6_R7_SUMMARY` supplies measured 10K results for descriptive cross-scale deltas. Reference paths never gate training. No automatic final winner or extra search is selected.

Validate parsing/compilation, run IDs, reused source pipeline, initialization equality, R6/R7 loss and detached BN-safe gating with small checks before expensive Kaggle execution. A single training seed and missing optional reference populations limit comparisons; no unmeasured scale-retention claim is made.
