# Train100K / Val15K scale validation: C, P3, Q3 (v1)

Status: frozen scale-validation design, 2026-10-03. This addendum extends `PAD_DATA_AND_EVALUATION_CONTRACT_v2_CROSSDOMAIN_BINARY.md` (SHA256 `2f1f6eb6351550d491e4fb59e2151b1ccd9b000c0985cd32f12621f9d58c5f91`). It does not change the parent localization, preprocessing, label, or evaluation contract. The question is whether the C baseline and the P3/Q3 training-only frequency mechanisms observed at Train10K remain effective at Train100K, without a new hyperparameter search.

## Sources and deterministic scale manifests

The source pools are v1 `final_train_full.csv` and `final_val50k.csv`, respectively. The v1 and v2 resource bundles must have identical frozen CelebA SCRFD cache SHA, the same CUDA detector SHA and parent provenance; no SCRFD inference or GT fallback is permitted. From each ordered pool, use `sklearn.model_selection.train_test_split` with `train_size=100000` or `15000`, `stratify=three_class_label`, `random_state=43`; retain selected rows in original pool order. Add `binary_label=int(three_class_label!=0)` and set `sampling_seed=43`. Each selected row must be unique, `VALID`, `usable_for_pad=True`, and `bbox_origin=SCRFD`; all three original classes must occur. Train and Val must be sample and subject disjoint. Save the two CSVs and their ordered-key, file, class, binary, and subject fingerprints once; existing files must match, never be silently replaced. The locally generated reference CSV SHA256 values are Train100K `5a4bab6ab7c11b275e215552c8dd6e4bb4524aac094804ef1492abfb5126cc4e` and Val15K `cefc7bb4be38928d7c8907c7c4e521c326b9f2d7481aa33ccdd9cd0febe2ba54`.

Val50K is sampled directly from its frozen v1 manifest. Some v1 Val rows are absent from the v2 *eligible Train* manifest because that manifest applies a Train-specific minimum-box filter; this does not change the v1 Val pool. The shared SCRFD cache and each selected record are still verified. All three methods consume the exact same ordered scale CSVs.

## Shared model and training recipe

Use official PAD-pretrained MiniFASNetV2 compatible feature weights, skip its historical probability classifier, expose the 128-D pre-classifier representation, and initialize the same Linear(128,2) PAD, Linear(128,11) spoof-type, Linear(128,5) lighting, and Linear(128,40) attribute heads with train seed 100. Assert identical initial-state SHA for all three runs. Real=0, Attack=1, and higher `z_real-z_attack` means more Real. Clean-only `L_aux=0.1 CE(spoof_type)+0.1 CE(lighting)+BCEWithLogits(attributes[Real])`; an all-Attack batch uses differentiable zero.

Use raw SCRFD bbox, Train jitter p=0.2/scale 0.95–1.05/center shift ±0.05 face size, 2.7× crop, 80×80 BGR float32 [0,1]. Training uses HorizontalFlip p=0.5; ISONoise color shift (0.15,0.35), intensity (0.2,0.5), p=0.2; RandomBrightnessContrast brightness/contrast limits 0.2, brightness_by_max=True, p=0.3; MotionBlur limit 5, p=0.2. Evaluation uses no jitter or stochastic augmentation. No normalization, label smoothing, weighted sampler, mixup, cutmix, gradient clipping, or differential LR.

All methods use SGD lr 0.005, momentum 0.9, weight decay 5e-4, batch 256, AMP as in the mini notebooks, max 20 epochs, MultiStepLR milestones [6,14] gamma 0.2 with step after each completed epoch, earliest stop epoch 8, and patience 4. Expected LR is 0.005 during epochs 1–6, 0.001 during 7–14, and 0.0002 during 15–20. Batch-256 OOM stops the run. Checkpoint selection minimizes Val15K ACER, tie-breaks on higher AUC, and uses an explicit no-improvement counter. C is eligible every epoch. P3/Q3 use epoch 1 clean-only warm-up, ineligible for best checkpoint or active patience; epoch 2 begins spectral training.

Keep separate deterministic streams for sample order, worker augmentation/bbox jitter, spectral gain, and Q3 band selection, and save/restore their state on resume. All three methods share the same base sample order, training seed, initialization, optimizer, schedule, and augmentation policy.

## Exactly three frozen runs

| Run ID | Active training objective | Selection |
|---|---|---|
| `C100k_copy_pretrained` | `L_PAD_clean + L_aux` | None; no FFT or KL |
| `P3_100k_worst_ce25` | `0.75 L_PAD_clean + 0.25 L_PAD_worst + L_aux` | Minimum aligned binary margin among LOW/MID/HIGH |
| `Q3_100k_random_ce25_kl01` | `0.75 L_PAD_clean + 0.25 L_PAD_random + 0.1 L_KL + L_aux` | Uniform independent LOW/MID/HIGH per sample |

P3/Q3 use the same smooth Gaussian radial generator: centers [0.15,0.45,0.75], sigma 0.10, gain Uniform(0.65,0.90), DC preserved, unchanged phase, common channel field, attenuation only. P3 candidate forwards are under `eval()` and `no_grad()` with original mode restored before differentiable training forwards. Q3 selects from its dedicated band RNG without candidate model forwards. `L_KL=F.kl_div(log_softmax(z_random.float()),softmax(z_clean.float().detach()),reduction='batchmean')`. There is no harmful-margin loss. Frequency has no inference-time component.

## Evaluation, reporting, and provenance

Train on Train100K and evaluate Val15K only during epochs. Reload the best eligible checkpoint, recalculate Val15K, lock its minimum-ACER threshold, write `evaluation_freeze.json`, then evaluate official CelebA Test once and official LCC once. Test/LCC threshold metrics use only the locked Val15K threshold. Preserve all candidates and identify scored SCRFD-valid rows. Report AUC, EER, TPR@FPR1%, APCER, BPCER, ACER/HTER, accuracy, threshold, candidate/scored counts, and coverage. Include CelebA Test original-class candidate/scored counts. LCC has already been seen during mechanism development and is an **external-development cross-domain stress benchmark**, not an untouched final benchmark.

Log every epoch's phase, LR, total and clean PAD loss, clean auxiliary terms, Val ACER/AUC/EER/threshold, best flag, patience count, wall time, and peak CUDA memory. P3/Q3 additionally log spectral loss, CE weights, gain and band counts/fractions, clean/spectral margins, harmful fraction and flip rate; Q3 logs KL and its weighted contribution plus cumulative band fractions. An unusual Q3 band imbalance produces a warning only. Save each run's config, initial-state audit, best/last states, optimizer/scheduler/scaler/RNG state, Val/Test/LCC predictions/metrics, freeze, and completion marker. Resume rejects any run ID, config hash, manifest, resource provenance, source/checkpoint SHA, or initial-state SHA mismatch.

The report compares C vs P3, C vs Q3, and P3 vs Q3 across Val15K, Test, and LCC with raw metrics and signed deltas. It cannot automatically select a winner or tune settings on any held-out outcome. A convincing, non-fragile frequency result would motivate a **separate** full-data protocol comparing the clean baseline with one frozen frequency method; no full-data training occurs here. These are single-seed scale contrasts and do not prove statistical significance.
