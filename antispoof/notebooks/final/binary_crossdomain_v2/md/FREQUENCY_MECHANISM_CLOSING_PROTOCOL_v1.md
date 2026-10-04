# Frequency mechanism closing protocol v1

Status: frozen design for four PAD-pretrained Mini Train10K/Val3K runs, 2026-10-02. This is an experiment-specific addendum to `PAD_DATA_AND_EVALUATION_CONTRACT_v2_CROSSDOMAIN_BINARY.md` (SHA256 `2f1f6eb6351550d491e4fb59e2151b1ccd9b000c0985cd32f12621f9d58c5f91`). The parent contract controls data, localization, model input, candidate accounting, metrics, and evaluation isolation. `CSMR_MECHANISM_PILOT_PROTOCOL_v1.md` defines the P references. This addendum authorizes only the four spectral-training mechanisms below.

## Fixed model, data, and training

All runs initialize the same official PAD-pretrained MiniFASNetV2 features and seed-100 binary/auxiliary heads as `C_copy_pretrained`: 128-D embedding; Linear(128,2) PAD, Linear(128,11) spoof type, Linear(128,5) lighting, and Linear(128,40) attributes. Primary labels are 0 Real and 1 Attack; the PAD score is `z_real-z_attack`. Auxiliary supervision applies to the clean view only: `L_aux=0.1 CE(spoof_type)+0.1 CE(lighting)+BCEWithLogits(attributes[Real])`; an all-Attack batch uses differentiable zero for the attribute term.

Use the unchanged, ordered `mini_train10k.csv` and `mini_val3k.csv`, plus official source Test and LCC manifests, all verified against the 00c resource bundle, raw CUDA SCRFD caches, and recorded fingerprints. Data seed 43; training seed 100. Train uses frozen raw SCRFD bbox with existing 20% jitter, 2.7× crop, 80×80 BGR float32 [0,1], HorizontalFlip 0.5, ISONoise color shift (0.15,0.35) and intensity (0.2,0.5) at 0.2, RandomBrightnessContrast limits 0.2 at 0.3, and MotionBlur limit 5 at 0.2. Evaluation has no jitter or random augmentation. No SCRFD inference occurs here.

All four use SGD lr 0.005, momentum 0.9, weight decay 5e-4, batch 256, MultiStepLR milestones [6,14] gamma 0.2, at most 20 epochs, AMP as in notebook 02, no label smoothing, weighted sampling, mixup, cutmix, or gradient clipping. Batch-256 OOM is fatal. Epoch 1 is clean `CE_PAD+L_aux` warm-up and is ineligible for final checkpoint selection and active-phase patience. Active training begins in epoch 2. Minimum Val3K ACER selects the checkpoint, with maximum Val3K AUC breaking ties; earliest stop epoch 8, patience 4, explicit no-improvement count.

## Fixed spectral generator and selection

Training-only spectral attenuation uses the tested LOW/MID/HIGH smooth Gaussian radial masks, centers [0.15,0.45,0.75], sigma 0.10, independent per-sample gain Uniform(0.65,0.90), unchanged phase, common attenuation field across color channels, preserved DC, and no amplification. For a binary PAD logit pair, aligned margin is `m=z_real-z_attack` for Real and `m=z_attack-z_real` for Attack. WORST chooses the band with minimum aligned margin using candidate forwards under `model.eval()` and `torch.no_grad()`, restores the original mode, then trains on a differentiable selected-view forward. RANDOM chooses LOW/MID/HIGH uniformly for each sample using a separate deterministic band RNG, independent of labels/model/margins, without candidate model forwards. Data-shuffle and augmentation seeds, gain RNG, and band RNG are separated. Record and restore the gain/band RNG states on resume.

## Exactly four runs

Let `L_c=CE(z_clean,y)`, `L_s=CE(z_selected_spectral,y)`, and `L_KL=KL(softmax(z_clean.detach()) || softmax(z_selected_spectral))`, computed with float32 `F.kl_div(F.log_softmax(z_spectral.float(),dim=1),F.softmax(z_clean.float().detach(),dim=1),reduction="batchmean")`. The clean prediction is a detached KL teacher. No run uses harmful-margin SmoothL1 or any other margin loss.

| Run ID | Selection | Active objective |
|---|---|---|
| `Q1_random_ce25` | RANDOM | `0.75 L_c + 0.25 L_s + L_aux` |
| `Q2_worst_ce25_kl01` | WORST | `0.75 L_c + 0.25 L_s + 0.1 L_KL + L_aux` |
| `Q3_random_ce25_kl01` | RANDOM | `0.75 L_c + 0.25 L_s + 0.1 L_KL + L_aux` |
| `Q4_worst_kl_only` | WORST | `1.0 L_c + 0.1 L_KL + L_aux` |

Q1 versus P3 isolates RANDOM versus WORST selection under matched CE weights, generator/gain, optimizer/data/initialization. Q2 versus Q3 differs only in spectral selection policy. Q2 versus Q4 shares WORST selection, generator/gain, and KL beta 0.1; only the allocation of clean versus spectral supervised CE changes. Q4 may calculate detached spectral CE as a diagnostic; its optimization weight remains exactly zero. KL beta 0.1 does not imply gradient strength comparable to historical harmful-margin lambda 0.1.

## Isolation, provenance, and reporting

Each run has an independent config hash, initialization audit, warm-up and best checkpoints, last state with optimizer/scheduler/AMP/global and spectral RNG states, history, patience and best metrics. Resume requires the same run ID, config hash, initial-state fingerprint, manifests, resource provenance, model-source hash, and pretrained-checkpoint hash. Unknown `RUN_FILTER` IDs fail before training. Prior C/P runs are not loaded by this notebook; the user will aggregate them separately.

During training use only Train10K and Val3K. Reload the selected eligible checkpoint, recalculate Val3K metrics and lock its minimum-ACER threshold, write `evaluation_freeze.json`, then evaluate official CelebA Test once and LCC once. Never adjust checkpoint, threshold, gain, beta, CE weights, band distribution, or recipe based on held-out results. Test and LCC threshold metrics use the locked Val threshold and report all candidates, scored rows, and detector coverage. Deployment remains SCRFD → 2.7× crop → 80×80 BGR [0,1] → spatial MiniFASNetV2 PAD logits; no FFT, frequency branch, spectral fusion, or extra inference head.

Each epoch logs LR, phase, total/clean/spectral CE, CE weights, beta, KL and weighted KL, clean-only auxiliary components, and Val ACER/AUC/EER. Active epochs also log policy, mean gain, band counts/fractions, clean/spectral aligned margins, mean drop, harmful fraction (descriptive for RANDOM), and clean-correct-to-spectral-wrong flip rate. The report gives raw Val/Test/LCC metrics and Q2↔Q3 and Q2↔Q4 paired deltas. It names the later P3↔Q1, P3↔Q2, and P6↔Q2 questions without loading earlier results. It must display conflicting metrics and must not automatically select a winner. These are single-seed pilot contrasts, not significance tests.
