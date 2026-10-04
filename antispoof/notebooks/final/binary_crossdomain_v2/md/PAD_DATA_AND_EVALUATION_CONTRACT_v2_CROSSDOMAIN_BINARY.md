# PAD data and evaluation contract v2: binary cross-domain

Status: frozen pilot specification, 2026-10-02. Scope: `binary_crossdomain_v2/` only. The parent `PAD_DATA_AND_EVALUATION_CONTRACT_v1.md` remains the contract for historical v1 notebooks. Parent SHA256: `aa67032a79627636d967dc885202540a34410c2b4d64522a5604d3aedff85b2d`.

## Localization and model input

Use only the frozen 00-v2 SCRFD-500M raw bbox caches. Detector SHA256 `5e4447f50245bbd7966bd6c0fa52938c61474a04ec7def48753668a9d8b4ea3a`, confidence 0.5, sizes 640, 480, 320, CUDA execution provenance. A scorable record has `status == VALID`, `usable_for_pad == True`, `bbox_origin == SCRFD`, `bbox_source` beginning `SCRFD`, and a finite, positive raw `bbox_xyxy`. No GT bbox fallback or new localization. Train hygiene rejects unscorable records and native minimum bbox side below 48 px, then enforces the inherited total, per-class, class-gap, and suspicious-rate guardrails. Retain failed candidates in source and target audits.

Use raw SCRFD bbox → optional Train jitter (20% chance, 0.95–1.05 scale, ±0.05 face-size center shift) → MiniFASNet 2.7× crop → 80×80 BGR float32 [0,1]. Evaluation has no jitter or augmentation. CSMR is training-only; inference has no FFT, spectral branch, selector, or margin loss.

## Labels and objectives

The v2 primary label is 0 Real and 1 Attack, where Attack combines historical 1 Physical and 2 Digital. Retain the original three-class labels for stratification and subgroup diagnostics. Copied v2 models emit `[z_real,z_attack]` and security score `d=z_real-z_attack` (higher is more Real). Historical comparators A/B emit three logits and use `d=z_real-logsumexp(z_physical,z_digital)`.

CelebA metadata supplies spoof type `int(labels[40])` in 0–10, lighting `int(labels[41])` in 0–4, and 40 binary attributes `labels[0:40]`. The notebook-local training-only heads are Linear(128,11), Linear(128,5), and Linear(128,40). For clean samples, `L_base=L_PAD+0.1 L_spoof+0.1 L_lighting+L_attr`, with binary CE, CE, CE, and BCEWithLogits respectively. Attributes contribute only for Real samples; an all-Attack batch uses differentiable zero.

CSMR E/F use warm-up epoch 1 with `L_base`, excluded from final checkpoint selection and patience. Active CSMR uses the historical radial masks and BN-safe candidate selection, but binary aligned margin: Real `m=d`, Attack `m=-d`; select the smallest candidate margin. The active objective is `0.5 L_PAD_clean + 0.5 L_PAD_worst + 0.2 L_margin + 0.1 L_spoof_clean + 0.1 L_lighting_clean + L_attr_clean`. `L_margin` is mean SmoothL1 from worst aligned margin to detached clean aligned margin, masked to harmful drops. Auxiliary losses use clean views only. Fixed CSMR: centers [0.15,0.45,0.75], sigma 0.10, gain [0.65,0.90], DC preserved, selection chunk 32.

## Pilot: six frozen runs

Mini Train10K samples from v1 `final_train_full.csv`; Mini Val3K samples from v1 `final_val50k.csv`. Seed 43; deterministic stratification by original three-class label; exact same ordered manifests for all runs; no sample or subject overlap. Training seed 100. A is legacy pretrained 3-logit AdamW; B is legacy scratch 3-logit AdamW; C and D are copied binary+auxiliary pretrained/scratch; E and F are C/D plus CSMR. A/B use historical 12 epochs, earliest stop 5, patience 3, label smoothing 0.1 and mild augmentation. C/D/E/F use the literature-adapted, compute-compressed recipe: SGD lr 0.005, momentum 0.9, weight decay 5e-4, batch 256, 20 epochs, MultiStepLR milestones [6,14], gamma 0.2, earliest stop 8, patience 4, no label smoothing/weighted sampling/mixup/cutmix. Copied augmentation: horizontal flip 0.5, ISO noise color shift (0.15,0.35), intensity (0.2,0.5), p 0.2; brightness/contrast limits 0.2, p 0.3; MotionBlur limit 5, p 0.2. Keep MiniFAS geometry and BGR [0,1].

Checkpoint selection uses minimum Mini-Val3K ACER, tie highest AUC, with explicit no-improvement counter. Threshold is minimum-ACER Mini-Val3K threshold only. Freeze checkpoint and threshold before one official CelebA Test and one official LCC evaluation per run. Do not use Test/LCC to choose any model, threshold, optimizer, schedule, augmentation, gain or lambda. LCC is pilot/pre-scale cross-domain evidence; primary pilot metrics are AUC and EER. Report AUC, EER, TPR@FPR1%, APCER, BPCER, ACER/HTER, binary accuracy, locked threshold, candidate_n, scored_n and detector coverage. Preserve all 67,170 official CelebA Test and 7,580 official LCC candidates (314 Real, 7,266 Spoof), including unscorable rows; metrics use scored rows only.

C vs E and D vs F isolate CSMR. A/B vs C/D compare training recipes with several simultaneous changes and are not single-factor causal ablations. Early stopping can prevent reaching the second LR decay; report this explicitly. No numeric result is implied by this contract.

## Future full cross-domain direction

Reconstruct eligible full Train from **all** official CelebA Train keys plus frozen 00-v2 records; do not concatenate v1 Train+Val, reserve v1 Val subjects, or exclude Train subjects solely because they appear in official Test. Audit Train/Test identity overlap. Official CelebA Test is source development for future checkpoint/threshold selection; official LCC is primary cross-domain test. This is a distinct protocol from the Mini pilot, whose source development set is Mini Val3K.

## Provenance and change control

00c verifies parent contract, metadata, detector/cache hashes and frozen CUDA provenance, then exports this exact v2 contract and ordered manifest fingerprints. 01 verifies those artifacts plus model source/checkpoint hashes and stores per-run configuration and resume hashes. A mismatch fails loudly. Any change to data, architecture, preprocessing, thresholds, splits, metrics, augmentation, or schedule requires a new explicit contract version. Kaggle runtime execution and resulting metrics must be verified from actual outputs.
