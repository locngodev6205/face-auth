# PAD Micro-Search Stages 0–2 Protocol

## 1. Research question and boundary

This screen asks whether a small frequency representation contributes useful face PAD evidence beyond one shared micro spatial model. Only the frequency representation changes across M0–M4. The notebooks stop after resource freezing, one shared spatial run, five frequency/concat screens, and a result comparison.

No gated fusion, attention, SE block, auxiliary loss, frequency augmentation, alternative spatial backbone, temporal PAD, RGB-D/IR input, ONNX export, quantization, edge timing, CASIA/Replay evaluation, or full-data retraining is included.

## 2. Data split and target boundary

Stage 0 selects 5,000 items from the frozen E1 Train membership and 2,000 from the frozen E1 Validation membership. The selection is deterministic, class-stratified by largest remainder, and remains a subset of each parent membership. The existing subject-disjoint Train/Val property is asserted; subjects are not resplit. The exact E1 NPZ and run-config fingerprints are checked before selection. The notebook accesses only `train_keys` and `val_keys`.

CelebA held-out Test is excluded from Stage 0–2 candidate selection. CASIA-FASD remains reserved for later confirmation so an external dataset stays outside architecture selection. LCC-FASD is labeled **external-dev / cross-domain stress set** because it has already been observed. It is not described as an untouched final benchmark.

There is no LCC fine-tuning, adaptation, threshold calibration, checkpoint selection, or hyperparameter selection. Each branch gets its own threshold from CelebA micro-Val and applies that unchanged threshold to LCC. Detector failures remain in the manifest and coverage denominators; model metrics use the valid-face subset and report both scored count and detector coverage.

## 3. Stage 0 — Shared resources

Stage 0 writes the micro manifest, exact key lists, selected CelebA bbox subset, LCC image-level manifest/cache, class-wise detector coverage, resource fingerprints, and a ZIP handoff. The compatible cross-dataset LCC cache is reused only if its detector policy, model SHA, and protocol snapshot match. A cache rebuild requires the matching SCRFD model and retains failed detections as status records.

The selected E1 bbox audit accepts only the artifact-observed statuses `VALID` (crop factor 1.55), `FALLBACK_DET_FAIL` (1.50), and `FALLBACK_SUSPICIOUS` (1.50); `INVALID`, missing factors, factor mismatches, and unknown statuses fail before training. Stage 0 reports overall/Train/Val status counts and crop-factor counts by status, and checks all selected image paths resolve.

The optional crop cache is deterministic and contains only base crops. It is disabled by default, has a size guard, is used only for non-augmented evaluation loading, and is removed if incomplete/oversized. The same bbox status/factor contract is checked before reading cached crops. Training reads raw images so augmentation remains available.

The compatible LCC cache is reused only when the full frozen policy, detector SHA, and sampled-manifest fingerprint match the attached protocol snapshot. Cache records are joined by normalized image path, then written under a self-consistent `frame_id` key. Stage 0 retains all source candidates and detector failures, verifies exactly one cache row per manifest path, checks raw LCC paths resolve, and fingerprints the ordered common `VALID` path list. If cache rebuilding is required, InsightFace/ONNX Runtime availability and the snapshot's InsightFace version are checked before SCRFD inference; cache reuse avoids those dependencies.

## 4. Stage 1 — One shared micro spatial E1

The backbone is ImageNet-pretrained MobileNetV3-Small. The E1 task projection/head starts from random initialization. Its architecture is 224 RGB → MobileNetV3-Small/GAP → 576→256 → BatchNorm → Hardswish → Dropout(0.2) → 256→128 → Hardswish → Dropout(0.2) → 128→3. It uses the frozen E1 normalization, 1.55 detector crop, reflect padding/letterbox, mild horizontal flip/brightness-contrast/bbox jitter, and gamma OFF.

Training uses seed 42, batch 128, AMP, AdamW (backbone LR 1e-5, head LR 1e-4, weight decay 1e-4), label smoothing 0.1, gradient clipping 5, up to six epochs, patience two, and no early stop before epoch three. Micro-Val ACER selects the checkpoint and calibrates the locked binary threshold. The one selected checkpoint SHA is required by every M0–M4 run; an incomplete or incompatible local ImageNet state dict is rejected rather than leaving random backbone layers.

Crop, RGB conversion, reflect padding, AREA/LANCZOS4 resize, 224 input, normalization, jitter parameters, and augmentation probabilities match executed E1 v5.3. The notebooks use a self-contained RGB flip and NumPy brightness/contrast implementation with the same ranges/probabilities in Stage 1 and M0–M4; that implementation is documented because it is not the original Albumentations call. Gamma is off and Val/LCC are unaugmented. Each Stage 2 sample derives spatial and frequency inputs from the same augmented RGB object in one dataset read.

A shared spatial model avoids five redundant spatial retrains and keeps the comparison budget aligned. The search checkpoint is not a final independent comparison model.

## 5. Stage 2 — Frequency screening protocol

### Phase A: frequency-only

Every frequency representation maps to 64 dimensions, followed by 64→128→Hardswish→Dropout(0.2)→3. Use seed 42, batch 128, AMP, AdamW LR 3e-4 / weight decay 1e-4, label smoothing 0.1, gradient clipping 5, up to five epochs, patience two, earliest stop at epoch three. Micro-Val ACER selects the checkpoint; its own micro-Val threshold is locked for LCC.

### Phase B: spatial plus frequency

Initialize the spatial branch from Stage 1, the frequency encoder from that candidate's Phase A best state, and the fusion head randomly. Concatenate the 256-D spatial and 64-D frequency vectors, then apply 320→128→Hardswish→Dropout(0.2)→3.

B1 runs two epochs with spatial and frequency frozen and fusion trainable. B2 runs two epochs with spatial frozen and frequency plus fusion trainable. Use frequency LR 1e-4 and fusion LR 3e-4. The best concat state is selected by micro-Val ACER; its threshold is calibrated on that same micro-Val and locked before LCC.

This warm-start reuses the Phase A frequency representation to screen fusion cheaply; it is only an architecture-search speed measure, not a final comparison protocol. A candidate that advances must later be retrained independently under a clean protocol before supporting a final E1-versus-proposed-method claim.

## 6. Candidate definitions

| Candidate | Fixed representation change | Question |
|---|---|---|
| M0 | Current E2 global 224×224 luminance DCT, signed-log, per-sample full-map z-score, E2 Tiny CNN, GAP 1×1, 64-D | Control |
| M1 | M0 CNN with AdaptiveAvgPool 4×4 before its 64-D projection | Does coarse location on the DCT plane help? |
| M2 | Normalized DCT plus fixed U/V maps in [0,1] (U increases left-to-right; V top-to-bottom), lightweight CNN, pool 4×4, 64-D | Does explicit frequency position help? |
| M3 | Normalized DCT in disjoint radial masks covering every coefficient once: r<1/3, 1/3≤r<2/3, r≥2/3; pool 4×4, 64-D | Does exposing band identity help? |
| M4 | 224 luminance into non-overlapping 8×8 block DCT; K=16 deterministic low-to-mid zigzag channels; signed-log; Train-only channel mean/std; 1×1 K→32 bottleneck and local CNN to 64-D | Does preserving local block-frequency layout help? |

M4 normalization statistics use micro-Train deterministic base crops only and are saved for exact reuse on micro-Val and LCC. Candidate notebooks do not tune their representation using LCC and do not label themselves promising.

## 7. Evaluation and diagnostics

CelebA micro-Val reports APCER, BPCER, ACER, AUC, and 3-class accuracy. LCC reports AUC and HTER as primary, with APCER, BPCER, ACER, accuracy, scored count, candidate count, and detector coverage. The binary score is `d = real_logit - logsumexp([physical_logit, digital_logit])`; real is predicted when `d >= threshold`.

For each concat candidate on micro-Val and the common valid LCC set, evaluate FULL, FREQ_ZERO, FREQ_SHUFFLE, and SPATIAL_ZERO with frozen weights and the selected CONCAT checkpoint's locked micro-Val threshold. FREQ_SHUFFLE uses a label-independent seeded Sattolo derangement (zero fixed points); save the seed, ordered permutation SHA256, and fixed-point count. The permutation builder receives only `n` and the seed. The final Val/LCC spatial and frequency features are reused for these ablations, avoiding another model inference pass. Record AUC, HTER/ACER, APCER, BPCER, feature norm mean/std, first-layer spatial/frequency weight-slice norms, and projected branch-contribution norms.

Notebook 07 requires Stage 1 `micro_e1_v1_summary.json` and M0–M4 summaries. It verifies the shared Stage 1 checkpoint SHA, every Stage 0 fingerprint/count, the common LCC path fingerprint and scored counts, and the common budgets. It compares candidates against two separate references: M0 controls frequency-representation design; Stage 1 is the spatial-only reference for whether frequency fusion helps spatial E1. FREQ_ZERO remains an ablation inside a CONCAT model and is not treated as the independent Stage 1 spatial model.

`PROMISING`/`BORDERLINE`/`KILL` labels are descriptive screening heuristics, not statistical tests or promotions. The prespecified promising rule requires either at least +2.0 percentage points LCC concat AUC or at least −2.0 points LCC concat HTER versus M0 with some positive AUC change (so an HTER-only operating-point tradeoff is not called a robust representation gain), micro-Val ACER regression no more than +1.5 points, and frequency-branch evidence beyond both FREQ_ZERO and FREQ_SHUFFLE. Stage 1 deltas, AUC, HTER, APCER/BPCER tradeoffs, and weak branch evidence are surfaced separately; no composite score is created. A candidate is not automatically discarded only because it is slightly below Stage 1 while beating M0 with complementary branch evidence. The packet always allows **NO WINNER**.

## 8. Interpretation limits and next stage

A micro screen is a compute-allocation decision. It can identify a candidate for a later clean experiment but does not establish generalization or statistical significance. LCC has influenced architecture choice by design. Any later confirmation should preserve held-out boundaries, use an independent initialization/training protocol, freeze the treatment before consulting reserved datasets, and retain the exact configs, seeds, manifests, checkpoints, and hashes.

No full-data training is scheduled until a micro candidate shows a substantial, reproducible signal, avoiding expensive runs for weak treatments. No Stage 0–2 performance numbers exist until the user executes the notebooks on Kaggle and returns their artifacts. No local real-data training is part of this task.

