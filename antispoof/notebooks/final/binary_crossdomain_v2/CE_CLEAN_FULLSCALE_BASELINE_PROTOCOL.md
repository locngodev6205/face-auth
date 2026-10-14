# CE-clean-only full-scale baseline

## Question and hypothesis

Does the complete clean/spectral/auxiliary objective improve transfer relative to binary CE on clean augmented inputs alone? No gain is assumed. Train exactly one new run: `CE_CLEAN_FULL_crop15`. Historical `C_FULL_crop15` is CE_clean + L_aux, not CE-only.

## Treatment and controls

Treatment: total loss = CE_clean with coefficient 1; no CE_worst, harmful gating, FFT/selection, or auxiliary loss/forward. Preserve the notebook18 binary MiniFASNetV2 trunk and PAD head. Auxiliary modules remain in the state dict to preserve initialization ordering and checkpoint compatibility, but are frozen, excluded from the optimizer and never called. This is the basic-loss baseline, not a pretrained three-class classifier.

Match notebook18: full eligible official CelebA Train (484561 usable of 494405 candidates), full official Test-as-Val (66379 usable of 67170 candidates), existing SCRFD caches and eligibility, crop1.5, 80x80 BGR [0,1], same augmentation and bbox jitter, pretrained MiniFASNet trunk / seed100 binary head, DATA_SEED43, batch256 AMP, four workers by default, SGD lr.005 / momentum.9 / weight_decay5e-4, 12 epochs maximum, milestones[4,8], gamma.2. No sampling or population edits. CE-only checkpoints are eligible from epoch1 (no spectral warmup). Phase-aware patience starts epoch9, patience3, earliest stop11. Selection minimizes source calibrated ACER, breaks ties by AUC, exactly as notebook18. Epoch-boundary resume preserves model, optimizer, scaler, scheduler and RNG state.

## Evaluation

Reload best checkpoint; re-infer full usable CelebA Test and freeze its source threshold exactly as notebook18. The historical full-scale experiment uses Test-as-Val for selection/calibration: its Test metrics are **not independent held-out Test evidence**. Preserve this disclosed reuse for comparability rather than silently creating a different split.

Evaluate full LCC training/development/evaluation and pooled combined from the frozen 00d caches, with the frozen source threshold and no target fitting. Both project and existing reference-compatible metric conventions remain available.

CASIA follows notebook20: all extracted color train/test frames; combined deduplicates (video_id,frame_index), prefers test before detection, no fallback based on detection success. SCRFD confidence.5, 640→480→320, largest valid finite box with confidence tie-break; no GT/full-image fallback. Same crop1.5 and valid-frame accounting. Primary mean-score video metrics need at least4 valid frames; frame metrics are secondary. Record train/test copy overlap and coverage. Source-frozen metrics are retained. Default optional train-only frame/video threshold calibration transfers each threshold unchanged to train/test/combined, exports separate metrics, and never selects checkpoints or fits on test/combined. CASIA train and combined calibrated results are not independent validation; Kaggle train/test identity overlap also limits independence.

## Metrics, artifacts and success criterion

AUC, descriptive EER, TPR@FPR1%, APCER, BPCER, ACER, HTER, accuracy, counts/coverage; score = logit(real) - logit(attack), Real iff score >= threshold. Pooled binary ACER = HTER. Source-frozen and CASIA train-calibrated operating points are labeled separately. Single-class/empty CASIA populations produce null metrics where undefined.

Notebook `21_fullscale_CE_clean_baseline_crop15.ipynb`; new output root only. Save configs, population/manifests, initialization provenance, best/last checkpoints, history, evaluation freeze, predictions/metrics, comparison CSV/JSON, Markdown report and ZIP. RUN_FILTER supports None or the single new run ID. No dependency on previous training output directories or contract/README source files.

Success: obtain a traceable measured CE-only comparison, not a desired improvement. One seed supports descriptive comparison only. Objective reductions do not guarantee accuracy or deployment quality. Training cost is at most12 epochs over484561 images plus full source evaluation each epoch; CE-only uses one forward per batch and avoids auxiliary/spectral computation. CASIA adds detection and one forward per localized frame.

## Validation

Compile every Python cell, clear outputs, prove CE-only gradient equals direct binary CE and auxiliary modules are not called/updated, check initialization against notebook18, test source/LCC population accounting, CASIA aggregation/calibration and report paths on offline synthetic inputs / saved predictions. Do not launch expensive training or camera work locally. Actual new-model results require running this notebook on Kaggle with GPU and mounted datasets/resources. Existing notebooks, manifests, caches, checkpoints and deployment settings remain untouched.

## Completed offline checks

- JSON parses; all Python cells compile; every code cell has cleared outputs/execution count.
- Real MiniFASNetV2 initialization state SHA matches notebook18 using the local pretrained checkpoint. Real-model CE loss and PAD-head gradients match direct binary CE under replayed dropout RNG; auxiliary forwards are forbidden and all auxiliary parameters are frozen.
- Source/Test failure accounting and physical/pooled LCC prediction/metric exports pass on synthetic candidate tables.
- Embedded CASIA evaluation passes end-to-end with a fake detector and synthetic images: one model inference pass, train/test overlap and combined deduplication, minimum-frame coverage, six frame/video calibrated metric rows, train-only fitting, report and ZIP.
- Unified comparison exports include five source/LCC frame rows, six CASIA source frame/video rows and six separately labeled calibrated rows. ZIP excludes the shared crop array.
- Phase-aware stopping check confirms earliest stop11. Full CUDA training/resume and real SCRFD/CASIA evaluation have not been run locally. The notebook checks a real SCRFD CUDA session before starting expensive training on Kaggle.
