# PAD data and evaluation contract

Contract version: v1  
Status: FROZEN  
Date: 2026-10-01

## A. Purpose and scope

This contract governs CelebA-Spoof Train, Val and Test; the official LCC-FASD evaluation split; future external Face PAD datasets unless a new version says otherwise; Vanilla and CSMR models; and future evaluation notebooks. Future notebooks must reference and assert this contract. A deliberate deviation requires a new contract version and an explicit rationale. Silent protocol changes are prohibited.

## B. Face localization

Use SCRFD-500M only, with model SHA256 `5e4447f50245bbd7966bd6c0fa52938c61474a04ec7def48753668a9d8b4ea3a`. The model crop bbox must originate from SCRFD. A dataset GT or annotation bbox may be used for detection matching or diagnostics, never as a fallback model crop. If there is no reliable SCRFD detection, the candidate is unscorable. Every model input requires `usable_for_pad == True` and `bbox_origin == "SCRFD"`.

Localization statuses are `VALID`, `DET_FAIL`, `SUSPICIOUS`, `UNREADABLE`, and `INVALID_BBOX`. Only `VALID` is PAD-scorable. `TOO_SMALL` is a Train hygiene rejection for a valid SCRFD box with native minimum side below 48 px; such a box can remain scorable for evaluation. No other status is trainable or scorable. A `VALID` record must contain a finite positive-area raw SCRFD bbox.

The frozen CelebA selection policy matches detections to the annotation face using highest raw IoU, then confidence; it accepts a detection only if raw IoU is at least 0.5 and normalized center shift is at most 0.2. Detector threshold is 0.5; input sizes are 640, 480, then 320. GT participates only in this quality check. This makes crop provenance SCRFD-only, although GT-guided choice among multiple detected faces can differ from production selection without annotations. LCC has no annotation matching and uses the largest valid SCRFD face, then confidence and coordinates for ties, at the same threshold and sizes.

## C. Bbox and crop

The pipeline is raw SCRFD bbox → optional Train bbox jitter → 2.7× MiniFASNet context crop → 80×80 resize → BGR float32 in [0,1] → MiniFASNetV2. Cache the raw detection only. Never cache the expanded crop box. Apply the 2.7× crop in Dataset preprocessing. Val, Test, and external data use no bbox jitter. Train jitter remains Notebook 03's 20% chance, 0.95–1.05 scale, and ±0.05 face-size center shift. Mild augmentation remains horizontal flip probability 0.5 and brightness/contrast probability 0.3 with Notebook 03's ranges.

## D. Train hygiene

CelebA Train uses all eligible official-Train images after official-Test subject overlap exclusion and Val subject reservation. Drop detector failures, suspicious detections, invalid boxes, unreadable images, and valid SCRFD boxes with native minimum side below 48 px. Never rescue a sample with GT. Train size is dynamic; do not sample replacements to reach a target count. Preserve the existing class-dependent filtering guardrails and audit their rates.

## E. Validation

Val contains exactly 50,000 `VALID` SCRFD-localized samples from official CelebA Train. Apply the Test-subject exclusion and SCRFD validity first; then create a deterministic subject-disjoint Train/Val partition with `DATA_SEED = 42`. All three project classes must be present. Prefer retaining valid historical Val15K samples where subject integrity permits. Val alone controls checkpoint selection, early stopping, and minimum-ACER threshold calibration. Official Test never enters Val.

## F. CelebA Test

The official candidate population is 67,170; `scored_n <= 67,170`. Preserve every candidate in the manifest and localization audit, including failed detections. PAD metrics use only the SCRFD-scorable subset. Always report `candidate_n`, `scored_n`, and `detector_coverage = scored_n / candidate_n`. Never call the scored subset the full candidate population.

## G. LCC-FASD official evaluation

Use the official 7,580-candidate evaluation split: 314 Real and 7,266 Spoof. Localize with SCRFD only and retain failed candidates in the audit. Compute PAD metrics on scored candidates and report candidate count, scored count, and coverage. The historical approximately 3,766-candidate resource is the **legacy external-development resource**. It is separate from the official split; any overlap is disclosed, not removed.

## H. Labels and PAD score

CelebA project labels are 0 Real, 1 Physical Spoof, and 2 Digital Spoof. The binary decision is Real versus Attack `{Physical, Digital}` with Real positive. Use `d = z_real - logsumexp(z_physical, z_digital)` in every training and evaluation notebook. LCC is binary; do not invent attack subclasses.

## I. Threshold

Select the threshold from CelebA Val only, using the frozen minimum-ACER procedure. Never calibrate on CelebA Test, LCC, or another external target. External AUC, EER, and TPR at FPR 1% are threshold-free target ROC statistics. External APCER, BPCER, HTER, and binary accuracy use the locked CelebA-Val threshold. Report these roles separately.

## J. Required metrics

For CelebA Val/Test report candidate count where applicable, scored count, detector coverage, AUC, APCER, BPCER, ACER, binary accuracy, three-class accuracy, EER, TPR at FPR 1%, confusion matrix, and locked threshold. For external binary datasets report candidate count, scored count, detector coverage, AUC, APCER, BPCER, HTER, binary accuracy, EER, TPR at FPR 1%, and locked source threshold. A confusion matrix covers scored candidates only.

## K. Model and inference

Inference uses MiniFASNetV2, 3×80×80 BGR float32 [0,1], a 2.7× crop, and three logits. CSMR is training-only. Inference contains no FFT, spectral branch, worst-band selector, or margin loss.

## L. Pretrained initialization

MiniFASNet source SHA256: `e498c4ec5e1ddfaba62b941a126c19d65aa564999f3309661fe43ee8bf38acd7`. Official pretrained checkpoint SHA256: `a5eb02e1843f19b5386b953cc4c9f011c3f985d0ee2bb9819eea9a142099bec0`. Canonical project initialization state SHA256: `318dbe2a0fa2c80cc36d4f881aa3967b226255af71a45049529daa5430b93adc`. Load compatible PAD-pretrained feature weights, skip the incompatible original classifier, and deterministically initialize the project three-class classifier.

## M. Cache provenance

Each cache records detector SHA, policy version, source key/path, localization status, bbox origin, and PAD usability. A historical GT-fallback record is never accepted as an SCRFD bbox. Reprocess it with the frozen detector or leave it unscorable. Every usable record must assert `bbox_origin == "SCRFD"`; only `VALID` records may carry `usable_for_pad == True`.

## N. Literature-comparison wording

When source/target direction matches but preprocessing differs, describe a published result as **same source/target benchmark, different preprocessing** or a **contextual literature reference, not strictly protocol-identical**. Claim strict protocol equivalence only with the same source data role, target split, localization/preprocessing, threshold policy, and no target fine-tuning.

## O. Change control

Future notebooks must fail a contract assertion or create a new contract version before changing detector, bbox policy, crop factor, input resolution, validation protocol, threshold rule, target split, or metric definition. No silent changes are allowed.

## P. Notebook integration

Every notebook sets `EVALUATION_CONTRACT_VERSION = "PAD_DATA_AND_EVALUATION_CONTRACT_v1"`, prints its version and SHA256 at startup, asserts protocol fields against this document, and writes the version and SHA256 into config/provenance or resource fingerprints and reports. The resource notebook embeds and exports this exact Markdown so training notebooks can verify it from the mounted v2 bundle without another input mount.
