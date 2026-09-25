# E1 v5.2 — Final Detector-Aligned Retrain Protocol

## Purpose

This is the final preprocessing-focused retrain of the spatial-only E1 baseline before freezing the baseline protocol. The model architecture and optimization recipe remain intentionally unchanged from v5.1; the intervention is limited to face ROI construction, data hygiene, and preprocessing provenance.

## Frozen model/training recipe

- Backbone: MobileNetV3-Large, ImageNet initialization.
- Classes: `Real`, `Physical Spoof`, `Digital Spoof`.
- Input: RGB `224×224`.
- Normalization:
  - mean `[0.5931, 0.4690, 0.4229]`
  - std `[0.2471, 0.2214, 0.2157]`
- Gamma: OFF.
- SCRFD preprocessing library: `insightface==2.0` (pinned for final-cache reproducibility).
- Batch size: 128.
- Optimizer: AdamW, weight decay `1e-4`.
- LR: backbone `1e-5`, head `1e-4`.
- Warmup: 2 epochs; cosine decay afterwards.
- Max epochs: 24.
- Early stop: Validation ACER, patience 4, not before epoch 10.
- Label smoothing: 0.1.
- Dropout: 0.2.
- Gradient clip: 5.0 after AMP unscale.
- Threshold calibration: Validation only.

## v5.2 preprocessing contract

### 1. Exact experimental membership first

Train/Validation/Test membership is constructed before detector preprocessing. Only the exact selected keys are cached; the notebook does not scan the whole CelebA-Spoof image pool during a preliminary run.

### 2. SCRFD-conditioned bounding boxes

Primary bbox source is SCRFD-500M (`buffalo_s` detection).

Detection order:

```text
SCRFD 640×640
    ↓ if no reliable target match
SCRFD 480×480
    ↓ if still no reliable target match
SCRFD 320×320
```

The smaller fallback detector inputs are intended for the observed very-large-face tail. SCRFD outputs remain in original-image coordinates.

The CelebA bbox is used only to associate/sanity-check the intended target and as a fallback when a reliable detector bbox cannot be produced.

A SCRFD match is considered reliable when:

```text
raw IoU with CelebA bbox >= 0.50
normalized center shift <= 0.20
```

### 3. Crop context

For a valid SCRFD bbox:

```text
square crop expansion = 1.55×
```

For a CelebA annotation fallback:

```text
square crop expansion = 1.50×
```

The two factors are intentionally different: the train-only geometry study found that SCRFD boxes are slightly tighter on average, and SCRFD `1.55×` best approximated the context of the previous CelebA `1.50×` reference crop.

### 4. Small-face policy

Only TRAIN is filtered:

```text
min(face width, face height) < 48 px
→ remove from TRAIN
```

Validation and Test are never filtered by face size.

The 48 px threshold is intentionally conservative: it removes the lowest-resolution tail while limiting class-dependent selection bias.

### 5. Detector failures and suspicious geometry

A detector failure is not automatically removed from TRAIN. After the 640/480/320 attempts, the notebook falls back to the CelebA bbox. This avoids creating class-selection bias from SCRFD failure patterns.

A detected-but-geometrically suspicious sample is removed from TRAIN. The same sample is retained in Validation/Test using the CelebA fallback so evaluation membership does not become easier after preprocessing.

### 6. Train augmentation

Only mild augmentation is used:

- horizontal flip `p=0.5`;
- brightness/contrast ±10%, `p=0.3`;
- bbox geometry jitter `p=0.2`;
- bbox scale jitter `0.95–1.05`;
- bbox center translation ±5%.

No research gamma, mixup, cutmix, focal loss, weighted sampler, strong blur/JPEG, or aggressive erasing is introduced into E1.

## Guardrails before training

The notebook stops before expensive training if any of the following occurs:

- total TRAIN filtering > 5%;
- any class filtering > 8%;
- class filtering-rate gap > 3 percentage points;
- suspicious-geometry filtering > 1%;
- SCRFD coverage for Train, Validation, or Test < 90%;
- Train/Validation/Test subject/key leakage is detected;
- Validation/Test membership changes after preprocessing;
- the selected bbox cache or split is incompatible with a resumed checkpoint.

The notebook saves:

- exact split manifest;
- preprocessing audit JSON;
- selected-record bbox-cache fingerprint;
- source counts (`SCRFD_640`, `SCRFD_480`, `SCRFD_320`, fallback);
- per-class removal rates;
- final model metadata and runtime contract.

## Evaluation scope

Validation threshold calibration and checkpoint selection use Validation only.

Held-out Test is disabled by default. When enabled after protocol freeze:

- small faces remain in Test;
- detector failures are not silently removed;
- detector coverage and bbox-source counts are reported separately;
- PAD metrics are therefore **PAD-model metrics on detector-conditioned crops with annotation fallback**, not a pure end-to-end detector-failure metric.

## Preliminary vs official

The notebook defaults to:

```python
RUN_MODE = "preliminary"
```

for a cheap preflight.

For the final full-data E1 freeze run, change to:

```python
RUN_MODE = "official"
```

before building the final cache/training run. Do not resume a v5.1 checkpoint: the preprocessing contract changed.

## Runtime contract after the final model is accepted

Deploy the frozen E1 with:

```text
detector = SCRFD-500M
bbox expansion = 1.55×
input = 224×224 RGB
gamma = OFF
mean/std = CelebA values above
threshold = probability/logit pair calibrated on the final Validation split
```

Do not reuse the v5.1 threshold automatically; v5.2 must recalibrate its threshold from its own Validation predictions.

## Acceptance checklist before freezing E1

1. Preflight cell passes without guardrail exceptions.
2. Inspect source counts and class-specific filtering; no unexpected skew.
3. Train v5.2 from ImageNet initialization, not a v5.1 PAD checkpoint.
4. Select checkpoint by Validation ACER only.
5. Lock the v5.2 Validation threshold.
6. Export ONNX and require PyTorch↔ONNX max logit difference `<1e-4`.
7. Run held-out Test once after protocol/model selection is frozen.
8. Re-run the realtime camera diagnostic with the v5.2 ONNX and the new `1.55×` runtime crop.
9. Preserve the manifest, preprocessing audit, cache fingerprint, best metadata, ONNX, and runtime config together.

## Runtime note

The detector-cadence fix is separate from this training protocol. It should not change model preprocessing or threshold semantics. Continue validating optical-flow tracked bbox stability independently from E1 training.
