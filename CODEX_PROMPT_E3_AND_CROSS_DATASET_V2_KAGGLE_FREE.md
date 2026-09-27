# Codex Task — Build E3 Concat Fusion + Cross-Dataset Evaluation Notebooks for `face_auth`

You are working inside the `face_auth` repository. Follow the repo `AGENTS.md` and use the installed repo/research skills where appropriate (`repo-navigator`, `research-audit`, `experiment-planner`, `reproducibility-guard`, `surgical-coder`, `focused-tester`, `results-analysis`).

## Goal

Create the next research stage after frozen E1 and E2:

```text
E1 — MobileNetV3-Small spatial-only       [FROZEN]
E2 — DCT frequency-only diagnostic        [FROZEN]
E3 — Spatial + DCT Frequency + CONCAT     [BUILD NOW]
         ↓
Cross-dataset E1 vs E3
         ↓
Decide later whether E4 gated fusion is justified
```

The main scientific question is:

> Does the explicit DCT frequency branch add complementary information to the MobileNetV3-Small spatial representation, especially under domain shift, while keeping the model lightweight enough for edge deployment?

Do not change E1 or E2. Do not tune E1/E2 based on Test results.

---

# 0. Inspect first — do not code blindly

Before creating files, locate and inspect the actual current versions of the following files/artifacts in the repository/worktree. Filenames may differ slightly; use repo search rather than assumptions.

Priority sources:

```text
E1:
- antispoof_baseline_spatial_mnv3_small_v5_3_edge_final.ipynb
- antispoof_e1_heldout_test_v1.ipynb
- E1 v5.3 protocol / artifact guide / Test guide

E2:
- e2_frequency_only_dct_train.ipynb
- e2_heldout_test.ipynb
- antispoof_e2_frequency_only_dct_v1.ipynb, if retained
- E2_frequency_only_dct_protocol.md
- E1_artifacts_for_E2_kaggle_guide.md
- E1_E2_heldout_test_kaggle_guide.md

Research design:
- md/research/pad_spatial_frequency_architecture_v2.md
- md/research/pad_frequency_domain_references_v2.md
```

If the executed E2 notebook differs from an older design document, the **executed/frozen E2 implementation wins** for architecture and preprocessing.

Likewise, the frozen E1 v5.3 implementation wins over older documents that still mention MobileNetV3-Large or older gamma/crop behavior.

Before editing, write a short internal checklist of the exact frozen values you found:
- E1 MobileNetV3-Small architecture and 256-D spatial feature;
- E2 DCT transform and exact `FrequencyBranch`;
- E2 frequency feature dimension;
- split/cache filenames and fingerprints;
- crop factors;
- augmentation;
- training schedule;
- metric/threshold semantics;
- artifact naming conventions.

Do not silently invent missing values. If an essential frozen detail cannot be found, stop and report the missing artifact/file.

---

# 1. Deliverables

Create these files under the research/notebook area that best matches the current repo structure.

## Required notebooks

### A. E3 training notebook

```text
antispoof_e3_spatial_frequency_concat_v1.ipynb
```

Kaggle-ready, self-contained for E3 training.

### B. E3 standalone held-out Test notebook

```text
antispoof_e3_heldout_test_v1.ipynb
```

Keep training and final held-out Test separate, consistent with E1/E2.

### C. Cross-dataset evaluation notebook

```text
antispoof_cross_dataset_e1_vs_e3_v1.ipynb
```

Primary comparison:

```text
E1 spatial-only
vs
E3 spatial + frequency concat
```

E2 may be supported behind an optional flag as a diagnostic only, but must not distract from the E1-vs-E3 primary comparison.

## Required Markdown documentation

Create:

```text
E3_spatial_frequency_concat_protocol.md
E3_kaggle_run_and_artifact_guide.md
cross_dataset_evaluation_protocol.md
cross_dataset_kaggle_run_and_result_guide.md
```

The guides must be practical enough that another team member can run the notebooks on Kaggle without reading the code.

---

# 2. Frozen E1/E2 data protocol — MUST reuse exactly

E3 is not allowed to resample the dataset or rerun SCRFD for CelebA-Spoof.

It must load the exact E1 artifacts:

```text
celeba_scrfd_bbox_cache_v5_3_<mode>_seed42.json
celeba_spoof_<mode>_v5_3_edge_mnv3_small_seed42.npz
```

Reuse the exact final E1:

```text
train_keys
val_keys
test_keys
bbox records
bbox source
crop factor
TRAIN filtering decisions
```

Frozen crop policy:

```text
valid SCRFD bbox          → 1.55×
CelebA annotation fallback → 1.50×
TRAIN min-face filtering   → already encoded by final train_keys
Val/Test                   → no small-face filtering
gamma                      → OFF
input                      → 224×224
seed                       → 42
```

Do not run SCRFD again on CelebA-Spoof in E3.

E3 must verify:
- cache schema/policy;
- exact split fingerprint;
- exact selected-cache fingerprint when metadata is available;
- zero Train/Val/Test overlap;
- same class counts as the frozen manifest.

Fail loudly on mismatch.

Support:

```python
E1_RUN_MODE = "preliminary"  # or "official"
E1_ARTIFACT_DIR = ""
```

with recursive exact-name discovery under `/kaggle/input` when the path is blank.

Do not mix preliminary and official artifacts.

---

# 3. E3 architecture — main experiment

E3 is the simple CONCAT model. Do **not** add gates, attention, SE-after-fusion, transformers, auxiliary losses, contrastive losses, or other extensions.

## Spatial branch

Use the exact E1 v5.3 spatial representation:

```text
224×224 RGB
→ MobileNetV3-Small ImageNet backbone
→ Global Average Pool
→ 576 → 256
→ BN
→ Hardswish
→ Dropout(0.2)
→ 256-D spatial feature
```

Use the exact frozen E1 RGB normalization.

## Frequency branch

Use the exact frozen E2 transform and exact frozen E2 `FrequencyBranch`.

The executed E2 implementation is the source of truth.

Expected conceptual flow:

```text
same augmented RGB face crop
→ resize/pad to 224×224
→ luminance
→ 2D DCT
→ sign(C) * log1p(abs(C))
→ E2 normalization
→ exact E2 Tiny Frequency CNN
→ 64-D frequency feature
```

Do not add a handcrafted frequency mask if frozen E2 used none.

## Critical alignment rule

Create **one face crop and one stochastic augmentation result per sample**, then derive both branches from that same augmented image.

Correct:

```text
raw image
→ cached bbox crop
→ ONE bbox jitter
→ ONE appearance augmentation
→ shared augmented RGB crop
        ├─ spatial preprocessing
        └─ luminance → DCT preprocessing
```

Incorrect:

```text
independent crop jitter for spatial and frequency
independent brightness/contrast for each branch
```

The branches must see the same underlying sample geometry.

## Fusion

Use:

```text
spatial feature   256-D
frequency feature  64-D
       ↓ concat
      320-D
       ↓
Linear 320 → 128
Hardswish
Dropout(0.2)
Linear 128 → 3
```

Output classes remain:

```text
0 Real
1 Physical Spoof
2 Digital Spoof
```

PAD score remains:

```text
d = real_logit - logsumexp([physical_spoof_logit, digital_spoof_logit])
REAL iff d >= locked threshold
```

---

# 4. Main E3 initialization policy — avoid confounding the ablation

For the **primary E3 experiment**, default to independent training:

```python
INIT_MODE = "independent"
```

Meaning:

```text
MobileNetV3-Small backbone → ImageNet pretrained, same as E1 start
spatial projection         → random init
E2 FrequencyBranch         → random init, same architecture as E2
fusion head                → random init
```

Do NOT preload task-trained E1/E2 PAD weights in the primary E3 run.

Reason: loading the already trained E1 and E2 branches would add staged task-specific pretraining and extra optimization exposure, confounding the main question "does the architecture/frequency branch help?"

You may implement an optional non-primary mode:

```python
INIT_MODE = "warm_start_e1_e2"
```

behind an explicit flag, disabled by default, for a later optimization/init ablation.

If supported:
- verify E1/E2 checkpoint architecture and fingerprints before loading;
- clearly label outputs as `warm_start`, never overwrite primary E3 artifacts;
- do not use warm-start results as the main E1-vs-E3 architecture claim unless explicitly requested later.

---

# 5. E3 training protocol

Keep the frozen E1 training discipline unless the actual E1 notebook shows a different value.

Expected:

```text
batch size                 = 128
optimizer                  = AdamW
weight decay               = 1e-4
MobileNetV3 backbone LR    = 1e-5
all new E3 layers LR       = 1e-4
warmup                     = 2 epochs
scheduler                  = cosine
max epochs                 = 24
early-stop metric          = Validation ACER
patience                   = 4
earliest early stop        = epoch 10
gradient clip              = 5.0 after AMP unscale
dropout                    = 0.20
label smoothing            = 0.10
class weights              = [1,1,1]
seed                       = 42
```

"New E3 layers" include:
- spatial projection;
- frequency branch;
- concat fusion classifier.

Use the exact same mild TRAIN augmentation as E1/E2.

Do not add resolution degradation/downsample-upsample augmentation to the primary
E3 run. A later face-size robustness experiment must use a new protocol version
and include E1 + the same augmentation as a control, so augmentation effects are
not attributed to fusion.

Checkpoint selection:
```text
Validation ACER only
```

Threshold calibration:
```text
Validation only
```

Never tune from held-out Test.

Save both:
```text
locked logit threshold
locked probability threshold = sigmoid(logit threshold)
```

---

# 6. E3 Kaggle artifact requirements

Use a clear run name, for example:

```text
e3_concat_<mode>_v1
```

Save at minimum:

```text
e3_concat_<mode>_v1_best.pth
e3_concat_<mode>_v1_checkpoint_last.pth
e3_concat_<mode>_v1_best_meta.json
e3_concat_<mode>_v1_run_config.json
e3_concat_<mode>_v1_training_history.json
e3_concat_<mode>_v1_training_history.png

e3_concat_<mode>_v1_best.onnx
e3_concat_<mode>_v1_runtime_config.json
```

Also save:
- model parameter count;
- spatial/frequency/fusion parameter counts;
- FP32 ONNX size;
- host-CPU model-only ORT latency diagnostic;
- exact split/cache fingerprints;
- exact DCT preprocessing spec;
- initialization mode.
- batch-1 preprocessing and total PAD latency, separately from model-only latency.

## ONNX design

Prefer a transparent two-input ONNX:

```text
rgb_input       [N,3,224,224]
frequency_map   [N,1,224,224]
→ logits        [N,3]
```

DCT preprocessing remains external, exactly like E2, unless the repo already has a proven exportable DCT module.

The current `AntiSpoofPredictor`/`app.py` does not support this two-input E3
contract. The exported runtime config must identify both inputs and the external
DCT contract. Do not imply that switching `PAD_RUNTIME_CONFIG_PATH` alone enables
E3 in the current app. Runtime integration is a separate later task; do not modify
the app or frozen E1/E2 runtime in this notebook-building task.

Do not introduce a fragile custom ONNX DCT implementation just to force single-input export.

Run PyTorch↔ONNX parity and require:

```text
max |Δlogit| < 1e-4
```

If export needs `onnxscript`, install it in the notebook dependency cell.

---

# 7. E3 held-out Test notebook

`antispoof_e3_heldout_test_v1.ipynb` must follow the same pattern as the standalone E1/E2 Test notebooks.

It must:
- load frozen E3 ONNX;
- load the exact E1 split manifest/cache;
- use only `test_keys`;
- load the locked E3 Validation threshold from `runtime_config` / `best_meta`;
- fail if threshold metadata conflicts;
- never scan Test thresholds;
- never update weights.

Save:

```text
e3_<mode>_heldout_test_results/
├── e3_heldout_test_summary.json
├── e3_test_protocol_snapshot.json
├── e3_test_predictions.csv
├── e3_attack_breakdown.csv
├── e3_binary_confusion_matrix.png
└── e3_score_distribution.png

e3_<mode>_heldout_test_results.zip
```

Per-sample CSV columns should align with E1/E2 result CSVs so later paired analysis is straightforward:
- key;
- subject;
- attack code;
- bbox source;
- native face minimum side in pixels, original image width/height, and crop factor;
- true class;
- 3 logits;
- `d`;
- `p_real`;
- binary prediction/correctness;
- 3-class prediction/correctness.

After E3 Test, do not alter E3 based on held-out Test.

Also save `e3_face_size_breakdown.csv`, with sample counts and APCER/BPCER/ACER
for frozen native-face-size bins: `<48`, `48–99`, `100–139`, `140–179`, and `>=180`
pixels. Use `face_min_side` from the shared bbox cache, not the resized 224 input.
Report Real/Attack counts separately; undefined metrics must be null/NaN, never
zero. Apply each model's global source-locked threshold, without bin-specific
calibration. Do not claim size causes the observed error without controlling
subject/attack/crop-source composition.

---

# 8. Cross-dataset evaluation scope

Build:

```text
antispoof_cross_dataset_e1_vs_e3_v1.ipynb
```

Primary models:

```text
E1 — frozen MobileNetV3-Small spatial-only
E3 — frozen spatial + DCT concat
```

Optional:
```python
EVALUATE_E2 = False
```

E2 may be included only as a diagnostic if its frozen artifacts are mounted.

Main targets should prioritize datasets that are free/publicly accessible for research
and practically mountable on Kaggle without a manual institutional approval workflow:

```text
PRIMARY 1: LCC-FASD
PRIMARY 2: CASIA-FASD
OPTIONAL : Replay-Attack, if a legally usable Kaggle/public copy is already available
```

Do **not** make OULU-NPU or SiW mandatory targets in this notebook. Their official
distribution requires license/request workflows, so they should remain optional future
benchmarks if the team later obtains authorized access.

Do not automatically download external datasets.

Expose editable Kaggle paths:

```python
LCC_FASD_ROOT = ""
CASIA_FASD_ROOT = ""
REPLAY_ATTACK_ROOT = ""  # optional
E1_ARTIFACT_DIR = ""
E3_ARTIFACT_DIR = ""
E2_ARTIFACT_DIR = ""  # optional
```

---

# 9. Cross-dataset protocol — zero-shot target evaluation

The main cross-dataset experiment is:

```text
Train/calibrate: CelebA-Spoof only
Evaluate: LCC-FASD + CASIA-FASD
Optional : Replay-Attack
```

Strict rules:

```text
NO target fine-tuning
NO target adaptation
NO target threshold calibration
NO target checkpoint selection
NO target-specific hyperparameter tuning
```

Each model uses its own threshold locked on the **CelebA-Spoof Validation split**.

Do not calculate a target-optimized threshold and use it as the main result.

The notebook should label the experiment accurately as:

> Project cross-dataset / zero-shot stress test.

Do not claim it is an official OCIM or official OULU/SiW protocol unless the exact official train/dev/test protocol has actually been reproduced.

---

# 10. External dataset adapters

LCC-FASD, CASIA-FASD, and optional Replay-Attack may be mounted in different
Kaggle layouts.

Implement clean dataset adapter/config sections instead of hard-coding one fragile
directory tree.

Important: these targets are heterogeneous:
- LCC-FASD is commonly distributed as image folders / image-level samples;
- CASIA-FASD and Replay-Attack are video-oriented in their original form.

Therefore the evaluator must support both:
1. image-level datasets;
2. video/session-level datasets.

Do not force an artificial `video_id` onto an image-only dataset.

Before inference, print a manifest audit:

```text
dataset
number of videos/sessions
number Real
number Attack
attack/subtype counts when derivable
number of candidate frames
```

If labels cannot be inferred unambiguously from the mounted structure/metadata, fail and explain what mapping must be configured.

Do not silently guess labels.

Dataset-access guardrail:
- prefer public/research-usable Kaggle datasets with clear provenance;
- do not depend on unauthorized mirrors of datasets whose official distribution
  requires an EULA/DRA;
- print the mounted Kaggle dataset path and any available license/source metadata
  into the protocol snapshot;
- if provenance/license is unclear, warn and require user confirmation before
  treating that dataset as a publishable benchmark result.

Save the resolved external manifest to JSON/CSV in the output package.

Record the exact dataset identity/version, source, partition and label mapping.
Verify that CASIA-FASD has not been confused with CASIA-SURF or another CASIA
dataset. Kaggle availability alone is not proof of provenance or permission.
Resolve these details before inference; do not pool an unverified distribution
into the primary benchmark.

---

# 11. Cross-dataset image/video sampling

The evaluator must support both image-level and video-level target datasets.

## A. Image-level datasets, especially LCC-FASD

For an image-level dataset:

```text
one image = one evaluation sample
```

Do not invent video/session aggregation.

Evaluate directly at image level using the model's source-locked CelebA-Spoof
Validation threshold.

If the mounted LCC-FASD distribution provides official training/development/evaluation
folders, use the **evaluation** partition as the default target stress-test partition.
Do not train or calibrate on its training/development folders in the zero-shot experiment.

## B. Video-level datasets, especially CASIA-FASD / optional Replay-Attack

Avoid overweighting long videos.

Implement a frozen deterministic sampling policy:

```python
FRAMES_PER_VIDEO = 16
FRAME_SAMPLE_SEED = 42
```

Use uniformly spaced frames across each video after frame count is known.

If a video dataset is already extracted into frames:
- recover/group source video/session only when that mapping is unambiguous;
- uniformly select up to `FRAMES_PER_VIDEO`.

Save the exact sampled frame manifest.

Primary evaluation unit for video datasets should be **video/session level**:

1. infer each sampled valid frame;
2. compute frame PAD logit score `d`;
3. aggregate per video using arithmetic mean of `d`;
4. apply the model's source-locked threshold to the video mean score.

Also save frame-level predictions as diagnostics.

Freeze this valid-frame policy before inspecting target predictions:

```python
MIN_VALID_FRAMES_PER_VIDEO = 1
```

Use the common successfully preprocessed sampled-frame set for both models.
Videos with zero valid frames are `NO_VALID_FACE`, with no PAD score/prediction;
retain them in the manifest/coverage outputs and exclude them explicitly from
conditional PAD metrics. Report candidate/evaluable/excluded video counts by
Real/Attack, valid sampled-frame count per video, and frame-level coverage. Videos
with only a few valid frames remain included under this frozen policy but must be
flagged as low coverage. Do not choose a different minimum using target results.
Model inference errors must fail the evaluation or be reported explicitly; never
silently change the common comparison set for one model.

Do not choose aggregation using target performance.

---

# 12. SCRFD preprocessing on cross datasets

LCC-FASD, CASIA-FASD and optional Replay-Attack do not reuse the CelebA cache.

Use SCRFD to build a **new external evaluation cache**, but with detector behavior aligned to E1:

```text
primary detector input 640×640
fallback 480×480
fallback 320×320
confidence threshold consistent with E1
crop factor 1.55×
```

There is no CelebA annotation fallback on the external datasets.

For each sampled frame cache:
- dataset;
- video/session id;
- frame id/path;
- image size;
- native face minimum side and crop factor;
- bbox xyxy;
- confidence;
- detector input size used;
- number of faces;
- status.

Face selection must be deterministic.

For typical single-subject PAD videos:
- choose the largest valid detected face;
- tie-break by confidence;
- record `n_faces`;
- flag ambiguous multi-face frames.

Do not silently delete failures.

Report:
```text
detector coverage overall
coverage by real/attack
coverage by dataset
multi-face rate
failure count
```

For E1-vs-E3 fairness, both models must evaluate the exact same successfully preprocessed frame set.

Primary PAD HTER/AUC should be computed on the common valid-face set, with detector coverage reported separately.

Do not hide detection failures inside the PAD metric.

Save the external SCRFD cache so reruns do not redetect every frame.

---

# 13. Cross-dataset preprocessing for each model

## E1

```text
external SCRFD crop 1.55×
→ RGB 224×224
→ frozen E1 mean/std
→ gamma OFF
→ E1 ONNX
```

## E3

From the same external crop:

```text
shared RGB crop
        ├─ E1 spatial preprocessing → rgb_input
        └─ E2 luminance/DCT preprocessing → frequency_map
→ E3 ONNX
```

The E3 DCT implementation must be byte/algorithmically consistent with frozen E2:
- same luminance weights;
- same OpenCV/PyTorch DCT convention used in E2;
- same signed-log;
- same normalization;
- same no-mask policy.

Do not alter target preprocessing based on observed external target results.

---

# 14. Cross-dataset metrics

Primary, at **image level for image datasets** and **video/session level for video datasets**:

```text
HTER at source-locked threshold
AUC
```

Also report, as supplementary:
```text
APCER
BPCER
ACER
binary Accuracy
```

For clarity define:

```text
HTER = (FAR + FRR) / 2
```

under the same binary Real-vs-Attack interpretation.

Because the source model has a fixed threshold, state in output metadata:

```text
threshold_origin = CelebA-Spoof Validation
target_threshold_tuning = false
```

Do not call a target-label-optimized EER threshold the main result.

Optional target EER may be omitted entirely to reduce snooping.

Save face-size breakdowns using the same frozen bins as the E3 held-out notebook,
with Real/Attack counts and each model's global source-locked threshold. For video
datasets, these are frame-level diagnostics, not additional independent video
samples. Do not interpret native pixels as a common physical camera distance
across datasets. Cross-dataset evaluation measures domain shift; a controlled
near/far camera experiment with Real and print/replay attacks is a separate later
test, and must not be claimed as completed by these notebooks.

---

# 15. Cross-dataset comparison outputs

For each target dataset, produce a table with the correct evaluation unit:

For LCC-FASD / image-level targets:

```text
Model | N images | Detector coverage | AUC | HTER | APCER | BPCER | ACER
E1    | ...
E3    | ...
```

For CASIA-FASD / video-level targets:

```text
Model | N videos | Detector coverage | AUC | HTER | APCER | BPCER | ACER
E1    | ...
E3    | ...
```

Also compute descriptive deltas:

```text
ΔAUC  = E3 - E1
ΔHTER = E3 - E1
```

These are descriptive measured differences, not significance claims.

If E1/E3 in-domain Test summary JSONs are mounted, additionally create:

```text
in-domain vs LCC-FASD degradation
in-domain vs CASIA-FASD degradation
```

for AUC and threshold-based error, clearly noting the metric definitions.

Only treat differences as directly comparable when evaluation units match.
CelebA image-level versus CASIA video-mean metrics must be presented side by side
with an explicit unit/aggregation caveat, not as a pure domain-shift delta.

Save:
```text
cross_dataset_summary.json
cross_dataset_model_comparison.csv
cross_dataset_face_size_breakdown.csv
lcc_fasd_image_predictions.csv
casia_fasd_video_predictions.csv
casia_fasd_frame_predictions.csv

# optional when Replay-Attack is enabled
replay_attack_video_predictions.csv
replay_attack_frame_predictions.csv

external_dataset_manifest.csv/json
external_scrfd_cache.json
roc_lcc_fasd.png
roc_casia_fasd.png
score_distribution_lcc_fasd.png
score_distribution_casia_fasd.png
```

Create one ZIP:

```text
cross_dataset_e1_vs_e3_results.zip
```

---

# 16. Cross-dataset result interpretation guardrails

The notebook/docs must not assume E3 wins.

Use neutral wording:

```text
If E3 improves AUC and/or reduces HTER consistently across targets,
that supports the hypothesis that the frequency branch adds robust
complementary information.

If E3 only improves in-domain but not cross-domain,
frequency fusion may be exploiting source-specific shortcuts.

If E3 is similar to E1 with modest overhead,
the result is inconclusive rather than automatically negative.

If E3 degrades cross-domain,
do not proceed directly to E4 as if frequency was validated;
inspect branch behavior / domain sensitivity first.
```

Do not implement E4 in this task.

---

# 17. Efficiency accounting

Because the project targets edge deployment, E3 documentation must report:

```text
total params
spatial branch params
frequency branch params
fusion-head params
FP32 ONNX size
host CPU model latency
batch-1 crop/resize + spatial preprocessing latency
batch-1 luminance/DCT preprocessing latency
batch-1 total PAD latency (preprocessing + model inference)
host process memory / peak memory, with measurement method
```

Cross-dataset notebook may report Kaggle inference timing as diagnostic only.

Do not present Kaggle CPU/GPU timing as Raspberry Pi latency.

Keep Raspberry Pi benchmarking as a later target-device experiment.

Compare E1/E3 using the same host, provider, thread budget, batch size, warmup and
timed sample count. Record these settings and report p50/p95, not only a mean.
Do not count GPU asynchronous launch time as completed inference latency.
Synthetic host timings are diagnostics and do not substitute for target-device
camera latency or memory measurements.

---

# 18. Markdown guide requirements

## `E3_spatial_frequency_concat_protocol.md`

Explain:
- research question;
- frozen E1/E2 dependencies;
- exact architecture;
- independent-init main policy;
- shared augmentation/crop rule;
- training schedule;
- threshold/calibration;
- artifact list;
- E1/E2/E3 fairness;
- frozen face-size bins and later resolution-augmentation control;
- two-input deployment contract and separate future runtime integration;
- what would justify E4 later.

## `E3_kaggle_run_and_artifact_guide.md`

Explain step-by-step:
- which E1 data artifacts to mount;
- which optional E1/E2 model artifacts are not required for independent E3;
- Kaggle path configuration;
- expected preflight messages;
- preliminary vs official;
- outputs to download;
- recommended local archive structure;
- how to run E3 held-out Test after freeze.

## `cross_dataset_evaluation_protocol.md`

Explain:
- why evaluation occurs after E3;
- zero-shot scope;
- LCC-FASD / CASIA-FASD roles;
- video/frame sampling;
- SCRFD external cache;
- source-locked thresholds;
- video-level HTER/AUC;
- detector coverage;
- zero-valid-frame and low-coverage video handling;
- face-size diagnostics and image/video evaluation-unit limitations;
- why this is a project cross-dataset stress test, not automatically an official benchmark.

## `cross_dataset_kaggle_run_and_result_guide.md`

Explain:
- required E1/E3 artifacts;
- optional E2 artifacts;
- LCC-FASD / CASIA-FASD dataset attachment;
- dataset path/label configuration;
- cache reuse;
- exact run order;
- expected output files;
- ZIP archive;
- recommended long-term folder structure;
- how to preserve manifests/caches/prediction CSVs for later E4/paper analysis.

---

# 19. Artifact archive structure to document

Recommend something like:

```text
pad_experiments/
├── E1_v5_3/
│   ├── training/
│   ├── data_protocol/
│   └── test/
│
├── E2_dct_v1/
│   ├── training/
│   └── test/
│
├── E3_concat_v1/
│   ├── training/
│   └── test/
│
└── cross_dataset_v1/
    ├── manifests/
    ├── detector_cache/
    ├── predictions/
    ├── plots/
    └── cross_dataset_e1_vs_e3_results.zip
```

Never overwrite `preliminary` with `official`.

---

# 20. Validation before finishing

Before reporting completion:

1. Parse every generated `.ipynb` as JSON.
2. Syntax-check every Python code cell, ignoring notebook magics/shell lines.
3. Search generated files for stale `MobileNetV3-Large` main-backbone references.
4. Search for accidental `gamma=True`.
5. Verify E3 main init defaults to `independent`.
6. Verify E3 uses exact E2 DCT/FrequencyBranch implementation.
7. Verify the same augmented crop feeds both E3 branches.
8. Verify no Test threshold calibration.
9. Verify cross-dataset target fine-tuning/adaptation is absent.
10. Verify external evaluation uses the same valid frame set for E1 and E3.
11. Verify video-level aggregation is deterministic.
12. Verify outputs include per-frame and per-video predictions.
13. Verify all artifacts use clear `preliminary` / `official` names.
14. Run `git diff --check`.
15. Do not modify E1/E2 frozen artifacts.
16. Verify face-size fields/breakdowns and undefined-bin metric handling.
17. Verify zero-valid-frame videos remain in coverage outputs and are explicitly excluded from conditional PAD metrics.
18. Verify E3 artifact docs do not claim compatibility with the current one-input app predictor.
19. Verify edge efficiency accounting includes external DCT/preprocessing and records timing settings.
20. Verify primary E3 augmentation remains identical to E1/E2 and stale OULU/SiW target references are absent.

If possible, run lightweight synthetic/unit checks without the real Kaggle datasets:
- E3 forward shape `[B,3]`;
- E3 two inputs share batch size;
- DCT tensor shape `[B,1,224,224]`;
- ONNX export/parity on dummy input;
- PAD score semantics;
- video mean-score aggregation;
- source threshold application;
- zero/one/multiple valid-frame video handling and undefined face-size-bin metrics;
- manifest/cache mismatch guards.

---

# 21. Final report to me

When finished, report:

```text
1. Files created
2. Exact E3 architecture
3. E3 initialization policy
4. Frozen E1/E2 settings reused
5. E3 artifact names
6. Cross-dataset target protocol
7. LCC-FASD/CASIA-FASD adapter assumptions requiring user configuration
8. Metrics/output artifacts
9. Static/tests performed
10. Any unresolved risks
```

Do not claim an E3 or cross-dataset result because these notebooks have not yet been run on Kaggle.

Stop after creating/reviewing the notebooks and Markdown guides. Do not start E4.
