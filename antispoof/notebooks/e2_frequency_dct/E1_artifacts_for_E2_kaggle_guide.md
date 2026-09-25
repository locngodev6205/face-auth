# E1 v5.3 Artifacts — Save and Reuse Guide for E2 on Kaggle

## 1. Why these artifacts must be preserved

E2 must reuse the exact E1 data protocol rather than rebuilding it.

The two most important reusable data artifacts are:

```text
SCRFD bbox cache
split manifest
```

Without them, E2 may:

- run SCRFD again unnecessarily;
- produce different detector/fallback decisions;
- accidentally resample Train/Validation/Test;
- break the fairness of E1/E2/E3 ablations.

E1 model artifacts must also be retained because they are needed later for E3 initialization/analysis and deployment, even though E2 itself does not load the spatial model.

---

## 2. E1 artifact checklist

For E1 v5.3, download and archive the following Kaggle outputs.

### A. Required to run E2

For `preliminary`:

```text
celeba_scrfd_bbox_cache_v5_3_preliminary_seed42.json
celeba_spoof_preliminary_v5_3_edge_mnv3_small_seed42.npz
```

For `official`, use the corresponding `official` files instead.

These two files are mandatory.

### B. Strongly recommended for E2 verification

```text
celeba_spoof_<mode>_v5_3_preprocess_audit.json
mnv3s_e1_<mode>_v5_3_edge_run_config.json
```

The E2 notebook uses `run_config.json`, when available, to verify:

```text
split fingerprints
SCRFD cache fingerprint
crop policy
```

### C. Preserve for E1 freeze / E3 / deployment

```text
mnv3s_e1_<mode>_v5_3_edge_best.pth
mnv3s_e1_<mode>_v5_3_edge_checkpoint_last.pth
mnv3s_e1_<mode>_v5_3_edge_best_meta.json
mnv3s_e1_<mode>_v5_3_edge_training_history.json
mnv3s_e1_<mode>_v5_3_edge_training_history.png
mnv3s_e1_<mode>_v5_3_edge_run_config.json
mnv3s_e1_<mode>_v5_3_edge_best.onnx
mnv3s_e1_<mode>_v5_3_edge_runtime_config.json
```

If Held-out Test was run after freeze, also preserve:

```text
mnv3s_e1_<mode>_v5_3_edge_heldout_test_results.json
```

### Minimum long-term E1 archive

At absolute minimum, keep:

```text
best.pth
best.onnx
best_meta.json
run_config.json
SCRFD cache JSON
split manifest NPZ
preprocess audit JSON
```

Do not keep only the ONNX model.

---

## 3. Suggested local archive structure

```text
smartface_pad_artifacts/
└── E1_v5_3_mnv3_small/
    ├── data_protocol/
    │   ├── celeba_scrfd_bbox_cache_v5_3_<mode>_seed42.json
    │   ├── celeba_spoof_<mode>_v5_3_edge_mnv3_small_seed42.npz
    │   └── celeba_spoof_<mode>_v5_3_preprocess_audit.json
    │
    ├── pytorch/
    │   ├── mnv3s_e1_<mode>_v5_3_edge_best.pth
    │   └── mnv3s_e1_<mode>_v5_3_edge_checkpoint_last.pth
    │
    ├── deployment/
    │   ├── mnv3s_e1_<mode>_v5_3_edge_best.onnx
    │   └── mnv3s_e1_<mode>_v5_3_edge_runtime_config.json
    │
    └── metadata/
        ├── mnv3s_e1_<mode>_v5_3_edge_best_meta.json
        ├── mnv3s_e1_<mode>_v5_3_edge_run_config.json
        ├── mnv3s_e1_<mode>_v5_3_edge_training_history.json
        └── mnv3s_e1_<mode>_v5_3_edge_training_history.png
```

The Kaggle input does not need this exact directory hierarchy; the E2 notebook searches recursively for the exact filenames.

---

## 4. Saving E1 outputs from Kaggle

The E1 notebook writes artifacts to its working/output directory.

After training:

1. Save a Kaggle notebook version so the run outputs are preserved.
2. Download the complete E1 output set locally, not only the ONNX file.
3. Verify that the mandatory cache JSON and split NPZ are present.
4. Keep a copy outside the transient Kaggle session.

Do not assume a temporary interactive session is your permanent artifact store.

---

## 5. Make the E1 artifacts available to E2

Two practical options are valid.

### Option A — Attach the saved E1 notebook output

Create/open the E2 Kaggle notebook and attach the output of the completed E1 notebook as an input.

The mounted files will appear somewhere under:

```text
/kaggle/input/...
```

### Option B — Create a private Kaggle Dataset

Upload the E1 artifact folder as a private Kaggle Dataset, for example:

```text
smartface-e1-v5-3-artifacts
```

Then attach it to the E2 notebook.

A typical mounted path may look like:

```text
/kaggle/input/smartface-e1-v5-3-artifacts/
```

The exact slug/path depends on how the dataset is named.

---

## 6. E2 notebook configuration

At the top of the E2 notebook:

```python
E1_RUN_MODE = "preliminary"
E1_ARTIFACT_DIR = ""
```

### Auto-discovery

If `E1_ARTIFACT_DIR=""`, E2 recursively searches `/kaggle/input` for the exact expected filenames.

This is convenient when only one E1 artifact set is mounted.

### Explicit path

If multiple versions are attached, set:

```python
E1_ARTIFACT_DIR = "/kaggle/input/<your-e1-artifact-dataset>"
```

The notebook then requires the exact files inside that directory.

---

## 7. Attach CelebA-Spoof too

E2 reuses the E1 bbox geometry, but it still needs the original images.

Attach the same CelebA-Spoof Kaggle dataset used by E1.

The notebook currently expects:

```python
DATA_ROOT = (
    "/kaggle/input/datasets/attentionlayer241/"
    "celeba-spoof-for-face-antispoofing/"
    "CelebA_Spoof_/CelebA_Spoof"
)
```

If Kaggle mounts it elsewhere, edit only `DATA_ROOT`.

Do not regenerate the split.

---

## 8. What E2 actually reuses

E2 loads:

```text
split manifest NPZ
        ↓
train_keys
val_keys
test_keys

SCRFD cache JSON
        ↓
bbox_xyxy
bbox_source
crop_factor
status
```

Then E2 reads the raw image and performs:

```text
same E1 crop
→ same E1 TRAIN augmentation
→ luminance
→ DCT
→ FrequencyOnlyPAD
```

E2 does **not**:

```text
rerun SCRFD
resample subjects
change small-face filtering
load E1 ONNX
run the MobileNetV3-Small spatial branch
```

This is intentional.

---

## 9. Which E1 model artifacts are reused later, not in E2?

### E1 `best.pth`

Not required by E2.

It is important for E3 because E3 uses a MobileNetV3-Small spatial branch. The E1 checkpoint can be used to initialize or freeze/unfreeze that branch if this is chosen in the E3 protocol.

### E1 `best.onnx`

Not required by E2 training.

Keep it for:

```text
runtime deployment
latency comparison
E1 vs E3 deployment benchmark
```

### E1 threshold / metadata

Not reused as E2 threshold.

E2 must calibrate its own threshold on the same Validation split.

Never copy the E1 PAD threshold into E2.

---

## 10. Preliminary vs official must not be mixed

Correct:

```text
E1 preliminary cache
+ E1 preliminary split
→ E2 preliminary
```

Correct:

```text
E1 official cache
+ E1 official split
→ E2 official
```

Incorrect:

```text
preliminary cache
+ official manifest
```

or:

```text
E1 preliminary split
→ claim E2 as official full-data experiment
```

The E2 notebook names its outputs using the selected `E1_RUN_MODE` to make this visible.

---

## 11. Recommended Kaggle flow

```text
E1 Kaggle run
    ↓
Save Version
    ↓
Download/archive all E1 artifacts
    ↓
Attach E1 output or private artifact Dataset to E2
    ↓
Attach the same CelebA-Spoof dataset
    ↓
Open E2 notebook
    ↓
Set E1_RUN_MODE
    ↓
Run artifact verification cell
    ↓
Expected:
"=== E1 artifact verification PASS ==="
    ↓
Run E2 training
```

If artifact verification fails, do not bypass it. Fix the mounted artifact set first.

---

## 12. Artifacts to save after E2

After E2, preserve:

```text
e2_freq_only_<mode>_dct_v1_best.pth
e2_freq_only_<mode>_dct_v1_checkpoint_last.pth
e2_freq_only_<mode>_dct_v1_best_meta.json
e2_freq_only_<mode>_dct_v1_run_config.json
e2_freq_only_<mode>_dct_v1_training_history.json
e2_freq_only_<mode>_dct_v1_training_history.png

e2_freq_only_<mode>_dct_v1_frequency_branch_best.pth
e2_freq_only_<mode>_dct_v1_frequency_branch_spec.json

e2_freq_only_<mode>_dct_v1_best.onnx
e2_freq_only_<mode>_dct_v1_runtime_config.json
```

The most important new artifact for E3 is:

```text
e2_freq_only_<mode>_dct_v1_frequency_branch_best.pth
```

However, whether E3 preloads this branch must be decided explicitly in the E3 protocol before training.

---

## 13. Before starting E3

Archive together:

```text
E1 best.pth
E1 best_meta / run_config
E1 SCRFD cache
E1 split manifest
E1 preprocess audit

E2 best.pth
E2 frequency_branch_best.pth
E2 frequency_branch_spec.json
E2 best_meta / run_config
```

This gives E3 everything needed to reproduce the exact two branches and data protocol.
