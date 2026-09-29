# PAD Micro-Search Stages 0–2 — Kaggle Input Setup

This is an operational setup checklist for all eight notebooks. Use private Kaggle Datasets for licensed data and model inputs. Never upload `.env` files, credentials, database dumps, or unrelated personal files.

## A. One-time raw inputs

| Input | Purpose | Needed by | Expected files | Recommended private Dataset | Required |
|---|---|---|---|---|---|
| CelebA-Spoof raw | Read frozen micro Train/Val images and labels | 00, 01, 02–06 | `metas/intra_test/train_label.json` and `Data/train/...` images | `smartface-celeba-spoof-raw` | Yes |
| Frozen E1 v5.3 bundle | Supplies exact parent Train/Val memberships, run-config fingerprints, and matching CelebA bbox cache | 00 | exact NPZ, run config, and bbox cache below | `smartface-pad-e1-v5-3-frozen` | Yes |
| LCC-FASD raw | Image-level external-dev inference | 00, 01, 02–06 | images at the paths recorded in the LCC evaluation manifest | `smartface-lcc-fasd-raw` | Yes |
| LCC manifest/cache bundle | Reuse compatible LCC detector results without rerunning SCRFD | 00 | `evaluation_unit_manifest.csv`, `external_scrfd_cache.json`, `cross_dataset_protocol_snapshot.json` | `smartface-pad-lcc-scrfd-cache-v1` | Strongly recommended |
| SCRFD `det_500m.onnx` | One-time LCC cache rebuild when no compatible cache is attached | 00 only | exact model file; SHA must match `5e4447f50245bbd7966bd6c0fa52938c61474a04ec7def48753668a9d8b4ea3a` | `smartface-scrfd-500m` | Conditional |

The frozen E1 NPZ manifest is mandatory. The bbox cache does not replace the split manifest. Include the run-config metadata used to verify Train/Val fingerprints. The source Stage 0 notebook reads only NPZ `train_keys` and `val_keys`; it does not access `test_keys`. Raw LCC images remain necessary for model evaluation even when the detector cache is reused. SCRFD is needed only when Stage 0 must rebuild an absent or incompatible cache.

## B. Recommended Kaggle Dataset bundles

The names below are recommendations, not hard-coded slugs. Keep versions immutable after downstream notebooks start.

### Frozen E1 bundle — `smartface-pad-e1-v5-3-frozen`

Use these exact files from the repository:

```text
smartface-pad-e1-v5-3-frozen/
├── data_protocol/
│   ├── celeba_spoof_preliminary_v5_3_edge_mnv3_small_seed42.npz
│   └── celeba_scrfd_bbox_cache_v5_3_preliminary_seed42.json
└── metadata/
    └── mnv3s_e1_preliminary_v5_3_edge_run_config.json
```

The NPZ is gitignored in this checkout and must be copied from the frozen E1 artifact store into the private Kaggle bundle. Do not regenerate it from detector-cache rows. The full deployment model is not needed for Stages 0–2.

### LCC cache bundle — `smartface-pad-lcc-scrfd-cache-v1`

```text
smartface-pad-lcc-scrfd-cache-v1/
├── manifests/
│   └── evaluation_unit_manifest.csv
├── detector_cache/
│   └── external_scrfd_cache.json
└── predictions/
    └── cross_dataset_protocol_snapshot.json
```

Pair these artifacts from the same cross-dataset protocol version. Stage 0 compares cache policy, detector SHA, and sampled-manifest fingerprint before reuse.

### Stage 0 bundle — `smartface-pad-micro-stage0-v1`

Download `pad_micro_stage0_resources.zip` from notebook 00, extract it, then publish the resulting tree:

```text
smartface-pad-micro-stage0-v1/
└── stage0/
    ├── micro_manifest_v1_seed42.npz
    ├── micro_manifest_v1_seed42.json
    ├── micro_train_keys.json
    ├── micro_val_keys.json
    ├── micro_resource_fingerprints.json
    ├── stage0_summary.json
    ├── celeba_micro_bbox_cache.json
    ├── lcc_external_dev_manifest.csv
    ├── lcc_scrfd_cache_micro_v1.json
    ├── lcc_coverage_summary.json
    └── lcc_resource_fingerprints.json
```

An optional `crop_cache/` folder appears only if the size-guarded option was enabled and completed.

### Stage 1 bundle — `smartface-pad-micro-stage1-v1`

Download and extract `pad_micro_stage1_shared_e1.zip`:

```text
smartface-pad-micro-stage1-v1/
└── stage1_shared_micro_e1/
    ├── micro_e1_v1_best.pth
    ├── micro_e1_v1_checkpoint_last.pth
    ├── micro_e1_v1_meta.json
    ├── micro_e1_v1_config.json
    ├── micro_e1_v1_training_history.csv
    ├── micro_e1_v1_val_predictions.csv
    ├── micro_e1_v1_lcc_predictions.csv
    └── micro_e1_v1_summary.json
```

M0–M4 require the best checkpoint, metadata/config, and Stage 0 resource bundle. The last checkpoint and predictions are retained for review but are not needed to warm-start candidates.

### Candidate bundles — `smartface-pad-micro-m0-v1` through `...-m4-v1`

Each ZIP has a candidate-specific directory with `summary.json`, `branch_ablation.csv`, `feature_diagnostics.json`, `metrics.csv`, `config.json`, and phase checkpoint/prediction/history artifacts. Notebook 07 needs only the summary, ablation, diagnostics, metrics, and config files. It does not need the large `.pth` checkpoints or any raw dataset.

## C. Notebook-by-notebook input matrix

| Notebook | Raw CelebA | Frozen E1 | Raw LCC | SCRFD | Stage 0 | Stage 1 | M0 result | M1 result | M2 result | M3 result | M4 result | GPU | Internet |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 00 Stage 0 | Yes | Yes | Yes | Conditional | No | No | No | No | No | No | No | No | Conditional |
| 01 Stage 1 | Yes | No | Yes | No | Yes | No | No | No | No | No | No | Yes | Conditional |
| 02 M0 | Yes | No | Yes | No | Yes | Yes | No | No | No | No | No | Yes | No |
| 03 M1 | Yes | No | Yes | No | Yes | Yes | No | No | No | No | No | Yes | No |
| 04 M2 | Yes | No | Yes | No | Yes | Yes | No | No | No | No | No | Yes | No |
| 05 M3 | Yes | No | Yes | No | Yes | Yes | No | No | No | No | No | Yes | No |
| 06 M4 | Yes | No | Yes | No | Yes | Yes | No | No | No | No | No | Yes | No |
| 07 compare | No | No | No | No | No | Yes | Yes | Yes | Yes | Yes | Yes | No | No |

Notebook 00 only needs the SCRFD model when a compatible cache is missing/incompatible. Notebook 01 needs ImageNet weights from the torchvision local cache or an explicitly mounted local state dict; Internet is needed only if neither is available. Candidate runs do not need raw SCRFD after Stage 0.

## D. Exact workflow for each notebook

Use a Kaggle GPU accelerator for Stage 1 and all five candidates. Keep the accelerator off for Stage 0 cache reuse and notebook 07. All paths shown below are examples; replace the dataset slug with your mounted input path. Leave a path blank only when recursive exact-name discovery finds exactly one match.

### 00_stage0_build_micro_resources.ipynb

1. Create/open a Kaggle Notebook and attach the raw CelebA Dataset, raw LCC Dataset, frozen E1 bundle, and compatible LCC cache bundle.
2. Select no accelerator when reusing the detector cache. If rebuilding the cache, allow additional CPU time.
3. Add the four inputs above. Add SCRFD only if the compatible detector cache is unavailable.
4. Internet: off for compatible cache reuse. A rebuild needs the matching SCRFD file, InsightFace, and ONNX Runtime CPUExecutionProvider. The notebook checks dependencies before SCRFD inference and prints the exact install command; the current attached snapshot records InsightFace 2.0, so use `%pip install -q insightface==2.0 onnxruntime` when that is the required version. If the snapshot records another version, follow the version printed by the notebook.
5. Top config: set `E1_ARTIFACT_DIR`, `CELEBA_ROOT`, `LCC_FASD_ROOT`, and `LCC_CACHE_DIR` if discovery is ambiguous. Set `SCRFD_MODEL_PATH` only for a rebuild.
6. Preflight prints resolved input roots, 5,000/2,000 counts, frozen E1 hashes/fingerprints, class allocations, selected bbox status counts overall/Train/Val and crop factors by status, LCC policy/hash compatibility, ordered common VALID path fingerprint, and detector coverage by class. It checks all selected raw image paths and fails on unknown/invalid bbox statuses or missing crop factors.
7. Download `pad_micro_stage0_resources.zip`.
8. Extract and publish as `smartface-pad-micro-stage0-v1`.
9. Attach that Stage 0 bundle plus raw CelebA and raw LCC to notebook 01 and each candidate notebook.

### 01_stage1_train_shared_micro_e1.ipynb

1. Create a new Kaggle Notebook and attach raw CelebA, raw LCC, and Stage 0.
2. Select a GPU accelerator.
3. Add those three datasets; frozen E1 parent artifacts are no longer needed.
4. Internet is needed only when ImageNet MobileNetV3-Small weights are not cached and no complete compatible torchvision state dict is mounted. Partial backbone states fail preflight; random backbone initialization is never used.
5. Set `STAGE0_RESOURCE_DIR`, `CELEBA_ROOT`, and `LCC_FASD_ROOT` if automatic discovery is ambiguous; optionally set `IMAGENET_WEIGHTS_PATH`.
6. Before training, preflight rechecks that all 7,000 selected CelebA images and every LCC candidate path resolve under the roots attached to this Kaggle session. It also prints manifest/cache hashes, class counts, roots, CUDA/AMP and trainable parameter count.
7. Download `pad_micro_stage1_shared_e1.zip` and record the printed checkpoint SHA.
8. Extract and publish as `smartface-pad-micro-stage1-v1`.
9. Attach the same Stage 0 bundle, Stage 1 bundle, raw CelebA, and raw LCC to each of M0–M4.

### 02_m0_global_dct_gap.ipynb

1. Create/open a Kaggle Notebook.
2. Select a GPU.
3. Attach raw CelebA, raw LCC, Stage 0, and Stage 1.
4. Internet: not required.
5. Set `STAGE0_RESOURCE_DIR`, `STAGE1_RESOURCE_DIR`, `CELEBA_ROOT`, and `LCC_FASD_ROOT` only when exact discovery is ambiguous.
6. Before GPU training, preflight rechecks every selected CelebA and LCC candidate image path for the roots attached to this session; it then prints the shared hashes, 5K/2K classes, LCC coverage, CUDA/AMP, and Phase A parameter count.
7. Download `micro_m0_global_dct_gap_results.zip`.
8. Extract and publish as `smartface-pad-micro-m0-v1`.
9. Attach its result folder to notebook 07.

### 03_m1_dct_pool4x4.ipynb

1. Create/open a Kaggle Notebook.
2. Select a GPU accelerator.
3. Attach raw CelebA, raw LCC, and the same Stage 0/Stage 1 versions used by M0.
4. Internet: not required.
5. Set the four path overrides only when recursive discovery is ambiguous.
6. Confirm preflight prints M1, the shared resource hashes, class counts, CUDA/AMP, and Phase A parameter count.
7. Download `micro_m1_dct_pool4x4_results.zip` from the M1 output directory.
8. Extract and publish as `smartface-pad-micro-m1-v1`.
9. Attach that result folder to notebook 07.
### 04_m2_coord_dct_pool4x4.ipynb

1. Create/open a Kaggle Notebook.
2. Select a GPU accelerator.
3. Attach raw CelebA, raw LCC, and the exact Stage 0/Stage 1 versions used by M0/M1.
4. Internet: not required.
5. Set the four path overrides only when automatic discovery is ambiguous.
6. Confirm preflight prints M2, the common hashes, class counts, CUDA/AMP, and Phase A parameter count.
7. Download `micro_m2_coord_dct_pool4x4_results.zip` from the M2 output directory.
8. Extract and publish as `smartface-pad-micro-m2-v1`.
9. Attach that result folder to notebook 07.
### 05_m3_band_aware_dct.ipynb

1. Create/open a Kaggle Notebook.
2. Select a GPU accelerator.
3. Attach raw CelebA, raw LCC, and the exact Stage 0/Stage 1 versions used by the other candidates.
4. Internet: not required.
5. Set the four path overrides only when automatic discovery is ambiguous.
6. Confirm preflight prints M3, the common hashes, class counts, CUDA/AMP, and Phase A parameter count.
7. Download `micro_m3_band_aware_dct_results.zip` from the M3 output directory.
8. Extract and publish as `smartface-pad-micro-m3-v1`.
9. Attach that result folder to notebook 07.
### 06_m4_block_dct_8x8.ipynb

1. Create/open a Kaggle Notebook.
2. Select a GPU accelerator.
3. Attach raw CelebA, raw LCC, and the exact shared Stage 0/Stage 1 versions.
4. Internet: not required.
5. Set the four path overrides only when discovery is ambiguous.
6. Confirm preflight prints M4, the common hashes, class counts, CUDA/AMP, Phase A parameters, and that normalization uses micro-Train only.
7. Download `micro_m4_block_dct_8x8_results.zip`, including its saved normalization-stat JSON.
8. Extract and publish as `smartface-pad-micro-m4-v1`.
9. Attach that result folder to notebook 07.
### 07_compare_m0_m4_results.ipynb

1. Create/open a CPU notebook; no accelerator is required.
2. Attach the extracted Stage 1 result bundle and all five extracted M0–M4 result folders.
3. Do not attach raw CelebA/LCC images, Stage 0 data, or model checkpoints; notebook 07 reads only Stage 1 summary metadata and candidate result summaries/diagnostics.
4. Internet is not required.
5. Set `STAGE1_RESULT_DIR` and/or `RESULT_DIRS` only when recursive exact-name discovery finds an ambiguous copy.
6. Preflight validates the Stage 1 schema/metrics/checkpoint SHA, every candidate ID, all Stage 0 resource fingerprints/counts, common LCC valid-path fingerprint/scored counts, branch diagnostics, and the shared micro budget.
7. Download `micro_m0_m4_comparison.csv`, `micro_m0_m4_ablation_comparison.csv`, and `micro_m0_m4_review_packet.json`.
8. The packet reports Stage 1 spatial and M0 representation references separately and surfaces APCER/BPCER and AUC/HTER tradeoffs. This screen reports `NO WINNER`; promising labels only guide a later independent confirmation.

## E. Publishing outputs between Kaggle sessions

Two handoff methods are supported:

- **Download/upload:** download a notebook ZIP, extract it locally, create a new private Kaggle Dataset, then attach that Dataset to the next notebook.
- **Kaggle-native publication:** use the notebook output/version publishing flow available in the current Kaggle UI and create a private Dataset version.

Kaggle `/kaggle/working` storage is ephemeral across sessions. Download or publish each required ZIP before ending that session. Attach extracted folder trees to later notebooks; do not rely on an old working directory.

## F. Path resolution

Examples:
- `/kaggle/input/<dataset-slug>/CelebA_Spoof`
- `/kaggle/input/<dataset-slug>/stage0`
- `/kaggle/input/<dataset-slug>/stage1_shared_micro_e1`

Blank overrides trigger recursive exact-name discovery under `/kaggle/input`. Exactly one match is accepted. If zero or multiple copies are found, the notebook fails before training and prints the expected explicit path setting. Do not attach old and new versions with duplicate artifact names unless you set the appropriate directory override.

## G. Minimal files to pass forward

- Stage 0 → Stage 1/M0–M4: micro NPZ/JSON, Train and Val key lists, fingerprint JSON, CelebA micro bbox cache, LCC manifest/cache/coverage.
- Stage 1 → M0–M4: `micro_e1_v1_best.pth`, meta/config, resource fingerprints through meta, and summary.
- Stage 1 → notebook 07: `micro_e1_v1_summary.json` only; the summary contains its resource hashes, shared checkpoint SHA, spatial Val/LCC metrics, and common LCC counts.
- M0–M4 → notebook 07: `summary.json`, `branch_ablation.csv`, `feature_diagnostics.json`, `metrics.csv`, and `config.json`. The complete ZIP may be published; notebook 07 does not load checkpoints or raw images.

## H. Failure checklist

| Failure | Expected fix |
|---|---|
| Multiple E1 files match | Set `E1_ARTIFACT_DIR` to the frozen bundle. |
| Frozen NPZ missing | Restore the exact v5.3 frozen NPZ; never regenerate memberships from bbox rows. |
| E1 run mode/fingerprints mismatch | Attach the preliminary v5.3 bundle whose Train/Val fingerprints match. |
| Stage 0 hash mismatch | Reattach one immutable Stage 0 version to every downstream run. |
| Stage 1 checkpoint hash mismatch | Reattach the matching Stage 1 ZIP and meta; do not rename checkpoints. |
| LCC raw image path missing | Set `LCC_FASD_ROOT` to the directory containing the manifest-relative images. |
| LCC policy/cache mismatch | Rebuild once in Stage 0 with the correct SCRFD hash or attach the compatible cache bundle. |
| ImageNet weights unavailable | Enable Internet for the run or mount a compatible torchvision MobileNetV3-Small state dict. Random fallback is disabled. |
| Stage 1 summary is missing/ambiguous | Attach one extracted Stage 1 result bundle or set `STAGE1_RESULT_DIR`; notebook 07 checks its schema and checkpoint/resource hashes. |
| Wrong candidate summary attached | Set that candidate's explicit `RESULT_DIRS` entry; notebook 07 also checks `candidate_id`. |
| Optional crop cache exceeds limit | Stage 0 deletes the partial cache and continues with raw image loading. |

## I. Security and reproducibility

- Do not upload `.env` files, database credentials, tokens, or unrelated model/data files.
- Do not rename frozen artifacts unless you record the mapping and update the explicit path.
- Do not edit micro key lists between candidate runs.
- Do not overwrite a Stage 0 or Stage 1 Dataset after M0–M4 start; publish a new version if the resources intentionally change.
- Preserve every SHA256, split fingerprint, configuration, checkpoint hash, and Stage 0/1 version used for each run.
