# Binary cross-domain MiniFASNetV2 pilot

This package creates a new v2 protocol inside this directory only. Historical v1 contracts, notebooks, and outputs remain separate. The question is whether a literature-adapted binary PAD and auxiliary recipe improves the MiniFASNetV2 pilot, and whether CSMR adds cross-domain robustness to that recipe.

## Execution order on Kaggle

1. Use the completed `00-v2_prepare_minifasnet_full_resources.ipynb` output with frozen CUDA SCRFD bbox caches. Its provenance must match the parent contract SHA `aa67032a79627636d967dc885202540a34410c2b4d64522a5604d3aedff85b2d`.
2. Attach the v1 resource bundle and raw CelebA-Spoof. Edit the v1 resource and CelebA paths in the CONFIG cell of `00c_prepare_binary_crossdomain_resources.ipynb`, then run it. The contract and README are embedded in 00c and exported automatically. It does **not** run SCRFD. It verifies metadata/cache hashes and produces `/kaggle/working/minifasnet_binary_crossdomain_resources_v2.zip` with a FileLink.
3. Upload or mount the unzipped `minifasnet_binary_crossdomain_resources_v2` bundle as a Kaggle Dataset.
4. Attach raw CelebA and official LCC images, the MiniFASNet source, and official `2.7_80x80_MiniFASNetV2.pth`. Edit the CONFIG cell of `01_mini_ablation_binary_crossdomain_10k.ipynb`, select `RUN_FILTER` if needed, then run it. It does **not** run SCRFD. Use a GPU and the frozen batch 256 for C/D/E/F; an OOM stops the run. Changing copied batch size requires a new, consistently restarted C/D/E/F protocol.
5. Inspect all six run directories, the comparison summary, saved predictions, provenance, and checkpoint/threshold history. Kaggle execution is required before reporting measured results.
6. Only after this pilot, create separate future full-data Vanilla and CSMR notebooks under the full cross-domain protocol.

00c emits exactly:

```text
minifasnet_binary_crossdomain_resources_v2/
  PAD_DATA_AND_EVALUATION_CONTRACT_v2_CROSSDOMAIN_BINARY.md
  README.md
  config.json
  provenance.json
  manifest_fingerprints.json
  bbox/celeba_bbox_cache_full.json
  bbox/lcc_official_evaluation_bbox_cache.json
  labels/celeba_auxiliary_labels.csv
  manifests/mini_train10k.csv
  manifests/mini_val3k.csv
  manifests/crossdomain_train_full.csv
  manifests/crossdomain_celeba_source_dev_full.csv
  manifests/lcc_official_evaluation_full.csv
```

The six runs are A/B historical three-logit pretrained/scratch, C/D binary+auxiliary pretrained/scratch, and E/F C/D plus CSMR. Mini Train10K and Mini Val3K are shared identically. Mini Val3K alone selects checkpoints and calibrates thresholds. Official CelebA Test and LCC are evaluated once after each run is frozen. Mini LCC is pilot evidence only. LCC never selects checkpoints, thresholds, or hyperparameters. C vs E and D vs F are the controlled CSMR pairs; A/B vs C/D change multiple training factors.

The copied recipe is literature-adapted and compute-compressed: 20 epochs with milestones [6,14] instead of the source's 71 epochs and [20,50]. The source repository is [Assessing-Efficient-FAS-CVPR2024](https://github.com/Inria-CENATAV-Tec/Assessing-Efficient-FAS-CVPR2024), especially its [config](https://github.com/Inria-CENATAV-Tec/Assessing-Efficient-FAS-CVPR2024/blob/main/configs/config_large_075.py), [training](https://github.com/Inria-CENATAV-Tec/Assessing-Efficient-FAS-CVPR2024/blob/main/train.py), and [CelebA loader](https://github.com/Inria-CENATAV-Tec/Assessing-Efficient-FAS-CVPR2024/blob/main/datasets/celeba_spoof.py). This package keeps the project MiniFASNet architecture, SCRFD boxes, MiniFAS crop, BGR input, and candidate accounting.

The run notebook steps `MultiStepLR` after each completed epoch; milestones 6 and 14 therefore change the LR used by epochs 7 and 15. Early stopping may end before the second decay; the report records that. Resuming requires an exact per-run configuration hash and preserved RNG, optimizer, scheduler, scaler, best checkpoint, and patience state.

Albumentations has both legacy `*_limit` and current `*_range` APIs. Notebook 01 checks the installed signatures, uses the frozen numeric ranges, records the version and API variant, and fails on an unknown API. ISO noise is applied in RGB space; its output is converted back to BGR before the MiniFASNet tensor. See the [current ISO noise API](https://albumentations.ai/docs/api-reference/albumentations/augmentations/pixel/noise/).
