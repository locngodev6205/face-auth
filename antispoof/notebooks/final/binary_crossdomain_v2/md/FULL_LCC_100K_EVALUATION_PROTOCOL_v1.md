# Full LCC 100K evaluation protocol v1

Evaluate the frozen `C100k_copy_pretrained`, `P3_100k_worst_ce25`, and `Q3_100k_random_ce25_kl01` checkpoints on LCC training, development, official evaluation, and their concatenation. Load each run's `best.pth` and its own `evaluation_freeze.json` Val15K threshold. Do not train, choose checkpoints, or fit thresholds on LCC.

Read the completed 00d manifests and SCRFD bbox caches. Score only `VALID`, `usable_for_pad=True` rows with a raw SCRFD bbox; retain failed detections as unscored candidates. Use notebook 04's MiniFASNetV2 wrapper, 2.7× crop, 80×80 BGR float input divided by 255, `z_real-z_attack` score, and metric definitions. No detector is run.

Training/development paths are relative to the common LCC root; official evaluation paths are relative to its `LCC_FASD_evaluation` directory. Infer each physical split once per model. Build combined results by concatenating physical predictions and recomputing metrics from pooled scores. Report candidate/scored counts, AUC, EER, TPR@FPR1%, APCER, BPCER, HTER, and accuracy. If an original 100K official-evaluation metric file is available, require agreement within 1e-6.

`MODEL_FILTER` and `LCC_SPLIT_FILTER` can restrict work. A combined request uses all three physical splits. `REUSE_EXISTING` reuses a prediction only when its input file stamps, threshold, inference settings, and candidate rows match; otherwise that split is recomputed. No contract or SHA audit is required. The pooled diagnostic does not select a model or reproduce legacy model/preprocessing results.
