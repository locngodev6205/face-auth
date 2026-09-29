# PAD Micro-Search Stages 0–2 — Final Pre-Run Audit

Audit date: 2026-09-29  
Scope: the eight Stage 0–2 notebooks and their three operational documents.

## A. Issues found

- **MAJOR — resolved:** Stage 1 and M0–M4 could reject E1 fallback bbox records or silently use the wrong crop factor. The shared crop contract now accepts `VALID`/1.55, `FALLBACK_DET_FAIL`/1.50, and `FALLBACK_SUSPICIOUS`/1.50; rejects `INVALID`, unknown statuses, missing factors, and mismatched factors. Stage 0 audits selected Train/Val status and factor counts before producing resources.
- **MAJOR — resolved:** `FREQ_SHUFFLE` used a one-position cyclic roll. It now uses the same seeded, label-independent Sattolo derangement in every candidate, guarantees zero fixed points, and records the seed, permutation SHA256, and fixed-point count. The serialized diagnostic description now matches the implementation.
- **MAJOR — resolved:** Notebook 07 lacked the independent Stage 1 spatial reference. It now loads exactly one `micro_e1_v1_summary.json`, validates the Stage 1 SHA and resource hashes, and reports spatial-reference metrics/deltas alongside M0 comparisons. M0 remains the frequency-representation control; no composite score or automatic promotion was added.
- **MAJOR — resolved:** a partial local ImageNet state dict could have left MobileNetV3-Small layers randomly initialized. Stage 1 now fails on any missing/unexpected backbone key and has no random fallback. Candidate diagnostics also reuse the ordered Val/LCC features from the selected CONCAT inference pass.
- **MAJOR — resolved:** LCC source-cache dictionary keys were not safe to treat as record IDs. Stage 0 now joins source cache records to evaluation candidates by normalized path, checks exact path-set equality and one record per path, then writes a self-consistent cache keyed by its standardized `frame_id`.
- **MAJOR — resolved:** downstream Kaggle sessions could attach a different/misconfigured raw-data root after Stage 0 and discover missing image paths only during evaluation. Stage 1 and M0–M4 now recheck all 7,000 selected CelebA images and every LCC candidate path before training starts.
- **MINOR — resolved:** all eight generated notebooks lacked nbformat 4.5 cell IDs; code cells also lacked explicit `execution_count` and `outputs` fields. These fields and unique IDs are now present.
- **MINOR — resolved:** machine-handoff JSON could emit non-standard NaN/Infinity. Stage 0, Stage 1, candidate, and review-packet JSON serialization now uses `allow_nan=False` for persisted outputs.
- **NOTE:** the local environment does not include the `nbformat` package. Notebook JSON, required 4.5 cell structure/IDs, and Python syntax were checked directly; `nbformat.validate()` itself could not be invoked.

No BLOCKER or MAJOR issue remains in the inspected source. No Kaggle training or real-data inference has been run as part of this audit.

## B. Fixes made

- Added strict bbox-status/factor validation before optional crop-cache reads in Stage 1 and all five candidate pipelines. Stage 0 records selected-key status counts overall, by split, and crop-factor counts by status.
- Added a downstream raw-image path preflight to Stage 1 and M0–M4 so a mismatched Kaggle attachment fails before GPU training.
- Added Stage 0 checks for the frozen 5K/2K membership, seed 42, E1 label/attack-code mapping, split and subject separation, bbox records, selected CelebA image paths, and all three classes. The frozen NPZ remains authoritative; held-out Test keys are not loaded.
- Hardened LCC cache reuse/rebuild: exact policy, detector SHA, and sampled-manifest SHA matching; early InsightFace/ONNX Runtime preflight on rebuild; normalized path joins independent of the source dictionary key; exact candidate coverage and common ordered VALID-path fingerprint; per-class detector coverage retained.
- Added common ordered-key/count checks for Stage 1 and candidate Val/LCC inference. Missing samples or changed order now fail instead of producing model-specific subsets.
- Made all machine-handoff JSON strict. Added notebook 4.5 cell IDs and empty execution/output fields so notebooks have complete code-cell structure.
- Added Stage 1 reference fields and interpretation to notebook 07 while preserving separate M0 and Stage 1 roles, APCER/BPCER tradeoffs, and AUC-versus-HTER interpretation.
- Updated the protocol, Kaggle guide, and input setup guide to describe the validated cache contract, shared augmentation treatment, shuffle policy, Stage 1 handoff, and recommended run order.

## C. Known design choices intentionally NOT changed

- Stage 0 still uses only the frozen E1 NPZ Train/Val memberships: exactly 5,000 Train and 2,000 Val samples, seed 42. The parent split is not reconstructed from metadata or cache rows.
- The model remains ImageNet MobileNetV3-Small with the agreed E1 head. M0 remains the E2-style global DCT/GAP control; M1 changes the pooling grid; M2 adds fixed coordinates; M3 adds fixed radial bands; M4 retains its 8×8 blocks and K=16 coefficients.
- M4 normalization is estimated only from deterministic micro-Train base crops and reused for Val/LCC. Stage 1 and M0–M4 use the same self-contained RGB augmentation implementation; Val and LCC remain unaugmented. The implementation is NumPy-based rather than the original Albumentations call, with the agreed flip, brightness/contrast, and bbox-jitter probabilities/ranges documented in the protocol.
- The approved micro budgets, Val-only checkpoint/threshold selection, locked LCC threshold, class mapping, binary score, and branch-ablation modes remain unchanged. No full training, held-out CelebA Test, CASIA/Replay evaluation, ONNX export, quantization, or app integration was introduced.
- Notebook 07 remains a descriptive screen: M0 controls frequency representation, Stage 1 is the spatial-only reference, and the output always permits `NO WINNER`.

## D. Static/synthetic tests run

- Parsed all eight notebook files as JSON; checked nbformat 4.5 top-level/cell fields, valid unique cell IDs, empty code outputs/counts, and compiled every code cell. No `nbformat` package was available for its own validator.
- Scanned active code for MobileNetV3-Large, `torch.roll`, ONNX export, CASIA, and Replay-Attack references; none were present. Confirmed M0–M4 shared source and candidate result-schema source are identical.
- Executed the actual Stage 0 bbox guard on synthetic `VALID`/fallback records and confirmed rejection of `INVALID`, unknown status, and missing factor.
- Ran actual M0–M4 representation/model code on synthetic RGB input: frequency encoders returned `[1, 64]`; candidate CONCAT heads returned `[1, 3]`. M2/M3 map channels and M4 `[16, 28, 28]` layout matched the configured contract.
- Checked M3 masks cover every coefficient exactly once. Compared M4's custom 8×8 block DCT with `cv2.dct` on 12 random blocks; maximum agreement was within `atol=2e-6`, `rtol=2e-6`.
- Tested the actual shuffle helper: same seed gives the same permutation and SHA, another seed changes the permutation, fixed-point count is zero, and the helper accepts only `(n, seed)`.
- Tested actual PAD scoring/metrics with synthetic logits: `d = real_logit - logsumexp(spoof_logits)`, APCER 0, BPCER 0.5, ACER 0.25, AUC 0.75. The selected threshold matched an independent brute-force search over score intervals and predicted Real for scores at/above threshold.
- Tested the actual raw-image preflight with temporary synthetic roots: it accepted complete CelebA/LCC paths and failed after a required image was removed.
- Ran the actual Phase B epoch helper on synthetic CPU batches. B1 trained only fusion and left spatial/frequency BatchNorm running state unchanged; B2 trained frequency plus fusion, left spatial BatchNorm unchanged, and updated frequency BatchNorm. Optimizer membership matched `requires_grad` parameters.
- Ran notebook 07 against synthetic Stage 1 and M0–M4 result bundles. It accepted matching resource/checkpoint hashes, wrote all three review artifacts, and retained `NO WINNER` with automatic promotion disabled.
- Round-tripped a synthetic summary through strict JSON parsing and confirmed `allow_nan=False` rejects NaN.
- `git diff --check`: see final verification; it passes.

## E. Remaining assumptions and limits

- The audit did not run on Kaggle and did not access raw CelebA/LCC images, the frozen E1 NPZ, a Kaggle GPU, or ImageNet download/cache. Stage 0 selected-key counts/statuses and image-path resolution must be confirmed by its runtime preflight.
- The observed local source caches contained 125,116 CelebA bbox records: 123,663 `VALID`/1.55, 1,249 `FALLBACK_DET_FAIL`/1.50, and 204 `FALLBACK_SUSPICIOUS`/1.50. The LCC cache contained 3,766 candidates: 3,762 `VALID` and 4 `NO_VALID_FACE`; its ordered sampled-manifest fingerprint matched the protocol snapshot. These are source-cache totals, not the selected 5K/2K Stage 0 audit counts.
- A rebuild of the LCC detector cache still depends on a compatible SCRFD model and working InsightFace/ONNX Runtime CPU provider. Stage 0 now checks that dependency before inference and prints the required install command; using the compatible frozen cache remains preferred.
- Stage 0 requires the exact frozen E1 NPZ and run-config/bbox artifacts plus raw CelebA and LCC roots. It intentionally fails early if an artifact is absent, ambiguous, incompatible, or an image path does not resolve.
- Synthetic checks establish code-path behavior and tensor/math contracts only. They do not establish training convergence, Kaggle runtime, data correctness beyond the inspected caches, or PAD performance. No Stage 0–2 metrics exist yet.

## F. Kaggle preflight checklist

- Attach the raw CelebA-Spoof and LCC-FASD datasets, exact frozen E1 NPZ/run-config/bbox-cache bundle, and the compatible LCC manifest/cache/snapshot bundle. Set explicit roots only when recursive exact-name discovery is ambiguous.
- Prefer cache reuse. If Stage 0 says a rebuild is required, attach the matching SCRFD model and satisfy the printed InsightFace/ONNX Runtime version/provider check before rerunning.
- Run Stage 0 and inspect: 5,000/2,000 counts, seed and frozen fingerprints, Train/Val and subject-overlap checks, class/attack-code counts, bbox status/factor counts by split, resolved image paths, LCC candidate and VALID counts by class, detector coverage, and common ordered VALID-path SHA.
- Publish the Stage 0 bundle unchanged. Run Stage 1 once with a GPU; its preflight must resolve all selected CelebA/LCC images under the current attached roots. Verify ImageNet weights loaded, Val/LCC counts and metrics, locked Val threshold, and checkpoint SHA. Publish that bundle unchanged.
- Run M0 only next. Its preflight also checks the current raw roots before GPU training. Verify its resource/checkpoint hashes, output schemas, common LCC scored count, and branch diagnostics before running M1–M4.
- Run notebook 07 only after Stage 1 and all five candidate bundles are available. It needs summaries/diagnostics, not raw images or checkpoints.

## G. Verdict

**READY FOR STAGE 0 KAGGLE SMOKE RUN**

This verdict is based on source audit, early runtime guards, and offline synthetic checks. It does not claim real-data training correctness or any Stage 0–2 result.

Recommended first-run sequence:

1. Run `00_stage0_build_micro_resources.ipynb`.
2. Inspect hashes, bbox status/factor counts, and LCC coverage/path fingerprint.
3. Run `01_stage1_train_shared_micro_e1.ipynb`.
4. Inspect Val and LCC baseline metrics and the Stage 1 checkpoint SHA.
5. Run M0 only.
6. Inspect M0 artifacts and branch diagnostics.
7. Only then launch M1–M4, optionally in parallel.
8. Run `07_compare_m0_m4_results.ipynb` with Stage 1 and M0–M4 result bundles.
