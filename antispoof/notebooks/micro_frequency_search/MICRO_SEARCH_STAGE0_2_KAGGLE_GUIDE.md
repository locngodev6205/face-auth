# PAD Micro-Search Stages 0–2 — Kaggle Run Guide

This is the short run-order guide. The detailed input matrix, upload trees, and per-notebook setup checklist are in [MICRO_SEARCH_STAGE0_2_KAGGLE_INPUT_SETUP_GUIDE.md](MICRO_SEARCH_STAGE0_2_KAGGLE_INPUT_SETUP_GUIDE.md). The frozen scientific choices are recorded in [MICRO_SEARCH_STAGE0_2_PROTOCOL.md](MICRO_SEARCH_STAGE0_2_PROTOCOL.md).

## Run order

1. Run `00_stage0_build_micro_resources.ipynb` with the frozen E1 membership, CelebA-Spoof raw images, LCC-FASD raw images, and the compatible LCC manifest/SCRFD cache bundle. It writes `pad_micro_stage0_resources.zip`.
2. Publish or upload the extracted Stage 0 directory as `smartface-pad-micro-stage0-v1`.
3. Run `01_stage1_train_shared_micro_e1.ipynb` once with a Kaggle GPU. It writes `pad_micro_stage1_shared_e1.zip`. Record the printed checkpoint SHA.
4. Run M0 first with the same Stage 0 and Stage 1 versions. Inspect its hashes, metrics, diagnostics, and complete result bundle before spending time on the remaining candidates.
5. After M0 passes review, run M1–M4 in any order (or in parallel) with those exact same Stage 0/Stage 1 versions. Each notebook writes one candidate result ZIP.
6. Publish or upload the extracted Stage 1 folder and all five candidate result folders, then run `07_compare_m0_m4_results.ipynb`. It needs Stage 1 plus M0–M4 summaries, but no raw image datasets.

## Artifact handoff

Kaggle working storage is temporary between sessions. Download each ZIP or publish the notebook output as a private Kaggle Dataset before starting the next stage. For the next notebook, publish/attach an extracted folder tree; notebooks do not read ZIP files directly. They validate exact file hashes and fail on ambiguous duplicate artifacts.

Use a new Dataset version if a resource intentionally changes. Never replace Stage 0 or Stage 1 midway through M0–M4.

## Execution limits

- Stage 0 builds only the frozen 5K/2K Train/Val micro resources. It never reads CelebA held-out Test.
- LCC-FASD is labeled external-dev / cross-domain stress because it has already been observed. Its threshold stays locked from CelebA micro-Val.
- Stage 1 trains one spatial baseline. All candidates warm-start from the same saved checkpoint.
- Stages 0–2 do not export ONNX, run device benchmarks, use CASIA/Replay-Attack, or make a final scientific winner claim.
- Notebook 07 reports M0 frequency-control and Stage 1 spatial-reference comparisons separately, uses prespecified screening heuristics, never auto-promotes a candidate, and reports `NO WINNER`.

