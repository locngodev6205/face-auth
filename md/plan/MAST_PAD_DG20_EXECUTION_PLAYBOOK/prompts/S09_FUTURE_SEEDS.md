# CODEX PROMPT — Post-Kaggle 3-seed evidence validation and document update plan

## Your role and non-negotiable rules
Work in `face_auth` and execute **S09 CONDITIONAL only**. Review `00_DATA_LOCK_FOR_CODEX.md`, `01_MASTER_EXECUTION_PLAYBOOK.md`, DG20, Word `(2).docx`, approved latest report revision, article and evidence audit 00–08. Determine actual file paths instead of assuming they exist. Preserve original Word, article/release, prior stage revisions, code/model artifacts, Stage1 locks and raw CSV/JSON. Create a new version, never overwrite an earlier approved revision.

The report must use natural academic Vietnamese; the article, if touched in its **optional** stage, stays English. The unfinished **Adaptive Threshold feature is out of scope**: do not inspect, implement, describe or imply results from it. Do not fabricate numeric values, images, citations, GPU measurements, model inference, a new seed or a statistical significance claim. Cite exact source, unit, population, and seed. `Pnnnn`/`Tn` are original OOXML locators, not physical pages. Use a localized change log and perform DOCX→PDF visual QA on changed pages where available; if not, mark rendered page count NOT VERIFIED and do not claim conformance.

## Stage-specific tasks
**CONDITIONAL:** Only run when the user provides new raw saved artifacts. **Allowed writes initially:** evidence/review and aggregate tables only, **NOT** training code, raw outputs, Word/article. Do NOT launch training/resume Kaggle or promise future runs.

1. Verify inventory by strategy and training seed 100/101/102. Current confirmed status: CE-only three completed source/LCC/CASIA; Clean+Aux three completed source/LCC, CASIA101/102 MISSING; WBST100 completed, 101/102 interruptions only reported; MAST100 completed,101/102 pending. If states haven't changed, state `INSUFFICIENT_FOR_ALL_FOUR_3SEED` and stop any requested aggregate/claim upgrade.
2. For each new run validate saved `train_seed`, `data_seed=43`, usable source train484561, manifest SHA, pretrained/init state, crop1.5, BGR80, augmentation, optimizer SGD.005/.9/5e-4, batch256 AMP, milestones4/8 gamma.2, warmup and source checkpoint policy, sigma1/18 WBST or0.10 MAST, harm objective. Resume needs optimizer/scheduler/scaler/RNG state; absence means restart or qualify run as incompatible, never mix interrupted output as completed.
3. Match checkpoint SHA/config/completion best epoch to source/evaluation freeze. Evaluate LCC exactly same scored path IDs, pooled image AUC not split average. CASIA video exactly same reconstructed dedup, min4 frames and 358 comparable valid video IDs, scores=mean valid-frame logits; fixed source threshold per run. Reject frame/video mismatches and fit-on-target rate leaks.
4. From raw predictions recompute AUC/EER/APCER/BPCER/ACER and compare to JSON. Per-seed report comes before aggregates. When all three COMPLETE comparable seeds exist for **a given method+population**, report mean ± sample SD ddof=1 and paired seedwise method gaps. Do not infer strong significance from n3; optional video CI is distinct from training variation.
5. Prepare source-backed proposed changes for Word Ch3, article Abstract/TableII/Fig4/Results/Conclusion, and DG20 D21–D23. Report exact old/new values and any ranking reversals without selecting the best seed. Do not edit the documents until the author approves those changes in a separate turn.

**Outputs:** `review/S09_SEED_INVENTORY.md`, `evidence/three_seed_per_run.csv`, `evidence/three_seed_aggregates.csv` only for fully qualified cells, `review/S09_DOCUMENT_DELTA_PROPOSAL.md`, `review/S09_COMPLETION_REPORT.md`.
**Acceptance:** no fake/pending results, reproducibility ledger for every run, raw score recomputation, author approval gate for manuscript edits. STOP.

## Acceptance and stop rule
Write `S09_COMPLETION_REPORT.md` with original hash checks, outputs that actually exist, exact validations, unresolved evidence, scope decisions, and review checklist. Do not initiate subsequent stages. Stop for author approval after S09 CONDITIONAL.
