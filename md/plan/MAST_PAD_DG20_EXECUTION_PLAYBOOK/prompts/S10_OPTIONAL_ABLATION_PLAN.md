# CODEX PROMPT — Gate-isolation and frequency ablation preregistration (no runs)

## Your role and non-negotiable rules
Work in `face_auth` and execute **S10 OPTIONAL only**. Review `00_DATA_LOCK_FOR_CODEX.md`, `01_MASTER_EXECUTION_PLAYBOOK.md`, DG20, Word `(2).docx`, approved latest report revision, article and evidence audit 00–08. Determine actual file paths instead of assuming they exist. Preserve original Word, article/release, prior stage revisions, code/model artifacts, Stage1 locks and raw CSV/JSON. Create a new version, never overwrite an earlier approved revision.

The report must use natural academic Vietnamese; the article, if touched in its **optional** stage, stays English. The unfinished **Adaptive Threshold feature is out of scope**: do not inspect, implement, describe or imply results from it. Do not fabricate numeric values, images, citations, GPU measurements, model inference, a new seed or a statistical significance claim. Cite exact source, unit, population, and seed. `Pnnnn`/`Tn` are original OOXML locators, not physical pages. Use a localized change log and perform DOCX→PDF visual QA on changed pages where available; if not, mark rendered page count NOT VERIFIED and do not claim conformance.

## Stage-specific tasks
**OPTIONAL planning only, not execution. Allowed writes:** `paper/research_plans/` and review report. Do not change training code, train, upload to Kaggle or use adaptive inference-threshold module.

1. Read notebook18 implementation and propose a **matched gate ablation** specifically correcting the confound between WBST sigma1/18 and MAST sigma.10. Freeze source 484561, CelebA Test-as-Val selection, crop1.5, sigma=.10 for matched comparisons, LOW/MID/HIGH centers1/12,1/4,5/12, shared per-image gain U(.65,.90), same augmentation/initialization/data seed/SGD batch256/warmup and source-only checkpoint rule.
2. Experimental cells required: (A) standard MAST strict detached h=1[m_worst<m_clean]; (B) **same-sigma no-gate** h=1 for all selected worst views, CE mean/B; (C) **random-H gate** randomly choose exactly H of B samples per batch using the same harmful sample count H measured online but a margin-independent subset, CE mean/H. All three MUST train-forward all B selected views so BN semantics are matched. H=0 differentiable-zero case remains for random gate. Use separate RNG for selection not changing data/gain streams.
3. Label primary RQs: A vs B tests gate condition under matched Gaussian width; A vs C tests targeting harmful rather than merely CE sparsification/normalization. Recommend matched seeds100/101/102 if budget permits; single seed is exploratory and report such.
4. Secondary variants only after P0+gate tests: δ-grid `m_clean−m_worst > δ` **as training gate sensitivity** (not adaptive threshold); LOW-only/MID-only/HIGH-only or leave-one-band-out; define how candidate argmin behaves with 1 or2 bands, keep comparability limits.
5. Include costs as **unknown until hardware logs**: actual GPU-hours/run, checkpoint and dataset storage, CPU CASIA inference costs. Do not invent minutes, money or Kaggle quota. Rank priorities P0 complete spectral seeds > P1 same sigma + random H > P2 δ/band experiments > new independent target as scientific next step.
6. Pre-register target endpoints, avoid tuning hyperparameters on LCC exposed pool; if aiming strong publication evidence define a fresh subject/session holdout with source and target manifest, failures/dedup and restricted scope. Log per-epoch harmful fraction/flip/band stats, optimizer state and actual training walltime if running later.

**Outputs:** `paper/research_plans/DG20_GATING_ABLATION_PREREGISTRATION.md`, `DG20_GPU_BUDGET_DECISION_TABLE.md` (no invented durations), `review/S10_COMPLETION_REPORT.md`.
**Acceptance:** sigma held fixed in A/B/C, same all-B forward, random-H matched count, reproducible RNG, plan distinguished from completed experiments. STOP and do not run anything.

## Acceptance and stop rule
Write `S10_COMPLETION_REPORT.md` with original hash checks, outputs that actually exist, exact validations, unresolved evidence, scope decisions, and review checklist. Do not initiate subsequent stages. Stop for author approval after S10 OPTIONAL.
