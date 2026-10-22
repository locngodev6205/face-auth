# MAST-PAD DG20 — Execution Playbook v2

## Start here

This folder replaces the previous high-level plan with detailed, executable Codex stages, a common verified numeric lock, a DG20 point-by-point response matrix, and post-Kaggle experimental work plan. It does **not** contain modified Word/article files or run any training. All Adaptive Threshold work is excluded.

### Main files

1. `01_MASTER_EXECUTION_PLAYBOOK.md` — 3-chapter layout (11/20/16 target pages), old Word paragraph/table locator mapping, ready-made results, figure/table plan, completion criteria and S00–S10 execution stages.
2. `00_DATA_LOCK_FOR_CODEX.md` — source evidence table: four-method AUC, fixed-threshold errors, partial 3-seed means, population counts, offline bootstrap CI, inference contract and profiling limits.
3. `02_DG20_RESPONSE_MATRIX_DETAILED.md` — faculty response proposal and exact audit matrix D01–D46 with follow-up mapping.
4. `03_EXPERIMENT_EXECUTION_PLAN.md` — Kaggle task matrix, 3-seed completion, matched gate ablation, YAML contract templates, resource-dependent decisions; planning only.
5. `prompts/S00_...md` through `prompts/S10_...md` — standalone Codex prompts; S00–S07 for the Word revision workflow, S08 optional article review, S09 post-Kaggle update, S10 optional experimental design only.

### How to use with Codex

1. Place this complete directory in the repository, for example `paper/prompts/dg20_v2/`. Keep its internal relative file layout.
2. Send one stage at a time. For Stage 0:

```text
Read paper/prompts/dg20_v2/00_DATA_LOCK_FOR_CODEX.md and
paper/prompts/dg20_v2/prompts/S00_FREEZE_INVENTORY.md.
Execute Stage S00 only. Locate the original Word, DG20, article, codebase
and audit evidence. Do not edit the original files, do not run experiments,
and STOP for author approval after producing the specified deliverables.
```

3. Review each stage's PDF/Word/report, approve before launching the next prompt. For S01, substitute `S01_CHAPTER1_RELATED_WORK.md` and repeat. Never launch all prompts at once.
4. S08 is conditional on explicit permission to edit the article; S09 requires new complete raw seed outputs; S10 produces a preregistration plan only.

### Scientific lock reminders

- Source training: 484,561 CelebA-Spoof usable images.
- Seed100 headline: CE-only LCC84.40/CASIA89.83; Clean+Aux84.60/92.04; WBST86.08/92.23; MAST86.03/93.52 AUC (%).
- There is no isolated harmful-gate experiment: WBST sigma1/18 vs MAST0.10.
- Existing seed101/102 outputs are partial. No spectral 3-seed mean or Clean+Aux 3-seed CASIA aggregate yet.
- Current Word still requires CE-only CASIA EER23.05→16.54% correction.
- The institute page limit is verified only after rendering a revised Word to PDF. Targets are Ch1 11≤15, Ch2 20≤25, Ch3 16≤20, main 47≤50.

### Snapshot scope

Created 2026-10-22 from the user-provided files and evidence audit reports. No external research or hidden experiments were used to produce these planning documents. Source/target claims, raw CSV relationships and current code evidence remain those of the supplied audit; where the audit did not establish a result, the plan labels it outstanding.
