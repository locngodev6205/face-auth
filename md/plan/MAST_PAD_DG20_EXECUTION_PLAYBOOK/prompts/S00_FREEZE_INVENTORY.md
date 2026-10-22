# CODEX PROMPT — Freeze inputs, indexes and acceptance ledger

## Your role and non-negotiable rules
You are an academic research assistant, computer-vision engineer, and careful Word/LaTeX editor working in the existing `face_auth` repository. Execute **S00 only**. Read `00_DATA_LOCK_FOR_CODEX.md`, `01_MASTER_EXECUTION_PLAYBOOK.md`, original DG20, latest Word `(2).docx`, paper, and relevant `paper/review/dg20_evidence_audit/00–08` reports. Resolve actual locations from the repo, do not invent paths. Source of truth hierarchy: matching raw prediction + saved run configuration/manifest > per-run JSON > verified audit > old Word or summary. Every numerical value must have population, unit, source, and seed.

**Language:** revised academic Word and DG20 response in polished Vietnamese; any article text stays academic English. **Preserve:** original Word, previous versions, article/release, Stage-1 locks, code, artifacts, hashes, source/reference PDFs. Create incremented output versions, never overwrite originals. **Excluded:** the unfinished Adaptive Threshold module, its code, design, results and claims; do not discuss or edit it. No new inference or training unless this stage explicitly says offline reanalysis from already stored predictions. No fake results, missing metadata, independent-holdout or causal-gate claims. Do not say you performed tasks you did not perform.

Use a precise change log mapping old OOXML `Pnnnn` / `Tn` locator (not physical page) to new section, action, provenance, and verification. Reuse approved vector figures where possible; do not fabricate experimental graphics. If editing DOCX, use a reversible method, rebuild heading/caption fields where applicable, render a PDF and inspect impacted pages when a converter is available. If not, explicitly mark pagination and visual layout **UNVERIFIED**; don't invent page counts. Restrict edits to the paths allowed below. Stop at the stage gate.

## Stage-specific instructions
**Allowed writes:** only `report_revision/evidence/` and `report_revision/review/`; no Word or article modifications.

1. Locate exact Word `(2)`, DG20, article `(1)`, 20-Oct MD, audit reports 00–08 and raw results. Record absolute/repo-relative paths, file hashes, file type and whether currently accessible. Treat the local reference locations in `07_EVIDENCE_INDEX.md` as historical evidence, not guaranteed current paths.
2. Make protected copies or immutable hash inventories of the original Word/PDF/LaTeX and the nine audit reports. Never duplicate massive training data unnecessarily; hash and link to existing artifacts.
3. Build a 46-row ledger `dg20_acceptance_matrix.md`, one ID D01–D46: reviewer request, current Word/Article status, verified source path/locator, required edit or new evidence, assigned stage S01–S10, approval state (not started).
4. Inspect Word source: extract heading order, old Ch1–Ch5, T15/T17/T18/T21/T22/T23, figure placeholder and bibliography anchors. Build `word_migration_map.md` with target Ch1 §1.1–1.4, Ch2 §2.1–2.6 and Ch3 §3.1–3.6, exact original locations.
5. Independently confirm immediately important data against audit CSV/JSON if available: CE CASIA video EER16.5423%, MAST CASIA APCER1.8657/BPCER51.1111%, four-method seed100 AUC, missing Clean CASIA101/102/spectral101/102. If source CSV unavailable, label copied audit rather than independently verified.
6. Document current DOCX rendering availability. DO NOT modify Word headings yet.

**Deliverables:** `report_revision/evidence/source_inventory.md`, `dg20_acceptance_matrix.md`, `word_migration_map.md`, `numeric_assertions.md`, `report_revision/review/S00_COMPLETION_REPORT.md`.
**Acceptance:** D01–D46 all present, no fake links, original hashes recorded, unresolved sources explicitly named. No Word changes.

## Completion requirements
Write a clear `S00_COMPLETION_REPORT.md` including input hashes, changed files, command/validation results, unresolved issues, source evidence anchors, and a stage-specific author-approval checklist. Make sure outputs physically exist; do not present success merely because you wrote a plan. State what was not verified. **STOP immediately after S00, and do not start another stage, training, upload, external publication, or Word/article rewrite outside the allowed scope.**
