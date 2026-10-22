# CODEX PROMPT — DG20 response matrix and independent final QA

## Your role and non-negotiable rules
Work in `face_auth` and execute **S07 only**. Review `00_DATA_LOCK_FOR_CODEX.md`, `01_MASTER_EXECUTION_PLAYBOOK.md`, DG20, Word `(2).docx`, approved latest report revision, article and evidence audit 00–08. Determine actual file paths instead of assuming they exist. Preserve original Word, article/release, prior stage revisions, code/model artifacts, Stage1 locks and raw CSV/JSON. Create a new version, never overwrite an earlier approved revision.

The report must use natural academic Vietnamese; the article, if touched in its **optional** stage, stays English. The unfinished **Adaptive Threshold feature is out of scope**: do not inspect, implement, describe or imply results from it. Do not fabricate numeric values, images, citations, GPU measurements, model inference, a new seed or a statistical significance claim. Cite exact source, unit, population, and seed. `Pnnnn`/`Tn` are original OOXML locators, not physical pages. Use a localized change log and perform DOCX→PDF visual QA on changed pages where available; if not, mark rendered page count NOT VERIFIED and do not claim conformance.

## Stage-specific tasks
**Prerequisite:** author-approved S06. **Allowed writes:** DG20 response docs and a NEW incremented final candidate Word/PDF if tiny approved corrections needed. No new experiment, article or main source changes.

1. Read original DG20 (two pages) and `01_DG20_COVERAGE_MATRIX.md` D01–D46, stage S00 ledger, S01–S06 reports, current `report_candidate.docx` and article. Make **one row for every D01–D46**: exact DG20 concern; prior state; tangible change with new chapter/section/table/page; evidence hash/code/prediction reference; remaining gap; truthful state `DONE_DOCUMENT`, `DONE_ALREADY_IN_ARTICLE`, `OFFLINE_VERIFIED_INCLUDED`, `PENDING_NEW_TRAIN`, `PENDING_NEW_DOMAIN`, `NOT_CLAIMED`, or `NOT_APPLICABLE`. Do not collapse requirements into only broad topics.
2. Write a **Vietnamese group response** organized in DG20 original sequence (paper structure, related work, proposed method, novelty/results, dataset/protocol, formatting/references). Each point starts with what group actually achieved and what was edited, then what requires more evidence. Avoid false defensive phrases or claiming reviewer accepted it.
3. Verify all salient numerical assertions against Data Lock: four seed100 AUC, Δpp, EER CE CASIA16.54, MAST CASIA APCER1.87/BPCER51.11, partial baseline n3 sampleSD, source/target populations, harmful last-epoch, complexity. Cross-check article's claim scope and Word's PAD method. Retain 8-page article scientific core unless separately authorized.
4. Explicitly address major OPEN DG20 issues: same-sigma gate-off, random-H, gate-margin δ, each band, multi-seed spectral, independent subject/session target, device trace/whole recognition. For seeds, state **plan** to complete 3 seeds100/101/102 for four strategies when quota returns, not already done. CE-only n3 complete, Clean source/LCC n3 complete, Clean CASIA101/102 absent, WBST101/102 interrupted reported and MAST101/102 pending.
5. Do not talk about Adaptive Threshold. Existing fixed source thresholds may be named as reporting protocol only.
6. Independently inspect final Word PDF: count chapter pages and overall main, verify visuals/captions/references, check spelling, no placeholders/undefined tables, no 10K/100K/micro results, no unsupported hardware or SOTA and no hidden pseudo-workflow text. Preserve unchanged originals and all preceding revisions.
7. Produce a short, readable 1–2-page faculty-facing DG20 response in addition to detailed 46-row evidence matrix; include a discreet 'not yet completed' section.

**Outputs:** `review/DG20_RESPONSE_DETAILED.md`, `review/DG20_RESPONSE_FACULTY_BRIEF.md`, `review/DG20_46_ITEM_STATUS.csv`, `report_final_candidate.docx/pdf` only if needed, `review/S07_FINAL_QA.md`, `review/S07_COMPLETION_REPORT.md`.
**Acceptance:** D01–D46 addressed, completed vs planned truthful, Word 3 chapters and page caps verified, scientific data intact; STOP without submitting or editing paper.

## Acceptance and stop rule
Write `S07_COMPLETION_REPORT.md` with original hash checks, outputs that actually exist, exact validations, unresolved evidence, scope decisions, and review checklist. Do not initiate subsequent stages. Stop for author approval after S07.
