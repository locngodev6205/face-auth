# CODEX PROMPT — Minimal author-approved IEEE article delta

## Your role and non-negotiable rules
Work in `face_auth` and execute **S08 OPTIONAL only**. Review `00_DATA_LOCK_FOR_CODEX.md`, `01_MASTER_EXECUTION_PLAYBOOK.md`, DG20, Word `(2).docx`, approved latest report revision, article and evidence audit 00–08. Determine actual file paths instead of assuming they exist. Preserve original Word, article/release, prior stage revisions, code/model artifacts, Stage1 locks and raw CSV/JSON. Create a new version, never overwrite an earlier approved revision.

The report must use natural academic Vietnamese; the article, if touched in its **optional** stage, stays English. The unfinished **Adaptive Threshold feature is out of scope**: do not inspect, implement, describe or imply results from it. Do not fabricate numeric values, images, citations, GPU measurements, model inference, a new seed or a statistical significance claim. Cite exact source, unit, population, and seed. `Pnnnn`/`Tn` are original OOXML locators, not physical pages. Use a localized change log and perform DOCX→PDF visual QA on changed pages where available; if not, mark rendered page count NOT VERIFIED and do not claim conformance.

## Stage-specific tasks
**CONDITIONAL:** Do not edit the article unless the author has explicitly asked to make an article revision and has supplied the approved S07 Word result. Otherwise perform a no-edit **proposal only** and STOP. **Allowed writes if approved:** COPY of LaTeX source and newly named PDF/ZIP, not original release.

1. Treat the already reviewed **8-page IEEE PAD-only article** as baseline. Compare claims/tables to the final Word and Data Lock. The article's method Eq1–13, all 12 seed100 AUC, figures, source/crop/spectral definition, LCC Combined pooled and CASIA reconstructed protocol, sigma+gate qualification, 434434 params are verified. Do not rewrite into whole 6-stage system paper.
2. Propose only genuinely needed textual changes: if source exposure nuance absent, add one sentence that LCC endpoints were inspected during development; if literature benchmark provenance still ambiguous, refine citation/note. Keep TableII 4 methods; maintain CASIA reconstructed video scope and no causal gate claim. NO new adaptive-threshold, CI/bootstrap, training seeds or latency inserted without explicit author approval and verified new sources.
3. If edits authorized, copy `paper/` LaTeX release to `paper/article_revision_DG20_vnext/`; preserve author block and IEEEtran formatting. Apply minimal surgical diffs, update BibTeX only with verified metadata, compile PDF, render all pages, check table/figure/equations/citations, page count and no overlap.
4. If adding offline CI has been separately approved, accurately describe n358 reconstructed video paired bootstrap2000, interval crossing zero for ΔMAST−WBST, and subject correlation caveat. Do not call it training-seed variance or independent subject CI.
5. If no change needed, report `NO_ARTICLE_CHANGE_RECOMMENDED` with exact justification and a list of event triggers that would justify a vNext (completed 3 seeds, gate ablation, new independent evaluation).

**Outputs:** `review/S08_ARTICLE_DELTA_PROPOSAL.md`; if authorized, `paper/article_revision_DG20_vnext/main.pdf`, diff, exact build logs and review report.
**Acceptance:** original final PDF and ZIP unchanged, scientific text claims scope correct, same 8pp target or explicit author approval for length change; STOP.

## Acceptance and stop rule
Write `S08_COMPLETION_REPORT.md` with original hash checks, outputs that actually exist, exact validations, unresolved evidence, scope decisions, and review checklist. Do not initiate subsequent stages. Stop for author approval after S08 OPTIONAL.
