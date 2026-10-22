# CODEX PROMPT — Complete Chapter 3 evaluation and scientific discussion

## Your role and non-negotiable rules
Work in `face_auth` and execute **S05 only**. Review `00_DATA_LOCK_FOR_CODEX.md`, `01_MASTER_EXECUTION_PLAYBOOK.md`, DG20, Word `(2).docx`, approved latest report revision, article and evidence audit 00–08. Determine actual file paths instead of assuming they exist. Preserve original Word, article/release, prior stage revisions, code/model artifacts, Stage1 locks and raw CSV/JSON. Create a new version, never overwrite an earlier approved revision.

The report must use natural academic Vietnamese; the article, if touched in its **optional** stage, stays English. The unfinished **Adaptive Threshold feature is out of scope**: do not inspect, implement, describe or imply results from it. Do not fabricate numeric values, images, citations, GPU measurements, model inference, a new seed or a statistical significance claim. Cite exact source, unit, population, and seed. `Pnnnn`/`Tn` are original OOXML locators, not physical pages. Use a localized change log and perform DOCX→PDF visual QA on changed pages where available; if not, mark rendered page count NOT VERIFIED and do not claim conformance.

## Stage-specific tasks
**Prerequisite:** S04 evidence and S03 report approved. **Allowed writes:** new `report_revision/report_s5.docx`, its PDF, Ch3 content/tables/figures and related review files. Leave Ch1/Ch2 method unchanged except reference numbering generated automatically.

1. Write full Chapter 3 with target **16 pages, max20**: §3.1 datasets/protocol4pp, §3.2 objectives/optimizer2pp, §3.3 results4pp, §3.4 operating point/diagnostics/complexity3pp, §3.5 discussion/limits2pp, §3.6 conclusion1pp. Report total design target11+20+16=47 pages ≤50, not just per-chapter maxima.
2. Construct T3.1: CelebA train **484561/494405**, source Test-as-Val **66379/67170**, LCC combined **18794/18827 pooled images** with splits8283/2947/7564; CASIA reconstructed Combined **358/360 videos,2660frames,90 real+268 attack**, minimum4 valid frames, dedup test-copy priority, copied overlaps20subjects/240videos/1403frame IDs. Explain source Test-as-Val not independent; LCC was seen during development; no target gradient adaptation. No official subject-disjoint CASIA claim.
3. Construct MAIN T3.2 seed100 AUC (%) **exact report values**:
   CE-only Source99.21, LCC84.40, CASIA89.83;
   Clean+Aux99.07,84.60,92.04;
   WBST99.22,**86.08**,92.23;
   MAST-PAD**99.23**,86.03,**93.52**.
   Explain ΔMAST−CE LCC+1.63pp/CASIA+3.69pp; ΔMAST−Clean+Aux +1.43/+1.48pp; ΔMAST−WBST LCC−0.0559/CASIA+1.2935pp. No false claim that MAST leads on pooled LCC. Plot 4 methods with AUC % labels and correct target evaluation units.
4. Construct T3.3 source-frozen error rates, source threshold **per strategy**: MAST LCC EER22.06/APCER7.98/BPCER46.26/ACER27.12; MAST CASIA EER16.92/APCER1.87/BPCER51.11/ACER26.49; CE-only CASIA EER **16.54 not23.05**. If listing all four, use Data Lock §D and source-verified CSV. AUC/EER don't change when only operating threshold changes under identical scores/unit; APCER/BPCER do. Discuss cost of genuine rejections neutrally, no unfounded 'calibration fixes EER' language.
5. Present separate **PARTIAL** n3 baseline table with `sample SD ddof=1`: CE-only source99.1963±0.0220, LCC83.6818±0.6267, CASIA video89.5688±1.9431; Clean+Aux source99.0817±0.0874, LCC85.0980±0.4342. Clean CASIA101/102 missing, WBST/MAST101/102 not complete. The four-model headline is common seed100, not these incomparable means. Avoid claiming gate stable on 3 seeds.
6. Diagnostics: last executed epoch11 harmful83.2135%, flip0.3077%, LOW42.6452/MID27.3035/HIGH30.0513%. State these describe training, best source-selected checkpoint is epoch7. Nonharmful have zero direct spectral CE, all B selected views forwarded. No causal inference from rates.
7. Complexity: PAD-only 434434 parameters,40810892 MACs,0.081621784GFLOPs Conv/Linear; raw FP32=1737736 bytes ~1.74MB **not actual ONNX file**. No unverified PAD2.22ms/whole pipeline67.98ms/FPS/hardware cache claims. Training FFT/selector costs occur before deployment.
8. Related-work context may use LCC Graph83.30/AENet86.80 and CASIA MADDG84.51/AMEL94.39/PatchNet94.58/GAC95.16/SA95.37 ONLY with captions making source→target and protocol differences explicit; no ranking over methods with mismatched evaluation. Do not duplicate the entire Chapter1 pros/cons table.
9. Use optional S04 bootstrap/attack breakdown only if the author has approved adding; if so report paired n358 video CI with no subject-level claims and that ΔMAST−WBST interval crosses zero. Otherwise leave for response appendix as optional evidence. Close with one concise scope paragraph, not defensive repetitive caveats.
10. Delete all historical micro/10K/100K result tables, plots and appendix results. Preserve artifacts in repo. Update references to removed figures/tables; don't leave dangling placeholders.

**Outputs:** `report_s5.docx`, `report_s5.pdf` if possible, `review/S05_NUMERICAL_AUDIT.md`, `review/S05_RESULTS_WORDING_REVIEW.md`, `review/S05_COMPLETION_REPORT.md`.
**Acceptance:** every table correct and captioned; no cross-unit or source/threshold confusion; Ch3≤20 after rendering; no change to approved Ch1/Ch2 or article.

## Acceptance and stop rule
Write `S05_COMPLETION_REPORT.md` with original hash checks, outputs that actually exist, exact validations, unresolved evidence, scope decisions, and review checklist. Do not initiate subsequent stages. Stop for author approval after S05.
