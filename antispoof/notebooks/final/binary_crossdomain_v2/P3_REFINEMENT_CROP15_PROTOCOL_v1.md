# Bounded P3 refinement at crop1.5 + SF, v1

## Scope and frozen reference

Use notebook13's P3_SF_crop15 implementation as the primary base. Frozen Train10K/Val3K, data seed43/train seed100, official PAD-pretrained MiniFASNetV2/128-D/binary+11/5/40 heads, source sample order, per-sample/epoch bbox and augmentation streams, gain RNG, auxiliary losses, batch256/AMP/SGD(.005,.9,5e-4), max20/milestones[6,14]/gamma.2, one clean warm-up epoch, early-stop earliest8/patience4, Val minACER/higher-AUC selection remain unchanged. No phase-aware stopping modification. All Train/Val/LCC use crop1.5.

Mandatory runs:

| Run | Geometry | Clean / selected-worst CE |
|---|---|---|
| P3_SF_W35_crop15 | SF | .65 / .35 |
| P3_SF_W50_crop15 | SF | .50 / .50 |

SF centers are [.15,.45,.75]*(1.5/2.7) = [.0833333333,.25,.4166666667]; sigma=.10*(1.5/2.7)=.0555555556. Gain Uniform(.65,.90), shared across bands/channels, preserved phase/DC and no amplification. BN-safe eval/no_grad worst selection and differentiable selected view are preserved. Aux is .1 spoof CE+.1 lighting CE+Real-only attributes BCE, clean-only. Warm-up is clean CE+aux with no spectral weighting.

Notebook13 currently hard-codes .75/.25 in the loss. Notebook15 changes only this active objective to read configured coefficients. Deterministic checks verify reference scalar and backward equivalence at .75/.25, and changed scalar/gradient at W35/W50. Notebook13 itself remains unchanged.

W35/W50 already exist at crop2.7 as notebook06 R3_worst_ce35/R4_worst_ce50. This round tests their interaction with crop1.5 and SF geometry, not a generic new CE-weight optimum. No additional weight, geometry, gain, crop, loss, seed or augmentation search.

## DSF: mandatory source-only preflight before enabling a third run

Preflight reads only frozen Train10K and physical source image headers to obtain dimensions, without detector or target data. Audit the exact per-sample training jitter at epoch2 (first spectral-active epoch): bbox_jitter(Random(BBOX_SEED+epoch*100000+sample_index)). Also record unjittered geometry for context. Preflight is not training and does not consume the global training/augmentation/gain RNG streams.

The original crop helper is instrumented to return its existing continuous `used=min((H-1)/bbox_height,(W-1)/bbox_width,1.5)`. This is s_eff relative to the actual bbox supplied after jitter, before integer rounding. Pixel-size ratios must not replace it. Its original resize result and geometry remain unchanged. Metadata exposes requested boundary interaction, actual inward window translation, and true scale reduction separately. Translation can occur while s_eff remains exactly1.5.

Use header-derived shape-only arrays and the same helper's geometry-only return; no second crop geometry implementation. Save per-sample CSV with dimensions, original/jittered box, effective scales, translation and reduction flags, plus JSON count/mean/std/min/p05/median/p95/max, fraction |s_eff-1.5|>.01, fraction below1.49/below1.45, boundary/translation/reduction fractions. Both source populations have count10000. The epoch2 distribution controls the decision.

Prespecified source-only gate: enable DSF iff at least1% of audited training samples deviate by more than.01 and effective-scale std>.005. These thresholds define geometric non-degeneracy, not a tuned performance threshold. They are frozen in config/report. Translation alone never enables DSF. This audit covers epoch2; later jitter draws can differ, an explicit limitation. If the gate fails, report exactly: “not run — geometry effectively degenerate under current crop helper”. Do not substitute a third experiment.

Creation-time local resources do not include source image files/dimensions, so no real audit statistics or DSF-enabled claim is fabricated. Kaggle preflight runs before `P3_DSF_crop15` can be appended to enabled RUNS. Explicit DSF filtering fails clearly if preflight does not enable it. W35/W50 can run independently of DSF training, with the required audit saved in the overall artifacts.

If enabled, DSF uses ratio_i=s_eff_i/2.7, centers_i=[.15,.45,.75]*ratio_i and sigma_i=.10*ratio_i; clean/worst=.75/.25. Only training preprocessing geometry is used. Inference remains ordinary crop1.5. A separate dynamic-mask generator/selection path handles B×3×80×80 masks; the static W35/W50 generator/selection helper remains untouched. Check s_eff=1.5 geometry and all-1.5 batch equivalence to static SF, batch/band alignment, and no static-run entry into the dynamic branch.

## Evaluation, resume and artifacts

Inside epochs infer Val3K only. Reload best, recompute/freeze Val3K minACER threshold, then infer physical LCC train/dev/eval at crop1.5 and recompute pooled metrics from concatenated scored predictions. Preserve failed candidates for coverage reporting. Raw counts8299/2948/7580/18827. No full CelebA Test, target fitting, checkpoint/early-stop selection from LCC, detector, GT fallback, harmful gate, R7 or severity.

Keep original crop/CE/band/margin/drop/flip/gain diagnostics. Record DSF effective-scale diagnostics descriptively if enabled. Per-run config/init audit/best/last/history/diagnostics/freeze/metrics/predictions and independent resume retain run/config/geometry/init checks and committed best recovery. Completed outputs are exported immediately; per-run ZIP and overall ZIP include preflight and protocol, with FileLinks.

Output root: /kaggle/working/micro_P3_refine_crop15_v1/. Root artifacts include dsf_preflight_samples.csv, dsf_preflight_summary.json, implementation_sanity_checks.json, p3_refinement_comparison.csv/json, p3_refinement_report.md and this protocol. RUN_FILTER=None runs mandatory W35/W50 and DSF only if source preflight enables it; explicit subsets are supported. No historical control is retrained.

## References, interpretation and bounded recommendation

Optional C_crop15 and P3_SF_crop15 CSV/JSON summaries supply measured deltas across source/LCC endpoints. User-provided historical pooled AUC values .830642 and .837525 remain labeled as provided references when no full artifact is attached; missing metrics remain unavailable. Descriptive screening target >=.840642 is not a stop/search rule. Optional notebook06 summaries provide only a contextual crop2.7 W35/W50 table; they never change run definitions.

Discuss the CE-weight×crop1.5×SF interaction separately from generic crop2.7 weight behavior, and source guard/target AUC versus calibration-only HTER changes. At most one completed new variant can be suggested as a candidate for full-scale validation based on the prespecified descriptive AUC comparison; no automatic new run or final winner. LCC does not influence training/checkpoint/calibration. Only if DSF and W35 independently show useful pooled-AUC evidence versus SF, with non-worsened source ACER available, the report may flag a future P3_DSF_W35 as a conditional option; it is not implemented or executed.

Limitations: Train10K/Val3K screening, one seed, repeatedly used LCC external-development benchmark, no significance claim, multi-seed robustness deferred by compute constraints, configured/realized geometry approximation and epoch2 preflight scope. No expensive local training. Validate parse/compile/cleared outputs, exact static controls/weights/warmup, scalar+backward equivalence, conditional DSF gate/masks, source-only selection/calibration and independent resume/export using tiny offline checks.
