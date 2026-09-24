---
name: results-analysis
description: Aggregate and analyze experimental results for this research repository using reproducible computations rather than manual log reading. Use for multi-seed summaries, paired comparisons, mean/std or confidence intervals, tables, figures, ablations, and reviewer-facing quantitative analysis.
---

# Results Analysis

Compute first; interpret second.

## Workflow

1. Identify the experiment specification and compatible run set.
2. Verify provenance when traceability is uncertain.
3. Prefer CSV, JSON, parquet, or generated summaries over long raw logs.
4. Aggregate with scripts/data tooling rather than mental arithmetic.
5. Preserve per-seed values alongside aggregates.
6. Use paired analysis when methods share evaluation seeds and pairing is valid.
7. Report sample count and statistic used.
8. Generate tables/figures from the same processed source where practical.
9. Separate measured values, derived statistics, and interpretation.
10. Flag outliers, missing runs, protocol mismatches, or suspicious inconsistencies.

## Guardrails

- Do not cherry-pick seeds or scenarios.
- Do not combine incompatible experiments.
- Do not round intermediate values before aggregation.
- Do not infer statistical significance from appearance alone.
- Keep raw data immutable.
