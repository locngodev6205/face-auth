---
name: research-audit
description: Perform a final consistency audit across this research repository, tracing manuscript claims and reviewer responses to tables/figures, processed results, raw runs, experiment configs, and implementation. Use before submission, after major result updates, or when checking whether revision evidence is internally consistent.
---

# Research Audit

Audit the full claim-to-code chain.

## Claim to Evidence

For each important claim:
1. locate the manuscript statement;
2. locate supporting table/figure/result;
3. verify the number or statement;
4. trace it to processed and raw evidence;
5. trace the experiment to config and implementation.

## Evidence to Claim

For each reportable result:
1. identify where it is used;
2. verify consistent rounding and labels;
3. verify interpretation does not exceed evidence.

## Reviewer Closure

For each reviewer comment:
1. verify the promised action exists;
2. verify manuscript location and response agree;
3. verify experimental evidence follows protocol.

## Check For

- stale numbers after reruns;
- table/text mismatches;
- inconsistent method names;
- incompatible seed/scenario mixtures;
- undocumented parameter changes;
- missing provenance;
- claims based on exploratory runs;
- conclusions stronger than evidence.

## Output

Report:
- blocking inconsistency;
- needs correction;
- needs clarification;
- verified.
