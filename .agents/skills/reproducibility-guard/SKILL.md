---
name: reproducibility-guard
description: Audit experimental provenance in this research repository. Use before accepting results as reportable, when comparing runs, when reproducing a number, or when checking that configs, seeds, code revision, checkpoints, and outputs can be traced consistently.
---

# Reproducibility Guard

A reportable result must be traceable from code to evidence.

## Audit Chain

Verify, where applicable:

`code revision -> command -> effective config -> training seed -> checkpoint -> evaluation seeds/scenario -> raw output -> processed result`

## Checks

1. Identify the exact run or result.
2. Confirm implementation/code revision.
3. Confirm effective parameters rather than assuming defaults.
4. Confirm training and evaluation seeds separately.
5. Confirm evaluation mode and scenario.
6. Confirm checkpoint.
7. Confirm raw outputs are preserved.
8. Confirm derived statistics can be regenerated.
9. Detect overwritten, ambiguous, missing, or mixed artifacts.
10. Classify the result as traceable, partially traceable, or not traceable.

## Guardrails

- Never reconstruct missing provenance by guessing.
- Never treat a filename as proof of configuration.
- Never silently merge incompatible protocols.
- Flag deviations from the approved experiment plan.
