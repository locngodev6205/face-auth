# AGENTS.md

## Project
`face_auth` is a computer-vision research/codebase focused on face detection, tracking, face recognition, and face presentation attack detection (PAD). Keep research changes scoped to CV/PAD; do not expand into attendance/check-in logic unless explicitly requested.

## Working Rules
- Inspect the relevant call path before editing. Prefer `repo-navigator` for codebase discovery.
- Make the smallest correct change. Prefer `surgical-coder`; avoid broad refactors unless required.
- After code changes, run focused tests with `focused-tester`.
- Do not silently change model contracts, preprocessing, thresholds, dataset splits, metrics, or experiment protocols.
- Preserve reproducibility: fixed seeds/configs/manifests, explicit model/data versions, and logged experiment settings.
- Treat train/val/test separation strictly; never tune preprocessing or thresholds on the official test set.
- For realtime PAD, distinguish raw model inference from detector, crop, tracker, temporal smoothing, and runtime effects.

## Research Workflow
Use the repo-local skills when appropriate:
- `research-audit` — verify claims, implementation, protocol, and evidence.
- `experiment-planner` — define hypotheses, controls, ablations, metrics, and stopping criteria before experiments.
- `experiment-runner` — execute planned experiments without changing protocol mid-run.
- `results-analysis` — analyze results and separate measured evidence from interpretation.
- `reproducibility-guard` — check seeds, manifests, configs, artifacts, and fair comparisons.
- `paper-editor` — edit manuscript wording without overstating results.
- `reviewer-response` — prepare evidence-based reviewer responses.

Use `literature-research` for external research and citations.

## Safety for Experiments
Before expensive training or camera-dependent work, exhaust deterministic/offline checks first. If a conclusion requires a real camera or hardware run, stop and give the user the exact command, test protocol, and outputs to return instead of guessing results.

## Output
For non-trivial tasks, summarize:
1. what was inspected,
2. what changed,
3. tests/evidence,
4. unresolved risks or next experiment.
