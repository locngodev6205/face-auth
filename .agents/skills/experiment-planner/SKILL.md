---
name: experiment-planner
description: Design fair, reproducible experiments and ablations for this research repository before code or expensive runs are started. Use for reviewer-requested experiments, baseline comparisons, ablations, sensitivity studies, seed plans, evaluation protocols, or any change whose scientific validity depends on controlled experimental design.
---

# Experiment Planner

Design the comparison before modifying code.

## Workflow

1. State the exact research question.
2. Define the hypothesis without assuming the outcome.
3. Identify treatment variables.
4. Identify control variables.
5. Select methods/baselines and justify comparability.
6. Define training budget, training seeds, evaluation seeds, scenarios, and evaluation policy.
7. Define metrics and aggregation before running.
8. Define success as an answerable question, not a desired numeric result.
9. Estimate computational cost and use staged validation when appropriate.
10. Produce an experiment specification before implementation.

## Fairness Checks

Verify:
- identical environment/scenario conditions where required;
- same evaluation seeds for paired comparisons;
- controlled training budget unless budget is the treatment;
- no hidden hyperparameter advantage;
- identical metric computation;
- only intended treatment variables differ.

## Required Output

Produce:
- question and hypothesis;
- compared methods;
- treatment variables;
- controlled variables;
- training configuration;
- evaluation configuration;
- metrics/statistics;
- expected artifacts;
- validation stages;
- known limitations.
