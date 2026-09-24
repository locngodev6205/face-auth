---
name: experiment-runner
description: Execute validated research experiments in this repository reproducibly and without silently changing the approved plan. Use when launching training or evaluation runs, generating run commands/configs, collecting outputs, or organizing experimental artifacts after an experiment specification exists.
---

# Experiment Runner

Execute the approved experiment specification exactly.

## Before Running

1. Read `AGENTS.md`.
2. Locate the experiment specification or explicit approved parameters.
3. Inspect the actual entrypoint, CLI arguments, config defaults, checkpoint behavior, and output paths.
4. Confirm inputs and outputs are unambiguous.
5. Run focused smoke validation before an expensive run when practical.

## Execution

- Materialize the exact command and effective configuration.
- Preserve training and evaluation seeds explicitly.
- Preserve scenario and evaluation policy explicitly.
- Record checkpoint and output locations.
- Capture failures without silently changing parameters.
- Do not overwrite reportable artifacts unless explicitly requested.

## Run Classification

Mark runs as:
- exploratory;
- smoke/sanity;
- reportable.

## After Running

Record:
- command;
- effective config;
- seeds;
- code revision if available;
- checkpoint;
- raw output;
- processed output;
- run status.
