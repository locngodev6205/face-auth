# Offline validation — 2026-09-26

Run `python3 antispoof/notebooks/validation/validate_e3_cross_dataset.py` from repo root.

9 checks passed on local Python 3.11, PyTorch 2.11.0, torchvision 0.26.0, ONNX Runtime 1.24.4:

- All three notebooks parse as JSON; every code cell compiles after excluding notebook commands; outputs/execution counts are cleared.
- E2 luminance, DCT, resize, frequency tensor, DepthwiseSeparableBlock and FrequencyBranch match the frozen source AST exactly.
- Shared-crop tensors have RGB `[3,224,224]` and frequency `[1,224,224]`; normalized DCT is finite; E3 forward `[B,3]`; mismatched input batch sizes fail.
- Random-init E3 has 1,125,187 parameters, including the exact 8,480-parameter frequency branch. ONNX export/checker and batch 1/2 parity pass; last run max absolute logit difference `7.450580596923828e-09`.
- Restored preliminary E1 NPZ split hashes and selected-cache fingerprint match actual frozen run config. Mutated cache produces a different hash; duplicate keys and subject overlap guards fail as intended.
- PAD score sign, Validation calibration, source threshold application, undefined metrics and all five native face-size boundaries pass.
- Zero/one/multiple valid-frame videos retain the right status, score, coverage and deterministic mean aggregation.
- Synthetic image adapter preserves image evaluation without video IDs; provenance rejection, invalid labels and escaping paths fail.
- Actual local frozen E1 ONNX/runtime/best-meta load succeeds; deliberately conflicting threshold metadata fails.
- Static protocol checks confirm independent initialization, no training Test loader, no held-out/target calibration, shared valid-frame set, external DCT contract and app incompatibility declaration.

Additional review: no stale backbone/gamma/target references in generated notebooks/docs; `git diff --check` clean. E1/E2 artifacts and app/runtime were not changed. Existing unrelated workspace changes remain untouched.

Limits: no Kaggle training or real target inference has run. Local synthetic parity uses random weights, not trained E3; the notebook must repeat parity after training. Local environment lacks Albumentations, so full DataLoader/augmentation execution and end-to-end checkpoint continuation were not exercised here. Notebook setup installs that dependency on Kaggle. Official artifacts, mounted dataset layout/labels/provenance, external SCRFD model and target-device timing need verification at run time. No PAD performance result, camera test, Raspberry Pi benchmark or E4 experiment is claimed.
