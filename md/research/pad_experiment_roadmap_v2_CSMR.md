# PAD Experiment Roadmap v2 — CSMR, Backbone Freeze, Scale and Final Training

_Last updated: 2026-09-29_

> This document replaces the old “M0–M4 search before full training” roadmap.
>
> M0–M4 are now a completed historical diagnostic stage. The active research direction is frozen CSMR-v1 plus a final backbone decision between MiniFASNetV2 and MobileNetV3-Small.

---

# 1. Current state

Completed:

```text
E1 spatial baseline
E2 frequency-only diagnostic
E3 naïve spatial+frequency fusion
M0–M4 explicit frequency representation search
CSMR-v1 seed42 pilot
MNV3 seed100 paired freeze check
fixed Test-2000 descriptor
MiniFASNetV2 Vanilla vs CSMR transfer check
```

Pending:

```text
MNV3-S 2.7× crop / 12-epoch fair check
→ choose final backbone
```

Then:

```text
10–20K scale sanity
→ 100K final
→ CSMR mechanism ablation
→ efficiency
→ CASIA untouched
→ paper
```

---

# 2. Completed finding from M0–M4

Original question:

> Which explicit frequency representation provides robust complementary evidence?

Candidates:

```text
M0 Global DCT + GAP
M1 DCT + Pool4×4
M2 Coord-DCT
M3 Band-aware DCT
M4 Block-DCT
```

Current conclusion:

- explicit frequency signal exists;
- source performance can look promising;
- cross-domain complementarity is not stable enough;
- no M0–M4 candidate is promoted as the final inference branch;
- M0–M4 are retained as motivation / diagnostic evidence.

Do not reopen M0–M4 tuning on LCC.

---

# 3. Active method — frozen CSMR-v1

Frozen:

```text
warmup = 1 epoch
lambda = 0.20
centers = [0.15, 0.45, 0.75]
sigma = 0.10
gain = Uniform(0.65, 0.90)
DC preserved
phase unchanged
same gain all channels
worst = argmin(label-aligned PAD margin)
margin penalty = harmful-only SmoothL1
```

PAD score:

```text
d = real_logit - logsumexp(physical_logit, digital_logit)
```

Aligned margin:

```text
Real   → +d
Attack → -d
```

No more LCC-driven CSMR hyperparameter tuning.

---

# 4. Development evidence summary

## MNV3 seed100

| Metric | Vanilla | CSMR | Delta |
|---|---:|---:|---:|
| Val ACER | 2.1062% | 2.1357% | +0.0295 pp |
| LCC HTER | 34.0274% | 33.1171% | −0.9103 pp |
| LCC AUC | 0.690181 | 0.714238 | +2.4057 pp |

Result:

```text
FREEZE CHECK = PASS
```

## MiniFASNetV2 seed100

| Metric | Vanilla | CSMR | Delta |
|---|---:|---:|---:|
| Val ACER | 1.6909% | 2.1691% | +0.4782 pp |
| LCC HTER | 31.8242% | 28.7548% | −3.0694 pp |
| LCC AUC | 0.819484 | 0.846264 | +2.6779 pp |

Result:

```text
TRANSFER_STRONG
```

MiniFAS currently leads on LCC, but current cross-backbone comparison is confounded by crop, input, pretraining and epoch budget.

---

# 5. Stage A — Final fair backbone check

Run:

```text
MNV3-S Vanilla
MNV3-S + CSMR
```

Config:

```text
seed = 100
Train = frozen 5K
Val = frozen 2K
crop = 2.7×
input = 224×224
max epochs = 12
patience = 3
earliest stop = 6
```

Keep:

- ImageNet initialization;
- same MNV3 head;
- same RGB normalization;
- same augmentation;
- same optimizer/LRs;
- same checkpoint/threshold logic;
- same LCC valid set;
- frozen CSMR-v1.

Only compare after both checkpoints are selected on Val.

Do not use LCC every epoch.

---

# 6. Backbone decision

Use Pareto reasoning, not a fabricated composite score.

Primary:

```text
LCC AUC
LCC HTER
```

Secondary:

```text
Val ACER/AUC
opened Test-2000 descriptor
params
latency
memory
model size
```

## CASE A — MiniFASNetV2 selected

Choose if:

- MiniFAS keeps a clear LCC advantage after MNV3 gets 2.7× context and equal 12-epoch budget;
- no severe source-domain collapse;
- deployment metrics are acceptable.

Freeze:

```text
BACKBONE = MiniFASNetV2
CROP = 2.7×
INPUT = 80×80
INIT = official PAD pretrained
CLASSIFIER = reinitialized 128→3
```

## CASE B — MNV3-S selected

Choose if:

- wider-context MNV3 closes most of LCC gap;
- or MNV3 provides a better overall Pareto trade-off.

Freeze:

```text
BACKBONE = MobileNetV3-Small
CROP = 2.7×
INPUT = 224×224
INIT = ImageNet
HEAD = 576→256→128→3
```

Once selected:

> no more backbone search.

---

# 7. Stage B — Scale sanity, not final training

Default:

```text
Train = 20K
Seed = 100
Vanilla + CSMR
```

Optional speed mode:

```text
Train = 10K
```

Use only if compute/time is constrained.

Purpose:

> Does the paired CSMR advantage survive when Train increases substantially beyond 5K?

Do not use scale sanity to tune CSMR.

## Validation

Prefer reusing a fixed development validation set so only Train scale changes.

If a larger Val is created, create it once before comparing methods and freeze its manifest/hash.

## Budget

### If MiniFAS selected

Suggested:

```text
max epochs = 12
patience = 3
earliest stop = 6
```

### If MNV3 selected

Suggested:

```text
max epochs = 12
patience = 3
earliest stop = 6
```

The 20K trajectory is used to pre-register the **100K optimization budget**, not to alter CSMR.

---

# 8. Stage C — Final data budget

100K is the final main training budget.

Target:

```text
Train = 100,000
Val = 15,000
training seed = 100
```

This preserves continuity with the old E1 v5.3 preliminary scale:

```text
100K Train
15K Val
```

Create:

```text
final_manifest_v1
```

with:

- deterministic sampling;
- stratification by Real / Physical / Digital;
- disjoint Train/Val membership;
- manifest SHA256;
- bbox/cache provenance;
- frozen before training either branch.

---

# 9. Final optimization budget — two cases

## CASE A — MiniFASNetV2

Do not copy MNV3’s historical 24 epochs automatically.

Use 20K trajectory to pre-register final max epochs.

Default if 20K plateaus by 12:

```text
MAX_EPOCHS = 12
PATIENCE = 3
EARLIEST_STOP = 6
```

If 20K is still improving at epoch 12:

- pre-register a larger final max epoch before any 100K result is seen;
- use exactly the same schedule for Vanilla and CSMR.

## CASE B — MNV3-S

Historical E1 v5.3 used:

```text
EPOCHS = 24
EARLY_STOP_PATIENCE = 4
EARLY_STOP_MIN_EPOCHS = 10
```

If MNV3 is selected, this is the default final schedule because it keeps continuity with the 100K E1 recipe.

Both:

```text
Vanilla
CSMR
```

must use the same schedule.

---

# 10. Stage D — CSMR mechanism ablation

This is more important than rerunning all M0–M4.

Use a fixed medium budget:

```text
Train = 20K
same frozen backbone
same Val
seed = 100
```

Minimum:

| ID | Method | Question |
|---|---|---|
| A0 | Vanilla | baseline |
| A1 | Random spectral augmentation | is augmentation alone enough? |
| A2 | Generic KL/logit consistency | is generic consistency enough? |
| A3 | Worst-band CE, lambda=0 | does worst-view selection help? |
| A4 | Full CSMR | full method |

Optional:

```text
A5 random-band + margin
A6 average-all-bands
```

Primary evaluation:

```text
Val ACER/AUC
LCC HTER/AUC
```

Do not select variants using Test-2000 or CASIA.

---

# 11. M0–M4 rerun policy

## CASE A — MiniFASNetV2 final

Do **not** rerun full M0–M4.

Existing M0–M4 remain historical MNV3-based motivation.

If the paper needs a same-backbone explicit-frequency comparison:

run only one representative explicit-frequency baseline:

```text
MiniFAS Vanilla
MiniFAS + representative explicit-frequency method
MiniFAS + CSMR
```

Candidate representative:

- M3 band-aware, because it is conceptually closest to CSMR’s LOW/MID/HIGH intervention; or
- best previous M0–M4 by documented development criterion.

Freeze the choice before running.

## CASE B — MNV3-S final

No M0–M4 rerun needed.

Existing M0–M4 directly belong to the same backbone family and remain sufficient as historical diagnostic evidence.

---

# 12. Opened Test-2000 policy

The fixed CelebA Test-2000 has already been opened.

Rules:

- same 2,000 sample identities only;
- no resampling;
- descriptive benchmark only;
- never use it for:
  - backbone selection;
  - CSMR tuning;
  - epoch budget;
  - threshold calibration;
  - ablation selection.

If evaluated on new models, label:

> **FIXED OPENED TEST-2000 DESCRIPTIVE BENCHMARK**

---

# 13. Stage E — Final 100K paired training

For the selected backbone:

```text
final Vanilla
final Full CSMR
```

Both must share:

- same final 100K/15K manifest;
- same initialization policy;
- same training seed;
- same optimizer family;
- same schedule;
- same augmentation;
- same crop/input contract;
- same checkpoint metric;
- same threshold calibration;
- same LCC candidate set.

Checkpoint:

```text
lowest Val ACER
```

Threshold:

```text
Val only
```

LCC:

```text
evaluate after checkpoint frozen
```

---

# 14. Stage F — Efficiency

Measure final Vanilla and CSMR deployment checkpoints.

Since CSMR is training-only, inference graph should be identical within a backbone.

Report:

```text
Parameters
MACs/FLOPs
model size
peak RAM
CPU latency
Raspberry Pi latency
FPS
```

Optional:

- ONNX;
- ONNX Runtime;
- quantization later.

Do not make quantization part of core novelty unless separately studied.

---

# 15. Stage G — CASIA untouched confirmation

Only after:

```text
backbone frozen
method frozen
100K checkpoints frozen
Val thresholds locked
```

open CASIA once.

Compare:

```text
Final Vanilla
Final CSMR
```

Metrics:

```text
HTER
AUC
APCER
BPCER
coverage
```

Do not tune anything after CASIA.

If CASIA contradicts LCC:

> report the failure / domain dependence honestly.

---

# 16. Run order

```text
0. Archive current MNV3/MiniFAS development artifacts

1. Run MNV3 2.7× / 12ep paired fair check

2. Select CASE A or CASE B
   → freeze final backbone

3. 20K scale sanity
   → pre-register 100K optimization budget

4. 20K CSMR mechanism ablation

5. Build 100K/15K final manifest

6. Train final Vanilla

7. Train final CSMR

8. LCC development report

9. Efficiency benchmark

10. Open CASIA once

11. Final paper tables/figures
```

---

# 17. Stop rules

Do not:

1. change CSMR bands/lambda/gain after backbone freeze;
2. resample Test-2000;
3. use CASIA for design;
4. reopen full M0–M4 search;
5. add gated fusion “for novelty”;
6. add another module unless a required ablation exposes a specific failure;
7. train 100K candidate architectures before the final backbone gate.

---

# 18. Artifact contract

Each paired run should save:

```text
config.json
meta.json
manifest fingerprint
initialization fingerprint
best checkpoint
training_history.csv
val_predictions.csv
lcc_predictions.csv
paired_metrics.csv
report.md
```

CSMR additionally:

```text
training_diagnostics.csv
class_diagnostics.csv
frequency_masks.png
intervention_gallery.png
```

Final 100K:

```text
final_protocol.json
final_manifest_sha256
checkpoint_sha256
efficiency.json
```

---

# 19. Decision tree

```text
MNV3 2.7× fair check
        ↓
MiniFAS still clearly stronger on LCC?
      /   \
    yes    no
    /       \
MiniFAS     MNV3
CASE A      CASE B
    \       /
     backbone frozen
          ↓
       20K sanity
          ↓
    20K ablations
          ↓
      100K final
          ↓
      efficiency
          ↓
     CASIA untouched
          ↓
        paper
```

---

# 20. Operational conclusion

The project is no longer searching for “a better frequency branch.”

The active question is:

> **Which lightweight spatial backbone should carry a frozen CSMR training strategy into the final 100K experiment?**

Once the `MNV3 2.7× / 12ep` check is complete, choose one backbone and stop architecture development.
