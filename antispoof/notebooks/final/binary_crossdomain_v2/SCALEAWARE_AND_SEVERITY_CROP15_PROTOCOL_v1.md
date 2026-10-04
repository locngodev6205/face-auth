# Scale-aware bands and severity weighting at crop 1.5, protocol v1

## Questions, hypotheses and isolated treatments

At fixed matched train/evaluation crop 1.5, test whether the historical fixed FFT geometry is misaligned with face-relative structure and whether equally weighting all harmful samples limits R7. Prior reported crop results motivate these questions; they do not prove either mechanism.

The approximate face-scale ratio is 1.5/2.7 = 0.5555555556. Scaled centers are original [.15,.45,.75] times this ratio: [.0833333333,.25,.4166666667]. Scaled sigma is .10 times this ratio: .0555555556. This approximation uses configured context scale, not per-image boundary-adjusted scale; frozen crop clipping makes it a hypothesis rather than exact physical normalization.

Exactly five new runs:

| Notebook | Run | Centers | Sigma | Spectral loss |
|---|---|---|---|---|
| 13 | P3_SC_crop15 | scaled | .10 | unconditional selected-worst CE |
| 13 | P3_SF_crop15 | scaled | .0555555556 | unconditional selected-worst CE |
| 14 | R7_SC_crop15 | scaled | .10 | original binary harmful CE |
| 14 | R7_SF_crop15 | scaled | .0555555556 | original binary harmful CE |
| 14 | R7_SEV_crop15 | original [.15,.45,.75] | .10 | normalized raw-severity CE |

No fixed-geometry P3/R7/C control is retrained. No SC+SEV/SF+SEV combination, crop/gain/center/sigma/CE-weight search, KL, SmoothL1, margin regression, random/all-band supervision or new run is added.

## Frozen controls and training

Reuse notebook 12's matched crop-1.5 micro implementation and notebook 06/07 mechanisms. Read existing mini_train10k.csv and mini_val3k.csv with split seed43/train seed100; no resampling. Preserve binary-v2 MiniFASNetV2/80x80/BGR/[0,1], 2/11/5/40 heads and deterministic official PAD-pretrained initialization. Record identical initialization SHA. Preserve sample/epoch bbox and augmentation seeds plus the independent gain generator.

Use frozen valid raw SCRFD boxes, never detector or GT fallback. Jitter p=.20, scale .95–1.05, shift ±.05 face size; then original crop helper at scale1.5 with its existing boundary scale limit/inward translation. Keep HorizontalFlip .5, ISONoise .2, brightness/contrast limits .2 with p=.3 and MotionBlur limit5 with p=.2. No alignment/normalization/mixup/cutmix/smoothing/weighted sampling. Train, Val and every LCC split use crop1.5.

SGD lr .005/momentum .9/weight decay5e-4, batch256, at most20 epochs, milestones[6,14]/gamma.2 stepped after each epoch. Keep earliest early-stop epoch8/patience4, source-Val3K minimum ACER and higher AUC tie-break; no full-scale phase-aware change. One clean warm-up epoch is excluded from spectral best selection/patience. Only Val3K is evaluated inside epochs.

Auxiliary loss is .1 spoof-type CE + .1 lighting CE + Real-only attribute BCE; all auxiliary supervision is clean only. Warm-up loss is clean CE+aux. Active loss is .75 clean CE+.25 the run's spectral CE+aux.

## Selection and severity definition

Reuse the tested spectral generator: one gain per sample Uniform(.65,.90), shared over LOW/MID/HIGH and channels; preserve phase/DC, no amplification. Only radial centers/sigma differ in SC/SF. Build each run's masks from its own config, avoiding global geometry mutation.

Worst-band selection temporarily uses eval/no_grad, compares label-aligned margins (z_real-z_attack for Real, negative for Attack), picks the lowest spectral margin, then restores the previous mode. Selection does not update BatchNorm. The tested micro helper also gathers clean margin; original P3 optimization does not use a harmful gate. Perform differentiable clean and selected-worst forwards only for the loss.

Original R7 SC/SF retain detached harmful=(m_worst<m_clean), normalized CE over harmful samples and differentiable zero when none.

For SEV only:

```
w_i = max(0, m_clean_i - m_worst_i).detach()
eps = 1e-8
CE_severity = sum(w_i * CE_worst_i) / (sum(w_i) + eps), if sum(w_i) > eps
CE_severity = differentiable zero, otherwise
L_SEV = .75 CE_clean + .25 CE_severity + L_aux
```

No gradients flow through selection margins, positive delta or weights. No severity clipping/square/exponent/temperature or tuning; positivity is the specified max(0,delta). SEV retains original centers and sigma to isolate weighting.

## Evaluation and diagnostics

Reload best, infer Val3K at crop1.5, derive/freeze its source minimum-ACER threshold, then evaluate physical LCC training/development/evaluation and combined, crop1.5. No CelebA full Test is loaded or inferred; no target fitting. Raw populations8299/2948/7580/18827; pooled metrics are recomputed from concatenated predictions including unscored failures, never averaged. Report AUC/EER/TPR@FPR1%, locked-source APCER/BPCER/ACER/HTER/accuracy, threshold, candidate/scored counts and coverage. Primary endpoint is pooled AUC; source ACER/AUC act as guard.

Per-epoch diagnostics retain clean/worst or harmful CE, total loss, worst LOW/MID/HIGH fractions, selection clean/worst margins, margin drop, flips, gains and harmful count/fraction. SEV additionally records actual epoch-wide mean/median/max positive delta (zero if none), sum of raw weights, positive fraction, effective weighted count (sum(w)^2/sum(w^2)), harmful unweighted mean CE and weighted CE. The latter is the sample-weighted mean of actual per-batch normalized loss, not a new epoch-global training objective. Geometry diagnostics remain descriptive.

## Independence, resume and artifacts

Notebook13 output root: /kaggle/working/micro_P3_scaleaware_crop15_v1/. Notebook14 output root: /kaggle/working/micro_R7_scaleaware_severity_crop15_v1/. Each uses runs/<run_id>/ with config, best, last_state, history, spectral/crop diagnostics, freeze, source/LCC metrics and predictions. No shared writable dependency.

RUN_FILTER=None selects exactly the runs in that notebook; any non-empty unique subset is allowed, unknown/cross-notebook IDs fail. Export root CSV/JSON/report and full root ZIP after every completed selected run, also a standalone run ZIP, so other unfinished variants do not block outputs.

Resume restores model/optimizer/scheduler/scaler/epoch, committed best payload, ACER/AUC/patience, histories/geometry and RNG state. One restore_rng definition only; run/config/geometry/crop/init checks prevent cross-variant resume.

Optional HISTORICAL_C_CROP15_SUMMARY / HISTORICAL_P3_CROP15_SUMMARY / HISTORICAL_R7_CROP15_SUMMARY are read-only CSV/JSON summary paths loaded after evaluation. Reports include relevant historical C and own-method crop15 controls and measured deltas; missing references remain pending. Training never depends on them; no old model is trained or inferred.

## Interpretation and validation

Answer separately: SC versus historical fixed (possible center misalignment), SF versus SC (possible bandwidth role), and fixed-geometry SEV versus historical binary R7 (possible suboptimal equal harmful weighting). SC beating SF may indicate that narrowing removes useful perturbation coverage. HTER-only SEV improvement may primarily reflect calibration; stronger AUC supports separability. Single-seed micro results do not establish causal proof, significance or a final method winner. Do not create a combined geometry+severity experiment now; a future single combination needs independent useful evidence from both questions.

Before expensive Kaggle training, perform only parse/compile/clear-output checks and tiny offline checks for exact run/filter/geometry definitions, reused preprocessing/model/streams, matched crop, P3/original R7 formulas, detached normalized SEV/zero case, BN safety, source-only calibration and pooled LCC. Static/offline checks do not prove Kaggle CUDA performance or measured research results.
