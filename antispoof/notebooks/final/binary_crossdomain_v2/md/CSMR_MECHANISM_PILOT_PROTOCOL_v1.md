# CSMR mechanism pilot protocol v1

Status: frozen design for the six-run, PAD-pretrained Mini Train10K/Val3K mechanism diagnostic. Date: 2026-10-02. This file is an **addendum** to `PAD_DATA_AND_EVALUATION_CONTRACT_v2_CROSSDOMAIN_BINARY.md` (SHA256 `2f1f6eb6351550d491e4fb59e2151b1ccd9b000c0985cd32f12621f9d58c5f91`), which continues to govern data, localization, model input, candidate accounting, metrics, and evaluation isolation. It authorizes only the spectral-training variations below.

## Research question and controls

Why did training-only CSMR fail to add sufficient benefit to the binary, auxiliary, literature-adapted C baseline in the reported mini pilot? Do not select a winner or tune any setting on LCC. P1–P6 all start from official PAD-pretrained MiniFASNetV2 features and newly initialized binary/auxiliary heads with seed 100. They use the exact same ordered 00c Mini Train10K/Val3K manifests, seed 43, raw SCRFD cache, 2.7× crop, 80×80 BGR [0,1], bbox jitter, copied augmentation, SGD 0.005/momentum 0.9/weight decay 5e-4, batch 256, MultiStepLR [6,14] gamma 0.2, at most 20 epochs, earliest stop 8, patience 4, and Mini-Val3K-only minimum-ACER checkpoint/threshold selection. Auxiliary losses are clean-view only: `L_aux=0.1 L_spoof+0.1 L_lighting+L_attr`, with attribute BCEWithLogits only on Real samples and differentiable zero for an all-Attack batch.

All active runs use the same LOW/MID/HIGH smooth Gaussian radial spectral generator (centers [0.15,0.45,0.75], sigma 0.10, DC preserved, phase unchanged, common channel gain, attenuation only), and sample-specific worst-band selection by smallest binary aligned PAD margin. Candidate forwards use `eval()` and `no_grad()` and cannot update BN. Restore training mode for differentiable clean and selected-worst forwards. No FFT, frequency branch, selector, margin, or KL is used at inference.

## Frozen run matrix

Let `L_c=CE(z_clean,y_binary)`, `L_w=CE(z_worst,y_binary)`, `d=z_real-z_attack`, `m=d` for Real and `m=-d` for Attack. Let `L_margin=mean(SmoothL1(m_worst,m_clean.detach())·1[m_worst<m_clean.detach()])`. For P6, `L_KL=KL(softmax(z_clean).detach() || softmax(z_worst))` computed as `F.kl_div(log_softmax(z_worst),softmax(z_clean).detach(),reduction='batchmean')`.

| Run ID | Warm-up | Gain | Active objective | Mechanism |
|---|---:|---|---|---|
| `P1_worst_ce_lambda0` | 1 | [0.65,0.90] | `0.5 L_c + 0.5 L_w + L_aux` | Worst-view classification without margin penalty |
| `P2_csmr_lambda01` | 1 | [0.65,0.90] | `0.5 L_c + 0.5 L_w + 0.1 L_margin + L_aux` | Reduced harmful-margin strength |
| `P3_mild_worst_ce_lambda0` | 1 | [0.65,0.90] | `0.75 L_c + 0.25 L_w + L_aux` | Reduced worst-view CE pressure; compare with P1 |
| `P4_mild_gain_lambda01` | 1 | [0.80,0.95] | `0.5 L_c + 0.5 L_w + 0.1 L_margin + L_aux` | Milder attenuation; differs from P2 only in gain |
| `P5_warmup3_lambda01` | 3 | [0.65,0.90] | `0.5 L_c + 0.5 L_w + 0.1 L_margin + L_aux` | Longer clean warm-up; differs from P2 only in duration |
| `P6_generic_kl_lambda01` | 1 | [0.65,0.90] | `0.5 L_c + 0.5 L_w + 0.1 L_KL + L_aux` | Generic consistency with **no** harmful-margin penalty |

Every warm-up epoch uses `L_c+L_aux`. Warm-up epochs cannot become the final best checkpoint or increment active patience. The first active epoch initializes the eligible best state. Active epochs use minimum Val ACER, tie highest Val AUC, with an explicit no-improvement counter. All six retain the same worst-band selection; P6 changes only the regularizer relative to P2.

## Evaluation and interpretation

Freeze the selected Val checkpoint and minimum-ACER Val threshold before evaluating official CelebA Test once and official LCC once. Test/LCC threshold metrics use that locked source threshold; report AUC, EER, TPR@FPR1%, APCER, BPCER, ACER/HTER, accuracy, candidate_n, scored_n, and coverage. LCC is pilot evidence only. Preserve failed detections in candidate predictions; never use LCC for checkpoint, threshold, lambda, gain, or formulation selection. Interpret C vs P1, P1 vs P2, P1 vs P3, P2 vs P4, P2 vs P5, and P2 vs P6 as the planned mechanism contrasts. C may appear only as an optional read-only metrics reference after matching resource/model/data fingerprints; do not retrain C in this notebook.

## Diagnostics and provenance

Per active epoch retain clean/worst CE, weights, margin and KL terms and coefficients, harmful fraction, mean margins/drop, mean harmful-only drop (zero when none), spectral flip rate (clean correct to worst incorrect), gain bounds/mean, selected LOW/MID/HIGH counts/fractions, auxiliary components, LR, total loss, and Val metrics. Each run owns its own config hash, initialization audit, best/last checkpoints, optimizer/scheduler/AMP/RNG state, warm-up checkpoint/metrics, patience state, frozen evaluation record, predictions, and completion marker. Unknown run IDs or mismatched resume/resource/config fingerprints fail. Batch 256 OOM is fatal; a changed batch size requires a new explicit protocol. No actual outcome is implied by this design.
