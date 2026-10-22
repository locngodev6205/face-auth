# MAST-PAD — KHÓA SỐ LIỆU VÀ QUY ƯỚC CHO MỌI STAGE

> Version: 2026-10-22; **read-only reference**. Nguồn: `03_METRIC_AND_DATA_AUDIT.md`, `02_CODE_AND_METHOD_EVIDENCE.md`, `04_COMPLEXITY_AND_SYSTEM_EVIDENCE.md`, `06_OPEN_ITEMS_AND_NEXT_WEEK.md` cùng `metrics_lock.md`, prediction/JSON đã kiểm. Không biến tập tin tổng hợp này thành bằng chứng gốc. Khi Codex có raw artifacts, kiểm tra thay vì mặc định tin số chép ở đây. **Không chỉnh Stage-1 locks hoặc artifacts gốc.**

## A. Bốn chiến lược và objective

| Method | Objective | Auxiliary | 3-band worst-view | Gate | Gaussian sigma |
|---|---|---|---|---|---|
| CE-only | `L_clean` | Không | Không | Không | — |
| Clean + Aux | `L_clean+L_aux` | Có, clean only | Không | Không | — |
| WBST | `0.75 L_clean+0.25 L_worst+L_aux` sau 1 epoch warmup | Có | Có | Không | `1/18` (0.0555556) |
| MAST-PAD | `0.75 L_clean+0.25 L_harmful+L_aux` sau 1 epoch warmup | Có | Có | Strict detached harmful | `0.10` |

`L_aux=0.1 CE_spoof11+0.1 CE_lighting5+BCE_attr40_real_only`. Trunk embedding=128, PAD head 128→2, spoof head 128→11, lighting 128→5, attribute 128→40. Tất cả objectives so sánh cùng MiniFASNetV2 PAD graph. CE-only có auxiliary-compatible state dict nhưng không dùng auxiliary forward/optimizer. WBST và MAST khác **cả sigma và gate**, không gọi gate-only ablation.

Selector: `d=z_real-z_attack`, `m=d` nếu Real, `m=-d` nếu Attack; `k*=argmin_{LOW,MID,HIGH} m(x_k,y)` (clean không nằm trong argmin), `h_i=1[m_worst<m_clean]` (strict, detached). Selection temporary eval/no_grad; **train forward toàn bộ B selected views**, kể cả `h_i=0`; nonharmful có 0 spectral CE trực tiếp nhưng vẫn trong batch/BN. `H=sum h`; `L_worst=Σ CE_selected/B`; `L_harmful=Σ h_i CE_selected/H` nếu H>0, differentiable-zero nếu H=0; hệ số clean vẫn 0.75 nếu H=0. First spectral epoch warmup `L_clean+L_aux`.

Spectral profile Gaussian smooth và **overlapping**, centers `[1/12,1/4,5/12]`, gain per sample `U(0.65,0.90)` chung 3 band/3 channel, DC preserved trước inverse reconstruction/clamp; no Fourier at inference. Input `80×80 BGR float [0,1]`; SCRFD + crop boundary-aware `1.5×`, không đảm bảo physical box đúng1.5 nếu chạm mép frame.

## B. Source, datasets, assessment units

| Population | Count | Meaning / guardrail |
|---|---|---|
| CelebA-Spoof official train | **484,561 usable / 494,405 candidate** | Source optimization only; loại 9,844 sau preprocessing |
| CelebA official Test-as-Val | **66,379 scored / 67,170 candidate** | Select checkpoint/fit source fixed threshold, **không independent source test**; Train/Test identity overlap 3 IDs: 5028,7332,9735 |
| LCC train | 8,283 / 8,299 | Images, not people |
| LCC dev | 2,947 / 2,948 | Images |
| LCC evaluation | 7,564 / 7,580 | Images |
| LCC Combined | **18,794 / 18,827** | Pooled image scores train+dev+eval; 1,941 real +16,853 attack; **không** mean of split AUC; LCC endpoints đã dùng trong phát triển crop/sigma |
| CASIA reconstructed Combined | **358 scorable / 360 videos**, **2,660 frames** | 90 real video +268 attack video; dedup `(video_id, frame_idx)` ưu tiên test trước detect; ≥4 valid frames / video; video score mean frame `(z_real-z_attack)`; extracted-color copies overlap **20 subject IDs/240 video IDs/1,403 frame IDs**, không official subject-disjoint |

Không coi LCC blind holdout: notebooks 11–14/17 từng đọc target LCC endpoint. Không đánh đồng CASA video AUC với image AUC, nor threshold-frozen with train-fitted rates. `source-frozen` là per-run threshold fit từ CelebA Test-as-Val, **không adaptive threshold**. Không bàn module adaptive threshold đang phát triển.

## C. AUC của cả 4 method, seed100 (AUC %; primary table)

| Method | CelebA Test-as-Val | LCC Combined image | CASIA Combined reconstructed video |
|---|---:|---:|---:|
| CE-only | 99.214398 | 84.398794 | 89.830017 |
| Clean + Aux | 99.073553 | 84.601127 | 92.039801 |
| WBST | 99.224181 | **86.083099** | 92.226368 |
| MAST-PAD | **99.228027** | 86.027208 | **93.519900** |

Report table rounding: 99.21/84.40/89.83; 99.07/84.60/92.04; 99.22/86.08/92.23; 99.23/86.03/93.52. Calculate differences from raw, **percentage points**: MAST−CE `LCC +1.628414; CASIA +3.689884`; MAST−Clean+Aux `+1.426081; +1.480100`; MAST−WBST `−0.055891; +1.293532`. **Do not falsely bold MAST as best LCC Combined.** On LCC evaluation split alone: MAST **84.005952%**, WBST **81.968680%**; not untouched test.

## D. Threshold-based rates at SOURCE-FROZEN threshold (seed100)

All rates %, Real-positive score. Per-row threshold is model-specific; EER is ROC-derived and threshold-independent for fixed scores.

| Method | Target/unit | Threshold | EER | APCER | BPCER | ACER | Attack accepted | Real rejected |
|---|---|---:|---:|---:|---:|---:|---:|---|
| CE-only | LCC image pooled | 4.5966796875 | 24.4623 | 7.3043 | 51.7259 | 29.5151 | 1231/16853 | 1004/1941 |
| CE-only | CASIA reconstructed video | 4.5966796875 | **16.5423** | 0.3731 | 73.3333 | 36.8532 | 1/268 | 66/90 |
| Clean+Aux | LCC image pooled | 5.130859375 | 23.5793 | 8.9064 | 47.1922 | 28.0493 | 1501/16853 | 916/1941 |
| Clean+Aux | CASIA reconstructed video | 5.130859375 | 16.7289 | 0.7463 | 55.5556 | 28.1509 | 2/268 | 50/90 |
| WBST | LCC image pooled | 5.3037109375 | 21.7906 | 7.7256 | 45.8527 | 26.7891 | 1302/16853 | 890/1941 |
| WBST | CASIA reconstructed video | 5.3037109375 | 17.4751 | 1.4925 | 56.6667 | 29.0796 | 4/268 | 51/90 |
| MAST-PAD | LCC image pooled | 4.66259765625 | 22.0559 | **7.9808** | **46.2648** | **27.1228** | 1345/16853 | 898/1941 |
| MAST-PAD | CASIA reconstructed video | 4.66259765625 | **16.9154** | **1.8657** | **51.1111** | **26.4884** | 5/268 | 46/90 |

**Mandatory Word fix:** CE-only CASIA Combined EER `23.05% → 16.54%` (`P0847`). **Word T21** fill MAST CASIA APCER **1.87%**, BPCER **51.11%**, ACER **26.49%**. EER ↑/↓ need not track AUC. Cannot claim calibration changes AUC: previous 3-seed summary mislabeled **CASIA frame AUC 87.4331%** versus **video AUC 89.8300%** as two policies. These differ by aggregation unit, not only threshold.

## E. Verified multi-seed partials (100,101,102; AUC %)

| Method | Seed | Source CelebA Test-as-Val | LCC Combined images | CASIA Combined videos |
|---|---:|---:|---:|---:|
| CE-only | 100 | 99.214398 | 84.398794 | 89.830017 |
| CE-only | 101 | 99.202743 | 83.407865 | 87.508292 |
| CE-only | 102 | 99.171879 | 83.238604 | 91.368159 |
| Clean + Aux | 100 | 99.073553 | 84.601127 | 92.039801 |
| Clean + Aux | 101 | 98.998677 | 85.404207 | **NOT_FOUND** |
| Clean + Aux | 102 | 99.172895 | 85.288759 | **NOT_FOUND** |
| WBST | 100 | 99.224181 | 86.083099 | 92.226368 |
| WBST | 101,102 | **REPORTED INTERRUPTED** | **NOT VERIFIED COMPLETE** | **NOT VERIFIED COMPLETE** |
| MAST-PAD | 100 | 99.228027 | 86.027208 | 93.519900 |
| MAST-PAD | 101,102 | **PENDING** | **PENDING** | **PENDING** |

**`n=3; sample SD ddof=1` only for fully complete rows:**

| Method | Population | Mean AUC (%) | Sample SD (percentage points) |
|---|---|---:|---:|
| CE-only | CelebA Test-as-Val | 99.196340 | 0.021971 |
| CE-only | LCC combined images | 83.681754 | 0.626715 |
| CE-only | CASIA combined videos | 89.568823 | 1.943145 |
| Clean + Aux | CelebA Test-as-Val | 99.081708 | 0.087395 |
| Clean + Aux | LCC combined images | 85.098031 | 0.434186 |

Clean+Aux−CE-only LCC means: +1.416277 pp. This cannot prove gate. Sample SD numbers in fractional notation: CE LCC 0.006267, Clean LCC 0.004342, CE CASIA 0.019431. Some old summary uses population SD ddof0: do not mix. No n=3 for spectral or Clean CASIA until raw, completed, compatible runs are independently verified.

## F. Offline uncertainty (CASIA reconstructed video, seed100 checkpoint only)

Audit used **2,000 paired video bootstrap resamples**, `n=358`, 95% percentile CI; pairing of same video IDs and labels all 4 models. These are **conditional video-resampling intervals**, not across-training-seed intervals and not fully independent subject CI because videos within same subject may correlate.

| Method | AUC point (%) | 95% video-bootstrap CI (%) |
|---|---:|---|
| CE-only | 89.83 | 85.58–93.41 |
| Clean + Aux | 92.04 | 88.93–94.84 |
| WBST | 92.23 | 89.19–94.90 |
| MAST-PAD | 93.52 | **90.76–95.90** |

Paired differences MAST−WBST `+1.2935 pp; CI [−0.0847,+2.8193]`; MAST−Clean `+1.4801 pp; CI [+0.2846,+2.8344]`; MAST−CE-only `+3.6899 pp; CI [+1.7499,+6.1430]`. **MAST-WBST CI spans 0; do not claim statistically confirmed gate advantage**. Results can appear with a clear caveat only upon author approval; no automatic manuscript insertion.

Training diagnostic final executed epoch11: `harmful=83.2135%`, `flip=0.3077%`, `LOW/MID/HIGH=42.6452/27.3035/30.0513%`. This is NOT best checkpoint's epoch7 evaluation. About 16.7865% selected views have zero *direct spectral CE* weight; all B forward.

## G. Runtime implementation, not benchmark outcomes

| Contract field | Value from audited current `.env`/code | Scope |
|---|---|---|
| Detector/PAD cadence | 0.2 s / 0.2 s | file config, not observed FPS |
| Recognition cadence | 1.0 s | when eligible, async worker |
| Tracking IoU | 0.3 | .env; Word 0.5 outdated |
| PAD vote window | max 5 | ready after min 3 valid votes |
| Spoof vote criterion | ratio ≥0.6 over **len(window)** | only once min_votes reached |
| Stale timeout | 3 s | current config |
| Inference order | SCRFD → IoU/LK track → PAD temporal verdict → 5-point alignment112 → ArcFace512 → stored centroid match | 6 conceptual stages; asynchronous queue/poll; stale-generation result invalidated |
| Stored runtime fixed PAD threshold | 1.6586914 | CASIA train-frame fitted contract; **different** from research source-frozen threshold4.66259765625; record existence only; no adaptive work |

Head/output real-time assertions not present in artifact: **no validated latency trace** for PAD 2.22ms, whole pipeline67.98ms, 14.7/15.5FPS, detector13.72ms or ArcFace8.30ms on this exact pipeline. Delete measured-speed claims unless attributable raw logs available. Track queue/poll and per-component timer scopes; async `recognition_ms` in loop is submit step, not ArcFace+DB model latency.

## H. Model complexity, valid scope

Deployed PAD-only graph `434,434 parameters`; training multi-head `441,658` (not CE-only trainable count); `40,810,892 MACs`, `81,621,784 FLOPs = 0.081621784 GFLOPs` counting Conv/Linear ×2 only, input 80×80; excludes SCRFD/ArcFace/tracking/normalization/pooling/activation/FFT/training overhead. `434,434×4=1,737,736 bytes ≈1.737736 MB decimal (~1.657 MiB)` raw FP32 weights only, **not measured ONNX export, process RAM, L3 cache or edge-device latency**.

## I. Literature contextual comparison (not same protocol)

| LCC contextual method | AUC (%) | Params (reported) | Citation provenance |
|---|---:|---:|---|
| Graph for Transformer Feature | 83.30 | ~10.84M | ESANN 2023, own Table 1 |
| AENet | 86.80 | 11.22M | `kprokofi/light-weight-face-anti-spoofing` benchmark repo row, **not automatically the ECCV-paper LCC result** |
| MAST-PAD | 86.03 | 0.434M PAD | CelebA→pooled LCC, protocols differ |

| CASIA contextual method | AUC (%) | Protocol | Backbone / reference |
|---|---:|---|---|
| MADDG |84.51| O+M+I → C | MADDG-based; SSDG architecture reference 3.35M **not original MADDG total** |
| AMEL |94.39| O+M+I → C | MADDG-based + experts |
| PatchNet |94.58| O+M+I → C | ResNet-18 |
| GAC-FAS |95.16| O+M+I → C | ResNet-18 |
| SA-FAS |95.37| O+M+I → C | ResNet-18 |
| MAST-PAD |93.52| CelebA → reconstructed C Combined | MiniFASNetV2 PAD0.434M |

No direct SOTA claims. AENet expands to **Auxiliary Information Embedding Network** under CelebA-Spoof (not 'Adversarial Erasing Network'); WBST is in-study internally implemented comparator, **not** a separate named publication. Do not invent ISO certification, DOI, bibliography pagination.

## J. Hard source/evidence boundaries

- `adaptive threshold` is excluded entirely; do not read/develop/discuss that feature. Fixed thresholds above are **existing protocols** only.
- No random gate, same-sigma gate-off, per-band or threshold-of-training-gate experiments have been run as final matched controls; merely **plans**, not proof.
- No official verified new subject-disjoint CASIA benchmark; no physical 3D mask/deepfake target verification; no end-to-end identity FAR/FRR or validated latency.
- Source of truth order: **predictions + per-run saved config/manifest → matched JSON → verified audit tables → old summary/Word wording**. Never silently overwrite locks or cherry-pick positive results.
