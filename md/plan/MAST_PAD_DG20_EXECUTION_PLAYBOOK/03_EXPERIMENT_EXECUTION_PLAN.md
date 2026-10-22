# Kế hoạch thí nghiệm bổ sung — bản chi tiết v2

**Kết quả chưa chạy phải để trạng thái PLANNED. Adaptive Threshold ngoài phạm vi.**

# Kế hoạch thí nghiệm bổ sung MAST-PAD sau khi có quota Kaggle

**Ngày soạn:** 22/10/2026 · **Loại:** phương án lựa chọn, CHƯA phê duyệt/chưa chạy · **Nguồn:** DG20, `03_METRIC_AND_DATA_AUDIT.md`, `06_OPEN_ITEMS_AND_NEXT_WEEK.md`, notebook18/21 và artifact statuses do Codex audit. **Không bao gồm Adaptive Threshold.**

## 1. Nguyên tắc tối ưu ngân sách

Ưu tiên hoàn thành **bốn chiến lược × ba seed** trước khi mở thí nghiệm mới. Mỗi experimental contrast phải giải quyết một câu hỏi cụ thể, không trộn các thay đổi `sigma`, gate, preprocessing, scheduler. **Không ước tính GPU-hours tùy ý** vì audit không có log vận hành đầy đủ (thời gian/epoch, loại GPU, quota còn lại). Trước khi submit Kaggle job, lấy thời gian 1 epoch thực của từng run và thời gian còn lại của quota để dự toán; bảo toàn checkpoint/optimizer/RNG state nếu resume.

| Ưu tiên | Hạng mục | Có cần train GPU? | Lý do |
|---|---|---|---|
| **P0** | Hoàn tất 3 seed ×4 methods và matching CASIA evaluation | **Có** với spectral seeds; CASIA Clean101/102 inference | Trực tiếp giải quyết ý kiến repeatability DG20 |
| **P0-offline** | Validation artifact, correct frame/video/unit, mean±SD, bootstrap, attack breakdown | Không | Khắc phục sai số/thiếu kiểm chứng mà không tốn Kaggle |
| **P1** | Same-sigma no-gate control (σ=0,10) | Có | Cô lập gate tốt hơn so WBST σ=1/18 |
| **P1** | Random gate đúng harmful count H (σ=0,10) | Có | Kiểm tra chọn harmful có giá trị so random masking cùng tỷ lệ |
| **P2** | Sensitivity của training-gate margin threshold `δ` | Có | Hiểu độ nhạy điều kiện mở cổng |
| **P2** | LOW/MID/HIGH one-band hoặc leave-one-band-out | Có | Đánh giá đóng góp band |
| **P2 độc lập** | Target data mới được freeze và subject/session/video-disjoint | Tốn data/inference; có thể không cần retrain | Giải quyết DG20 về external validity mạnh hơn |
| **Optional** | Device/end-to-end profiling có log | Không nhất thiết Kaggle | Xác minh claim lightweight thực tế nếu muốn nêu tốc độ |

## 2. P0: ma trận seed và công việc cụ thể

| Method | S100 | S101 | S102 | Cần làm khi quota phục hồi |
|---|---|---|---|---|
| CE-only | Source/LCC/CASIA **verified** | **verified** | **verified** | Không retrain; rà cùng scoring protocol và provenance |
| Clean + Aux | Source/LCC/CASIA verified | Source/LCC verified, **CASIA missing** | Source/LCC verified, **CASIA missing** | Inference matching CASIA từ best checkpoint frozen cho101/102, không fit ngưỡng lên test |
| WBST | all source/LCC/CASIA verified | **Reported interrupted epoch8** | **Reported interrupted epoch7** | Kiểm completion/log; resume **chỉ khi** đủ optimizer/scheduler/AMP/RNG/dataloader state; nếu không resume tương đương, run từ đầu với config đã khóa |
| MAST-PAD | all verified | **Chưa chạy** | **Chưa chạy** | Train source full484561 và freeze checkpoint, sau đó LCC/CASIA evaluation |

**Mục tiêu:** n=3 cho từng phương pháp trên **cùng target unit** (LCC pooled image, CASIA deduplicated reconstructed video). Không gộp frame vs video AUC hoặc source-frozen vs train-calibrated thành các seed khác. Không dùng incomplete WBST checkpoints như final. Không suy notebook đang `TRAIN_SEED=101` chứng minh run101 đã chạy; đọc saved config/completion.

### 2.1. Khóa tính tương thích giữa runs

- 484.561 usable official CelebA Train; official source checkpoint selection Test-as-Val 66.379; data seed43 và manifest/membership nhất quán; train seed100,101,102; pretrained initial state theo design seed pairing được lưu hash.
- Input crop1,5×, SCRFD/preprocessing,80×80 BGR float[0,1], augmentation, SGD lr.005/momentum.9/wd5e-4, batch256 AMP, milestone[4,8], gamma.2, max12, warmup1 epoch spectral và early-stop rule thống nhất.
- Objective frozen theo method: CE-only clean CE; Clean+Aux cleanCE+aux; WBST σ=1/18 all-worst-selected; MAST σ=.1 strict harmful gate; 0.75/0.25 + aux after warmup.
- Target inference chỉ sau source checkpoint freeze; LCC pooled 18.794 scored images, CASIA Combined 358 valid reconstructed videos/2.660 frames theo dedup/video≥4 logic. Kết quả mới phải nối checkpoint SHA/config/hash với predictions.

### 2.2. Trình tự chạy đề nghị

1. **Không tốn GPU:** kiểm inventory và restore state WBST101/102; thử tìm đủ optimizer/scheduler/scaler/RNG/loader state. Nếu không thể khôi phục nghiêm ngặt, chọn restart từ đầu thay vì ghép training khác recipe. Không thử tùy tiện checkpoint giữa chừng rồi báo cáo n=3.
2. **Ưu tiên chạy việc inference còn thiếu**: CASIA Clean+Aux101/102. Khóa checkpoint; compute video scores với source threshold, xuất frame/video metrics riêng. Nếu Kaggle mới có GPU nhưng local CPU inference được thì cân nhắc local để tiết kiệm quota (đo thời gian trước).
3. **Hoàn tất spectral source training**: WBST101/102 hợp lệ, MAST101/102; mỗi run ghi checkpoint best, epochs, completed flag, logs và manifest.
4. **Đánh giá target**: source-frozen LCC split+Combined, CASIA dedup frame+video và video Combined theo cùng danh sách IDs. Không tối ưu lại crop, sigma hay gate dựa vào LCC mới.
5. **Audit offline**: tái tính AUC/EER/APCER/BPCER/ACER từ CSV với code độc lập, hash source, check sample count, paired path/videoIDs, source thresholds; so metric JSON; nếu lỗi dừng trước aggregate.
6. **Thống kê**: bảng từng seed trước, sau đó `mean ± sample SD (ddof=1, n=3)`; tính paired per-seed delta cùng seed cho tương quan đúng thiết kế. Không lấy CI 2000 bootstrap video làm CI training-seed; với n=3 không khẳng định kiểm định significance mạnh.
7. **Chỉ sau khi nhóm duyệt** mới cập nhật Word abstract/results/conclusion và article version mới. Giữ version seed100 cũ làm lịch sử.

### 2.3. Bảng cần tạo sau chạy (không điền dữ liệu giả)

| Method | Seed | CelebA Test-as-Val AUC (%) | LCC Combined image AUC (%) | CASIA Combined video AUC (%) | Checkpoint/hash/status |
|---|---|---|---|---|---|
| CE-only / Clean+Aux / WBST / MAST-PAD | 100 / 101 / 102 | từ predictions | từ predictions | từ predictions | VERIFIED/INCOMPLETE/INVALID |
| Mỗi method (n=3) | **mean ± sample SD** | sau đủ3 | sau đủ3 | sau đủ3 | chỉ khi cả3 comparable |

Ngay bây giờ có thể điền tham chiếu các mean đã kiểm: CE-only LCC83,6818±0,6267%; CASIA89,5688±1,9431%; Clean+Aux LCC85,0980±0,4342%. **Không** ước lượng Clean CASIA hoặc WBST/MAST mean3seed từ seed100.

## 3. P1: matched ablation tách vai trò harmful gate

Đây là yêu cầu **khác** repeat multi-seed. Đối chứng cần giữ source, model/init, augmentation/crop1,5, three-view generator, **sigma=.10**, gain/distribution, coefficient loss, warmup/checkpoint rule, train-mode forward của **B selected images**, BatchNorm và training budget.

| Nhánh | Selection | Spectral CE mask/reduction | Câu hỏi trả lời |
|---|---|---|---|
| **MAST-PAD chuẩn** | worst aligned margin trong 3 bands | strict harmful `H`; mean trên H | mốc đối chiếu |
| **Same-sigma no-gate** | y hệt | tất cả B selected views; mean trên B | Gate giúp gì khi Gaussian width giống nhau? |
| **Random-H gate** | vẫn chọn worst band | mỗi batch chọn ngẫu nhiên **đúng H** trong B, độc lập margin; mean trên H | Nhắm harmful có hơn random selection với cùng sparsity? |

**Bảo đảm fairness:** với random-H, H lấy từ gate MAST được tính ở cùng thời điểm/model không update BN trong selector; random mask dùng RNG riêng, preserve data/sample-order/gain stream và full-B forward như MAST. Nếu biến thể chọn different H do model trajectories, báo H time-series rõ. Nên chạy ít nhất một seed exploratory cho từng đối chứng sau khi P0 xong, và nhiều matched seeds nếu định viết causal gate claim. Một run mới chỉ nâng bằng chứng sơ bộ, không hợp thức hóa claim “gate chứng minh là nguyên nhân”.

**Metric/diagnostics ghi bắt buộc:** CelebA source selection, LCC pooled + evaluation partition AUC, CASIA Combined reconstructed video AUC, EER/rates frozen, training cost và harmful fraction, mean margin drop, flip fraction, band shares, H distribution, peak memory nếu đo được; không chuyển training-extra-cost thành inference overhead.

## 4. P2: nghiên cứu cơ chế phổ (nếu còn quota)

### 4.1. Gate-margin threshold sensitivity

Thay điều kiện binary `m_clean−m_worst>δ` với vài giá trị `δ` **đăng ký trước từ source-only**, gồm δ=0 chuẩn. Đây là **ngưỡng trong training gate**, hoàn toàn khác ngưỡng vận hành PAD; không giới thiệu/đánh giá adaptive threshold. Giữ mọi thành phần khác như MAST và không chọn δ tốt nhất theo pooled LCC/CASIA chính thức.

### 4.2. Single/leave-one-band controls

Đối chiếu LOW-only, MID-only, HIGH-only hoặc LOO (drop LOW/MID/HIGH), mọi mức gain, sigma, model, budget và H reduction giống nhau. Single-band one candidate làm `argmin` tầm thường; phải diễn giải là **band availability** chứ không trực tiếp đo tầm quan trọng causal từng band. Hình Gaussian profile vốn là miền chồng lấn, không kết luận từng band đại diện riêng cho Moiré/print/blur vật lý khi chưa có dữ liệu.

### 4.3. Complexity/training compute

Nếu cần lập luận trade-off, đo actual GPU-hours/epoch và training throughput cho cùng GPU, crop/pipeline/batch. MAST có selector eval/no_grad + full-B selected train forward; gate không có nghĩa chỉ forward 83% selected; tránh suy giảm training cost từ gating fraction.

## 5. Protocol độc lập mới (P2, rất giá trị học thuật)

- Freeze architecture/preprocessing/sigma/gate/checkpoint rule trước khi chạm target mới, giữ change log. Chọn target dataset/session/subjects với manifest gốc và quyền sử dụng hợp lệ.
- Đối chiếu de-dup theo **subject/video/frame**, công bố failed-face-detection coverage, classification denominators và unit AUC. Không gọi current reconstructed CASIA Combined official subject-disjoint hoặc LCC pooled target-blind vì trong development đã quan sát LCC.
- Nếu muốn so sánh directly O&M&I→C literature, thực hiện **chính giao thức đó** cho methods so sánh; không đổi tên CelebA→CASIA thành OMI→C.
- Có thể tái sử dụng 4 frozen checkpoints chỉ để infer holdout mới nếu compatible input/permissions; không nhất thiết train mới. Nhưng không dùng holdout mới để tuning sau khi xem results.

## 6. Phân tích offline và kiểm tra trước khi tốn quota

**Đã có/không train:** paired bootstrap video CASIA n358,2000 resamples; CI MAST93,52% [90,76–95,90], paired ΔMAST−WBST+1,2935pp [-0,0847,+2,8193]; `casia_attack_breakdown.csv`; LCC partition AUC; CASIA overlap/dedup manifest; baseline3seed metrics. Các CI video là conditional theo checkpoint và còn subject correlation; không claim training variance. File summary cũ đang trộn frame/video vào “Zero-Shot/Domain Calibrated”; phải sửa cách trình bày trước aggregate.

**Cần đo log nếu bảo vệ claim runtime:** PAD-only/whole pipeline/camera FPS/identity decision distinct; hardware/provider/model SHA, 50 warmups hoặc con số thực, samples, mean/median/p95, asynchronous queue/cache states. Không bắt buộc dùng Kaggle nếu đo webcam/local. Nếu không làm, chỉ giữ 434.434 params/40,81M MACs/0,08162GFLOPs.

## 7. Release gates của kết quả thí nghiệm mới

- **G0 — Pre-run lock:** phiên bản code/notebook hash/torch-CUDA version/config/data/checkpoint expected hashes, design matrix và target endpoint ghi trước.
- **G1 — Source completion:** log chứng minh completed, best epoch, optimizer/scheduler/RNG provenance nếu resume; source selection đúng; không report interrupted as final.
- **G2 — Target validation:** matched LCC paths and split sizes; CASIA subject/video/frame dedup và unit; source-frozen threshold; no target adaptation.
- **G3 — Independent recomputation:** CSV scores/labels → AUC/EER/rates so JSON; record coverage/failed detections; preserve per-seed tables.
- **G4 — Interpretation:** update claim chart; separate (a) robustness four-strategy, (b) gate causality control, (c) independent-target, (d) deployment; không trộn kết luận.
- **G5 — Publishing:** sau tác giả duyệt mới revise report/article; addendum số liệu kèm provenance, không overwrite archived release seed100.

## 8. Quyết định theo quota (để nhóm chọn)

- **Quota vừa đủ cho ~4 spectral completions:** chỉ P0 (WBST101/102, MAST101/102) + CASIA Clean101/102 inference; tối ưu trước bằng CPU/GPU local nếu được. Đây là lựa chọn ưu tiên.
- **Quota dư cho ~2 extra trainings:** chạy same-sigma no-gate và random-H gate một seed mỗi nhánh, rồi đánh giá; chưa đủ độ ổn định để kết luận gate độc lập mạnh.
- **Quota dồi dào hơn:** lặp controls theo matched seeds, sau đó δ/per-band và independent benchmark. Chốt thứ tự theo câu hỏi DG20 và mức đóng góp dự kiến.

**Chú ý:** cụm “~4 completions” là **đếm số run**, không phải ước lượng quota/GPU-hours. Quyết định resume/restart thực tế dựa vào state và ngân sách Kaggle. Không có công việc nào trong kế hoạch này đã được tự động chạy hoặc đặt lịch.

## 9. Nguồn

Evidence audit `00–08`, đặc biệt `03_METRIC_AND_DATA_AUDIT.md` §4–6 và `06_OPEN_ITEMS_AND_NEXT_WEEK.md` §1–4; notebook18/21; DG20 §3–5; baseline source-of-truth seed100 và báo cáo multi-seed. Các chỉ tiêu thí nghiệm mới là **đề xuất của nhóm**, không phải kết quả đã công bố. **Adaptive Threshold hoàn toàn ngoài phạm vi.**


---

## 10. Ma trận task chính xác cho đợt có quota tiếp theo

**Ghi chú thời gian:** Kế hoạch thực hiện **tuần kế tiếp kể từ 22/10/2026** là dự kiến của nhóm và tùy quota; không có automation/scheduled run. Thuật ngữ “run” bên dưới là hoàn tất và kiểm chứng *nguồn + targets*, không phải một checkpoint tạm. **Không adaptive threshold**.

| Task ID | Artifact đang có | Việc thêm | GPU training? | Kiểm chứng bắt buộc | Ưu tiên |
|---|---|---|---|---|---|
| G0.1 | CE-only seeds100/101/102 | Check SHA/manifest và retain | Không | 3 source+LCC+CASIA video hoàn chỉnh | Đã có |
| G0.2 | Clean+Aux seeds100/101/102 source+LCC | CASIA Combined video inference seeds101/102 | **Không retrain**, inference GPU/CPU tùy thời gian | checkpoint selected SHA, same358 IDs, mean logits/video, source-threshold | P0 |
| G0.3 | WBST seed100 full; 101/102 interrupted reported at epochs8/7 | Kiểm tra resume proof; train complete101/102 khi có quota | **Có** nếu không thể resume hợp lệ | optimizer/scheduler/scaler/RNG state hoặc restart; paired source data | P0 |
| G0.4 | MAST-PAD seed100 full | Train101 và102 | **Có** | same fullsource recipe, sigma.1/gate, source checkpoint; target eval | P0 |
| G0.5 | Các target source-frozen hiện có | Evaluate LCC + CASIA for new spectral checkpoints | Inference | LCC18794 images, CASIA358 videoIDs/2660frame, no target fitting | P0 |
| G0.6 | Verifier audit seed100 and baselines | Recompute 3seed AUC/EER/rates from predictions | Không training | exact per-run provenance + JSON match; sample SD ddof1 | P0 |
| G1.1 | MAST standard sigma.10 | **Same-sigma gate-off** (1 exploratory seed) | **Có**, optional | only gate criterion changes; same full-B train forward | P1 |
| G1.2 | MAST standard sigma.10 | **Random-H gate** (1 exploratory seed) | **Có**, optional | exact H online, random H distinct indices, extra RNG, mean/H | P1 |
| G2.1 | final MAST sigma.10 | Prespecified gate-margin δ grid | Có | δ conditions defined source-only | P2 |
| G2.2 | LOW/MID/HIGH masks | LOW-only/MID-only/HIGH-only or LOO | Có | common sigma/gain/full-B semantics | P2 |
| G2.3 | current exposed LCC / reconstructed CASIA | Target mới với subject/session manifest độc lập | Có thể chỉ inference | lock architecture before any target scores | P2 |
| G2.4 | runtime timers in code; no logs | Local device end-to-end latency/FAR-FRR evaluation | Local resource | component vs pipeline vs end-user decision separate | Optional |

## 11. Templates mẫu để Codex ghi lại trạng thái trước khi chi quota

### 11.1. `pre_run_contract.yaml` — schema gợi ý, KHÔNG phải kết quả đã chạy

```yaml
study: MAST_PAD_full_source
train_seed: 101  # or 102, record actual
base_data_seed: 43
usable_celebA_train_n: 484561
crop_context_scale: 1.5
input: [3, 80, 80]
image_representation: BGR_float_0_1
optimizer: SGD
lr: 0.005
momentum: 0.9
weight_decay: 0.0005
batch_size: 256
amp: true
max_epochs: 12
lr_milestones: [4, 8]
lr_gamma: 0.2
method: MAST-PAD  # or WBST
sigma: 0.10  # WBST uses 0.0555555556
spectral_centers: [0.0833333333, 0.25, 0.4166666667]
spectral_gain_interval: [0.65, 0.90]
clean_ce_weight: 0.75
spectral_ce_weight: 0.25
spectral_warmup_epochs: 1
source_checkpoint_selection: CelebA_Test_as_Val_min_calibrated_ACER_then_max_AUC
artifact_fields_required:
  - code_commit
  - notebook_sha256
  - train_manifest_sha256
  - initial_state_sha256
  - selected_checkpoint_sha256
  - best_epoch
  - completion_status
  - source_threshold
  - raw_predictions_sha256
```

**Quan trọng:** Codex không được đặt giả `code_commit` hay SHA. Đây là schema để thu thông tin từ môi trường chạy; path phân nhánh seed và source manifest phải xác nhận thật. WBST và MAST giữ objective khác nhau và sigma như các runs cũ cho P0, **không âm thầm đổi WBST sigma thành .1**. Sigma=.1 cho both arms chỉ thuộc P1 gate isolation.

### 11.2. `evaluation_contract.yaml` — giao thức output mong muốn

```yaml
source:
  name: CelebA-Spoof Test-as-Val
  scored_images: 66379
  checkpoint_and_threshold_selection: source_only
lcc:
  primary_unit: image
  pool: [training, development, evaluation]
  scored_images: 18794
  auc: from_pooled_scores
casia:
  primary_unit: reconstructed_video
  candidate_videos: 360
  expected_scorable_videos: 358
  video_score: mean_valid_frame_real_minus_attack_logit
  minimum_valid_frames: 4
  dedup: video_id_and_frame_index_test_copy_first
  sample_ids_must_match_seed100: true
metrics:
  auc_positive_class: real
  source_threshold_policy: frozen_per_checkpoint
  eer_from_roc: true
  error_rate_units: percent
  seed_dispersion: sample_std_ddof1
```

Count = EXPECTED, not a reason to silently discard misdetected data; if newer evaluation yields different scorable set, declare population incompatibility and investigate, not force numbers.

### 11.3. Output table headers (CSV; do not fill missing)

```text
method,train_seed,data_seed,source_commit,checkpoint_sha256,manifest_sha256,complete,best_epoch,population,unit,scored_n,candidate_n,auc_percent,eer_percent,apcer_percent,bpcer_percent,acer_percent,source_threshold,predictions_sha256
```

### 11.4. CI và so sánh paired seeds

- Với **n=3 train seeds**, report 3 hàng/raw values cho từng method trước khi tính mean, rồi tính `sample_std(ddof=1)`. Mỗi delta dùng đúng **same seed pair** trên cùng population, không so seed100 MAST với mean3 CE để kết luận cải thiện tổng thể.
- Với **n=358 CASIA videos** của seed100, paired-video bootstrap hiện có 2,000 resamples MAST CI90.76–95.90% và ΔMAST−WBST [−0.0847,+2.8193] pp. Khoảng này không phải 95% CI của mean across train seeds và chưa hiệu chỉnh video clustering theo người.
- Test mới phải xác nhận subject/video manifests trước khi dùng subject-bootstrap hoặc claim independent. Raw label/path ordering của all four checkpoints phải tương ứng khi paired bootstrap.

## 12. Resource decision tree — không ước lượng giờ GPU khi chưa đo

| Quota trở lại | Quyết định nhóm nên ưu tiên | Chưa được khẳng định |
|---|---|---|
| Đủ cho hoàn tất 4 spectral runs, có khả năng local CASIA eval | Hoàn thành P0, mới cân nhắc publish mean3seed | Không nói gate causal |
| Chỉ đủ cho 2 run | Ưu tiên MAST101/102 **hoặc** kiểm thực tế chi phí resume WBST và công việc hoàn thành nhanh nhất; báo vẫn n<3 ở phương pháp thiếu | Không ghép 2 seed của model này với 3 seed model kia thành matched mean |
| Dư thêm 2 experiments | Same-sigma no-gate + random-H exploratory seed | Không tuyên bố reproducible gate effect |
| Dư nhiều hơn + target mới | Lặp matched gate controls across seeds, rồi independent target | Không dùng LCC exposure target để tune thêm hyperparam |
| Không có GPU quota | Sửa Word/chapter/figures, audit offline/CI/attack type, thu log runtime local nếu có thiết bị | Không thêm seeded results mới |

**Quota priority không tương đương time estimate.** Baseline compute thời gian/GB có thể lấy từ logs cũ nếu đã đo, sau đó build realistic quota budget table; không bịa T4-hours.

## 13. Prompts liên quan trong gói playbook

- `prompts/S09_FUTURE_SEEDS.md`: read-only validation và proposed document delta **sau khi có artifacts mới**.
- `prompts/S10_OPTIONAL_ABLATION_PLAN.md`: design preregistered same-sigma / random-H / gate δ / per-band; **không chạy**.
- `prompts/S04_OFFLINE_EVIDENCE.md`: làm ngay khi không có GPU; xử lý frame/video mismatch và chuẩn hóa số liệu.
- `prompts/S05_CHAPTER3_RESULTS.md`: đưa bảng/thống kê đã duyệt vào Word; không điền missing cells.
