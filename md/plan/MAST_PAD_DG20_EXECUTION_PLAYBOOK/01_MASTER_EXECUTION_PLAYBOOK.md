# MAST-PAD — Kế hoạch thực thi chi tiết DG20 (v2)

**22/10/2026 | Dùng để giao Codex, chờ tác giả duyệt | Chưa sửa Word/article.** Đọc **cùng** `00_DATA_LOCK_FOR_CODEX.md` để dùng thống nhất số liệu. Căn cứ: `Báo cáo Thị Giác Máy Tính (2).docx`, article 8 trang, DG20 16/10, MD review 20/10, audit `00–08` ngày 22/10 và các raw artifacts được audit xác minh. Không đặt giả thiết rằng tất cả file CSV audit đã được đính kèm cuộc hội thoại; Codex phải tìm chúng trong repo và ghi rõ khi thiếu.

**Ngoài phạm vi tuyệt đối:** Adaptive Threshold đang phát triển. Không đưa module này vào mô tả hệ thống, thí nghiệm, đánh giá hoặc bài báo. Chỉ có thể ghi nhận **fixed source threshold** và **existing fixed runtime configuration** khi thật sự cần phân biệt hai giao thức đã có.

## 1. Mục tiêu, sản phẩm và giới hạn

1. **Báo cáo học phần Word:** chính xác **3 chương**, vẫn giải thích pipeline xác thực sáu giai đoạn, nhưng MAST-PAD là trọng tâm nghiên cứu. Tổng nội dung tính điểm **≤50 trang**, Ch1 **≤15**, Ch2 **≤25**, Ch3 **≤20**. Target **11 + 20 + 16 = 47** trang để có 3 trang dự phòng. Chỉ xác nhận số trang sau khi **render Word thành PDF và đếm thật**; TOC cache không được coi là minh chứng. Mục lục/phụ lục được loại theo yêu cầu giảng viên; vì phạm vi loại trừ tài liệu tham khảo và mở đầu chưa rõ, Codex phải báo **hai cách đếm** và không tự kết luận nếu vướng sát 50.
2. **Article IEEE 8 trang:** giữ nguyên làm bản PAD-only chuẩn; không nhét sáu-stage runtime, không sửa nếu không có bằng chứng/kết quả mới hoặc lỗi kiểm chứng. Mọi sửa article chỉ theo stage riêng đã được nhóm phê duyệt.
3. **DG20 response:** theo dõi đủ **D01–D46**, không nhầm “đã code”, “đã audit” với “đã sửa Word”, “đã train” hoặc “đã kiểm chứng độc lập”. Tài liệu phản hồi viết tiếng Việt, tích cực, minh bạch.
4. **Vùng dữ liệu cuối:** chỉ 4 chiến lược CE-only, Clean + Aux, WBST, MAST-PAD trên source train **484.561 ảnh**; không có bảng/hình/mô tả kết quả micro/10K/100K hoặc R7_SEV trong bản nộp, **kể cả phụ lục kết quả**. Không xóa các artifacts lịch sử khỏi repository.
5. **Không tuyên bố:** Harmful Gate có tác dụng nhân quả đã cô lập; LCC là untouched target; CASIA reconstructed là official subject-disjoint; 2,22ms/67,98ms là measured trace; Pi/Jetson realtime; 1,74MB là kích thước file ONNX; compliance ISO; robustness 3D mask/deepfake hoặc FAR/FRR end-to-end nếu không có evidence.

### Chỉ mục file đi kèm

- `00_DATA_LOCK_FOR_CODEX.md`: **dữ liệu sẵn để điền** (AUC, rates, seeds, CI, counts, optimizer, runtime, models, citations).
- `01_MASTER_EXECUTION_PLAYBOOK.md`: hướng dẫn tác giả duyệt và kiểm soát stage.
- `02_DG20_RESPONSE_MATRIX_DETAILED.md`: response D01–D46, phản hồi trình bày theo từng nhóm, phân biệt done/pending.
- `03_EXPERIMENT_EXECUTION_PLAN.md`: lộ trình chạy GPU/post-quota, evidence và acceptance gates.
- `prompts/S00_...` đến `prompts/S10_...`: prompt tiếng Anh sẵn dán cho Codex, mỗi lần **một stage**.
- `MAST_PAD_DG20_PLAYBOOK_AND_PROMPTS.zip`: gói giữ nguyên toàn bộ cấu trúc, có thể chép vào repo.

### Input repo mà Codex phải xác minh (không giả định đường dẫn tồn tại)

- Word đích `(2).docx`, DG20 `.docx`, article `(1).pdf`, MD review cũ.
- Audit `paper/review/dg20_evidence_audit/00_EXECUTIVE_VERDICT.md` đến `08_COMPLETION_REPORT.md`, và `01_DG20_COVERAGE_MATRIX.md` D01–D46.
- Audit derived `recomputed_metrics.csv`, `baseline_three_seed_metrics.csv`, `baseline_three_seed_summary.csv`, `paired_video_bootstrap.json`, `casia_attack_breakdown.csv`, `runtime_whitelist.json`, `added_seed_validation.json`, `development_markdown_evidence.txt` (nếu có trong repo).
- Code `app.py`, `tracking/tracker.py`, `vision_pipeline.py`, `recognition/worker.py`, `recognition/embedder.py`, `alignment/aligner.py`, `database/vector_db.py`, notebooks 18/21, raw predictions / config / completion / metrics JSON.
- Article LaTeX `paper/main.tex`, `paper/sections/`, `paper/references.bib`, `paper/release/` chỉ đọc trừ **optional article stage**.

**Locator** `Pnnnn` chỉ là paragraph OOXML, `Tn` là bảng document.xml, **không** phải trang. Audit Word mới: 1.315 paragraphs / 23 tables. Không lấy số trang cached TOC làm số trang nộp.

## 2. Layout ba chương — phân bổ đến tiểu mục

### Chương 1 — Tổng quan, nghiên cứu liên quan, ưu/nhược (11/15 trang)

| Tiểu mục | Số trang target | Nguồn cũ → nội dung cuối | Checklist khoa học |
|---|---:|---|---|
| 1.1 Bài toán, động lực và phạm vi | 2 | `P0027–0030`, `P0245–0282`; PAD print/replay, generalization gap, ràng buộc mô hình nhỏ | Không khẳng định physical moiré là nguyên nhân đo được; không claim đã đo latency |
| 1.2 Tổng quan các hướng | 4 | Khảo sát backbone/detector/PAD từ old Ch2; Related Work article | Phân nhóm Spatial/MiniFASNet, CelebA-AENet & auxiliary, MADDG/SSDG, PatchNet/SA-FAS/GAC/AMEL, frequency Chen/Bi-FPNFAS/FSDA/Cao–Ma |
| 1.3 **Bảng ưu/nhược và nghiên cứu gap** | 3 | §4.4 T19/T20 và papers; tách khỏi “bảng xếp hạng” trực tiếp | Tạo bảng 6–9 hướng, principle, strengths, limitations with citation, khác biệt nguồn/protocol, gap MAST. WBST là **internal comparator** không phải bài đã công bố |
| 1.4 Mục tiêu và đóng góp | 2 | old §1.2 + method intro | Ba đóng góp: worst spectral margin; harmful gate + H-normalization; giữ deploy PAD 434.434 và target AUC. Không chứng minh riêng gate bởi WBST khác sigma |

**Bảng nghiên cứu liên quan phải có nguồn từng dòng:** MiniFASNet/Silent-Face; CelebA-Spoof/AENet; MADDG; SSDG; PatchNet; SA-FAS; GAC-FAS/AMEL; Chen spectral; FSDA; Cao/Ma shortcut. Có thể nhóm theo họ để không vượt trang. **AENet = Auxiliary Information Embedding Network** theo CelebA-Spoof, không phải “Adversarial Erasing Network”. Benchmark repo LCC 86,80% không gán thành kết quả LCC gốc của ECCV paper. Không xây “AUC/params” thành metric chất lượng chuẩn.

**Hình đề nghị**: Threat → domain shift → PAD decision; sử dụng sơ đồ khái niệm có nhãn *illustrative*, không tạo ROC giả 99→82 như dữ liệu quan sát. Table 1.1 so sánh ưu/nhược/gap; không sao chép biểu đồ từ literature khi chưa rõ quyền.

### Chương 2 — Giải pháp hệ thống và MAST-PAD (20/25 trang)

| Mục | Trang | Nội dung cuối | Figure/table |
|---|---:|---|---|
| 2.1 Kiến trúc tổng thể | 2 | 6-stage online pipeline và offline spectral training chỉ thuộc PAD | F2.1 6 stage + async worker |
| 2.2 SCRFD, tracker và temporal PAD | 4 | face detection, IoU/LK, valid face crop, PAD temporal state, cadence | F2.2 votes/state machine; cấu hình runtime |
| 2.3 Alignment, ArcFace, centroid matching | 2 | five landmarks →112×112; ArcFace512-D L2-normalized; stored centroid, worker queue/poll | F2.3 workflow worker/invalidation |
| 2.4 DFT và spectral intervention | 2 | 2D orthonormal FFT, Gaussian radial attenuation, DC/phase behavior | F2.4 LOW/MID/HIGH vector curves |
| 2.5 **MAST-PAD đầy đủ** | **7** | 128-D miniFAS embedding, auxiliary heads, margin, worst of 3, detached gate, reduction H, warmup, BN, full-B forward, four methods | F2.5 training/inference, F2.6 mechanism, T2.1 4 methods, Algorithm 1 |
| 2.6 Implementation, contracts | 3 | crop geometry, optimizer, sigma, gain, output/threshold fixed policy distinction, compute scope | T2.2 training settings |

**Không biến hệ thống 6 giai đoạn thành stage “PAD rồi recognition chạy tất cả mỗi frame”.** Theo audit code:

- Detector & PAD configured cadence **0.2 s / 0.2 s**; recognition **1.0 s** khi đủ điều kiện; không phải observed camera FPS.
- IoU threshold **0.3**, LK propagation, stale timeout **3 s**.
- PAD voting capacity **5**, `min_votes=3`, spoof vote fraction **≥0.6** trên **số vote đã có** `len(window)` sau khi đủ min3; không luôn chia5. Không hứa “loại bỏ hoàn toàn false alarms”.
- PAD `1.5×` face crop `80×80 BGR [0,1]`, ArcFace align `112×112` là **hai preprocessing khác nhau**.
- Recognition **async worker**, bounded queue, submit+poll, generation/track continuity invalidation, UNKNOWN/ERROR/PENDING không đồng nghĩa REAL/SPOOF. Cache centroid không thay toàn gallery và enrollment vector đã được lưu, không tính lại mọi sample mỗi frame.
- Main loop `recognition_ms` chủ yếu phản ánh submit/alignment, không phải actual ArcFace+DB finish latency.

**Cơ chế MAST phải đúng code, không chỉ đúng lời văn:**

```text
Real=0, Attack=1; d(x)=z_real(x)-z_attack(x)
m(x,y)= d(x) nếu Real, -d(x) nếu Attack
k*=argmin_{LOW,MID,HIGH} m(x_k,y)   # clean KHÔNG tham gia argmin
h=1[m(x_k*,y) < m(x_clean,y)]        # strict, detached
L_worst=Σ CE(x_selected_i,y_i)/B
L_harmful=Σ h_i CE(x_selected_i,y_i)/H, H>0; differentiable 0 if H=0
L_aux=0.1 CE_spoof11+0.1 CE_light5+BCE_attr40(real-only)
L_MAST = 0.75 L_clean+0.25 L_harmful+L_aux after epoch1 warmup
```

Selector temporary `eval()+no_grad()` để không cập nhật running BatchNorm; sau đó **train forward tất cả B selected views**, h_i chỉ vô hiệu hạng CE trực tiếp cho nonharmful, không loại sample khỏi statistics BatchNorm. Nếu H=0 không tự đổi clean coefficient 0.75→1.0. Gain per sample `U(0.65,0.90)` dùng chung 3 bands/channel; centers1/12,1/4,5/12; overlapping Gaussian; sigma WBST **1/18** còn MAST **0.10**. Phase/DC giữ trước clamp, không khẳng định output phase bất biến sau clamp.

**Bảng T15 phải thay toàn bộ**: 4 methods `CE-only`, `Clean + Aux`, `WBST`, `MAST-PAD`. Dùng bảng objectives trong Data Lock §A; bỏ R7_SEV/phiên bản severity và các hàng trọng số mẫu `1/N`/`1/H` gây nhầm với chọn band. MAST cột spectral loss **L_harmful**, không `L_worst`; spoof head 128→**11** (`P0558`), không 128→1. Một worst view mỗi ảnh, không mean 3 frequency views.

### Chương 3 — Thực nghiệm, đánh giá và kết luận (16/20 trang)

| Mục | Trang | Nguồn và nội dung bắt buộc | Bảng/hình |
|---|---:|---|---|
| 3.1 Datasets & protocol | 4 | source Train/Test-as-Val, LCC splits/combined/exposure, CASIA reconstructed dedup, threshold policy | T3.1 population audit |
| 3.2 Four strategies & training | 2 | seed100 comparison, 12-epoch max, SGD/warmup, same graph, sigma caveat | training settings |
| 3.3 AUC comparisons and partial 3 seeds | 4 | 4×3 AUC %, context baseline3 seeds **chỉ đủ nơi đã verified**, literature protocol differences | T3.2 AUC, T3.3 partial seeds, F3.1 AUC chart |
| 3.4 Error/diagnostics/complexity | 3 | EER/APCER/BPCER, optional conditional bootstrap & actual attack types, last epoch11 stats, Conv/Linear complexity | T3.4 rates, F3.2 margin diagnostics, T3.5 complexity |
| 3.5 Interpretation/limits | 2 | MAST improves over 2 clean controls, WBST slight LCC lead, gate/sigma confound, exposure, no latency trace | short external contextual comparisons |
| 3.6 Conclusion & future | 1 | evidence-backed achievement then future seed/gate-only/independent target | no fourth chapter |

#### Đúng số liệu để điền ngay

**T3.1 — dữ liệu/giao thức**: CelebA Train **484.561 usable / 494.405 candidates**, CelebA Test-as-Val **66.379 / 67.170**; LCC Training **8.283**, Dev **2.947**, Eval **7.564**, Combined **18.794 images/18.827 candidates**; CASIA Combined **2.660 frames, 358/360 reconstructed videos** (90 real, 268 attack), min 4 valid frames; overlapping copies **20 subject IDs/240 video IDs/1.403 frame IDs**, dedup test priority. LCC Combined pooled image-level AUC, CASIA Combined video-level mean frame-logit AUC. LCC endpoint đã xuất hiện trong development; không xem như blind target test.

**T3.2 — 4 phương pháp seed100, ROC-AUC (%)**:

| Cấu hình | CelebA Test-as-Val | LCC Combined images | CASIA Combined reconstructed videos |
|---|---:|---:|---:|
| CE-only |99,21|84,40|89,83|
| Clean + Aux |99,07|84,60|92,04|
| WBST |99,22|**86,08**|92,23|
| MAST-PAD |**99,23**|86,03|**93,52**|

**Chênh lệch exact pp**: MAST−CE LCC +**1,6284** / CASIA +**3,6899**; MAST−Clean +**1,4261** / +**1,4801**; MAST−WBST LCC **−0,0559** / CASIA **+1,2935**. Trong abstract/conclusion dùng 2 chữ số; tính từ raw không từ AUC làm tròn.

**T3.3 — Baseline một phần 3 seed, n=3 sample SD (`ddof=1`, AUC%, SD theo điểm %):** CE-only CelebA **99,1963 ± 0,0220**, LCC Combined **83,6818 ± 0,6267**, CASIA Combined video **89,5688 ± 1,9431**; Clean + Aux CelebA **99,0817 ± 0,0874**, LCC Combined **85,0980 ± 0,4342**; Clean+Aux CASIA seed101/102 chưa có; WBST & MAST seed101/102 chưa hoàn chỉnh. CE-only LCC per-seed 84,3988/83,4079/83,2386; Clean LCC 84,6011/85,4042/85,2888; CE CASIA 89,8300/87,5083/91,3682. **Không có spectral mean±SD** hiện tại. LCC baseline mean gain Clean−CE +**1,4163pp**.

**T3.4 — source-frozen operating rates, seed100 (%, same population/unit for pair):**

| Target | Model | EER | APCER | BPCER | ACER |
|---|---|---:|---:|---:|---:|
| LCC pooled images | MAST-PAD |22,06|7,98|46,26|27,12|
| CASIA Combined video | MAST-PAD |16,92|1,87|51,11|26,49|
| CASIA Combined video | CE-only |**16,54**|0,37|73,33|36,85|

**Word `P0847` mandatory correction**: CE-only CASIA video EER **23,05% → 16,54%**. **Word `T21` mandatory completion:** MAST CASIA APCER/BPCER **1,87%/51,11%**. T3.4 có thể liệt kê cả 4 methods ×2 targets theo Data Lock §D. Chú thích rõ ngưỡng per model lấy từ CelebA, Real-positive, AUC/EER theo ROC không thay chỉ bằng chọn lại ngưỡng, APCER/BPCER tại ngưỡng đang dùng có thể đổi. Giữ thảo luận công bằng: BPCER 46,26/51,11 vẫn cao tại ngưỡng cố định nguồn, chưa đủ tuyên bố an toàn thực tế.

**T3.5 — profile PAD-only**: **434.434 deployed parameters**, **40.810.892 MACs** và **0,081621784 GFLOPs** Conv/Linear ×2 (80×80 input), raw weights FP32 **1.737.736 bytes ≈1,74MB** (*không* phải actual ONNX file/RAM). SCRFD/ArcFace/tracking/activation/norm/pooling không nằm trong FLOP count. Không có log hợp lệ cho PAD 2,22ms, pipeline 67,98ms, camera14,7/15,5FPS: bỏ claim nếu nhóm chưa cung cấp raw log đúng source/model/provider.

**T3.6 — contextual literature**: LCC Graph Transformer AUC83,30%, ~10,84M; AENet AUC86,80%,11,22M do repo benchmark ghi; MAST86,03%,0,434M. CASIA MADDG84,51%, AMEL94,39%, PatchNet94,58%, GAC-FAS95,16%, SA-FAS95,37% trong **O+M+I→C** so với MAST93,52% **CelebA→reconstructed CASIA Combined**. Không so trực tiếp thứ hạng; không ghi số param của ResNet18 như complete method. Nếu bảng tổng quan ở Ch1 đã đầy đủ, Ch3 chỉ giữ compact context, tránh lặp.

**Diagnostics**: epoch cuối thực chạy 11, harmful **83,2135%**, prediction flip **0,3077%**, LOW/MID/HIGH được chọn **42,6452/27,3035/30,0513%**. Đây **không phải** diagnostic của checkpoint tốt nhất epoch7. Gate không “sparse hard-example mining” theo nghĩa rất ít sample: ~16,79% có zero direct spectral CE ở epoch cuối; tất cả selected views vẫn train-forward.

**CI đã có từ audit, chỉ đưa nếu tác giả duyệt:** 2.000 paired video bootstrap resamples trên **358 video IDs**: MAST AUC93,52%, 95% CI **[90,76;95,90]%**; ΔMAST−WBST +1,2935pp 95% CI **[−0,0847;+2,8193]pp**, bao gồm 0. CI là conditional-on-checkpoint video sampling, không phải training-seed/subject-independent significance. `casia_attack_breakdown.csv` đã được Codex audit export trong repo; **kiểm mapping/count trước khi đưa bất cứ tỷ lệ nào**, không bịa thêm 3D mask.

## 3. Migration map và lỗi cụ thể theo tài liệu Word cũ

| Mã | Locator audit | Hiện trạng Word `(2)` | Thao tác bắt buộc | Stage |
|---|---|---|---|---|
| M01 | old `P0027–30`, `P0245–282` | lời mở đầu/Ch1 lặp | thu gọn về Ch1 1.1, 1.4 | S1 |
| M02 | old `P0283–393` | dataset đặt ở Ch1 | chuyển protocol vào Ch3 3.1 | S1/S5 |
| M03 | old `P0395–593` | 6-stage runtime đôi chỗ khác code | chỉnh thành Ch2 2.1–2.3, 2.6 | S2 |
| M04 | `P0471` | tracking IoU 0,5 | từ cấu hình hiện tại **0,3**, ghi phạm vi config | S2 |
| M05 | `P0479–483` | window luôn 5 / 5 votes; cam kết no false alarms | capacity5/min3; ratio≥.6 trên len(window); bỏ tuyệt đối hóa | S2 |
| M06 | `P0490–491/P1067` | cadence PAD0,1/recognition0,5; nơi khác0,2/1 | .2/.2 detector/PAD; 1.0 recognition, current file config | S2 |
| M07 | `P0558` | spoof head128→1 | **128→11**, giữ PAD128→2/light128→5/attr128→40 | S3 |
| M08 | `P0594–684` | MAST theory và FFT phân mảnh | nhập Ch2 2.4–2.5, điều chỉnh BN/H=0 | S3 |
| M09 | `T15/P0688–734` | R7_SEV; MAST `L_worst`; hàng 1/N gây nhầm | thay bảng 4 method, MAST `L_harmful`; xóa severity | S3 |
| M10 | `P0736/P0881` | kỳ vọng thêm seed tự cô lập gate | sửa: seeds đo tính ổn định 4 configs, gate-only cần same sigma | S3/S5 |
| M11 | old §3.6/T16 | crop 2.7/1.5 screening | giữ crop1.5 cuối, không đưa AUC 10K/100K | S3 |
| M12 | `P0800–819` | experimental/setup | chuyển vào Ch3 với source/count/units đúng | S5 |
| M13 | old §4.2.1–4.2.3, `T23` | micro/100K histories kể cả phụ lục | **loại khỏi bản báo cáo nộp**, không xóa repo | S1/S5 |
| M14 | `P0847` | CE CASIA EER=23,05% | **16,54%** | S5 |
| M15 | `T17/T18` | method run-code/SD drift | AUC% four methods, partial verified 3seed ddof1 | S5 |
| M16 | old T19/T20; `P0967` | literature context bị hiểu matched | ghi rõ khác source/target/protocol, param scope | S1/S5 |
| M17 | `T21` | thiếu MAST CASIA APCER/BPCER | điền 1,87%/51,11%, ACER26,49 | S5 |
| M18 | `P1037–1039` | OLED/lighting causal speculation | chỉ là hypothesis hoặc dùng type breakdown verified | S4/S5 |
| M19 | `P1042–1072` | latency 2,22ms/67,98ms/cache/ONNX | bỏ nếu không có log, giữ profile PAD scope | S5 |
| M20 | `P1078/1085` | “mọi cấu hình 1 seed” | phân biệt common 4-method seed100 với baseline 3seed hiện đã verified | S5 |
| M21 | `P1080` | LCC independent/untouched | ghi nhận LCC-used-in-development | S5 |
| M22 | ref Word `[1]` | AENet “Adversarial Erasing Network” | **Auxiliary Information Embedding Network**; đúng source paper/benchmark | S1/S6 |
| M23 | misc refs/ISO | MADDG wrong title, unverified DOI, ISO standard claim | truy primary citation, unknown means unknown | S1/S6 |
| M24 | headings `3.2/3.12/2.3.4` | lặp nhãn | native multilevel heading, 3 new chapters | S6 |
| M25 | old Fig4.3 caption/T17 | chart chỉ 3 baseline và run IDs | chart 4 methods with % axes | S5/S6 |
| M26 | `P1072` | demo placeholder | ảnh minh chứng có nguồn tác giả hoặc **bỏ**, không tự chế | S6 |
| M27 | Word assignment table | phân công trống | giữ placeholder có nhãn chờ nhóm hoặc bỏ nếu không bắt buộc; **không tự bịa tỷ lệ** | S6 |

**DG20 lỗi `4.4.1` bị lặp và thiếu giá trị trước 99,21% đã được audit xác nhận không còn trong Word `(2)`.** Đánh dấu resolved in newer draft, không “sửa” một lỗi đã mất. Các locator P/T là từ audit, Codex phải đối chiếu anchor text trong Word mới trước khi thay, phòng paragraph index drift.

## 4. Thứ tự triển khai theo stage

Mọi stage: **one stage only**, không làm stage kế; giữ source không đổi, tạo bản `report_sN.docx` mới, `stage_N_log.md`, `issues.md`, khi sửa Word thì cố render PDF và inspect những trang ảnh hưởng. Chấp thuận chỉ khi outputs thực sự tồn tại và đúng criteria. Nếu không có renderer, ghi **NOT VERIFIED** thay vì khẳng định page cap.

| Stage | Prompt riêng | Phạm vi | Output chính | Gate review |
|---|---|---|---|---|
| **S00** Inventory/freeze | `prompts/S00_FREEZE_INVENTORY.md` | chỉ đọc originals, backup/ledger | source inventory hashes; 46 DG20 IDs matrix; saved originals | đường dẫn thực và originals không đổi |
| **S01** Restructure + Ch1 | `prompts/S01_CHAPTER1_RELATED_WORK.md` | Word working copy Ch1, provisional 3-chapter frame | report_s1.docx+PDF; literature strengths/limits matrix | Ch1 ≤15 khi render; citations verified |
| **S02** Six-stage runtime | `prompts/S02_CHAPTER2_PIPELINE.md` | Ch2 2.1–2.3/2.6 | report_s2.docx+PDF; code-concordance | states/async/cadence/votes all right |
| **S03** MAST method | `prompts/S03_CHAPTER2_MAST_METHOD.md` | Ch2 2.4–2.5/T15/method figures | report_s3.docx+PDF; equations table | strict gate, H=0, full B & BN, sigma correct |
| **S04** Offline evidence | `prompts/S04_OFFLINE_EVIDENCE.md` | audit outputs only; do not edit Word | validated metric tables/CI/manifest plan | units/source parity; no fabricated values |
| **S05** Ch3 results | `prompts/S05_CHAPTER3_RESULTS.md` | Ch3/Results figures/tables | report_s5.docx+PDF; all 4 seed100 + partial n3 baseline | numeric consistency, correct thresholds, honest limits |
| **S06** Layout/TOC/refs | `prompts/S06_FORMAT_AND_PAGINATE.md` | entire Word formatting, references, captions | report_candidate.docx+PDF; actual page audit | 3 chapters; 15/25/20; main ≤50; no placeholder |
| **S07** DG20 final and QA | `prompts/S07_DG20_FINAL_QA.md` | response notes and tiny approved Word fixes | report_final_candidate.docx/PDF; DG20 46-line response; final issues | completed ≠ planned, preserve article |
| **S08** Optional article delta | `prompts/S08_OPTIONAL_ARTICLE.md` | only after explicit approval; isolated LaTeX release copy | diff proposal and optional vNext | approved PDF preserved, no retroactive claim |
| **S09** Post-Kaggle seed update | `prompts/S09_FUTURE_SEEDS.md` | new raw artifacts only after exists | complete per-seed/provenance and proposal for paper/report | 3×4 compatible runs, paired delta & ddof1 |
| **S10** Optional ablation design | `prompts/S10_OPTIONAL_ABLATION_PLAN.md` | experiment plan only; no run | gate-only/random-H/per-band study matrix | same sigma held, cost justified, no auto-run |

## 5. DG20 acceptance mapping (grouped, underlying D01–D46 in audit)

| Nhóm nhận xét | Giải quyết bằng report/edit | Chỉ cần evidence mới | Phụ trách |
|---|---|---|---|
| Paper PAD-only vs system report, shorten round history D01–D05/D45 | giữ article PAD-only, Word 3 chapters, tables final | — | S01,S02,S05 |
| Novelty/WBST/citations D06–D11,D37–D42 | related works strengths/limits, metadata đúng, contextual comparison | literature validation vẫn theo nguồn hiện có | S01,S06 |
| Correct harmful method D16–D20 | Ch2 math, H=0, BN, sigma qualifier | isolated gate proof chưa có | S03 |
| Ablations D12–D15 | mô tả kế hoạch rõ, không claim tested | same-sigma no gate, random-H, delta gate, per-band | S10 / future GPU |
| Seed+CI D21–D23,D44 | baseline verified partial n3, 2000 video-bootstrap if approved | WBST/MAST 101/102, Clean CASIA 101/102 | S04,S05,S09 |
| Latency/full-system D24–D25,D36,D46 | bỏ speed unsupported, runtime code match | actual benchmark identity/session/device | S02,S05 |
| Source/target protocol D26–D31,D43 | counts, dedup, LCC exposure ledger | truly new independent domain | S04,S05 |
| Operating errors/attack D32–D35 | EER fix, threshold rates, typed analysis verified | unseen 3D/deepfake | S04,S05 |

## 6. Figure/table inventory — chỉ vẽ lại khi có giá trị

| Item | Vị trí | Điều kiện hình/bảng chuẩn |
|---|---|---|
| T1.1 related-work pros/cons | Ch1 | Method/year/citation, principle, ưu, nhược, gap, không giả metadata |
| F2.1 6-stage pipeline | Ch2 | SCRFD→track→temporal PAD→align→ArcFace→centroid, async queue/poll, invalidation |
| F2.2 voting/timing | Ch2 | max5/min3, .6 over available votes, .2/.2/1 config chứ không measured timing |
| F2.3 train vs infer MAST | Ch2 | Fourier/aux/select only training, 434434 PAD graph deploy separate SCRFD |
| F2.4 gate mechanism | Ch2 | eval/no_grad + train selected B, strict detached h, H normalization |
| F2.5 spectral profiles | Ch2 | overlapping Gaussian, DC override, sample gain=.775 illustrative, sigma=.1 |
| T2.1 four objective table | Ch2 | thay T15, `L_harmful` correct with no unsupported sigmoid claim |
| T3.1 protocol population | Ch3 | count/units, dedup, Test-as-Val, overlap |
| T3.2 4 method AUC | Ch3 | % and winners correctly bolded |
| F3.1 AUC % dot plot | Ch3 | 4 labels, LCC/CASIA panel axis, no artificial error bars |
| T3.3 operating rates | Ch3 | source-frozen + EER distinctions |
| T3.4 3-seed baseline subset | Ch3 | only verified populations, SD ddof1, pending visible |
| F3.2 training diagnostics | Ch3 | epoch11 only; no causal proof |
| T3.5 profiler | Ch3 | PAD-only MAC/FLOPs/weight byte estimate scope |
| T3.6 literature context | Ch1/Ch3 | comparison descriptive and protocol-specific |

## 7. Mở rộng không cần Kaggle và điều kiện sử dụng

- Có sẵn audit **2.000 paired video bootstrap** CASIA reconstructed Combined `n=358`: MAST AUC93,52%, CI[90,76–95,90]; paired MAST−WBST+1,2935pp CI[−0,0847,+2,8193]. Chỉ dùng khi ghi rõ *conditional-on-checkpoint, video-resampling, not subject-independent*; CI of Δ qua0 ⇒ không claim MAST gate statistically better. Source JSON/CSV trong repo, không tự generate lại số khi chỉ có text.
- CASIA `attack_type` breakdown từ audit có file riêng; dùng đúng nhãn/count sau khi Codex kiểm theo video_ids, không tự suy 3D Mask. Một đoạn phân tích theo attack type hỗ trợ DG20 mà không cần train mới.
- LCC split Evaluation MAST84,006% > WBST81,969% seed100 là mô tả phụ theo split, không phải untouched holdout; không thay bảng main LCC Combined.
- Development exposure ledger đã xác nhận LCC-visible screening khi chọn crop/sigma; báo cáo trung thực mà không cần đưa screening results vào paper.

## 8. Kế hoạch thí nghiệm sau khi có quota (tóm tắt)

**P0:** CE-only 100/101/102 đã verified source/LCC/CASIA; Clean+Aux 100/101/102 verified source/LCC nhưng CASIA 101/102 missing; WBST 100 complete, 101/102 interrupted reported (resume only with optimizer/scheduler/RNG state) và MAST100 complete,101/102 pending. Do full-source compatible run completions, all target predictions at matching units, hash+checkpoint verification, sample SD ddof1, seedwise deltas. 3 seed **không** tách riêng gate khỏi sigma.

**P1:** same-sigma `σ=.10` always-selected no gate vs standard MAST, và random-H gate với cùng `H` per batch; keep all B train forward, same train recipe/manifest, separate RNG. Thử 1 seed exploratory nếu quota ít; repeat khi mục tiêu là nhân quả của gate. 

**P2:** gate-margin `δ` sensitivity (là *training harmful gate*, hoàn toàn khác adaptive inference threshold đang ngoài scope), leave-one-band-out, new genuinely independent target/manifest, optional pipeline latency/identity benchmark trên local hardware. Chưa chạy; không được mô tả là hoàn thành.

## 9. Mệnh lệnh chạy Codex từng stage

Sao chép cả thư mục này vào repo, ví dụ `paper/prompts/dg20_v2/`, và gửi:

```text
Read paper/prompts/dg20_v2/00_DATA_LOCK_FOR_CODEX.md and
paper/prompts/dg20_v2/prompts/S00_FREEZE_INVENTORY.md.
Execute Stage S00 only. Preserve the original Word, article and raw evidence.
Produce all deliverables and STOP for author approval.
```

Ở lượt tiếp theo thay `S00_FREEZE_INVENTORY.md` bằng tên prompt stage cần chạy. Mỗi prompt có mục Inputs, Allowed edits, Tasks, Output, Verification và STOP riêng. **Không chạy tất cả prompts một lần.** Stage S08–S10 là tùy chọn, chỉ kích hoạt sau khi được tác giả chấp thuận hoặc có artifact mới.

## 10. Checklist cuối

- [ ] Word 3 chương; có PDF đếm trang thật: Ch1≤15, Ch2≤25, Ch3≤20, main≤50; front/back tính riêng.
- [ ] T15 thay4 method, không R7_SEV, `L_harmful` và H=0/full-B/BN đúng; head128→11.
- [ ] Bảng headline 4-model seed100 AUC% đúng; WBST LCC 86,08 cao hơn MAST86,03; CASIA MAST93,52 cao nhất.
- [ ] CE CASIA EER 16,54; MAST CASIA APCER1,87/BPCER51,11/ACER26,49; LCC MAST7,98/46,26/27,12.
- [ ] Baseline partial 3seed thực tế; không spectral mean3seed giả; SD dùng raw ddof1.
- [ ] Runtime config .2/.2/1.0, IoU .3, window5/min3, recognition async and cache semantics.
- [ ] LCC exposure, CASIA extracted overlap, CelebA Test-as-Val minh bạch; context literature khác protocol.
- [ ] Không có unsupported 2,22ms/67,98ms/FPS/cache-fit/ONNX size, claim real-time/ISO/3D mask.
- [ ] Không có results 10K/100K/micro trong main hoặc result appendix; không có placeholder figure/blank attribution.
- [ ] DG20 response 46 ID hiện trạng rõ: already addressed vs docs correction vs new experiment.
- [ ] Article cũ nguyên trạng trừ khi autor duyệt delta; adaptive threshold ngoài phạm vi.
