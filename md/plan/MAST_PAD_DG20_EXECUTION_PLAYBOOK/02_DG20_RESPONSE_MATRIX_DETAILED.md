# Phản hồi DG20 theo từng hạng mục — bản tích hợp chi tiết v2

**Trạng thái:** dự thảo để tác giả duyệt, KHÔNG có nghĩa Word đã sửa hoặc thí nghiệm mới hoàn thành. Tài liệu này kết hợp bản phản hồi trước, matrix đủ D01–D46 do Codex audit và trách nhiệm sửa theo stage mới. Nguồn chính: DG20 gốc (16/10/2026), Word `(2)`, article 8 trang, audit `00–08` (22/10). **Không bao gồm Adaptive Threshold.**

## Cách dùng và quy tắc phản hồi

- Phần I là thư phản hồi có thể biên tập để gửi giảng viên, **chỉ gửi sau khi S07 hoàn tất**. Phần II là checklist từng ID dùng nội bộ; trạng thái audit phản ánh tài liệu tại 22/10, không tự nâng lên DONE sau khi viết kế hoạch.
- Chỉ ghi **“đã”** cho source/analysis verified, paper hiện có; ghi **“sẽ/đề xuất”** cho sửa Word và experiments chưa chạy.
- Các so sánh literature dùng đúng protocol; dùng AUC (%) và chênh lệch pp; không claim causal Harmful Gate khi sigma còn khác.
- Sau mỗi stage S01–S07, cập nhật cột `status_after_stage` trong **bản copy** ledger, không sửa DG20 gốc/coverage audit. D01–D46 phải có evidence/file/page mới hoặc chỉ rõ còn thiếu.

## I. Văn bản phản hồi tổng hợp để nhóm chỉnh và duyệt

## A. Lời phản hồi tổng quát đề nghị gửi giảng viên

> Nhóm xin cảm ơn những nhận xét về tính mới, cấu trúc bài viết, đối chứng Harmful Gate, giao thức đánh giá và hiệu năng triển khai. Nhóm đã rà soát lại mã nguồn cùng các tập dự đoán để đối chiếu công thức, đơn vị AUC/EER và các bộ dữ liệu. Bản article mới tập trung riêng vào MAST-PAD, còn báo cáo học phần sẽ được tổ chức lại thành ba chương và giữ phần pipeline xác thực sáu giai đoạn ở mức phù hợp. Nhóm sẽ chuẩn hóa các đối chứng CE-only, Clean + Aux, WBST và MAST-PAD, điều chỉnh những khẳng định vượt quá bằng chứng và bổ sung phân tích có thể tái lập từ artifacts. Dự kiến tuần kế tiếp khi khôi phục quota Kaggle, nhóm hoàn thiện ba seed cho cả bốn chiến lược; đồng thời xác định rõ đây là kiểm tra độ ổn định của các cấu hình, **không** phải ablation tách riêng ảnh hưởng của harmful gate. Những yêu cầu về gate-only controls và benchmark độc lập được đặt thành kế hoạch thực nghiệm bổ sung, chưa được coi là đã hoàn thành.

## B. Ma trận phản hồi theo cấu trúc DG20

| Nhóm nhận xét của DG20 | Tình trạng bằng chứng hiện có | Hướng xử lý/đáp lời nhóm | Mức hoàn thành kỳ vọng |
|---|---|---|---|
| **1. Tách paper nghiên cứu khỏi mô tả 6 bước** (D01,D45) | **Đã đáp ứng trong article**: paper8trang chỉ về PAD; Word vẫn5chương và chứa hệ 6 bước | Giữ article PAD-only; chuyển Word thành **3 chương**, pipeline rõ trong Ch2, MAST-PAD là trọng tâm | Word: sẽ sửa; article: giữ |
| **2. Câu hỏi trung tâm và giảm inference cost** (D02,D03) | Mô tả selector/gate/loss và 434.434 PAD params được xác minh; chưa có benchmark device | Trình bày lợi ích kiến trúc: không có FFT/selection ở inference, **không suy ra tốc độ đo được** | Đã có cơ sở; sửa claim Word |
| **3. Rút vòng chạy và bảng cuối ngắn** (D04,D05) | Word còn lịch sử micro/10K/100K ở thân và phụ lục; article bảng4model chuẩn | Bỏ các kết quả screening khỏi bản nộp, chỉ 4 final configs full-source; bảng AUC% | Sửa văn bản ngay |
| **4. Nguồn gốc WBST, novelty và liên hệ phổ** (D06–D08) | WBST là đối chứng nội bộ worst-band; method mới kết hợp 3-band+aligned-margin+harmful gate+H normalization; literature trong article đầy đủ hơn | Bảng nghiên cứu liên quan có advantages/limitations; không gán WBST cho paper chưa có; sửa AENet/MADDG/citations | Sửa Word; đối chiếu metadata |
| **5. Literature protocol, AUC/params** (D09–D11) | Article TableIII/IV đã nói source/protocol khác nhau, không dùng AUC/param như chuẩn | Giữ contextual comparison; trong Word bỏ claim matched winner, throughput/latency superiority | Sửa Word ngay |
| **6. Ablation: random gate cùng tỷ lệ** (D12) | **Chưa có matched experiment**; harmful epoch11 ~83,21% | Đề xuất random-H-per-batch với đúng H từ gate, giữ mọi yếu tố và forward B tương đương | Cần train mới; chưa khắc phục |
| **7. Ablation: same-sigma no-gate** (D13) | WBST sigma1/18; MAST sigma0,10 ⇒ chưa cô lập gate | Train no-gate sigma0,10 cùng recipe với MAST; random gate là đối chứng tiếp theo | Cần train mới |
| **8. Gate-margin threshold / single bands** (D14,D15) | Có mask/bands trong code, chưa có final matched control | Đăng ký trước δ-grid `m_clean−m_worst>δ` và one-band/leave-one-band, không tune trên target | Tùy quota, cần train mới |
| **9. CE-only tách auxiliary** (D16) | Có baseline CE-only; CE-only/Clean+Aux seed100 cùng source/crop, baseline seeds101/102 đã xác minh một số target | Giữ bốn đối chứng, nêu sự khác biệt aux, không gọi CE-only gate ablation | Đã có seed100; Word cần sửa wording |
| **10. Loss normalization, stop-gradient, crop/spectral** (D17–D19) | Article đã trình bày; code xác nhận H=0, detached selection, whole selected-batch train BN, shared gain | Sửa Word T15, head128→11, `L_harmful`, sigma; thêm H=0 và BN qualifier | Sửa Word ngay |
| **11. MAST ≈WBST LCC; CASIA gain cần lặp lại** (D20,D21) | LCC86,03 vs WBST86,08; CASIA93,52 vs92,23 cho seed100, hai config khác sigma+gate | Giữ diễn giải balanced; tuần sau chạy hoàn tất 101/102 và đánh giá LCC+CASIA với unit đồng nhất | Phần ổn định chờ chạy; không claim causal gate |
| **12. CI theo người/video** (D22) | Audit đã 2000 paired-video bootstrap trên358 reconstructed videos; MAST CI90,76–95,90%; MAST−WBST ΔCI [−0,0847,+2,8193]pp | Có thể đưa appendix/thảo luận như **video-resampling conditional CI**; chưa kiểm subject independence | Phân tích offline đã có, chờ biên tập |
| **13. Multi-seed** (D23) | CE-only source/LCC/CASIA3seed; Clean+Aux source/LCC3seed; Clean CASIA101/102 chưa có; spectral chỉ seed100 hoàn tất | **Cam kết kế hoạch tuần sau:** 3 seeds 100/101/102 cho cả4 đối chứng; WBST101/102 xác minh interrupt/resume, MAST101/102 run mới; CASIA target eval toàn bộ | Chờ quota/hoàn thành; không nói đã có |
| **14. PAD latency vs whole pipeline** (D24,D25) | Không tìm được trace định danh cho2,22ms/67,98ms/14,7FPS; 0,08162GFLOPs là Conv/Linear PAD-only | Bỏ các latency khỏi khẳng định chính nếu chưa xuất được measurement log; ghi rõ provider/hardware khi đo mới | Sửa Word ngay; benchmark tùy chọn |
| **15. CelebA Test-as-Val, CASIA overlap** (D26,D27) | Article mô tả; CE source official Test dùng chọn checkpoint; CASIA reconstructed overlap20subject IDs/240videoIDs/1403frameIDs | Chuyển protocol vào Word, không gọi independent source test hoặc official subject-disjoint CASIA | Sửa văn bản ngay |
| **16. Manifest người/video; độc lập mới** (D28,D29) | Có manifest video_id/frame_index derived từ filenames, chưa có official independent holdout | Công bố quy trình dedup/coverage; thiết kế benchmark subject/session-disjoint riêng khi có data | Manifest offline có; independent test vẫn mở |
| **17. LCC exposure khi chọn crop/sigma** (D30,D31,D43) | Notebooks11–14,17 cho thấy target endpoint đã được xem; không có decision log chứng minh target-blind | Chủ động nói LCC được dùng trong phát triển thiết kế; đánh giá kết quả trên pool quan sát, không gọi confirmatory unseen; test mới phải được freeze | Sửa Word/article nếu thiếu qualifier |
| **18. Source threshold, APCER/BPCER** (D32,D33) | Có locked threshold riêng per run và rate CSV; MAST CASIA rate còn trống Word | Điền LCC7,98/46,26/27,12 và CASIA1,87/51,11/26,49, nêu threshold source cố định | Sửa Word ngay |
| **19. Attack-type / 3D mask** (D34,D35) | Audit xuất `casia_attack_breakdown.csv`; không có evidence matched 3D mask target | Chỉ phân tích loại thực sự có metadata; không nói robust với3D mask/deepfake; giới hạn phạm vi | Offline một phần, test mới còn thiếu |
| **20. End-to-end identity security** (D36,D46) | Runtime6stage implemented, chưa có identity FAR/FRR or session benchmark | Word chỉ claim prototype pipeline, không quy PAD AUC thành accuracy xác thực danh tính; nếu claim production cần benchmark | Sửa claim ngay, benchmark để sau |
| **21. Typo/heading/AUC/citation/ISO** (D37–D42) | Duplicate4.4.1 và missing99,21 cũ đã sửa, Word còn heading lỗi3.2/3.12; AENet gán nhầm, SD và units khác | Không sửa lỗi không còn; xử lý numbering, EER16,54, source484561, AUC%, pp, reference primary/DOI, ISO chính xác phiên bản | Biên tập Word ngay |
| **22. Khẳng định stability hoặc không false alarms** (D44) | Temporal-vote logic trong code có capacity5/min3; không có measured error-free guarantee | Thay claim tuyệt đối bằng đặc tả cơ chế và phạm vi kiểm chứng; seed dispersion chỉ cho nơi đủ kết quả | Sửa Word ngay |

Đối chiếu chi tiết từng mã D01–D46 có tại `01_DG20_COVERAGE_MATRIX.md`; bảng trên gộp theo **ý của DG20** để làm văn bản phản hồi đọc được, không giả định mọi hàng đã được reviewer nghiệm thu.

## C. Kết quả hiện có và cách trả lời chính xác

### C1. Bảng headline seed100 — AUC (%), đủ4 chiến lược

| Model | CelebA Test-as-Val | LCC Combined image | CASIA Combined reconstructed video |
|---|---:|---:|---:|
| CE-only |99,21|84,40|89,83|
| Clean + Aux |99,07|84,60|92,04|
| WBST |99,22|**86,08**|92,23|
| MAST-PAD |**99,23**|86,03|**93,52**|

Trình bày kết luận: **“Trong một seed, MAST-PAD cải thiện cả hai target so với CE-only và Clean + Aux; trên CASIA đạt điểm cao nhất trong bốn chiến lược, còn trên LCC WBST nhỉnh hơn 0,0559 điểm phần trăm. Kết quả so với WBST phản ánh thay đổi cấu hình cả sigma lẫn gating.”**

### C2. Chỉ số điểm vận hành được phép đưa vào Word

- Từ cùng scores và **source-frozen threshold**, MAST-PAD LCC Combined: EER22,06%, APCER7,98%, BPCER46,26%, ACER27,12%; CASIA Combined reconstructed video: EER16,92%, APCER1,87%, BPCER51,11%, ACER26,49%.
- CE-only CASIA EER đúng là **16,54%**, không phải23,05% của Word. AUC và EER không thay chỉ bởi đổi fixed threshold nếu score/population giữ nguyên; thay threshold chỉ ảnh hưởng operating metrics.
- CASIA reconstructed video n=358 từ image extraction có dedup; frame AUC và video AUC không phải phương pháp calibration khác nhau. Summary ba seed có conflict nhãn, phải sửa nguồn tổng hợp trước khi diễn giải.

### C3. Những bằng chứng offline bổ sung

- Video-resampling CI từ audit (2000 bootstrap) cho MAST CASIA93,52% là **90,76–95,90%**; MAST−WBST CI **[−0,0847,+2,8193] pp**, khoảng qua0. Đây là bất định trên reconstructed-video population với checkpoint cố định, **không phải** run-to-run/training-seed CI và chưa loại phụ thuộc các video cùng subject.
- CASIA `attack_type` đã xuất theo method; nhóm phải thẩm định ánh xạ loại/tổng N trước khi đưa biểu đồ. Manifest dedup/video_id và LCC split-level đã có. Không dùng các kết quả này thay cho independent domain test mới.
- AUC baseline n=3 đã xác minh ở scope: CE-only LCC **83,6818±0,6267%**, CASIA video **89,5688±1,9431%**; Clean+Aux LCC **85,0980±0,4342%** (sample SD). Clean CASIA còn thiếu101/102. Không có mean spectral n=3.

## D. Những yêu cầu chưa khắc phục và cam kết theo mốc

| Việc chưa đủ | Nhóm đề xuất làm | Khi nào có thể cập nhật phản hồi |
|---|---|---|
| **Repeat 3 seeds cho4 đối chứng** | Có verified baselines, hoàn tất WBST/MAST101/102 và CASIA Clean+Aux101/102; đối chiếu CSV+manifest+checkpoint | **Tuần kế tiếp (dự kiến từ 26/10/2026)** khi quota Kaggle khôi phục; không hứa xong trước khi artifact xác minh |
| **Same-sigma gate ablation** | No-gate σ0,10, random-H gate σ0,10, sau đó δ thresholds/per-band nếu budget | Sau hoàn tất4-method seeds hoặc khi bổ sung quota theo ưu tiên |
| **Evaluation độc lập** | Thiết kế/freeze một target holdout mới với subject/session/video manifest, không dùng LCC/CASIA Combined như test độc lập giả định | Chờ dataset/giám sát thí nghiệm, không thay bằng chỉnh văn bản |
| **Latency camera/end-to-end** | Thu log model SHA/device/provider/p95 + worker/session, hoặc bỏ các claim | Không cần chờ nếu chọn bỏ claim |
| **Phân loại 3D mask/identity security** | Chỉ đưa vào khi có target data và end-to-end session benchmark | Công việc mở rộng, không tuyên bố hoàn thành trong báo cáo hiện tại |

**Cần tránh cam kết sai:** đủ ba seed của bốn chiến lược chỉ đánh giá ổn định bốn cấu hình; không thể chứng minh harmful gate là nguyên nhân riêng biệt nếu WBST và MAST khác sigma. Kết quả bootstrap hiện có cũng không thay thế nhiều training seeds hoặc target độc lập.

## E. Câu phản hồi mẫu từng vấn đề lớn (có thể đưa vào bản gửi giảng viên sau khi thực hiện)

**Về tính mới và Harmful Gate:**
> Nhóm xác định đóng góp của MAST-PAD là phối hợp lựa chọn worst-band theo lề nhãn, harmful gate đã tách gradient và chuẩn hóa loss theo số mẫu harmful, trong một kiến trúc PAD suy luận không thêm nhánh phổ. Nhóm đã trình bày công thức/thuật toán để tái lập và thừa nhận rằng kết quả hiện có đánh giá cấu hình hoàn chỉnh. Nhóm dự kiến kiểm chứng riêng gate bằng same-sigma no-gate và random-rate gate nếu tài nguyên cho phép.

**Về đánh giá đa miền và giao thức:**
> Nhóm đã mô tả lại CelebA Test-as-Val, LCC pooled images và video CASIA tái dựng từ frame trích xuất, cùng quy tắc dedup/coverage. Việc không huấn luyện trên LCC không đồng nghĩa target-blind hoàn toàn vì một số quyết định thiết kế trước đây tham khảo kết quả LCC. Vì vậy nhóm không coi benchmark hiện tại là phép đánh giá độc lập mới, và định hướng đánh giá thêm trên holdout được khóa trước khi xem dữ liệu.

**Về chi phí suy luận:**
> Các số đo hiện xác minh được thuộc graph PAD: 434.434 tham số, 40.810.892 MACs và xấp xỉ 0,08162 GFLOPs theo quy ước Conv/Linear. Chúng không phải thời gian xử lý thực đo của webcam hay pipeline xác thực. Những số latency chưa kèm log phù hợp sẽ không được sử dụng làm chứng cứ về tốc độ thiết bị.

**Về repeatability:**
> Hai đối chứng clean đã có dữ liệu đa seed một phần hoặc đầy đủ; nhóm dự kiến hoàn tất các seed còn thiếu cho WBST và MAST-PAD, cùng đánh giá CASIA của Clean+Aux, sau đó tổng hợp mean ± sample SD trên ba seed được xác minh và cùng protocol. Nhóm sẽ báo cáo cả per-seed values, không chỉ chọn checkpoint/seed cho kết quả đẹp nhất.

## F. Quy trình sử dụng bản phản hồi

1. Đây là **dự thảo chưa gửi**; khi sửa Word xong, đổi từng câu “nhóm sẽ” thành “nhóm đã” **chỉ khi** có artifact/đoạn sửa thật.
2. Đính kèm bảng **DG20 item → sửa trong Word/Article → minh chứng** với locator sau revision; không gửi hash nội bộ dài nếu giảng viên không yêu cầu.
3. Các mục experiment mới chưa chạy vẫn để trạng thái **planned/remaining**, không tự đánh dấu resolved.
4. Quy tắc 3 chương và ≤50 trang áp dụng cho **Word**; article giữ cấu trúc IEEE.
5. Adaptive Threshold nằm **ngoài phạm vi phản hồi triển khai và kết quả hiện tại**; không nêu chi tiết thiết kế/trạng thái ở bản gửi nếu không được hỏi.

## G. Nguồn

DG20 gốc 16/10, MD đánh giá20/10, Word(2), article8trang, audit `00_EXECUTIVE_VERDICT.md`, `01_DG20_COVERAGE_MATRIX.md` (D01–D46), `02_CODE_AND_METHOD_EVIDENCE.md`, `03_METRIC_AND_DATA_AUDIT.md`, `04_COMPLEXITY_AND_SYSTEM_EVIDENCE.md`, `05_PROPOSED_WORD_ARTICLE_EDITS.md`, `06_OPEN_ITEMS_AND_NEXT_WEEK.md`, `07_EVIDENCE_INDEX.md`. Thuật ngữ **Đã có bằng chứng** ≠ **đã sửa bản Word**; phương án thực nghiệm ≠ kết quả đã thực hiện.


## II. Ma trận D01–D46 nguyên trạng audit và cách xử lý trong stage

**Lưu ý:** trạng thái `RESOLVED_IN_WORD`/`RESOLVED_IN_ARTICLE` bên dưới là **kết luận của audit**, không có nghĩa nhà trường chấp nhận hoặc toàn bộ research issue đã đóng. `Pnnnn` và `Tn` không phải số trang vật lý. Bản gốc nguyên trạng được nhúng dưới đây để không bỏ sót nhận xét.

| ID | DG20 locator | Yêu cầu | Word/article locator | Phân loại | Hành động | Chứng cứ |
| --- | --- | --- | --- | --- | --- | --- |
| D01 | P0007 | Tách bài báo PAD khỏi hệ sáu giai đoạn | Word Ch2; article I/III/IV | RESOLVED_IN_ARTICLE | Giữ article PAD-only; Word giữ hệ thống ở Ch2 mới | E01/E03; E08 app.py:650–725 |
| D02 | P0007 | Làm rõ đóng góp harmful gate | Word §3.4 P0654–0684; article III | RESOLVED_IN_ARTICLE | Định vị tổ hợp selector+gate+loss; không gọi hiệu ứng gate đã tách | E05 CELL2 training_terms; E03 III |
| D03 | P0007 | Tách chi phí training khỏi inference | Word P0618/T22; article III-E/V-F | RESOLVED_IN_ARTICLE | Giữ graph PAD gọn; bỏ suy luận latency từ graph | E12 shared_clean_inference |
| D04 | P0007 | Rút lịch sử các vòng thử nghiệm | Word §4.2 P0820–0847; T23 | DOCUMENT_EDIT_ONLY | CUT kết quả 10K/100K/micro cả phụ lục theo phạm vi hiện tại | E01; prompt constraints |
| D05 | P0007 | Bảng thực nghiệm chính ngắn, rõ | Word T17/T18; article Table II | RESOLVED_IN_WORD | T17 đủ bốn phương pháp; đổi AUC sang % và bỏ run code | E10/E11 recomputed_metrics.csv |
| D06 | P0009 | Xác minh WBST là phương pháp nào | Word §3.5/T15; article III-E | DOCUMENT_EDIT_ONLY | Gọi đối chứng nội bộ worst-band spectral training; chưa có nguồn bài báo riêng | E05 RUNS/CELL1; E16 reference_lock |
| D07 | P0009 | Truy nguồn thành phần phổ và mức mới | Word Ch3; article II/III | RESOLVED_IN_ARTICLE | Giữ đối chiếu spectral precedent; không first-ever/novel backbone | E16; E03 II |
| D08 | P0009 | Metadata citation chính xác | Word [1]/[4]/[6]/[7]/[10]/[12] | DOCUMENT_EDIT_ONLY | Sửa AENet/MADDG, đối chiếu tác giả/trang từ PDF; không đoán DOI | E17 local_reference_openings; E16 |
| D09 | P0009 | Không xếp hạng AENet giữa giao thức khác | Word T19/P0967; article Table III | RESOLVED_IN_ARTICLE | Word thêm nguồn benchmark, bỏ suy luận tốc độ | E05 CELL6 REFERENCE_AUDIT; E03 V-D |
| D10 | P0009 | Không xếp hạng PatchNet/GAC giữa giao thức khác | Word T20/P1008; article Table IV | RESOLVED_IN_WORD | Giữ câu không kết luận thứ hạng; %/protocol nhất quán | E01 P1008; E03 V-E |
| D11 | P0009 | Không dùng AUC/#params như metric chuẩn | Word §4.4/P0967; article V-D | RESOLVED_IN_ARTICLE | Giữ AUC và số tham số riêng, không suy matched superiority | E03 V-D; E12 |
| D12 | P0011 | Random gate cùng tỷ lệ kích hoạt | Word P1014; article V-F | NEEDS_MISSING_EVIDENCE_OR_NEW_EXPERIMENT | Thiết kế random subset cùng H trên cùng sigma; chưa chạy | E05 RUNS chỉ CLEAN/WORST/HARMFUL |
| D13 | P0011 | No-gate cùng sigma | Word T15/P0881; article III-E/V-F | NEEDS_MISSING_EVIDENCE_OR_NEW_EXPERIMENT | MAST sigma .1 với gate off; WBST hiện .05556 không đủ | E05 CELL1 RUNS |
| D14 | P0011 | Độ nhạy threshold của harmful gate | Word §3.4; article III-C | NEEDS_MISSING_EVIDENCE_OR_NEW_EXPERIMENT | Ngưỡng độ giảm margin trong training, không phải operating threshold; predeclare | E05 training_terms strict inequality |
| D15 | P0011 | Hiệu quả từng band | Word P1017; article V-C | NEEDS_MISSING_EVIDENCE_OR_NEW_EXPERIMENT | Band frequency có sẵn; cần single-band/remove-band controls | E13 epoch11; chưa có matched final controls |
| D16 | P0011 | CE-only để tách Laux | Word T17/T18/P0847; article Table II | RESOLVED_IN_WORD | Có đối chứng CE-only; chỉ mô tả chênh lệch quan sát, không gate causality | E06 CE-only; E10/E11 |
| D17 | P0011 | Công bố loss normalization | Word P0667–0680; article Eq9 | RESOLVED_IN_ARTICLE | Word thêm H=0 và giữ hệ số clean .75 | E05 training_terms |
| D18 | P0011 | Stop-gradient và chọn view | Word P0658; article III-B/C | RESOLVED_IN_WORD | Bổ sung gate.detach, full selected-batch train forward | E05 select_worst_view/training_terms |
| D19 | P0011 | Crop/spectral settings tái lập | Word §3.3/T16; article III-A/Table I | RESOLVED_IN_ARTICLE | Word ghi radial normalization, gain shared, sigma từng method | E05 CELL1/2 spectral_views |
| D20 | P0013 | LCC gần WBST không chứng minh gate | Word P0881; article V-A/V-F | RESOLVED_IN_ARTICLE | Sửa câu thêm seed để cô lập gate; phải cùng sigma | E10; E05 RUNS |
| D21 | P0013 | Lặp lại lợi ích CASIA | Word T18; article V-F | NEEDS_MISSING_EVIDENCE_OR_NEW_EXPERIMENT | Cần WBST/MAST seeds101/102, thêm CASIA Clean+Aux101/102; CE-only đã verified đủ | E18 summary §1; seed_config_inventory |
| D22 | P0013 | CI theo subject/video | Word §5.2; article V-F | OFFLINE_REANALYSIS_POSSIBLE | Đã tính paired video bootstrap; còn giới hạn tương quan subject | paired_video_bootstrap.json; E11 video_id |
| D23 | P0013 | Nhiều training seeds | Word T18/P1078/P1085 | NEEDS_MISSING_EVIDENCE_OR_NEW_EXPERIMENT | CE/Clean source/LCC3seed đã AVAILABLE_VERIFIED; CE CASIA3seed verified; spectral chưa đủ | E18 vs E20 added_seed_validation.json; spectral E10 config.train_seed |
| D24 | P0013 | Phân biệt 2.22ms PAD và 67.98ms pipeline | Word P1042/P1067; article V-F | DOCUMENT_EDIT_ONLY | Bỏ số đo khỏi bản nộp nếu chưa có trace; không thay bằng estimate | E12 không đo latency; E08 async worker |
| D25 | P0013 | Đúng CPU/GPU/provider, không ngoại suy edge | Word §4.6/P1069–1071 | NEEDS_MISSING_EVIDENCE_OR_NEW_EXPERIMENT | Cần log benchmark đúng model/hash/provider/hardware | E08 embedder.py CPU provider; E12 profiler scope |
| D26 | P0015 | CelebA Test-as-Val và identity overlap | Word P1080; article IV-A/V-F | RESOLVED_IN_ARTICLE | Word không gọi untouched source test; ghi 3 overlap IDs | E14 dataset_counts.json |
| D27 | P0015 | CASIA không verified subject-disjoint | Word P1079; article IV-C/V-F | RESOLVED_IN_ARTICLE | Ghi reconstructed/overlap; giữ denominator 358/360 | E15 evaluation_protocol.json |
| D28 | P0015 | Manifest subject/video kiểm tra được | Word §1.3/4.1; article IV | OFFLINE_REANALYSIS_POSSIBLE | Đã xác nhận CSV IDs/pairing; cung cấp manifest cho giao thức mới | E10 manifests; E11 video_id/frame_index |
| D29 | P0015 | Test độc lập mới/giao thức sạch | Word §5.3; article V-F | NEEDS_MISSING_EVIDENCE_OR_NEW_EXPERIMENT | Khóa thiết kế rồi đánh giá domain holdout mới | E15 overlap; E19 history |
| D30 | P0015 | LCC-informed development/exposure | Word P1075/P1080; article IV-B | DOCUMENT_EDIT_ONLY | Bỏ “LCC độc lập hoàn toàn”; công bố exposure ledger | E19 CELL0 notebooks11–18; E05 CELL6 |
| D31 | P0015 | Không gradient target không đồng nghĩa unseen | Word P1080; article IV-B | RESOLVED_IN_ARTICLE | Giữ qualifier article; sửa Word tương ứng | E03 IV-B; E19 |
| D32 | P0015 | Ngưỡng dev cố định khi đánh giá target | Word T21; article IV-D | RESOLVED_IN_WORD | Giữ source-frozen; tách cấu hình runtime CASIA-fit khỏi headline | E10 locked_threshold; E09 fixed contract |
| D33 | P0015 | APCER/BPCER đầy đủ | Word T21 P1032/1033; article không headline rates | OFFLINE_REANALYSIS_POSSIBLE | Đã đếm 5/268 attack accepted, 46/90 real rejected CASIA MAST | recomputed_metrics.csv; E11 |
| D34 | P0015 | Breakdown theo kiểu attack | Word P1037–1039; article V-F | OFFLINE_REANALYSIS_POSSIBLE | CASIA attack_type sẵn; LCC cần metadata category mapping | casia_attack_breakdown.csv; E11 |
| D35 | P0015 | Không suy robustness 3D mask | Word P1083; article PAD scope | RESOLVED_IN_WORD | Giữ giới hạn, không dùng hình taxonomy thành chứng cứ | E01 P1083; E11 attack_type |
| D36 | P0015 | Không suy end-to-end authentication từ PAD | Word P1084; article I/V-F | RESOLVED_IN_WORD | Giữ scope; cần end-to-end benchmark riêng cho claim hệ thống | E08; không có identity evaluation trace |
| D37 | P0017 | Sửa đánh số 4.4.1 bị lặp | Word P0942; TOC P0099 | RESOLVED_IN_WORD | Không còn lặp heading 4.4.1 trong body; sửa 3.2/3.12/2.3.4 còn lỗi | word_paragraphs.json |
| D38 | P0017 | Điền giá trị thiếu trước 99.21% | Word P0847 | RESOLVED_IN_WORD | Đã có .9921 (99.21%); không chép lỗi bản cũ sang mới | E01 P0847 |
| D39 | P0017 | Thống nhất fraction/percent và 484561/485K | Word T17/T18/T23; article Table II | DOCUMENT_EDIT_ONLY | AUC%; delta pp; ghi đúng 484561 usable source examples | E14; E10 recomputation |
| D40 | P0017 | Cache/hardware lập luận phải profiling | Word P1070/1071 | DOCUMENT_EDIT_ONLY | Bỏ cache-fit khẳng định và tương đương ONNX/raw weights | E12 chỉ Conv/Linear arithmetic |
| D41 | P0017 | DOI/URL đủ và chính xác | Word bibliography P1092–1113; article references | DOCUMENT_EDIT_ONLY | Đối chiếu từng entry; giữ unknown nếu chưa có primary source | E16/E17; không invent DOI |
| D42 | P0017 | Đúng phiên bản ISO, không compliance claim | Word P1110 và §4.1.2 | DOCUMENT_EDIT_ONLY | Nêu đúng phiên bản đã dùng; chưa có tài liệu ISO local để certify | E01 [19]; NOT_FOUND ISO normative source |
| D43 | P0019 | Ưu tiên protocol sạch/identity/LCC disclosure | Word §1.3/§5.2; article IV/V-F | NEEDS_MISSING_EVIDENCE_OR_NEW_EXPERIMENT | Disclosure sửa ngay; protocol sạch vẫn cần evidence mới | D26–D31 |
| D44 | P0020 | Không tuyên bố stability trước seeds/gate/CI | Word P0483/P0881; article V-F | DOCUMENT_EDIT_ONLY | Bỏ “loại bỏ hoàn toàn False Alarm”; tách 3 loại kiểm chứng | D12–D23; E08 |
| D45 | P0021 | Bài báo ngắn gọn riêng MAST-PAD | Word Ch2; article I–VI | RESOLVED_IN_ARTICLE | Không viết lại paper; giữ hệ sáu-stage trong Word | E03 |
| D46 | P0021 | Giao thức vận hành/end-to-end nếu claim xác thực | Word §2.6/§4.6/P1084 | NEEDS_MISSING_EVIDENCE_OR_NEW_EXPERIMENT | Benchmark subject/session, identity errors, temporal outcomes, latency | E08 implemented ≠ measured |
| X01 | User/prompt exclusion | Adaptive threshold đang phát triển | Không audit/không đưa vào Word/article | OUT_OF_SCOPE (adaptive threshold only) | Không đọc/đánh giá thiết kế hoặc kết quả của thành phần này | Ràng buộc người dùng, không phải một yêu cầu DG20 mới |


## III. Bản phân công stage đề xuất và phép thử thành công

| Dải DG20 | Stage trước tiên | Bằng chứng phải đưa vào báo cáo phản hồi | Trạng thái khoa học sau sửa Word |
|---|---|---|---|
| D01–D05 | S01/S02/S05 | Word 3 chương; article PAD-only 8tr; 4-model table | Editorial fix có thể hoàn tất |
| D06–D11 | S01/S06 | Table ưu/nhược với verified references; AENet provenance, WBST internal | Literature positioning vẫn phụ thuộc phạm vi các nguồn đã đọc |
| D12–D15 | S10 và experiment plan | same-sigma gate-off, random-H, δ-training-gate, per-band preregistration | **Không hoàn tất nếu chưa có new runs** |
| D16–D20 | S03/S05 | Eq exact, H=0, detached gate/full-B, sigma conflict | Định nghĩa method chính xác; không causal effect riêng gate |
| D21–D23 | S04/S05/S09 | CE 3 seeds, Clean source/LCC 3, spectral pending; video bootstrap | **Chờ 4 spectral completions + Clean CASIA eval** |
| D24–D25 | S02/S05 | 434434/40.81M/0.08162G; không có latency trace | Không claim realtime; benchmark thật chưa có |
| D26–D31 | S04/S05 | CelebA Test-as-Val; LCC exposure; CASIA overlap reconstruction; manifest | Giao thức mô tả đúng; clean independent target vẫn thiếu |
| D32–D35 | S04/S05 | source-frozen EER/APCER/BPCER; attack type chỉ sau raw-mapping | Error metrics giải quyết; unseen attack coverage chưa có |
| D36,D46 | S02/S05 | Full pipeline code flow, recognition async; không có FAR/FRR | Không claim hệ thống đạt accuracy/security đo độc lập |
| D37–D42 | S01/S06 | heading/ISO/citation/units; lỗi đã sửa không sửa lại | Editorial response đầy đủ |
| D43–D45 | S07 | summary issue scope và không overclaim | Chờ tác giả nghiệm thu |

## IV. Văn bản ngắn đề xuất gửi giảng viên (sau nghiệm thu)

> Nhóm đã tiếp thu các nhận xét DG20 và thực hiện việc đối chiếu mã nguồn, bộ dữ liệu, kết quả dự đoán và tài liệu tham khảo. Phiên bản báo cáo học phần sau chỉnh sửa sẽ được tổ chức thành ba chương, trong đó Chương 1 tổng quan và đối chiếu ưu/nhược các nghiên cứu liên quan, Chương 2 trình bày hệ thống sáu giai đoạn và phương pháp MAST-PAD, Chương 3 trình bày giao thức, kết quả và đánh giá. Bài báo IEEE được giữ riêng như một nghiên cứu về PAD. Chúng tôi đã xác minh các công thức chọn một worst spectral view trên mỗi ảnh, harmful gate theo lề quyết định, loss chuẩn hóa trên số mẫu harmful và graph suy luận PAD 434.434 tham số, đồng thời xác minh lại số liệu AUC/EER và sửa các chỗ chưa thống nhất về nguồn đánh giá.
>
> Về kiểm chứng thực nghiệm, kết quả chung hiện được trình bày từ seed 100 cho bốn chiến lược, MAST-PAD đạt 86,03% AUC trên LCC Combined mức ảnh và 93,52% trên CASIA Combined mức video tái dựng. Đối chứng WBST nhỉnh hơn trên LCC 0,0559 điểm phần trăm, trong khi MAST-PAD hơn WBST 1,2935 điểm phần trăm trên CASIA. Hai cấu hình khác cả độ rộng Gaussian và gate nên chưa quy riêng mức tăng cho gate. Nhóm dự kiến hoàn tất đánh giá ba seed cho cả bốn chiến lược trong tuần kế tiếp khi có quota Kaggle; các đối chứng same-sigma và random-H cùng kiểm thử miền độc lập được tách thành hướng thí nghiệm bổ sung. Chúng tôi ghi rõ CelebA Test-as-Val, lịch sử sử dụng LCC khi phát triển và giới hạn dữ liệu CASIA tái dựng, đồng thời không tiếp tục sử dụng các tuyên bố latency/tuân thủ tiêu chuẩn chưa có log hoặc bằng chứng tương ứng.

**Không gửi thư ngắn này nguyên văn nếu Word cuối chưa sửa.** Sau S07 thay thì tương lai bằng việc đã làm thật, đính kèm bảng vị trí chương/trang và artifact khi giảng viên cần.
