# Full-scale model results — source of truth

Gộp tài liệu: **2026-10-20**. Snapshot C/P3/R7 gốc: **2026-10-12**. Phạm vi: ba checkpoint full-scale của thí nghiệm 18, đánh giá CASIA train/test/combined từ notebook 20 và báo cáo baseline CE-clean-only của notebook 21. Không trộn với kết quả 100K hoặc E1/E3 thuộc protocol khác. **Phạm vi seed: toàn bộ kết quả hiện tại là seed100; seed101 và seed102 sẽ được bổ sung sau, chưa thuộc phạm vi tổng hợp này.**

Tài liệu tổng hợp số đo đã có, phục vụ slide/paper/report. JSON/CSV per-run là bằng chứng gốc; bảng C/P3/R7 được tính/đối chiếu từ các nguồn đó theo snapshot gốc, không chọn metric nhỏ nhất giữa các ngưỡng khác nhau. Đơn vị, ngưỡng và population luôn đi cùng số liệu.

## 1. Nguồn và thứ tự ưu tiên

1. `runs/<run_id>/metrics/*.json`, predictions, config, completion và evaluation_freeze của thí nghiệm 18.
2. `runs/<run_id>/<train|test|combined>/*_metrics.json` và `*_metrics_casia_calibrated.json` của đánh giá CASIA notebook20.
3. Báo cáo tổng hợp không thay thế per-run artifacts. CE-only đã được kiểm tra riêng ngày **2026-10-20** từ output notebook21: 17 populations project và 5 populations reference khớp predictions CSV (tolerance 1e-8); checkpoint/manifest hashes local khớp freeze. Kiểm tra này độc lập với snapshot 51 populations C/P3/R7 trước đó.

**Lưu ý dữ liệu:** `fullscale_project_metrics.json` và `fullscale_reference_compatible_metrics.json` ở root hiện chỉ chứa C_FULL; tài liệu này lấy đủ ba model trực tiếp từ thư mục runs, tránh thiếu P3/R7 do ghép các phiên RUN_FILTER.

| Nguồn | Vai trò |
| --- | --- |
| [Experiment18 output](../../antispoof/notebooks/final/binary_crossdomain_v2/output/18_fullscale_C_P3SF_R7SC_crop15) | CelebA + LCC, training provenance |
| [Notebook20 CASIA output](../../antispoof/notebooks/final/binary_crossdomain_v2/output/casia_train_test_C_P3SF_R7SC_100k_fullscale) | Frame/video; source-frozen + train-calibrated |
| [dataset_counts.json](../../antispoof/notebooks/final/binary_crossdomain_v2/output/18_fullscale_C_P3SF_R7SC_crop15/dataset_counts.json) | Official candidates/usable/exclusions |
| [complexity_report.json](../../antispoof/notebooks/final/binary_crossdomain_v2/output/18_fullscale_C_P3SF_R7SC_crop15/complexity_report.json) | Parameters/MACs/FLOPs |
| [evaluation_protocol.json](../../antispoof/notebooks/final/binary_crossdomain_v2/output/casia_train_test_C_P3SF_R7SC_100k_fullscale/evaluation_protocol.json) | CASIA split/threshold/coverage protocol |

## 2. Model và loss

C không phải CE-only: run cũ vẫn có auxiliary loss. R7 là MAST-PAD; P3 là WBST. Cả ba dùng chung binary MiniFASNetV2 trunk/PAD head; spectral và auxiliary heads chỉ thuộc training, không thêm vào clean inference.

`L_aux = 0.1 CE_spoof_type + 0.1 CE_lighting + BCE_attributes(real samples)`; auxiliary loss lấy từ clean branch. `CE_harmful` là CE trên worst views có aligned margin thấp hơn clean, chuẩn hóa theo số harmful samples; bằng 0 khi không có harmful sample.

| Model | Run ID | Loss sau warmup | Warmup / eligibility | Status kết quả |
| --- | --- | --- | --- | --- |
| Base / CE_clean | CE_CLEAN_FULL_crop15 | CE_clean (weight 1; không CE_worst/L_aux) | Không warmup; eligible epoch1 | **Đã đối chiếu metric/predictions và checkpoint/manifest local** |
| C / Clean | C_FULL_crop15 | CE_clean + L_aux | Không warmup; eligible epoch1 | Đã có kết quả |
| P3_SF / WBST | P3_SF_FULL_crop15 | 0.75 CE_clean + 0.25 CE_worst + L_aux | Epoch1 clean+aux; eligible epoch2 | Đã có kết quả |
| R7_SC / MAST-PAD | R7_SC_FULL_crop15 | 0.75 CE_clean + 0.25 CE_harmful + L_aux | Epoch1 clean+aux; eligible epoch2 | Đã có kết quả |

### Cấu hình chung

- Full eligible official CelebA Train; không sampling. Pretrained MiniFASNetV2 trunk, binary head seed100.
- Crop **1.5×**, **80×80 BGR float [0,1]**, không runtime gamma/mean-std. Train augmentation và bbox jitter theo notebook18; evaluation không augmentation.
- SGD lr0.005, momentum0.9, weight_decay0.0005; batch256 AMP, seed100, DATA_SEED43.
- Max12 epochs, MultiStepLR milestones[4,8], gamma0.2. Phase-aware patience bắt đầu epoch9, patience3, earliest stop11. Checkpoint selection không bị trì hoãn: calibrated source ACER thấp nhất, AUC phá hòa.
- P3/R7 centers=[0.08333333333333331, 0.24999999999999997, 0.41666666666666663], gain[0.65,0.90], preserve DC; sigma P3=0.05555555555555555, R7=0.1.

| Model | Best epoch | Epoch cuối thực chạy | Source logit threshold |
| --- | --- | --- | --- |
| Base / CE_clean † | 9 | 12 | 4.5966796875 |
| C / Clean | 9 | 12 | 5.130859375 |
| P3_SF / WBST | 6 | 11 | 5.3037109375 |
| R7_SC / MAST-PAD | 7 | 11 | 4.66259765625 |

## 3. Quy ước đọc metric

- **† Base / CE_clean** là `CE_CLEAN_FULL_crop15`, run seed100 từ output notebook21. Metric trong bảng được đọc từ JSON và đối chiếu lại predictions CSV ngày 2026-10-20; các rate được đổi từ fraction sang %. Bảng full precision ở mục 10 giữ nguyên 17 dòng báo cáo ban đầu, khớp JSON gốc. Provenance và giới hạn về phiên bản notebook được ghi ở mục 10.
- AUC hiển thị dạng **fraction** (0–1), 8 chữ số sau dấu thập phân. Các rate/accuracy/coverage trong bảng hiển thị **%**, 4 chữ số sau dấu thập phân. JSON gốc/appendix giữ full precision; không lấy bảng làm đầu vào tính lại.
- Score = logit(real) − logit(attack); Real iff score ≥ threshold.
- APCER = attack được nhận là real; BPCER = real bị nhận là attack. Pooled binary **ACER = HTER = (APCER+BPCER)/2**.
- AUC/EER/TPR@FPR1% mô tả ROC; EER không phải operating threshold.
- Bảng chính dùng source-frozen threshold. CASIA train-calibrated được tách riêng, không thay thế ngầm baseline. CASIA video mean không tương đương frame: không đưa ngưỡng video vào runtime frame.

## 4. CelebA full Test-as-Val

**Full Test đã dùng để chọn checkpoint và hiệu chỉnh ngưỡng trong notebook18; đây không phải untouched independent Test.** Các số dưới đây là kết quả tại source-calibrated threshold, đúng thiết kế thí nghiệm. Không thay bằng argmax metric ở phần reference.

Candidates67170, scored66379; không scored791. Train candidates494405, usable484561; exclusions9844 (DET_FAIL4580, SUSPICIOUS779, TOO_SMALL4484, INVALID_BBOX1). Train/Test identity overlap: 5028, 7332, 9735; giữ official membership.

| Model | AUC | EER % | APCER % | BPCER % | ACER/HTER % | Accuracy % | TPR@FPR1% % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Base / CE_clean † | 0.99214398 | 4.4757 | 5.0736 | 3.8260 | 4.4498 | 95.2997 | 85.4483 |
| C / Clean | 0.99073553 | 4.6764 | 5.3381 | 3.9116 | 4.6248 | 95.0888 | 82.6608 |
| P3_SF / WBST | 0.99224181 | 4.3924 | 4.6974 | 4.0123 | 4.3548 | 95.5076 | 86.4869 |
| R7_SC / MAST-PAD | 0.99228027 | 4.3519 | 4.4781 | 4.2137 | 4.3459 | 95.6010 | 87.1541 |

## 5. LCC full — source-frozen threshold

Evaluate toàn bộ physical training/development/evaluation; combined là pooled union, không infer lại hoặc fit threshold trên LCC. Đơn vị: image. Threshold giữ nguyên từ CelebA source. Coverage nằm trong từng metric JSON.

### LCC training

| Model | AUC | EER % | APCER % | BPCER % | ACER/HTER % | Accuracy % | TPR@FPR1% % | Scored / candidates | Coverage % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Base / CE_clean † | 0.87025207 | 22.5323 | 5.6366 | 50.9002 | 28.2684 | 87.6856 | 24.7954 | 8283 / 8299 | 99.8072 |
| C / Clean | 0.86922397 | 21.3434 | 6.4014 | 49.4272 | 27.9143 | 87.2510 | 15.5164 | 8283 / 8299 | 99.8072 |
| P3_SF / WBST | 0.88126169 | 19.9610 | 6.2881 | 46.9722 | 26.6301 | 87.7098 | 26.0728 | 8283 / 8299 | 99.8072 |
| R7_SC / MAST-PAD | 0.87748892 | 20.3780 | 5.9340 | 48.3633 | 27.1487 | 87.8064 | 22.2586 | 8283 / 8299 | 99.8072 |

### LCC development

| Model | AUC | EER % | APCER % | BPCER % | ACER/HTER % | Accuracy % | TPR@FPR1% % | Scored / candidates | Coverage % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Base / CE_clean † | 0.81761469 | 26.9698 | 9.7954 | 55.5556 | 32.6755 | 83.9158 | 16.2963 | 2947 / 2948 | 99.9661 |
| C / Clean | 0.85159202 | 23.4514 | 11.9197 | 40.0000 | 25.9599 | 84.2212 | 22.3259 | 2947 / 2948 | 99.9661 |
| P3_SF / WBST | 0.85283776 | 22.4659 | 9.5594 | 42.2222 | 25.8908 | 85.9518 | 26.1728 | 2947 / 2948 | 99.9661 |
| R7_SC / MAST-PAD | 0.85467747 | 22.9292 | 11.2116 | 41.4815 | 26.3466 | 84.6284 | 19.5062 | 2947 / 2948 | 99.9661 |

### LCC evaluation

| Model | AUC | EER % | APCER % | BPCER % | ACER/HTER % | Accuracy % | TPR@FPR1% % | Scored / candidates | Coverage % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Base / CE_clean † | 0.81268724 | 27.3356 | 8.0552 | 50.0000 | 29.0276 | 90.2036 | 17.9936 | 7564 / 7580 | 99.7889 |
| C / Clean | 0.81162223 | 26.4235 | 10.2897 | 47.7707 | 29.0302 | 88.1544 | 8.5987 | 7564 / 7580 | 99.7889 |
| P3_SF / WBST | 0.81968680 | 24.8273 | 8.4828 | 46.1783 | 27.3306 | 89.9524 | 15.2866 | 7564 / 7580 | 99.7889 |
| R7_SC / MAST-PAD | 0.84005952 | 24.1847 | 8.8414 | 44.2675 | 26.5544 | 89.6880 | 19.1083 | 7564 / 7580 | 99.7889 |

### LCC combined

| Model | AUC | EER % | APCER % | BPCER % | ACER/HTER % | Accuracy % | TPR@FPR1% % | Scored / candidates | Coverage % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Base / CE_clean † | 0.84398794 | 24.4623 | 7.3043 | 51.7259 | 29.5151 | 88.1079 | 20.1321 | 18794 / 18827 | 99.8247 |
| C / Clean | 0.84601127 | 23.5793 | 8.9064 | 47.1922 | 28.0493 | 87.1395 | 14.0134 | 18794 / 18827 | 99.8247 |
| P3_SF / WBST | 0.86083099 | 21.7906 | 7.7256 | 45.8527 | 26.7891 | 88.3367 | 22.5688 | 18794 / 18827 | 99.8247 |
| R7_SC / MAST-PAD | 0.86027208 | 22.0559 | 7.9808 | 46.2648 | 27.1228 | 88.0653 | 19.9897 | 18794 / 18827 | 99.8247 |

## 6. CASIA — population và source-frozen threshold

Nguồn CASIA là Kaggle extracted-color images, không phải full original videos. Video ID được dựng từ filename, score video là mean valid-frame score; ít nhất4 valid frames mới scorable. Frame metrics gồm mọi frame detect thành công. SCRFD confidence0.5,640→480→320, largest finite positive box/confidence tie-break; crop1.5; không GT/full-image fallback.

Combined dedup `(video_id,frame_index)` trước detection, ưu tiên test copy; không fallback sang train nếu test copy detect fail. Train/test overlap: **20 subject IDs,240 video IDs,1403 frame IDs**. Không khẳng định official subject-disjoint benchmark hoặc độc lập train/test.

| Pool | Frame scored / candidates | Video scored / candidates |
| --- | --- | --- |
| train | 1655 / 1655 | 237 / 240 |
| test | 2408 / 2408 | 352 / 360 |
| combined | 2660 / 2660 | 358 / 360 |

### CASIA video — source-frozen

| Pool | Model | AUC | EER % | APCER % | BPCER % | ACER/HTER % | Accuracy % | TPR@FPR1% % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | Base / CE_clean † | 0.91239764 | 19.1535 | 0.5618 | 67.7966 | 34.1792 | 82.7004 | 44.0678 |
| train | C / Clean | 0.92096743 | 20.2819 | 0.5618 | 50.8475 | 25.7046 | 86.9198 | 49.1525 |
| train | P3_SF / WBST | 0.90268520 | 21.6911 | 2.8090 | 59.3220 | 31.0655 | 83.1224 | 33.8983 |
| train | R7_SC / MAST-PAD | 0.91915826 | 16.9015 | 2.8090 | 47.4576 | 25.1333 | 86.0759 | 45.7627 |
| test | Base / CE_clean † | 0.88369490 | 17.9924 | 0.3788 | 73.8636 | 37.1212 | 81.2500 | 37.5000 |
| test | C / Clean | 0.90874656 | 20.2652 | 0.7576 | 59.0909 | 29.9242 | 84.6591 | 42.0455 |
| test | P3_SF / WBST | 0.91180269 | 17.9924 | 1.8939 | 57.9545 | 29.9242 | 84.0909 | 39.7727 |
| test | R7_SC / MAST-PAD | 0.92587810 | 18.1818 | 1.8939 | 53.4091 | 27.6515 | 85.2273 | 45.4545 |
| combined | Base / CE_clean † | 0.89830017 | 16.5423 | 0.3731 | 73.3333 | 36.8532 | 81.2849 | 42.2222 |
| combined | C / Clean | 0.92039801 | 16.7289 | 0.7463 | 55.5556 | 28.1509 | 85.4749 | 44.4444 |
| combined | P3_SF / WBST | 0.92226368 | 17.4751 | 1.4925 | 56.6667 | 29.0796 | 84.6369 | 42.2222 |
| combined | R7_SC / MAST-PAD | 0.93519900 | 16.9154 | 1.8657 | 51.1111 | 26.4884 | 85.7542 | 45.5556 |

### CASIA frame — source-frozen

| Pool | Model | AUC | EER % | APCER % | BPCER % | ACER/HTER % | Accuracy % | TPR@FPR1% % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | Base / CE_clean † | 0.90470000 | 19.0821 | 1.1191 | 62.3762 | 31.7477 | 83.9275 | 36.8812 |
| train | C / Clean | 0.91486019 | 18.7585 | 3.0376 | 43.3168 | 23.1772 | 87.1299 | 43.8119 |
| train | P3_SF / WBST | 0.90060922 | 20.3004 | 3.8369 | 50.4950 | 27.1660 | 84.7734 | 30.1980 |
| train | R7_SC / MAST-PAD | 0.91182697 | 18.2312 | 5.6755 | 40.0990 | 22.8872 | 85.9215 | 43.8119 |
| test | Base / CE_clean † | 0.86209395 | 21.6712 | 1.1558 | 71.7428 | 36.4493 | 81.5199 | 27.0728 |
| test | C / Clean | 0.88314490 | 20.8079 | 2.9169 | 54.1455 | 28.5312 | 84.5100 | 30.9645 |
| test | P3_SF / WBST | 0.89111996 | 20.1117 | 2.9169 | 56.5144 | 29.7156 | 83.9286 | 27.7496 |
| test | R7_SC / MAST-PAD | 0.90321061 | 18.7469 | 4.4029 | 47.0389 | 25.7209 | 85.1329 | 40.6091 |
| combined | Base / CE_clean † | 0.87433100 | 20.1847 | 1.1078 | 68.5460 | 34.8269 | 81.8045 | 30.2671 |
| combined | C / Clean | 0.89177245 | 20.0098 | 2.9708 | 50.2967 | 26.6338 | 85.0376 | 35.1424 |
| combined | P3_SF / WBST | 0.89866379 | 19.1605 | 2.9204 | 53.2641 | 28.0923 | 84.3233 | 30.2671 |
| combined | R7_SC / MAST-PAD | 0.90941412 | 18.2384 | 4.4814 | 44.3620 | 24.4217 | 85.4135 | 40.3561 |

## 7. CASIA — train-calibrated operating points

Mỗi model fit threshold riêng cho frame/video, **chỉ dùng train scores**, tối thiểu pooled ACER/HTER; ties chọn gần source threshold nhất. Sau fit, chuyển nguyên threshold sang test và combined. Không fit test/combined, không retrain/checkpoint selection, không đổi AUC/EER.

Train metrics là fitting-set diagnostics; combined chứa train calibration data. Kaggle train/test copy overlap làm test không được coi là xác nhận độc lập hoàn toàn. Đây chưa phải webcam-validated operating point.

| Model | Source threshold | CASIA train frame threshold | CASIA train video threshold |
| --- | --- | --- | --- |
| Base / CE_clean † | 4.5966796875 | -0.84521484 | -3.428955 |
| C / Clean | 5.130859375 | 2.375 | -1.1993669 |
| P3_SF / WBST | 5.3037109375 | 1.7167969 | 0.8730957 |
| R7_SC / MAST-PAD | 4.66259765625 | 1.6586914 | -0.033447266 |

### CASIA video — threshold fit trên train

| Pool | Model | AUC | EER % | APCER % | BPCER % | ACER/HTER % | Accuracy % | TPR@FPR1% % | Source ACER % | Giảm ACER (pp) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | Base / CE_clean † | 0.91239764 | 19.1535 | 29.2135 | 5.0847 | 17.1491 | 76.7932 | 44.0678 | 34.1792 | 17.0301 |
| train | C / Clean | 0.92096743 | 20.2819 | 29.2135 | 5.0847 | 17.1491 | 76.7932 | 49.1525 | 25.7046 | 8.5555 |
| train | P3_SF / WBST | 0.90268520 | 21.6911 | 14.6067 | 23.7288 | 19.1678 | 83.1224 | 33.8983 | 31.0655 | 11.8977 |
| train | R7_SC / MAST-PAD | 0.91915826 | 16.9015 | 17.4157 | 13.5593 | 15.4875 | 83.5443 | 45.7627 | 25.1333 | 9.6458 |
| test | Base / CE_clean † | 0.88369490 | 17.9924 | 31.4394 | 14.7727 | 23.1061 | 72.7273 | 37.5000 | 37.1212 | 14.0152 |
| test | C / Clean | 0.90874656 | 20.2652 | 28.7879 | 13.6364 | 21.2121 | 75.0000 | 42.0455 | 29.9242 | 8.7121 |
| test | P3_SF / WBST | 0.91180269 | 17.9924 | 11.3636 | 22.7273 | 17.0455 | 85.7955 | 39.7727 | 29.9242 | 12.8788 |
| test | R7_SC / MAST-PAD | 0.92587810 | 18.1818 | 20.4545 | 17.0455 | 18.7500 | 80.3977 | 45.4545 | 27.6515 | 8.9015 |
| combined | Base / CE_clean † | 0.89830017 | 16.5423 | 31.7164 | 12.2222 | 21.9693 | 73.1844 | 42.2222 | 36.8532 | 14.8839 |
| combined | C / Clean | 0.92039801 | 16.7289 | 27.9851 | 11.1111 | 19.5481 | 76.2570 | 44.4444 | 28.1509 | 8.6028 |
| combined | P3_SF / WBST | 0.92226368 | 17.4751 | 11.1940 | 22.2222 | 16.7081 | 86.0335 | 42.2222 | 29.0796 | 12.3715 |
| combined | R7_SC / MAST-PAD | 0.93519900 | 16.9154 | 19.0299 | 14.4444 | 16.7371 | 82.1229 | 45.5556 | 26.4884 | 9.7512 |

### CASIA frame — threshold fit trên train

| Pool | Model | AUC | EER % | APCER % | BPCER % | ACER/HTER % | Accuracy % | TPR@FPR1% % | Source ACER % | Giảm ACER (pp) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | Base / CE_clean † | 0.90470000 | 19.0821 | 13.4293 | 21.5347 | 17.4820 | 84.5921 | 36.8812 | 31.7477 | 14.2657 |
| train | C / Clean | 0.91486019 | 18.7585 | 12.6299 | 23.7624 | 18.1961 | 84.6526 | 43.8119 | 23.1772 | 4.9811 |
| train | P3_SF / WBST | 0.90060922 | 20.3004 | 13.3493 | 25.4950 | 19.4222 | 83.6858 | 30.1980 | 27.1660 | 7.7438 |
| train | R7_SC / MAST-PAD | 0.91182697 | 18.2312 | 13.7490 | 20.7921 | 17.2705 | 84.5317 | 43.8119 | 22.8872 | 5.6167 |
| test | Base / CE_clean † | 0.86209395 | 21.6712 | 16.7309 | 26.2267 | 21.4788 | 80.9385 | 27.0728 | 36.4493 | 14.9705 |
| test | C / Clean | 0.88314490 | 20.8079 | 10.9521 | 33.6717 | 22.3119 | 83.4718 | 30.9645 | 28.5312 | 6.2193 |
| test | P3_SF / WBST | 0.89111996 | 20.1117 | 11.8877 | 30.9645 | 21.4261 | 83.4302 | 27.7496 | 29.7156 | 8.2895 |
| test | R7_SC / MAST-PAD | 0.90321061 | 18.7469 | 14.0341 | 24.7039 | 19.3690 | 83.3472 | 40.6091 | 25.7209 | 6.3519 |
| combined | Base / CE_clean † | 0.87433100 | 20.1847 | 16.1128 | 23.8872 | 20.0000 | 81.9173 | 30.2671 | 34.8269 | 14.8269 |
| combined | C / Clean | 0.89177245 | 20.0098 | 11.1279 | 30.5638 | 20.8458 | 83.9474 | 35.1424 | 26.6338 | 5.7879 |
| combined | P3_SF / WBST | 0.89866379 | 19.1605 | 11.7825 | 28.3383 | 20.0604 | 84.0226 | 30.2671 | 28.0923 | 8.0319 |
| combined | R7_SC / MAST-PAD | 0.90941412 | 18.2384 | 13.9476 | 22.7003 | 18.3240 | 83.8346 | 40.3561 | 24.4217 | 6.0977 |

## 8. Reference-compatible metrics — bảng riêng

CE-only đã được bổ sung từ `metrics/*_reference_metrics.json` và đối chiếu từ logits CSV bằng đúng hàm reference của notebook, tách riêng với source-calibrated metrics.

Các file `*_reference_metrics.json` dùng softmax attack-positive cho AUC/ROC và argmax raw logits cho decision; EER = min(FPR,FNR) ở điểm crossing gần nhất. **Tên APCER/BPCER trong implementation reference bị đảo so với ISO/project:** reference APCER là real→attack, reference BPCER là attack→real. Không đổi tên hoặc ghép các cột này vào bảng project. Không coi argmax ACER là metric tại source-calibrated threshold. Bảng dưới chỉ là kết quả model của chúng ta theo convention này, không phải số baseline paper đã được reproduce.

### Reference convention: test

| Model | AUC (attack-positive) | EER % | Ref APCER (real→attack) % | Ref BPCER (attack→real) % | ACER % | Argmax accuracy % |
| --- | --- | --- | --- | --- | --- | --- |
| Base / CE_clean † | 0.99214398 | 4.4754 | 0.5437 | 19.7442 | 10.1439 | 86.0016 |
| C / Clean | 0.99073553 | 4.6759 | 0.4682 | 21.7801 | 11.1241 | 84.5975 |
| P3_SF / WBST | 0.99224181 | 4.3900 | 0.3977 | 23.2269 | 11.8123 | 83.6048 |
| R7_SC / MAST-PAD | 0.99228027 | 4.3491 | 0.6242 | 18.9423 | 9.7833 | 86.5394 |

### Reference convention: lcc_training

| Model | AUC (attack-positive) | EER % | Ref APCER (real→attack) % | Ref BPCER (attack→real) % | ACER % | Argmax accuracy % |
| --- | --- | --- | --- | --- | --- | --- |
| Base / CE_clean † | 0.87025224 | 22.5041 | 22.2586 | 22.8721 | 22.5654 | 77.2184 |
| C / Clean | 0.86922397 | 21.3284 | 16.0393 | 26.6251 | 21.3322 | 74.9366 |
| P3_SF / WBST | 0.88126169 | 19.9547 | 11.1293 | 31.0579 | 21.0936 | 71.8822 |
| R7_SC / MAST-PAD | 0.87748892 | 20.3764 | 18.2488 | 22.6738 | 20.4613 | 77.9790 |

### Reference convention: lcc_development

| Model | AUC (attack-positive) | EER % | Ref APCER (real→attack) % | Ref BPCER (attack→real) % | ACER % | Argmax accuracy % |
| --- | --- | --- | --- | --- | --- | --- |
| Base / CE_clean † | 0.81761469 | 26.9136 | 20.9877 | 32.0220 | 26.5048 | 69.4944 |
| C / Clean | 0.85159202 | 23.4461 | 10.1235 | 38.0409 | 24.0822 | 65.7957 |
| P3_SF / WBST | 0.85283776 | 22.4626 | 10.8642 | 41.5028 | 26.1835 | 62.7078 |
| R7_SC / MAST-PAD | 0.85467747 | 22.8954 | 13.3333 | 32.9662 | 23.1498 | 69.7319 |

### Reference convention: lcc_evaluation

| Model | AUC (attack-positive) | EER % | Ref APCER (real→attack) % | Ref BPCER (attack→real) % | ACER % | Argmax accuracy % |
| --- | --- | --- | --- | --- | --- | --- |
| Base / CE_clean † | 0.81268702 | 27.2828 | 26.1146 | 28.8690 | 27.4918 | 71.2454 |
| C / Clean | 0.81162223 | 26.4138 | 18.1529 | 35.9724 | 27.0626 | 64.7673 |
| P3_SF / WBST | 0.81968680 | 24.8138 | 18.4713 | 36.4276 | 27.4495 | 64.3178 |
| R7_SC / MAST-PAD | 0.84005952 | 24.1655 | 19.1083 | 28.5241 | 23.8162 | 71.8667 |

### Reference convention: lcc_combined

| Model | AUC (attack-positive) | EER % | Ref APCER (real→attack) % | Ref BPCER (attack→real) % | ACER % | Argmax accuracy % |
| --- | --- | --- | --- | --- | --- | --- |
| Base / CE_clean † | 0.84398803 | 24.4526 | 22.6172 | 26.8320 | 24.7246 | 73.6033 |
| C / Clean | 0.84601127 | 23.5626 | 15.1468 | 32.3681 | 23.7575 | 69.4105 |
| P3_SF / WBST | 0.86083099 | 21.7884 | 12.2617 | 34.9433 | 23.6025 | 67.3992 |
| R7_SC / MAST-PAD | 0.86027208 | 22.0505 | 17.3622 | 26.7430 | 22.0526 | 74.2258 |

## 9. Complexity và runtime artifacts

Output notebook21 chưa có số đo complexity, ONNX hoặc runtime config cho CE-only; bảng dưới giữ các số đo C/P3/R7 gốc, không gán số đo của model khác cho CE-only.

Cả ba dùng chung clean inference architecture. FLOPs đo bằng torch.profiler CPU cho Conv/Linear ở80×80; không gồm normalization/activation/pooling và training FFT/selection. Không có FPS/latency webcam được đo trong artifact này.

| Model | Deployment params | All training params | MACs | GFLOPs |
| --- | --- | --- | --- | --- |
| C / Clean | 434434 | 441658 | 40810892 | 0.081621784 |
| P3_SF / WBST | 434434 | 441658 | 40810892 | 0.081621784 |
| R7_SC / MAST-PAD | 434434 | 441658 | 40810892 | 0.081621784 |

- [R7_SC_FULL_crop15.onnx](../../antispoof/models/R7_SC_FULL_crop15.onnx)

- [R7_SC_FULL_crop15_runtime_config.json](../../antispoof/models/R7_SC_FULL_crop15_runtime_config.json)

- [R7_SC_FULL_crop15_runtime_config_casia.json](../../antispoof/models/R7_SC_FULL_crop15_runtime_config_casia.json)

Runtime source JSON R7_FULL threshold4.66259765625; CASIA variant threshold1.6586914 là **frame** threshold. Việc có config không chứng minh hiệu quả webcam. Active .env và deployment decisions không thuộc số liệu benchmark này.

## 10. Provenance, kiểm tra và trạng thái baseline mới

| Run | Checkpoint SHA256 | Best epoch / source threshold xác nhận |
| --- | --- | --- |
| C_FULL_crop15 | 43247a879487cffedbcfa92f17041427b34f9ccf529a2bcdac13b1f1d54a6df5 | 9 / 5.130859375 |
| P3_SF_FULL_crop15 | 4454aa473e3e1ff05b9039bc8c1a6402ad17fabf1879e59a463b954cc305b71a | 6 / 5.3037109375 |
| R7_SC_FULL_crop15 | 843a8a693c6135be2a532122df19cb0e0151ce83d6c49f23c414c34e52b570af | 7 / 4.66259765625 |

Shared initial-state SHA256: `f32f61b79635ba650f66815b1ef5cd2e5ef926c87c616b01b2497ed9a12e5029`.

Theo kiểm tra được ghi trong snapshot C/P3/R7 gốc (không chạy lại khi gộp): đã xác nhận local best.pth SHA khớp evaluation_freeze ở cả ba run; config/completion/source thresholds và CASIA provenance khớp best epoch/seed/threshold. Đã tính lại AUC/EER/TPR/APCER/BPCER/ACER/HTER/accuracy/coverage từ predictions cho **51 metric populations** (15 CelebA/LCC +18 CASIA source +18 CASIA calibrated), khớp JSON với tolerance1e-8. Threshold CASIA train được chuyển nguyên sang các pool. Đây là kiểm tra offline artifacts, không chạy lại model/training. CASIA provenance ghi đường dẫn uploaded checkpoint nhưng không có checkpoint hash để chứng minh byte equality của upload; không suy đoán git commit môi trường train khi artifact không ghi.

| Model | Config | Completion | Freeze | Raw predictions/metrics |
| --- | --- | --- | --- | --- |
| C_FULL_crop15 | [config](../../antispoof/notebooks/final/binary_crossdomain_v2/output/18_fullscale_C_P3SF_R7SC_crop15/runs/C_FULL_crop15/config.json) | [completion](../../antispoof/notebooks/final/binary_crossdomain_v2/output/18_fullscale_C_P3SF_R7SC_crop15/runs/C_FULL_crop15/completion.json) | [freeze](../../antispoof/notebooks/final/binary_crossdomain_v2/output/18_fullscale_C_P3SF_R7SC_crop15/runs/C_FULL_crop15/evaluation_freeze.json) | [run output](../../antispoof/notebooks/final/binary_crossdomain_v2/output/18_fullscale_C_P3SF_R7SC_crop15/runs/C_FULL_crop15) |
| P3_SF_FULL_crop15 | [config](../../antispoof/notebooks/final/binary_crossdomain_v2/output/18_fullscale_C_P3SF_R7SC_crop15/runs/P3_SF_FULL_crop15/config.json) | [completion](../../antispoof/notebooks/final/binary_crossdomain_v2/output/18_fullscale_C_P3SF_R7SC_crop15/runs/P3_SF_FULL_crop15/completion.json) | [freeze](../../antispoof/notebooks/final/binary_crossdomain_v2/output/18_fullscale_C_P3SF_R7SC_crop15/runs/P3_SF_FULL_crop15/evaluation_freeze.json) | [run output](../../antispoof/notebooks/final/binary_crossdomain_v2/output/18_fullscale_C_P3SF_R7SC_crop15/runs/P3_SF_FULL_crop15) |
| R7_SC_FULL_crop15 | [config](../../antispoof/notebooks/final/binary_crossdomain_v2/output/18_fullscale_C_P3SF_R7SC_crop15/runs/R7_SC_FULL_crop15/config.json) | [completion](../../antispoof/notebooks/final/binary_crossdomain_v2/output/18_fullscale_C_P3SF_R7SC_crop15/runs/R7_SC_FULL_crop15/completion.json) | [freeze](../../antispoof/notebooks/final/binary_crossdomain_v2/output/18_fullscale_C_P3SF_R7SC_crop15/runs/R7_SC_FULL_crop15/evaluation_freeze.json) | [run output](../../antispoof/notebooks/final/binary_crossdomain_v2/output/18_fullscale_C_P3SF_R7SC_crop15/runs/R7_SC_FULL_crop15) |

- CASIA C_FULL_crop15: [frame/video metrics, predictions và train calibration](../../antispoof/notebooks/final/binary_crossdomain_v2/output/casia_train_test_C_P3SF_R7SC_100k_fullscale/runs/C_FULL_crop15)

- CASIA P3_SF_FULL_crop15: [frame/video metrics, predictions và train calibration](../../antispoof/notebooks/final/binary_crossdomain_v2/output/casia_train_test_C_P3SF_R7SC_100k_fullscale/runs/P3_SF_FULL_crop15)

- CASIA R7_SC_FULL_crop15: [frame/video metrics, predictions và train calibration](../../antispoof/notebooks/final/binary_crossdomain_v2/output/casia_train_test_C_P3SF_R7SC_100k_fullscale/runs/R7_SC_FULL_crop15)

### Base / CE_clean — đã có số liệu báo cáo

Notebook: [21_fullscale_CE_clean_baseline_crop15.ipynb](../../antispoof/notebooks/final/binary_crossdomain_v2/21_fullscale_CE_clean_baseline_crop15.ipynb).

**Mức bằng chứng (kiểm tra offline 2026-10-20):** đã có artifact gốc tại [output notebook21](../../antispoof/notebooks/final/21_fullscale_CE_clean_baseline_crop15). Đã tính lại 17 populations project từ predictions CSV (AUC/EER/TPR/APCER/BPCER/ACER/HTER/accuracy/coverage), khớp JSON với tolerance 1e-8; 5 populations reference cũng khớp khi dùng đúng softmax float32 và convention của notebook. Cả 17 dòng báo cáo ban đầu bên dưới khớp JSON full precision. Đây là kiểm tra artifact, không chạy lại inference/training.

| Thông tin | CE_clean |
| --- | --- |
| Best epoch / epoch cuối | 9 / 12 |
| Train seed / data seed | 100 / 43 |
| Source threshold | 4.5966796875 |
| CASIA train frame threshold | -0.84521484 |
| CASIA train video threshold | -3.428955 |
| Best checkpoint SHA256 (đã hash local) | `979bf9e34a85d6e922215ca85758ce57c0514cec7a4efa5db7e6efdcbd0fb2dc` |
| Initial-state SHA256 (config/completion/audit) | `f32f61b79635ba650f66815b1ef5cd2e5ef926c87c616b01b2497ed9a12e5029` |

Đã hash cả bản root và per-run của `full_train.csv`/`full_test.csv`, khớp `evaluation_freeze.json`. `config.json` xác nhận CE_clean=1, CE_worst=0, L_aux=0, seed100, crop1.5 BGR80 và cấu hình huấn luyện; completion, freeze, CASIA config và training history nhất quán về best epoch/threshold.

**Phạm vi run:** kết quả CE_clean trong tài liệu này là seed100, nhất quán với các kết quả C/P3/R7 hiện tại. Theo kế hoạch của tác giả, seed101 và seed102 sẽ được bổ sung sau; cấu hình notebook chuẩn bị cho seed tiếp theo không được dùng để mô tả lại run seed100.

**Giới hạn provenance:** không xác lập được exact training git revision chỉ từ các artifact đã kiểm tra. Initial-state SHA được đối chiếu giữa metadata, chưa tái tạo khởi tạo; checkpoint best đã được hash trực tiếp. Thư mục chưa có complexity report, ONNX hoặc runtime config CE-only. Không suy ra latency/FPS hay hiệu quả webcam.

Nguồn trực tiếp:

- [Config](../../antispoof/notebooks/final/21_fullscale_CE_clean_baseline_crop15/runs/CE_CLEAN_FULL_crop15/config.json), [completion](../../antispoof/notebooks/final/21_fullscale_CE_clean_baseline_crop15/runs/CE_CLEAN_FULL_crop15/completion.json), [freeze](../../antispoof/notebooks/final/21_fullscale_CE_clean_baseline_crop15/runs/CE_CLEAN_FULL_crop15/evaluation_freeze.json), [training history](../../antispoof/notebooks/final/21_fullscale_CE_clean_baseline_crop15/runs/CE_CLEAN_FULL_crop15/training_history.csv).
- [Best checkpoint](../../antispoof/notebooks/final/21_fullscale_CE_clean_baseline_crop15/runs/CE_CLEAN_FULL_crop15/best.pth), [initialization audit](../../antispoof/notebooks/final/21_fullscale_CE_clean_baseline_crop15/runs/CE_CLEAN_FULL_crop15/initialization_audit.json), [manifest hashes](../../antispoof/notebooks/final/21_fullscale_CE_clean_baseline_crop15/manifest_sha256.json).
- [Project và reference metrics](../../antispoof/notebooks/final/21_fullscale_CE_clean_baseline_crop15/runs/CE_CLEAN_FULL_crop15/metrics), [CelebA/LCC predictions](../../antispoof/notebooks/final/21_fullscale_CE_clean_baseline_crop15/runs/CE_CLEAN_FULL_crop15/predictions).
- [CASIA frame calibration](../../antispoof/notebooks/final/21_fullscale_CE_clean_baseline_crop15/runs/CE_CLEAN_FULL_crop15/casia/frame_casia_calibration.json), [video calibration](../../antispoof/notebooks/final/21_fullscale_CE_clean_baseline_crop15/runs/CE_CLEAN_FULL_crop15/casia/video_casia_calibration.json), [CASIA predictions/metrics](../../antispoof/notebooks/final/21_fullscale_CE_clean_baseline_crop15/runs/CE_CLEAN_FULL_crop15/casia).

**Cách đọc bảng CE-only gốc:** metric dưới đây là fraction, không phải phần trăm như bảng so sánh. `test` là CelebA full Test-as-Val; `source-calibrated` tương ứng source-frozen khi transfer. `frame` ở CelebA/LCC là đơn vị ảnh; CASIA tách frame và reconstructed video. Các đường dẫn `runs/<run_id>/...` phía dưới là cấu trúc output tương đối với output notebook21 đã liên kết.

Loss = CE_clean only (weight1). No spectral selection, CE_worst, auxiliary forward or L_aux.
Historical C_FULL used CE_clean + L_aux. Auxiliary modules are frozen compatibility placeholders; not optimized.
Matched notebook18: full eligible Train, crop1.5 BGR80, pretrained trunk/seed100, SGD .005/.9/5e-4, batch256 AMP, max12 epochs, milestones[4,8], gamma.2, phase-aware patience9/3. Checkpoints eligible from epoch1.
CelebA full Test-as-Val is reused for selection and calibration, as in notebook18: not independent held-out Test evidence.
LCC uses the source-frozen threshold. CASIA source and optional train-only calibrated thresholds are separate; video and frame units are separate.
CASIA train and combined calibrated metrics include fitting data; Kaggle train/test copy overlap limits independence. No webcam validation or significance claim.

| Run | Dataset | Unit | Threshold policy | AUC | EER | APCER | BPCER | ACER | HTER | Scored | Candidates |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CE_CLEAN_FULL_crop15 | test | frame | source-calibrated | 0.9921439799017713 | 0.04475704217898091 | 0.05073632161668279 | 0.038260169150221505 | 0.04449824538345215 | 0.04449824538345215 | 66379 | 67170 |
| CE_CLEAN_FULL_crop15 | lcc_training | frame | source-calibrated | 0.8702520657603567 | 0.22532317742673103 | 0.056365953830902135 | 0.5090016366612111 | 0.28268379524605663 | 0.28268379524605663 | 8283 | 8299 |
| CE_CLEAN_FULL_crop15 | lcc_development | frame | source-calibrated | 0.8176146904838224 | 0.2696977202746938 | 0.0979543666404406 | 0.5555555555555556 | 0.32675496109799806 | 0.32675496109799806 | 2947 | 2948 |
| CE_CLEAN_FULL_crop15 | lcc_evaluation | frame | source-calibrated | 0.8126872391829563 | 0.27335646826268395 | 0.08055172413793103 | 0.5 | 0.2902758620689655 | 0.2902758620689655 | 7564 | 7580 |
| CE_CLEAN_FULL_crop15 | lcc_combined | frame | source-calibrated | 0.8439879397180328 | 0.2446227070073732 | 0.0730433750667537 | 0.5172591447707368 | 0.29515125991874525 | 0.29515125991874525 | 18794 | 18827 |
| CE_CLEAN_FULL_crop15 | casia_train | video | source-calibrated | 0.9123976385450391 | 0.19153494572462382 | 0.0056179775280898875 | 0.6779661016949152 | 0.34179203961150256 | 0.34179203961150256 | 237 | 240 |
| CE_CLEAN_FULL_crop15 | casia_train | frame | source-calibrated | 0.9047000023743381 | 0.19082061083806223 | 0.011191047162270184 | 0.6237623762376238 | 0.31747671169994696 | 0.31747671169994696 | 1655 | 1655 |
| CE_CLEAN_FULL_crop15 | casia_test | video | source-calibrated | 0.8836949035812672 | 0.17992424242424238 | 0.003787878787878788 | 0.7386363636363636 | 0.3712121212121212 | 0.3712121212121212 | 352 | 360 |
| CE_CLEAN_FULL_crop15 | casia_test | frame | source-calibrated | 0.8620939482067744 | 0.2167115054565501 | 0.011557512383048982 | 0.7174280879864636 | 0.3644928001847563 | 0.3644928001847563 | 2408 | 2408 |
| CE_CLEAN_FULL_crop15 | casia_combined | video | source-calibrated | 0.8983001658374792 | 0.1654228855721393 | 0.0037313432835820895 | 0.7333333333333333 | 0.36853233830845766 | 0.36853233830845766 | 358 | 360 |
| CE_CLEAN_FULL_crop15 | casia_combined | frame | source-calibrated | 0.8743309994890047 | 0.20184690459328053 | 0.011077542799597181 | 0.685459940652819 | 0.3482687417262081 | 0.3482687417262081 | 2660 | 2660 |
| CE_CLEAN_FULL_crop15 | casia_train | frame | CASIA train-calibrated | 0.9047000023743381 | 0.19082061083806223 | 0.1342925659472422 | 0.21534653465346534 | 0.1748195503003538 | 0.1748195503003538 | 1655 | 1655 |
| CE_CLEAN_FULL_crop15 | casia_test | frame | CASIA train-calibrated | 0.8620939482067744 | 0.2167115054565501 | 0.16730875068794718 | 0.2622673434856176 | 0.2147880470867824 | 0.2147880470867824 | 2408 | 2408 |
| CE_CLEAN_FULL_crop15 | casia_combined | frame | CASIA train-calibrated | 0.8743309994890047 | 0.20184690459328053 | 0.16112789526686808 | 0.23887240356083086 | 0.20000014941384947 | 0.20000014941384947 | 2660 | 2660 |
| CE_CLEAN_FULL_crop15 | casia_train | video | CASIA train-calibrated | 0.9123976385450391 | 0.19153494572462382 | 0.29213483146067415 | 0.05084745762711865 | 0.1714911445438964 | 0.1714911445438964 | 237 | 240 |
| CE_CLEAN_FULL_crop15 | casia_test | video | CASIA train-calibrated | 0.8836949035812672 | 0.17992424242424238 | 0.3143939393939394 | 0.14772727272727273 | 0.23106060606060608 | 0.23106060606060608 | 352 | 360 |
| CE_CLEAN_FULL_crop15 | casia_combined | video | CASIA train-calibrated | 0.8983001658374792 | 0.1654228855721393 | 0.31716417910447764 | 0.12222222222222222 | 0.21969320066334994 | 0.21969320066334994 | 358 | 360 |

Metrics are fractions. ACER=HTER=(APCER+BPCER)/2; Real iff logit(real)-logit(attack) >= threshold.
Source and LCC predictions/metrics: runs/<run_id>/predictions and metrics. CASIA predictions, threshold curves, coverage and report: runs/<run_id>/casia.
One new model, one seed (100). Metric đã được đối chiếu artifact gốc như mô tả ở đầu mục này.

## 11. Cách dùng khi viết báo cáo

- Ghi đầy đủ dataset/pool, image hoặc video, threshold policy và coverage. Ưu tiên các bảng project source-frozen để mô tả transfer; CASIA train-calibrated là phân tích operating point riêng.
- Không nói CelebA full Test là untouched holdout; không nói CASIA copy là official subject-disjoint.
- Không trộn 100K với full-scale; không lấy CASIA video AUC và frame ACER thành cùng một hàng mà thiếu đơn vị. Không chọn tốt nhất giữa source/target thresholds mà không ghi policy.
- Không khẳng định R7 tốt nhất trên mọi tập/chỉ số. Một seed, không std/CI/significance; các khác biệt chỉ là descriptive measured differences.
- LCC đã được xem nhiều lần trong quá trình thiết kế; không giả định fresh blind benchmark. Không suy ra hiệu quả webcam từ benchmark tĩnh.
- Số liệu raw chưa làm tròn dưới đây và các source links có ưu tiên cao hơn số phần trăm đã format trong bảng.

## 12. Snapshot số liệu full precision

Tất cả rate ở JSON dưới đây là fraction (không phải %). `threshold_policy` quyết định operating point; `unit` quyết định frame/image hoặc video. Có68 records project: 51 C/P3/R7 từ snapshot gốc và 17 CE-only đã đối chiếu CSV ngày 2026-10-20; reference-compatible metrics giữ ở JSON gốc đã liên kết, không trộn vào snapshot project này.

```json
[
{"acer": 0.04624830931382525, "apcer": 0.053380629904331935, "auc": 0.9907355300359331, "binary_accuracy": 0.95088808207415, "bpcer": 0.03911598872331857, "candidate_n": 67170, "dataset": "test", "detector_coverage": 0.9882239094834003, "eer": 0.04676356625894518, "hter": 0.04624830931382525, "locked_source_threshold": 5.130859375, "run_id": "C_FULL_crop15", "scored_n": 66379, "threshold_policy": "source-frozen", "tpr_at_fpr_1pct": 0.8266084373741441, "unit": "image"},
{"acer": 0.27914264078450335, "apcer": 0.0640135958079592, "auc": 0.8692239662274344, "binary_accuracy": 0.8725099601593626, "bpcer": 0.49427168576104746, "candidate_n": 8299, "dataset": "lcc_training", "detector_coverage": 0.9980720568743222, "eer": 0.21343426270626026, "hter": 0.27914264078450335, "locked_source_threshold": 5.130859375, "run_id": "C_FULL_crop15", "scored_n": 8283, "threshold_policy": "source-frozen", "tpr_at_fpr_1pct": 0.15516366612111293, "unit": "image"},
{"acer": 0.25959874114870185, "apcer": 0.11919748229740362, "auc": 0.8515920195044245, "binary_accuracy": 0.842212419409569, "bpcer": 0.4, "candidate_n": 2948, "dataset": "lcc_development", "detector_coverage": 0.9996607869742198, "eer": 0.23451447776126505, "hter": 0.25959874114870185, "locked_source_threshold": 5.130859375, "run_id": "C_FULL_crop15", "scored_n": 2947, "threshold_policy": "source-frozen", "tpr_at_fpr_1pct": 0.22325925925925927, "unit": "image"},
{"acer": 0.29030177904678234, "apcer": 0.10289655172413793, "auc": 0.811622227103009, "binary_accuracy": 0.881544156530936, "bpcer": 0.47770700636942676, "candidate_n": 7580, "dataset": "lcc_evaluation", "detector_coverage": 0.9978891820580474, "eer": 0.26423457061278277, "hter": 0.29030177904678234, "locked_source_threshold": 5.130859375, "run_id": "C_FULL_crop15", "scored_n": 7564, "threshold_policy": "source-frozen", "tpr_at_fpr_1pct": 0.08598726114649681, "unit": "image"},
{"acer": 0.28049297570319925, "apcer": 0.08906426155580609, "auc": 0.8460112694327802, "binary_accuracy": 0.8713951261040758, "bpcer": 0.47192168985059246, "candidate_n": 18827, "dataset": "lcc_combined", "detector_coverage": 0.9982471981728369, "eer": 0.23579327477380935, "hter": 0.28049297570319925, "locked_source_threshold": 5.130859375, "run_id": "C_FULL_crop15", "scored_n": 18794, "threshold_policy": "source-frozen", "tpr_at_fpr_1pct": 0.14013395157135497, "unit": "image"},
{"acer": 0.2317720081360654, "apcer": 0.030375699440447643, "auc": 0.9148601910550767, "binary_accuracy": 0.8712990936555891, "bpcer": 0.43316831683168316, "candidate_n": 1655, "dataset": "casia_train", "detector_coverage": 1.0, "eer": 0.1875845857967091, "hter": 0.2317720081360654, "locked_source_threshold": 5.130859375, "run_id": "C_FULL_crop15", "scored_n": 1655, "threshold_policy": "source-frozen", "tpr_at_fpr_1pct": 0.4381188118811881, "unit": "frame"},
{"acer": 0.1819613616037863, "apcer": 0.12629896083133493, "auc": 0.9148601910550767, "binary_accuracy": 0.8465256797583082, "bpcer": 0.2376237623762376, "candidate_n": 1655, "casia_calibrated_threshold": 2.375, "dataset": "casia_train", "detector_coverage": 1.0, "eer": 0.1875845857967091, "fit_split": "train", "hter": 0.1819613616037863, "run_id": "C_FULL_crop15", "scored_n": 1655, "source_threshold": 5.130859375, "threshold_policy": "CASIA train-calibrated", "tpr_at_fpr_1pct": 0.4381188118811881, "unit": "frame"},
{"acer": 0.25704627689963816, "apcer": 0.0056179775280898875, "auc": 0.9209674347743286, "binary_accuracy": 0.869198312236287, "bpcer": 0.5084745762711864, "candidate_n": 240, "dataset": "casia_train", "detector_coverage": 0.9875, "eer": 0.2028185107598553, "hter": 0.25704627689963816, "locked_source_threshold": 5.130859375, "run_id": "C_FULL_crop15", "scored_n": 237, "threshold_policy": "source-frozen", "tpr_at_fpr_1pct": 0.4915254237288136, "unit": "video"},
{"acer": 0.1714911445438964, "apcer": 0.29213483146067415, "auc": 0.9209674347743286, "binary_accuracy": 0.7679324894514767, "bpcer": 0.05084745762711865, "candidate_n": 240, "casia_calibrated_threshold": -1.1993669, "dataset": "casia_train", "detector_coverage": 0.9875, "eer": 0.2028185107598553, "fit_split": "train", "hter": 0.1714911445438964, "run_id": "C_FULL_crop15", "scored_n": 237, "source_threshold": 5.130859375, "threshold_policy": "CASIA train-calibrated", "tpr_at_fpr_1pct": 0.4915254237288136, "unit": "video"},
{"acer": 0.2853120602841932, "apcer": 0.029168959823885527, "auc": 0.8831448986680598, "binary_accuracy": 0.8450996677740864, "bpcer": 0.5414551607445008, "candidate_n": 2408, "dataset": "casia_test", "detector_coverage": 1.0, "eer": 0.20807852515302455, "hter": 0.2853120602841932, "locked_source_threshold": 5.130859375, "run_id": "C_FULL_crop15", "scored_n": 2408, "threshold_policy": "source-frozen", "tpr_at_fpr_1pct": 0.3096446700507614, "unit": "frame"},
{"acer": 0.22311930843034433, "apcer": 0.10952118877270225, "auc": 0.8831448986680598, "binary_accuracy": 0.834717607973422, "bpcer": 0.33671742808798644, "candidate_n": 2408, "casia_calibrated_threshold": 2.375, "dataset": "casia_test", "detector_coverage": 1.0, "eer": 0.20807852515302455, "fit_split": "train", "hter": 0.22311930843034433, "run_id": "C_FULL_crop15", "scored_n": 2408, "source_threshold": 5.130859375, "threshold_policy": "CASIA train-calibrated", "tpr_at_fpr_1pct": 0.3096446700507614, "unit": "frame"},
{"acer": 0.29924242424242425, "apcer": 0.007575757575757576, "auc": 0.9087465564738292, "binary_accuracy": 0.8465909090909091, "bpcer": 0.5909090909090909, "candidate_n": 360, "dataset": "casia_test", "detector_coverage": 0.9777777777777777, "eer": 0.20265151515151514, "hter": 0.29924242424242425, "locked_source_threshold": 5.130859375, "run_id": "C_FULL_crop15", "scored_n": 352, "threshold_policy": "source-frozen", "tpr_at_fpr_1pct": 0.42045454545454547, "unit": "video"},
{"acer": 0.21212121212121213, "apcer": 0.2878787878787879, "auc": 0.9087465564738292, "binary_accuracy": 0.75, "bpcer": 0.13636363636363635, "candidate_n": 360, "casia_calibrated_threshold": -1.1993669, "dataset": "casia_test", "detector_coverage": 0.9777777777777777, "eer": 0.20265151515151514, "fit_split": "train", "hter": 0.21212121212121213, "run_id": "C_FULL_crop15", "scored_n": 352, "source_threshold": 5.130859375, "threshold_policy": "CASIA train-calibrated", "tpr_at_fpr_1pct": 0.42045454545454547, "unit": "video"},
{"acer": 0.2663376573701369, "apcer": 0.029707955689828803, "auc": 0.8917724516720904, "binary_accuracy": 0.850375939849624, "bpcer": 0.5029673590504451, "candidate_n": 2660, "dataset": "casia_combined", "detector_coverage": 1.0, "eer": 0.2000980154852513, "hter": 0.2663376573701369, "locked_source_threshold": 5.130859375, "run_id": "C_FULL_crop15", "scored_n": 2660, "threshold_policy": "source-frozen", "tpr_at_fpr_1pct": 0.3514243323442136, "unit": "frame"},
{"acer": 0.20845846743226323, "apcer": 0.11127895266868076, "auc": 0.8917724516720904, "binary_accuracy": 0.8394736842105263, "bpcer": 0.3056379821958457, "candidate_n": 2660, "casia_calibrated_threshold": 2.375, "dataset": "casia_combined", "detector_coverage": 1.0, "eer": 0.2000980154852513, "fit_split": "train", "hter": 0.20845846743226323, "run_id": "C_FULL_crop15", "scored_n": 2660, "source_threshold": 5.130859375, "threshold_policy": "CASIA train-calibrated", "tpr_at_fpr_1pct": 0.3514243323442136, "unit": "frame"},
{"acer": 0.2815091210613599, "apcer": 0.007462686567164179, "auc": 0.9203980099502487, "binary_accuracy": 0.8547486033519553, "bpcer": 0.5555555555555556, "candidate_n": 360, "dataset": "casia_combined", "detector_coverage": 0.9944444444444445, "eer": 0.16728855721393038, "hter": 0.2815091210613599, "locked_source_threshold": 5.130859375, "run_id": "C_FULL_crop15", "scored_n": 358, "threshold_policy": "source-frozen", "tpr_at_fpr_1pct": 0.4444444444444444, "unit": "video"},
{"acer": 0.1954809286898839, "apcer": 0.2798507462686567, "auc": 0.9203980099502487, "binary_accuracy": 0.7625698324022346, "bpcer": 0.1111111111111111, "candidate_n": 360, "casia_calibrated_threshold": -1.1993669, "dataset": "casia_combined", "detector_coverage": 0.9944444444444445, "eer": 0.16728855721393038, "fit_split": "train", "hter": 0.1954809286898839, "run_id": "C_FULL_crop15", "scored_n": 358, "source_threshold": 5.130859375, "threshold_policy": "CASIA train-calibrated", "tpr_at_fpr_1pct": 0.4444444444444444, "unit": "video"},
{"acer": 0.043548464829030464, "apcer": 0.04697409437815758, "auc": 0.9922418127454586, "binary_accuracy": 0.9550761536027961, "bpcer": 0.040122835279903346, "candidate_n": 67170, "dataset": "test", "detector_coverage": 0.9882239094834003, "eer": 0.04392433472908691, "hter": 0.043548464829030464, "locked_source_threshold": 5.3037109375, "run_id": "P3_SF_FULL_crop15", "scored_n": 66379, "threshold_policy": "source-frozen", "tpr_at_fpr_1pct": 0.8648686065243657, "unit": "image"},
{"acer": 0.2663011897027331, "apcer": 0.06288061181135816, "auc": 0.8812616894024506, "binary_accuracy": 0.8770976699263552, "bpcer": 0.469721767594108, "candidate_n": 8299, "dataset": "lcc_training", "detector_coverage": 0.9980720568743222, "eer": 0.19960973707956692, "hter": 0.2663011897027331, "locked_source_threshold": 5.3037109375, "run_id": "P3_SF_FULL_crop15", "scored_n": 8283, "threshold_policy": "source-frozen", "tpr_at_fpr_1pct": 0.26072831423895254, "unit": "image"},
{"acer": 0.25890812133927793, "apcer": 0.0955940204563336, "auc": 0.8528377577682587, "binary_accuracy": 0.8595181540549711, "bpcer": 0.4222222222222222, "candidate_n": 2948, "dataset": "lcc_development", "detector_coverage": 0.9996607869742198, "eer": 0.2246588182727705, "hter": 0.25890812133927793, "locked_source_threshold": 5.3037109375, "run_id": "P3_SF_FULL_crop15", "scored_n": 2947, "threshold_policy": "source-frozen", "tpr_at_fpr_1pct": 0.2617283950617284, "unit": "image"},
{"acer": 0.2733055128486712, "apcer": 0.08482758620689655, "auc": 0.8196867999121459, "binary_accuracy": 0.8995240613432046, "bpcer": 0.46178343949044587, "candidate_n": 7580, "dataset": "lcc_evaluation", "detector_coverage": 0.9978891820580474, "eer": 0.24827278717329226, "hter": 0.2733055128486712, "locked_source_threshold": 5.3037109375, "run_id": "P3_SF_FULL_crop15", "scored_n": 7564, "threshold_policy": "source-frozen", "tpr_at_fpr_1pct": 0.15286624203821655, "unit": "image"},
{"acer": 0.2678914037811517, "apcer": 0.0772562748472082, "auc": 0.8608309944893372, "binary_accuracy": 0.8833670320314995, "bpcer": 0.4585265327150953, "candidate_n": 18827, "dataset": "lcc_combined", "detector_coverage": 0.9982471981728369, "eer": 0.21790647943931213, "hter": 0.2678914037811517, "locked_source_threshold": 5.3037109375, "run_id": "P3_SF_FULL_crop15", "scored_n": 18794, "threshold_policy": "source-frozen", "tpr_at_fpr_1pct": 0.22568778979907267, "unit": "image"},
{"acer": 0.2716598998029299, "apcer": 0.03836930455635491, "auc": 0.9006092155978187, "binary_accuracy": 0.8477341389728097, "bpcer": 0.504950495049505, "candidate_n": 1655, "dataset": "casia_train", "detector_coverage": 1.0, "eer": 0.20300393348687384, "hter": 0.2716598998029299, "locked_source_threshold": 5.3037109375, "run_id": "P3_SF_FULL_crop15", "scored_n": 1655, "threshold_policy": "source-frozen", "tpr_at_fpr_1pct": 0.30198019801980197, "unit": "frame"},
{"acer": 0.1942218502425782, "apcer": 0.1334932054356515, "auc": 0.9006092155978187, "binary_accuracy": 0.8368580060422961, "bpcer": 0.25495049504950495, "candidate_n": 1655, "casia_calibrated_threshold": 1.7167969, "dataset": "casia_train", "detector_coverage": 1.0, "eer": 0.20300393348687384, "fit_split": "train", "hter": 0.1942218502425782, "run_id": "P3_SF_FULL_crop15", "scored_n": 1655, "source_threshold": 5.3037109375, "threshold_policy": "CASIA train-calibrated", "tpr_at_fpr_1pct": 0.30198019801980197, "unit": "frame"},
{"acer": 0.3106551133117501, "apcer": 0.028089887640449437, "auc": 0.9026852028185108, "binary_accuracy": 0.8312236286919831, "bpcer": 0.5932203389830508, "candidate_n": 240, "dataset": "casia_train", "detector_coverage": 0.9875, "eer": 0.21691106455913162, "hter": 0.3106551133117501, "locked_source_threshold": 5.3037109375, "run_id": "P3_SF_FULL_crop15", "scored_n": 237, "threshold_policy": "source-frozen", "tpr_at_fpr_1pct": 0.3389830508474576, "unit": "video"},
{"acer": 0.1916777756617787, "apcer": 0.14606741573033707, "auc": 0.9026852028185108, "binary_accuracy": 0.8312236286919831, "bpcer": 0.23728813559322035, "candidate_n": 240, "casia_calibrated_threshold": 0.8730957, "dataset": "casia_train", "detector_coverage": 0.9875, "eer": 0.21691106455913162, "fit_split": "train", "hter": 0.1916777756617787, "run_id": "P3_SF_FULL_crop15", "scored_n": 237, "source_threshold": 5.3037109375, "threshold_policy": "CASIA train-calibrated", "tpr_at_fpr_1pct": 0.3389830508474576, "unit": "video"},
{"acer": 0.29715639192547916, "apcer": 0.029168959823885527, "auc": 0.8911199640172204, "binary_accuracy": 0.8392857142857143, "bpcer": 0.5651438240270727, "candidate_n": 2408, "dataset": "casia_test", "detector_coverage": 1.0, "eer": 0.20111710513695158, "hter": 0.29715639192547916, "locked_source_threshold": 5.3037109375, "run_id": "P3_SF_FULL_crop15", "scored_n": 2408, "threshold_policy": "source-frozen", "tpr_at_fpr_1pct": 0.27749576988155666, "unit": "frame"},
{"acer": 0.21426097013820405, "apcer": 0.11887727022564668, "auc": 0.8911199640172204, "binary_accuracy": 0.8343023255813954, "bpcer": 0.3096446700507614, "candidate_n": 2408, "casia_calibrated_threshold": 1.7167969, "dataset": "casia_test", "detector_coverage": 1.0, "eer": 0.20111710513695158, "fit_split": "train", "hter": 0.21426097013820405, "run_id": "P3_SF_FULL_crop15", "scored_n": 2408, "source_threshold": 5.3037109375, "threshold_policy": "CASIA train-calibrated", "tpr_at_fpr_1pct": 0.27749576988155666, "unit": "frame"},
{"acer": 0.29924242424242425, "apcer": 0.01893939393939394, "auc": 0.9118026859504134, "binary_accuracy": 0.8409090909090909, "bpcer": 0.5795454545454546, "candidate_n": 360, "dataset": "casia_test", "detector_coverage": 0.9777777777777777, "eer": 0.17992424242424238, "hter": 0.29924242424242425, "locked_source_threshold": 5.3037109375, "run_id": "P3_SF_FULL_crop15", "scored_n": 352, "threshold_policy": "source-frozen", "tpr_at_fpr_1pct": 0.3977272727272727, "unit": "video"},
{"acer": 0.17045454545454544, "apcer": 0.11363636363636363, "auc": 0.9118026859504134, "binary_accuracy": 0.8579545454545454, "bpcer": 0.22727272727272727, "candidate_n": 360, "casia_calibrated_threshold": 0.8730957, "dataset": "casia_test", "detector_coverage": 0.9777777777777777, "eer": 0.17992424242424238, "fit_split": "train", "hter": 0.17045454545454544, "run_id": "P3_SF_FULL_crop15", "scored_n": 352, "source_threshold": 5.3037109375, "threshold_policy": "CASIA train-calibrated", "tpr_at_fpr_1pct": 0.3977272727272727, "unit": "video"},
{"acer": 0.280922690286008, "apcer": 0.02920443101711984, "auc": 0.8986637919442029, "binary_accuracy": 0.8432330827067669, "bpcer": 0.5326409495548962, "candidate_n": 2660, "dataset": "casia_combined", "detector_coverage": 1.0, "eer": 0.19160533228145987, "hter": 0.280922690286008, "locked_source_threshold": 5.3037109375, "run_id": "P3_SF_FULL_crop15", "scored_n": 2660, "threshold_policy": "source-frozen", "tpr_at_fpr_1pct": 0.3026706231454006, "unit": "frame"},
{"acer": 0.20060378136570237, "apcer": 0.11782477341389729, "auc": 0.8986637919442029, "binary_accuracy": 0.8402255639097744, "bpcer": 0.28338278931750743, "candidate_n": 2660, "casia_calibrated_threshold": 1.7167969, "dataset": "casia_combined", "detector_coverage": 1.0, "eer": 0.19160533228145987, "fit_split": "train", "hter": 0.20060378136570237, "run_id": "P3_SF_FULL_crop15", "scored_n": 2660, "source_threshold": 5.3037109375, "threshold_policy": "CASIA train-calibrated", "tpr_at_fpr_1pct": 0.3026706231454006, "unit": "frame"},
{"acer": 0.2907960199004975, "apcer": 0.014925373134328358, "auc": 0.9222636815920399, "binary_accuracy": 0.8463687150837989, "bpcer": 0.5666666666666667, "candidate_n": 360, "dataset": "casia_combined", "detector_coverage": 0.9944444444444445, "eer": 0.17475124378109447, "hter": 0.2907960199004975, "locked_source_threshold": 5.3037109375, "run_id": "P3_SF_FULL_crop15", "scored_n": 358, "threshold_policy": "source-frozen", "tpr_at_fpr_1pct": 0.4222222222222222, "unit": "video"},
{"acer": 0.16708126036484244, "apcer": 0.11194029850746269, "auc": 0.9222636815920399, "binary_accuracy": 0.8603351955307262, "bpcer": 0.2222222222222222, "candidate_n": 360, "casia_calibrated_threshold": 0.8730957, "dataset": "casia_combined", "detector_coverage": 0.9944444444444445, "eer": 0.17475124378109447, "fit_split": "train", "hter": 0.16708126036484244, "run_id": "P3_SF_FULL_crop15", "scored_n": 358, "source_threshold": 5.3037109375, "threshold_policy": "CASIA train-calibrated", "tpr_at_fpr_1pct": 0.4222222222222222, "unit": "video"},
{"acer": 0.04345889087610218, "apcer": 0.04478125335913146, "auc": 0.9922802748683524, "binary_accuracy": 0.9560101839437172, "bpcer": 0.04213652839307289, "candidate_n": 67170, "dataset": "test", "detector_coverage": 0.9882239094834003, "eer": 0.0435187302248215, "hter": 0.04345889087610218, "locked_source_threshold": 4.66259765625, "run_id": "R7_SC_FULL_crop15", "scored_n": 66379, "threshold_policy": "source-frozen", "tpr_at_fpr_1pct": 0.8715414820781313, "unit": "image"},
{"acer": 0.27148671235534344, "apcer": 0.05934003682197989, "auc": 0.8774889199125415, "binary_accuracy": 0.8780635035615115, "bpcer": 0.483633387888707, "candidate_n": 8299, "dataset": "lcc_training", "detector_coverage": 0.9980720568743222, "eer": 0.20377990858710543, "hter": 0.27148671235534344, "locked_source_threshold": 4.66259765625, "run_id": "R7_SC_FULL_crop15", "scored_n": 8283, "threshold_policy": "source-frozen", "tpr_at_fpr_1pct": 0.2225859247135843, "unit": "image"},
{"acer": 0.2634656292799487, "apcer": 0.11211644374508262, "auc": 0.8546774679216327, "binary_accuracy": 0.846284356973193, "bpcer": 0.4148148148148148, "candidate_n": 2948, "dataset": "lcc_development", "detector_coverage": 0.9996607869742198, "eer": 0.22929160474400445, "hter": 0.2634656292799487, "locked_source_threshold": 4.66259765625, "run_id": "R7_SC_FULL_crop15", "scored_n": 2947, "threshold_policy": "source-frozen", "tpr_at_fpr_1pct": 0.19506172839506172, "unit": "image"},
{"acer": 0.26554447616955856, "apcer": 0.08841379310344828, "auc": 0.8400595211948167, "binary_accuracy": 0.8968799576943416, "bpcer": 0.4426751592356688, "candidate_n": 7580, "dataset": "lcc_evaluation", "detector_coverage": 0.9978891820580474, "eer": 0.24184669448715135, "hter": 0.26554447616955856, "locked_source_threshold": 4.66259765625, "run_id": "R7_SC_FULL_crop15", "scored_n": 7564, "threshold_policy": "source-frozen", "tpr_at_fpr_1pct": 0.1910828025477707, "unit": "image"},
{"acer": 0.27122793444407445, "apcer": 0.07980774936213138, "auc": 0.860272080856274, "binary_accuracy": 0.8806534000212833, "bpcer": 0.46264811952601753, "candidate_n": 18827, "dataset": "lcc_combined", "detector_coverage": 0.9982471981728369, "eer": 0.22055921750012597, "hter": 0.27122793444407445, "locked_source_threshold": 4.66259765625, "run_id": "R7_SC_FULL_crop15", "scored_n": 18794, "threshold_policy": "source-frozen", "tpr_at_fpr_1pct": 0.19989696032972695, "unit": "image"},
{"acer": 0.22887234766642134, "apcer": 0.05675459632294165, "auc": 0.9118269740643128, "binary_accuracy": 0.859214501510574, "bpcer": 0.400990099009901, "candidate_n": 1655, "dataset": "casia_train", "detector_coverage": 1.0, "eer": 0.18231157648138913, "hter": 0.22887234766642134, "locked_source_threshold": 4.66259765625, "run_id": "R7_SC_FULL_crop15", "scored_n": 1655, "threshold_policy": "source-frozen", "tpr_at_fpr_1pct": 0.4381188118811881, "unit": "frame"},
{"acer": 0.1727054000364065, "apcer": 0.1374900079936051, "auc": 0.9118269740643128, "binary_accuracy": 0.8453172205438066, "bpcer": 0.2079207920792079, "candidate_n": 1655, "casia_calibrated_threshold": 1.6586914, "dataset": "casia_train", "detector_coverage": 1.0, "eer": 0.18231157648138913, "fit_split": "train", "hter": 0.1727054000364065, "run_id": "R7_SC_FULL_crop15", "scored_n": 1655, "source_threshold": 4.66259765625, "threshold_policy": "CASIA train-calibrated", "tpr_at_fpr_1pct": 0.4381188118811881, "unit": "frame"},
{"acer": 0.25133307941344507, "apcer": 0.028089887640449437, "auc": 0.9191582555703675, "binary_accuracy": 0.8607594936708861, "bpcer": 0.4745762711864407, "candidate_n": 240, "dataset": "casia_train", "detector_coverage": 0.9875, "eer": 0.1690154256332127, "hter": 0.25133307941344507, "locked_source_threshold": 4.66259765625, "run_id": "R7_SC_FULL_crop15", "scored_n": 237, "threshold_policy": "source-frozen", "tpr_at_fpr_1pct": 0.4576271186440678, "unit": "video"},
{"acer": 0.1548752618548848, "apcer": 0.17415730337078653, "auc": 0.9191582555703675, "binary_accuracy": 0.8354430379746836, "bpcer": 0.13559322033898305, "candidate_n": 240, "casia_calibrated_threshold": -0.033447266, "dataset": "casia_train", "detector_coverage": 0.9875, "eer": 0.1690154256332127, "fit_split": "train", "hter": 0.1548752618548848, "run_id": "R7_SC_FULL_crop15", "scored_n": 237, "source_threshold": 4.66259765625, "threshold_policy": "CASIA train-calibrated", "tpr_at_fpr_1pct": 0.4576271186440678, "unit": "video"},
{"acer": 0.25720889474943825, "apcer": 0.04402861860209136, "auc": 0.903210606352674, "binary_accuracy": 0.8513289036544851, "bpcer": 0.4703891708967851, "candidate_n": 2408, "dataset": "casia_test", "detector_coverage": 1.0, "eer": 0.18746944397106846, "hter": 0.25720889474943825, "locked_source_threshold": 4.66259765625, "run_id": "R7_SC_FULL_crop15", "scored_n": 2408, "threshold_policy": "source-frozen", "tpr_at_fpr_1pct": 0.40609137055837563, "unit": "frame"},
{"acer": 0.19369006944192235, "apcer": 0.14034122179416622, "auc": 0.903210606352674, "binary_accuracy": 0.8334717607973422, "bpcer": 0.2470389170896785, "candidate_n": 2408, "casia_calibrated_threshold": 1.6586914, "dataset": "casia_test", "detector_coverage": 1.0, "eer": 0.18746944397106846, "fit_split": "train", "hter": 0.19369006944192235, "run_id": "R7_SC_FULL_crop15", "scored_n": 2408, "source_threshold": 4.66259765625, "threshold_policy": "CASIA train-calibrated", "tpr_at_fpr_1pct": 0.40609137055837563, "unit": "frame"},
{"acer": 0.2765151515151515, "apcer": 0.01893939393939394, "auc": 0.9258780991735538, "binary_accuracy": 0.8522727272727273, "bpcer": 0.5340909090909091, "candidate_n": 360, "dataset": "casia_test", "detector_coverage": 0.9777777777777777, "eer": 0.18181818181818182, "hter": 0.2765151515151515, "locked_source_threshold": 4.66259765625, "run_id": "R7_SC_FULL_crop15", "scored_n": 352, "threshold_policy": "source-frozen", "tpr_at_fpr_1pct": 0.45454545454545453, "unit": "video"},
{"acer": 0.1875, "apcer": 0.20454545454545456, "auc": 0.9258780991735538, "binary_accuracy": 0.8039772727272727, "bpcer": 0.17045454545454544, "candidate_n": 360, "casia_calibrated_threshold": -0.033447266, "dataset": "casia_test", "detector_coverage": 0.9777777777777777, "eer": 0.18181818181818182, "fit_split": "train", "hter": 0.1875, "run_id": "R7_SC_FULL_crop15", "scored_n": 352, "source_threshold": 4.66259765625, "threshold_policy": "CASIA train-calibrated", "tpr_at_fpr_1pct": 0.45454545454545453, "unit": "video"},
{"acer": 0.24421693695632035, "apcer": 0.04481369587109768, "auc": 0.909414118413464, "binary_accuracy": 0.8541353383458646, "bpcer": 0.443620178041543, "candidate_n": 2660, "dataset": "casia_combined", "detector_coverage": 1.0, "eer": 0.18238425656150925, "hter": 0.24421693695632035, "locked_source_threshold": 4.66259765625, "run_id": "R7_SC_FULL_crop15", "scored_n": 2660, "threshold_policy": "source-frozen", "tpr_at_fpr_1pct": 0.4035608308605341, "unit": "frame"},
{"acer": 0.18323965084971655, "apcer": 0.13947633434038267, "auc": 0.909414118413464, "binary_accuracy": 0.8383458646616542, "bpcer": 0.22700296735905046, "candidate_n": 2660, "casia_calibrated_threshold": 1.6586914, "dataset": "casia_combined", "detector_coverage": 1.0, "eer": 0.18238425656150925, "fit_split": "train", "hter": 0.18323965084971655, "run_id": "R7_SC_FULL_crop15", "scored_n": 2660, "source_threshold": 4.66259765625, "threshold_policy": "CASIA train-calibrated", "tpr_at_fpr_1pct": 0.4035608308605341, "unit": "frame"},
{"acer": 0.26488391376451076, "apcer": 0.018656716417910446, "auc": 0.9351990049751244, "binary_accuracy": 0.8575418994413407, "bpcer": 0.5111111111111111, "candidate_n": 360, "dataset": "casia_combined", "detector_coverage": 0.9944444444444445, "eer": 0.16915422885572134, "hter": 0.26488391376451076, "locked_source_threshold": 4.66259765625, "run_id": "R7_SC_FULL_crop15", "scored_n": 358, "threshold_policy": "source-frozen", "tpr_at_fpr_1pct": 0.45555555555555555, "unit": "video"},
{"acer": 0.1673714759535655, "apcer": 0.19029850746268656, "auc": 0.9351990049751244, "binary_accuracy": 0.8212290502793296, "bpcer": 0.14444444444444443, "candidate_n": 360, "casia_calibrated_threshold": -0.033447266, "dataset": "casia_combined", "detector_coverage": 0.9944444444444445, "eer": 0.16915422885572134, "fit_split": "train", "hter": 0.1673714759535655, "run_id": "R7_SC_FULL_crop15", "scored_n": 358, "source_threshold": 4.66259765625, "threshold_policy": "CASIA train-calibrated", "tpr_at_fpr_1pct": 0.45555555555555555, "unit": "video"},
{"acer": 0.04449824538345215, "apcer": 0.05073632161668279, "auc": 0.9921439799017713, "binary_accuracy": 0.9529971828439717, "bpcer": 0.038260169150221505, "candidate_n": 67170, "dataset": "test", "detector_coverage": 0.9882239094834003, "eer": 0.04475704217898091, "hter": 0.04449824538345215, "locked_source_threshold": 4.5966796875, "run_id": "CE_CLEAN_FULL_crop15", "scored_n": 66379, "threshold_policy": "source-frozen", "tpr_at_fpr_1pct": 0.8544829842931937, "unit": "image"},
{"acer": 0.28268379524605663, "apcer": 0.056365953830902135, "auc": 0.8702520657603567, "binary_accuracy": 0.8768562115175661, "bpcer": 0.5090016366612111, "candidate_n": 8299, "dataset": "lcc_training", "detector_coverage": 0.9980720568743222, "eer": 0.22532317742673103, "hter": 0.28268379524605663, "locked_source_threshold": 4.5966796875, "run_id": "CE_CLEAN_FULL_crop15", "scored_n": 8283, "threshold_policy": "source-frozen", "tpr_at_fpr_1pct": 0.2479541734860884, "unit": "image"},
{"acer": 0.32675496109799806, "apcer": 0.0979543666404406, "auc": 0.8176146904838224, "binary_accuracy": 0.8391584662368511, "bpcer": 0.5555555555555556, "candidate_n": 2948, "dataset": "lcc_development", "detector_coverage": 0.9996607869742198, "eer": 0.2696977202746938, "hter": 0.32675496109799806, "locked_source_threshold": 4.5966796875, "run_id": "CE_CLEAN_FULL_crop15", "scored_n": 2947, "threshold_policy": "source-frozen", "tpr_at_fpr_1pct": 0.16296296296296298, "unit": "image"},
{"acer": 0.2902758620689655, "apcer": 0.08055172413793103, "auc": 0.8126872391829563, "binary_accuracy": 0.9020359598096246, "bpcer": 0.5, "candidate_n": 7580, "dataset": "lcc_evaluation", "detector_coverage": 0.9978891820580474, "eer": 0.27335646826268395, "hter": 0.2902758620689655, "locked_source_threshold": 4.5966796875, "run_id": "CE_CLEAN_FULL_crop15", "scored_n": 7564, "threshold_policy": "source-frozen", "tpr_at_fpr_1pct": 0.17993630573248406, "unit": "image"},
{"acer": 0.29515125991874525, "apcer": 0.0730433750667537, "auc": 0.8439879397180328, "binary_accuracy": 0.8810790677875918, "bpcer": 0.5172591447707368, "candidate_n": 18827, "dataset": "lcc_combined", "detector_coverage": 0.9982471981728369, "eer": 0.2446227070073732, "hter": 0.29515125991874525, "locked_source_threshold": 4.5966796875, "run_id": "CE_CLEAN_FULL_crop15", "scored_n": 18794, "threshold_policy": "source-frozen", "tpr_at_fpr_1pct": 0.20132148377125195, "unit": "image"},
{"acer": 0.31747671169994696, "apcer": 0.011191047162270184, "auc": 0.9047000023743381, "binary_accuracy": 0.8392749244712991, "bpcer": 0.6237623762376238, "candidate_n": 1655, "dataset": "casia_train", "detector_coverage": 1.0, "eer": 0.19082061083806223, "hter": 0.31747671169994696, "locked_source_threshold": 4.5966796875, "run_id": "CE_CLEAN_FULL_crop15", "scored_n": 1655, "threshold_policy": "source-frozen", "tpr_at_fpr_1pct": 0.3688118811881188, "unit": "frame"},
{"acer": 0.1748195503003538, "acer_reduction": 0.14265716139959317, "apcer": 0.1342925659472422, "auc": 0.9047000023743381, "binary_accuracy": 0.8459214501510574, "bpcer": 0.21534653465346534, "candidate_n": 1655, "casia_calibrated_threshold": -0.84521484, "dataset": "casia_train", "detector_coverage": 1.0, "eer": 0.19082061083806223, "evaluation_policy": "CASIA train threshold transferred unchanged; no test/combined fitting", "fit_n": 1655, "fit_split": "train", "hter": 0.1748195503003538, "hter_reduction": 0.14265716139959317, "objective": "minimum pooled binary ACER/HTER", "run_id": "CE_CLEAN_FULL_crop15", "scored_n": 1655, "source_acer": 0.31747671169994696, "source_hter": 0.31747671169994696, "source_threshold": 4.5966796875, "split": "train", "status": "calibrated", "threshold_policy": "CASIA train-calibrated", "tie_break": "closest to source threshold", "tpr_at_fpr_1pct": 0.3688118811881188, "train_test_copy_overlap": {"frame_ids": 1403, "subjects": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], "video_ids": 240}, "unit": "frame"},
{"acer": 0.34179203961150256, "apcer": 0.0056179775280898875, "auc": 0.9123976385450391, "binary_accuracy": 0.8270042194092827, "bpcer": 0.6779661016949152, "candidate_n": 240, "dataset": "casia_train", "detector_coverage": 0.9875, "eer": 0.19153494572462382, "hter": 0.34179203961150256, "locked_source_threshold": 4.5966796875, "run_id": "CE_CLEAN_FULL_crop15", "scored_n": 237, "threshold_policy": "source-frozen", "tpr_at_fpr_1pct": 0.4406779661016949, "unit": "video"},
{"acer": 0.1714911445438964, "acer_reduction": 0.17030089506760615, "apcer": 0.29213483146067415, "auc": 0.9123976385450391, "binary_accuracy": 0.7679324894514767, "bpcer": 0.05084745762711865, "candidate_n": 240, "casia_calibrated_threshold": -3.428955, "dataset": "casia_train", "detector_coverage": 0.9875, "eer": 0.19153494572462382, "evaluation_policy": "CASIA train threshold transferred unchanged; no test/combined fitting", "fit_n": 237, "fit_split": "train", "hter": 0.1714911445438964, "hter_reduction": 0.17030089506760615, "objective": "minimum pooled binary ACER/HTER", "run_id": "CE_CLEAN_FULL_crop15", "scored_n": 237, "source_acer": 0.34179203961150256, "source_hter": 0.34179203961150256, "source_threshold": 4.5966796875, "split": "train", "status": "calibrated", "threshold_policy": "CASIA train-calibrated", "tie_break": "closest to source threshold", "tpr_at_fpr_1pct": 0.4406779661016949, "train_test_copy_overlap": {"frame_ids": 1403, "subjects": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], "video_ids": 240}, "unit": "video"},
{"acer": 0.3644928001847563, "apcer": 0.011557512383048982, "auc": 0.8620939482067744, "binary_accuracy": 0.8151993355481728, "bpcer": 0.7174280879864636, "candidate_n": 2408, "dataset": "casia_test", "detector_coverage": 1.0, "eer": 0.2167115054565501, "hter": 0.3644928001847563, "locked_source_threshold": 4.5966796875, "run_id": "CE_CLEAN_FULL_crop15", "scored_n": 2408, "threshold_policy": "source-frozen", "tpr_at_fpr_1pct": 0.2707275803722504, "unit": "frame"},
{"acer": 0.2147880470867824, "acer_reduction": 0.14970475309797393, "apcer": 0.16730875068794718, "auc": 0.8620939482067744, "binary_accuracy": 0.8093853820598007, "bpcer": 0.2622673434856176, "candidate_n": 2408, "casia_calibrated_threshold": -0.84521484, "dataset": "casia_test", "detector_coverage": 1.0, "eer": 0.2167115054565501, "evaluation_policy": "CASIA train threshold transferred unchanged; no test/combined fitting", "fit_n": 1655, "fit_split": "train", "hter": 0.2147880470867824, "hter_reduction": 0.14970475309797393, "objective": "minimum pooled binary ACER/HTER", "run_id": "CE_CLEAN_FULL_crop15", "scored_n": 2408, "source_acer": 0.3644928001847563, "source_hter": 0.3644928001847563, "source_threshold": 4.5966796875, "split": "test", "status": "calibrated", "threshold_policy": "CASIA train-calibrated", "tie_break": "closest to source threshold", "tpr_at_fpr_1pct": 0.2707275803722504, "train_test_copy_overlap": {"frame_ids": 1403, "subjects": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], "video_ids": 240}, "unit": "frame"},
{"acer": 0.3712121212121212, "apcer": 0.003787878787878788, "auc": 0.8836949035812672, "binary_accuracy": 0.8125, "bpcer": 0.7386363636363636, "candidate_n": 360, "dataset": "casia_test", "detector_coverage": 0.9777777777777777, "eer": 0.17992424242424238, "hter": 0.3712121212121212, "locked_source_threshold": 4.5966796875, "run_id": "CE_CLEAN_FULL_crop15", "scored_n": 352, "threshold_policy": "source-frozen", "tpr_at_fpr_1pct": 0.375, "unit": "video"},
{"acer": 0.23106060606060608, "acer_reduction": 0.14015151515151514, "apcer": 0.3143939393939394, "auc": 0.8836949035812672, "binary_accuracy": 0.7272727272727273, "bpcer": 0.14772727272727273, "candidate_n": 360, "casia_calibrated_threshold": -3.428955, "dataset": "casia_test", "detector_coverage": 0.9777777777777777, "eer": 0.17992424242424238, "evaluation_policy": "CASIA train threshold transferred unchanged; no test/combined fitting", "fit_n": 237, "fit_split": "train", "hter": 0.23106060606060608, "hter_reduction": 0.14015151515151514, "objective": "minimum pooled binary ACER/HTER", "run_id": "CE_CLEAN_FULL_crop15", "scored_n": 352, "source_acer": 0.3712121212121212, "source_hter": 0.3712121212121212, "source_threshold": 4.5966796875, "split": "test", "status": "calibrated", "threshold_policy": "CASIA train-calibrated", "tie_break": "closest to source threshold", "tpr_at_fpr_1pct": 0.375, "train_test_copy_overlap": {"frame_ids": 1403, "subjects": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], "video_ids": 240}, "unit": "video"},
{"acer": 0.3482687417262081, "apcer": 0.011077542799597181, "auc": 0.8743309994890047, "binary_accuracy": 0.8180451127819549, "bpcer": 0.685459940652819, "candidate_n": 2660, "dataset": "casia_combined", "detector_coverage": 1.0, "eer": 0.20184690459328053, "hter": 0.3482687417262081, "locked_source_threshold": 4.5966796875, "run_id": "CE_CLEAN_FULL_crop15", "scored_n": 2660, "threshold_policy": "source-frozen", "tpr_at_fpr_1pct": 0.3026706231454006, "unit": "frame"},
{"acer": 0.20000014941384947, "acer_reduction": 0.14826859231235862, "apcer": 0.16112789526686808, "auc": 0.8743309994890047, "binary_accuracy": 0.8191729323308271, "bpcer": 0.23887240356083086, "candidate_n": 2660, "casia_calibrated_threshold": -0.84521484, "dataset": "casia_combined", "detector_coverage": 1.0, "eer": 0.20184690459328053, "evaluation_policy": "CASIA train threshold transferred unchanged; no test/combined fitting", "fit_n": 1655, "fit_split": "train", "hter": 0.20000014941384947, "hter_reduction": 0.14826859231235862, "objective": "minimum pooled binary ACER/HTER", "run_id": "CE_CLEAN_FULL_crop15", "scored_n": 2660, "source_acer": 0.3482687417262081, "source_hter": 0.3482687417262081, "source_threshold": 4.5966796875, "split": "combined", "status": "calibrated", "threshold_policy": "CASIA train-calibrated", "tie_break": "closest to source threshold", "tpr_at_fpr_1pct": 0.3026706231454006, "train_test_copy_overlap": {"frame_ids": 1403, "subjects": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], "video_ids": 240}, "unit": "frame"},
{"acer": 0.36853233830845766, "apcer": 0.0037313432835820895, "auc": 0.8983001658374792, "binary_accuracy": 0.8128491620111732, "bpcer": 0.7333333333333333, "candidate_n": 360, "dataset": "casia_combined", "detector_coverage": 0.9944444444444445, "eer": 0.1654228855721393, "hter": 0.36853233830845766, "locked_source_threshold": 4.5966796875, "run_id": "CE_CLEAN_FULL_crop15", "scored_n": 358, "threshold_policy": "source-frozen", "tpr_at_fpr_1pct": 0.4222222222222222, "unit": "video"},
{"acer": 0.21969320066334994, "acer_reduction": 0.14883913764510773, "apcer": 0.31716417910447764, "auc": 0.8983001658374792, "binary_accuracy": 0.7318435754189944, "bpcer": 0.12222222222222222, "candidate_n": 360, "casia_calibrated_threshold": -3.428955, "dataset": "casia_combined", "detector_coverage": 0.9944444444444445, "eer": 0.1654228855721393, "evaluation_policy": "CASIA train threshold transferred unchanged; no test/combined fitting", "fit_n": 237, "fit_split": "train", "hter": 0.21969320066334994, "hter_reduction": 0.14883913764510773, "objective": "minimum pooled binary ACER/HTER", "run_id": "CE_CLEAN_FULL_crop15", "scored_n": 358, "source_acer": 0.36853233830845766, "source_hter": 0.36853233830845766, "source_threshold": 4.5966796875, "split": "combined", "status": "calibrated", "threshold_policy": "CASIA train-calibrated", "tie_break": "closest to source threshold", "tpr_at_fpr_1pct": 0.4222222222222222, "train_test_copy_overlap": {"frame_ids": 1403, "subjects": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], "video_ids": 240}, "unit": "video"}
]
```
