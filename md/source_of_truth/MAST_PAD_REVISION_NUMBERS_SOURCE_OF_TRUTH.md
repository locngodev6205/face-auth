# MAST-PAD — Số liệu đã xác minh phục vụ lượt chỉnh sửa Word và article

**Cập nhật: 22/10/2026.** Tài liệu tập trung số liệu dùng cho lượt sửa sau DG20, bao gồm dữ liệu CE-only và Clean + Aux seed101/102 vừa bổ sung. Không thay thế hay sửa các artifact, draft và scientific locks trước đây. Adaptive threshold hoàn toàn ngoài phạm vi.

## 1. Quy ước sử dụng

- Chỉ các cấu hình huấn luyện trên **484.561 ảnh nguồn khả dụng** được đưa vào so sánh chính; không đưa kết quả 10K/100K/micro/severity vào phần kết quả hoặc phụ lục kết quả.
- Tên trình bày: **CE-only**, **Clean + Aux** (C-clean), **WBST**, **MAST-PAD**. Mã run chỉ dùng để truy nguồn.
- Các cột AUC/EER/APCER/BPCER/ACER dưới đây có đơn vị **%**, trừ khi ghi rõ fraction. Chênh lệch là **điểm phần trăm (pp)**.
- Bảng chung bốn phương pháp vẫn là **seed100**. Các bảng ba seed chỉ áp dụng cho những baseline/population đã có đủ dữ liệu, không trộn ba seed baseline với một seed spectral trong cùng aggregate.
- Mean ± SD tính từ số chưa làm tròn, **sample SD, ddof=1, n=3**, seeds100/101/102. SD của tỷ lệ được biểu diễn bằng pp.
- Nhãn binary0=Real,1=Attack; score=`z_real-z_attack`; quyết định Real khi score≥ngưỡng nguồn đã khóa. AUC/EER không phụ thuộc operating threshold trên cùng score/population; APCER/BPCER/ACER có phụ thuộc.
- EER dùng đúng code hiện hữu: chỉ số ROC đầu tiên tối thiểu hóa |FPR−FNR|, lấy trung bình FPR/FNR tại đó; không nội suy giao điểm.

| Tên | Run ID | Objective sau warmup | Sigma |
| --- | --- | --- | --- |
| CE-only | CE_CLEAN_FULL_crop15 | CE_clean; không aux/spectral | — |
| Clean + Aux | C_FULL_crop15 | CE_clean + L_aux | — |
| WBST | P3_SF_FULL_crop15 | 0.75 CE_clean + 0.25 CE_worst + L_aux | 1/18 |
| MAST-PAD | R7_SC_FULL_crop15 | 0.75 CE_clean + 0.25 CE_harmful + L_aux | 0.10 |

`L_aux=0.1 CE_spoof11+0.1 CE_lighting5+BCE_attributes40(real only)`. WBST/MAST epoch1 chỉ clean+aux, không eligible best checkpoint; từ epoch2 dùng objective trên. MAST chia spectral CE cho số harmful H; H=0 cho differentiable zero, hệ số clean vẫn0.75. So WBST/MAST hiện thay cả sigma và gate, không phải ablation gate riêng.

## 2. Dữ liệu và population chuẩn

| Population | Candidate | Scored/usable | Real | Attack | Vai trò |
| --- | --- | --- | --- | --- | --- |
| CelebA official Train | 494405 | 484561 | 160073 | 324488 | Nguồn huấn luyện |
| CelebA official Test-as-Val | 67170 | 66379 | 19864 | 46515 | Chọn checkpoint/ngưỡng; không untouched test |
| LCC Combined images | 18827 | 18794 | 1941 | 16853 | Gộp training/development/evaluation |
| CASIA Combined frames | 2660 | 2660 | 674 | 1986 | Frame pool đã dedup |
| CASIA Combined reconstructed videos | 360 | 358 | 90 | 268 | Headline CASIA; ≥4 valid frames/video |

| LCC split | Candidate images | Scored images |
| --- | --- | --- |
| Training | 8299 | 8283 |
| Development | 2948 | 2947 |
| Evaluation | 7580 | 7564 |
| Combined | 18827 | 18794 |

CelebA loại9.844 ảnh Train: DET_FAIL4.580, TOO_SMALL4.484, SUSPICIOUS779, INVALID_BBOX1; Test-as-Val không chấm791 ảnh. Official Train/Test giữ membership và có3 identity overlap:5028,7332,9735.

CASIA extracted-image copy: Train1.655frame/240candidate videos, Test2.408frame/360candidate videos; chấm được237Train và352Test videos. Hai bản sao chồng lấn20subjects,240videoIDs,1.403frameIDs. Combined dedup theo(video_id,frame_index), ưu tiên test copy trước detection, không fallback sang train nếu test copy không được chấm. Video score là trung bình logit difference của các valid frames. Không gọi đây là official subject-disjoint protocol.

LCC pooled AUC tính trên score của18.794ảnh nối lại, không trung bình ba AUC split. LCC đã được theo dõi trong phát triển crop/spectral design; không mô tả là blind/unseen holdout hoàn toàn.

## 3. Bảng so sánh chính — đủ bốn phương pháp, seed100

| Phương pháp | CelebA Test-as-Val AUC (%) | LCC Combined image AUC (%) | CASIA Combined video AUC (%) |
| --- | --- | --- | --- |
| CE-only | 99.2144 | 84.3988 | 89.8300 |
| Clean + Aux | 99.0736 | 84.6011 | 92.0398 |
| WBST | 99.2242 | 86.0831 | 92.2264 |
| MAST-PAD | 99.2280 | 86.0272 | 93.5199 |

Hiển thị hai chữ số cho bài báo: CE-only **99.21 / 84.40 / 89.83**; Clean + Aux **99.07 / 84.60 / 92.04**; WBST **99.22 / 86.08 / 92.23**; MAST-PAD **99.23 / 86.03 / 93.52**. Thứ tự source/LCC/CASIA như bảng. Source là selection evidence; headline transfer là LCC/CASIA.

| So sánh seed100 | Δ LCC AUC (pp) | Δ CASIA AUC (pp) |
| --- | --- | --- |
| MAST-PAD − CE-only | +1.628414 | +3.689884 |
| MAST-PAD − Clean + Aux | +1.426081 | +1.480100 |
| MAST-PAD − WBST | -0.055891 | +1.293532 |

MAST-PAD đạt CASIA AUC cao nhất trong bốn strategy; WBST nhỉnh hơn MAST trên LCC0.055891pp. Không diễn giải thành ưu thế trên mọi metric hoặc causal gain riêng của gate.

## 4. EER và operating-point metrics — seed100, ngưỡng nguồn cố định

| Phương pháp | Population | Ngưỡng nguồn | EER (%) | APCER (%) | BPCER (%) | ACER (%) |
| --- | --- | --- | --- | --- | --- | --- |
| CE-only | CelebA Test-as-Val | 4.5966796875 | 4.4757 | 5.0736 | 3.8260 | 4.4498 |
| Clean + Aux | CelebA Test-as-Val | 5.130859375 | 4.6764 | 5.3381 | 3.9116 | 4.6248 |
| WBST | CelebA Test-as-Val | 5.3037109375 | 4.3924 | 4.6974 | 4.0123 | 4.3548 |
| MAST-PAD | CelebA Test-as-Val | 4.6625976562 | 4.3519 | 4.4781 | 4.2137 | 4.3459 |
| CE-only | LCC Combined images | 4.5966796875 | 24.4623 | 7.3043 | 51.7259 | 29.5151 |
| Clean + Aux | LCC Combined images | 5.130859375 | 23.5793 | 8.9064 | 47.1922 | 28.0493 |
| WBST | LCC Combined images | 5.3037109375 | 21.7906 | 7.7256 | 45.8527 | 26.7891 |
| MAST-PAD | LCC Combined images | 4.6625976562 | 22.0559 | 7.9808 | 46.2648 | 27.1228 |
| CE-only | CASIA combined video | 4.5966796875 | 16.5423 | 0.3731 | 73.3333 | 36.8532 |
| Clean + Aux | CASIA combined video | 5.130859375 | 16.7289 | 0.7463 | 55.5556 | 28.1509 |
| WBST | CASIA combined video | 5.3037109375 | 17.4751 | 1.4925 | 56.6667 | 29.0796 |
| MAST-PAD | CASIA combined video | 4.6625976562 | 16.9154 | 1.8657 | 51.1111 | 26.4884 |

**Số cần sửa trong Word:** CE-only CASIA Combined video EER **23.05% →16.54%** (giá trị chưa làm tròn16.5422885572%). MAST LCC EER22.055891% →22.06%; MAST CASIA EER16.915423% →16.92%. AUC cao hơn không bảo đảm EER thấp hơn: CE-only CASIA EER thấp hơn MAST dù AUC thấp hơn.

### Confusion matrix MAST-PAD tại ngưỡng nguồn4.66259765625

| LCC Combined images | Dự đoán Real | Dự đoán Attack | Tổng |
| --- | --- | --- | --- |
| Nhãn Real | 1043 | 898 | 1941 |
| Nhãn Attack | 1345 | 15508 | 16853 |

| CASIA combined video | Dự đoán Real | Dự đoán Attack | Tổng |
| --- | --- | --- | --- |
| Nhãn Real | 44 | 46 | 90 |
| Nhãn Attack | 5 | 263 | 268 |

LCC: APCER=1345/16853=7.9808%; BPCER=898/1941=46.2648%; ACER27.1228%. CASIA: APCER=5/268=1.8657%; BPCER=46/90=51.1111%; ACER26.4884%. Đây là số lỗi tại ngưỡng nguồn, không dựng từ EER.

## 5. Baseline ba seed — số từng seed

| Phương pháp | Seed | Best epoch | Ngưỡng nguồn | Source AUC (%) | LCC Combined AUC (%) | CASIA Combined video AUC (%) |
| --- | --- | --- | --- | --- | --- | --- |
| CE-only | 100 | 9 | 4.5966796875 | 99.214398 | 84.398794 | 89.830017 |
| CE-only | 101 | 9 | 4.6982421875 | 99.202743 | 83.407865 | 87.508292 |
| CE-only | 102 | 12 | 4.9775390625 | 99.171879 | 83.238604 | 91.368159 |
| Clean + Aux | 100 | 9 | 5.130859375 | 99.073553 | 84.601127 | 92.039801 |
| Clean + Aux | 101 | 9 | 5.23046875 | 98.998677 | 85.404207 | Chưa có |
| Clean + Aux | 102 | 9 | 5.0888671875 | 99.172895 | 85.288759 | Chưa có |

## 6. Baseline ba seed — mean ± sample SD

| Phương pháp | Population | AUC (%) | EER (%) | APCER (%) | BPCER (%) | ACER (%) |
| --- | --- | --- | --- | --- | --- | --- |
| CE-only | CASIA combined video | 89.568823 ± 1.943145 | 19.888060 ± 3.256498 | 1.368159 ± 1.723434 | 63.703704 ± 8.981004 | 32.535931 ± 3.850514 |
| CE-only | CelebA Test-as-Val | 99.196340 ± 0.021971 | 4.463381 ± 0.012207 | 5.007704 ± 0.137141 | 3.869647 ± 0.136235 | 4.438675 ± 0.010141 |
| CE-only | LCC combined images | 83.681754 ± 0.626715 | 24.452695 ± 0.282726 | 7.353785 ± 0.101452 | 52.962391 ± 2.053051 | 30.158088 ± 0.999351 |
| Clean + Aux | CelebA Test-as-Val | 99.081708 ± 0.087395 | 4.701483 ± 0.196473 | 5.155326 ± 0.455143 | 4.173379 ± 0.226708 | 4.664353 ± 0.216100 |
| Clean + Aux | LCC combined images | 85.098031 ± 0.434186 | 23.299847 ± 0.243454 | 7.213355 ± 1.476306 | 48.634724 ± 1.947559 | 27.924040 ± 0.669841 |

**Chưa có** aggregate CASIA ba seed cho Clean + Aux; không tính mean±SD WBST/MAST-PAD khi chưa đủ seed.

| Clean + Aux − CE-only, cùng3seed | Δ mean AUC (pp) |
| --- | --- |
| CelebA Test-as-Val | -0.114632 |
| LCC combined images | +1.416277 |

### Chi tiết theo phân vùng — mean ± sample SD, ba seed

| Phương pháp | Population | AUC (%) | EER (%) | APCER (%) | BPCER (%) | ACER (%) |
| --- | --- | --- | --- | --- | --- | --- |
| CE-only | CASIA combined frame | 86.9677 ± 1.7517 | 21.6072 ± 1.7797 | 2.2659 ± 1.6293 | 63.0564 ± 5.1031 | 32.6611 ± 1.8972 |
| CE-only | CASIA test frame | 85.9013 ± 1.8780 | 22.7580 ± 1.9544 | 2.3115 ± 1.6354 | 65.7642 ± 5.7662 | 34.0379 ± 2.1791 |
| CE-only | CASIA test video | 88.2475 ± 1.9592 | 20.9596 ± 3.3685 | 1.5152 ± 1.9682 | 65.1515 ± 9.7974 | 33.3333 ± 3.9908 |
| CE-only | CASIA train frame | 89.5163 ± 1.1612 | 19.9209 ± 0.8532 | 2.7178 ± 1.4653 | 60.3135 ± 1.8905 | 31.5157 ± 0.2126 |
| CE-only | CASIA train video | 90.3415 ± 1.3784 | 19.7169 ± 0.7459 | 0.9363 ± 1.1695 | 64.4068 ± 4.4843 | 32.6716 ± 1.7288 |
| CE-only | LCC development images | 82.0890 ± 0.5931 | 25.8185 ± 2.1193 | 10.1495 ± 1.1244 | 53.7449 ± 2.9250 | 31.9472 ± 0.9375 |
| CE-only | LCC evaluation images | 81.0742 ± 0.4427 | 26.2738 ± 0.9811 | 8.0322 ± 0.4763 | 52.5478 ± 2.5478 | 30.2900 ± 1.4002 |
| CE-only | LCC training images | 86.0130 ± 0.8787 | 22.9653 ± 0.3820 | 5.6508 ± 0.3189 | 52.8096 ± 2.3266 | 29.2302 ± 1.0621 |
| Clean + Aux | LCC development images | 84.7212 ± 0.4356 | 23.5254 ± 0.8378 | 11.1461 ± 0.8550 | 41.1523 ± 1.7862 | 26.1492 ± 0.5027 |
| Clean + Aux | LCC evaluation images | 82.0979 ± 0.8686 | 25.1543 ± 1.4130 | 7.6000 ± 2.3537 | 51.1677 ± 4.5746 | 29.3839 ± 1.6120 |
| Clean + Aux | LCC training images | 87.0062 ± 0.4100 | 21.7770 ± 0.5900 | 5.4006 ± 0.8760 | 50.4637 ± 1.3305 | 27.9321 ± 0.5549 |

## 7. Đính chính bảng CASIA trong summary ba seed

`FULLSCALE_3SEEDS_SUMMARY_REPORT.md` §2.3 đang đặt **frame-level/source-frozen** cạnh **video-level/stored train-fit** dưới tên hai policy. Số AUC khác nhau do đổi population/unit, không do đổi ngưỡng. Giữ cùng unit và seed, AUC/EER trong hai policy JSON giống hệt nhau cho đủ18cặp đã kiểm tra. Tài liệu này chỉ dùng source-frozen cho bảng rate; không đề xuất phép fit ngưỡng mới.

| CASIA pool | CE-only seed | Frame AUC (%) | Video AUC (%) |
| --- | --- | --- | --- |
| train | 100 | 90.470000 | 91.239764 |
| train | 101 | 88.223184 | 88.754523 |
| train | 102 | 89.855739 | 91.030280 |
| test | 100 | 86.209395 | 88.369490 |
| test | 101 | 83.888347 | 86.230200 |
| test | 102 | 87.606242 | 90.142906 |
| combined | 100 | 87.433100 | 89.830017 |
| combined | 101 | 85.030301 | 87.508292 |
| combined | 102 | 88.439664 | 91.368159 |

EER trong cột summary ba seed là **mean3seed**, không phải EER seed100. Ví dụ Combined frame EER mean21.607185%, video19.888060%; không kết luận các mean này sai chỉ vì khác EER seed100. Một số SD trong summary dùng ddof0; lượt sửa này thống nhất sample SD(ddof1).

## 8. Trạng thái dữ liệu dùng cho lượt sửa

| Phương pháp | Seed100 | Seed101 | Seed102 |
| --- | --- | --- | --- |
| CE-only | Đã xác minh source/LCC/CASIA | Đã xác minh source/LCC/CASIA | Đã xác minh source/LCC/CASIA |
| Clean + Aux | Đã xác minh source/LCC/CASIA | Đã xác minh source/LCC; thiếu CASIA | Đã xác minh source/LCC; thiếu CASIA |
| WBST | Đã xác minh source/LCC/CASIA | Summary báo dừng epoch8; chưa có raw để xác minh completion | Summary báo dừng epoch7; chưa có raw để xác minh completion |
| MAST-PAD | Đã xác minh source/LCC/CASIA | Pending; chưa có output | Pending; chưa có output |

Run config/completion/manifest/checkpoint của baseline ba seed đã đối chiếu; cùng seed CE-only/Clean + Aux dùng initial-state fingerprint giống nhau. Notebook hiện đặt TRAIN_SEED101 không thay đổi provenance output seed100 đã đóng băng. Chưa xác lập exact training git revision chỉ từ fingerprints.

## 9. Thông số phương pháp, diagnostics và complexity

| Thông số | Giá trị dùng khi sửa |
| --- | --- |
| Input PAD | 80×80×3, BGR, float[0,1], crop yêu cầu1.5×; scale hiệu dụng bị giới hạn bởi geometry |
| Data seed / model seeds | 43 / 100,101,102 theo các run đã có |
| Training recipe | Batch256; SGD lr0.005,momentum0.9,weight_decay0.0005; max12epochs; LR milestones4/8,gamma0.2 |
| Heads | PAD128→2; spoof128→11; lighting128→5; attributes128→40(real-only loss) |
| Spectral centers | 1/12,1/4,5/12 trên r=sqrt(u²+v²)/sqrt(2×40²) |
| Gain / FFT | Uniform[0.65,0.90] mỗi ảnh, shared3bands/channels; FFT ortho, DC preserved trước clamp; Gaussian overlapping |
| Deployment parameters | 434434, PAD trunk+binary head, không detector/ArcFace/aux |
| Multihead model parameters | 441658; không gọi toàn bộ số này là trainable CE-only |
| MACs / FLOPs | 40810892 /81621784; 0.081621784GFLOPs; Conv/Linear-only,2FLOP/MAC |
| Raw FP32 weights | 1737736bytes=1.737736MB≈1.657MiB; không phải ONNX file size/RAM đã đo |
| MAST best / final epoch | 7 /11 |
| Epoch11 harmful / flip fraction | 83.213465% /0.307701% |
| Epoch11 LOW / MID / HIGH selection | 42.645198% /27.303477% /30.051325% |

Diagnostics là hàng **epoch11**, không gán cho checkpoint best epoch7. Gate chỉ cho biết mẫu nào nhận trọng số spectral CE trực tiếp; selected-view forward vẫn chạy toàn batch ở train mode, nên không hứa mọi ảnh hưởng BN/gradient của non-harmful sample biến mất.

## 10. Cấu hình runtime hiện tại — tách khỏi kết quả huấn luyện

| Tham số | Giá trị file hiện tại |
| --- | --- |
| Detector/PAD/recognition cadence | 0.2s /0.2s /1.0s |
| PAD window / min votes / spoof ratio | 5 /3 /≥0.6 trên số phiếu đang có |
| PAD stale / tracker IoU | 3s /.3 |
| Detector min face / PAD detector confidence | 60px /.5 |
| Recognition threshold | .35, là cấu hình chưa phải kết quả FAR/FRR |
| Fixed source research threshold MAST | 4.66259765625 |
| Fixed contract mà .env đang trỏ tới | R7_SC_FULL_crop15_runtime_config_casia.json:1.6586914, từ CASIA train frame predictions |
| Measured latency có trace khớp MAST | Chưa tìm thấy trace đủ chứng minh2.22ms/67.98ms/14.7FPS/15.5FPS |

Các giá trị là cấu hình trên đĩa, không chứng minh một camera process đã dùng đúng cấu hình đó. Runtime contract CASIA-fit khác source-frozen research contract; không dùng rate của threshold4.6626 để mô tả threshold1.6587. 1000/67.98≈14.71 chỉ là số học, không là camera FPS đã đo. Không chuyển benchmark model khác sang MAST-PAD.

## 11. Các thay thế số/mô tả cần thực hiện trong lượt sửa

| Vị trí Word/summary | Giá trị/mô tả cũ | Giá trị/mô tả cần dùng |
| --- | --- | --- |
| Word P0847 | CE-only CASIA video EER23.05% | 16.54% seed100 Combined video |
| Word T21 CASIA | APCER/BPCER còn trống | 1.87% /51.11%; ACER26.49% |
| Word T18 CE LCC SD(fraction) | .006277 | .006267; tính raw,ddof1 |
| Word T18 Clean+Aux LCC SD(fraction) | .004336 | .004342; tính raw,ddof1 |
| Word T18 CE CASIA SD(fraction) | .019434 | .019431; tính raw,ddof1 |
| Word P0558 | Spoof head128→1 | 128→11 |
| Word T15 | MAST L_worst; cóseverity; thiếuCE-only | MAST L_harmful; bốn strategy cuối |
| Word P0479–0491 | PAD.1s,recognition.5s,min_votes5 | PAD.2s,recognition1s,min_votes3,capacity5 |
| Word P0881 | Thêm seeds sẽ cô lập hoàn toàn gate | Seeds đo repeatability; gate-specific cần same-sigma control |
| Word P1078/P1085 | Mọi strategy/CE-only chỉ1seed | Bảng chung4method seed100; baseline đã có3seed như mục8 |
| SummaryCASIA§2.3 | AUC thay vì calibration | Ghi rõ frame/video và threshold policy thành hai thuộc tính riêng |
| Word latency/ONNX | 1.74MB là ONNX,2.22/67.98ms đã đo | 1.737736MB raw FP32; bỏ latency claim nếu thiếu trace |

Locator P/T theo bản trích audit: P là số đoạn OOXML gồm đoạn trong bảng, T là thứ tự bảng trong document.xml, không phải số trang/caption. Không lấy số trang TOC cache làm pagination đã xác minh.

## 12. Số liệu bổ sung đã có — chỉ dùng khi quyết định đưa vào bản sửa

### Paired video bootstrap CASIA seed100 — audit bổ sung, chưa thuộc frozen manuscript

| Phương pháp | AUC (%) | Percentile95%CI (%) |
| --- | --- | --- |
| CE-only | 89.8300 | 85.5847–93.4124 |
| Clean + Aux | 92.0398 | 88.9296–94.8434 |
| WBST | 92.2264 | 89.1916–94.8966 |
| MAST-PAD | 93.5199 | 90.7625–95.9009 |

| Chênh lệch | Point(pp) | 95%CI(pp) |
| --- | --- | --- |
| MAST-PAD - CE-only | +3.689884 | [+1.749946, +6.142977] |
| MAST-PAD - Clean + Aux | +1.480100 | [+0.284568, +2.834435] |
| MAST-PAD - WBST | +1.293532 | [-0.084706, +2.819282] |

Seed bootstrap20261022,2000resamples,paired358reconstructedvideoIDs và labels. Đây là video-resampling uncertainty có điều kiện trên score seed100, không phải subject-independent CI hoặc training-seed variability. Video cùng subject vẫn tương quan; không suy significance tổng quát. CI MAST−WBST chứa0. Các kết quả này chưa tự động được thêm vào article.

### Breakdown CASIA Combined seed100 tại ngưỡng nguồn

| Phương pháp | Cut APCER%,n89 | Replay APCER%,n89 | Warped APCER%,n90 | Real BPCER%,n90 |
| --- | --- | --- | --- | --- |
| CE-only | 0.0000 | 1.1236 | 0.0000 | 73.3333 |
| Clean + Aux | 0.0000 | 0.0000 | 2.2222 | 55.5556 |
| WBST | 0.0000 | 1.1236 | 3.3333 | 56.6667 |
| MAST-PAD | 1.1236 | 2.2472 | 2.2222 | 51.1111 |

Attack type dựa trên mapping filename-derived của bản CASIA trích frame; không suy robust3Dmask/Deepfake. LCC chưa có subject/camera/category mapping đầy đủ cho toàn pooled evaluation nên không báo subject-cluster CI hoặc per-attack LCC khi thiếu nhãn.

## 13. Truy nguồn và kiểm tra

Ưu tiên raw prediction + matched config/metadata > metricJSON > tổng hợp đã xác minh > prose Word/summary. Bảng này được sinh từ dữ liệu audit và đối chiếu lại **70 tổ hợp method/seed/population ×5metric** với JSON gốc ở lần tạo file, sai số<1e-12. Mean/sampleSD được kiểm lại từ từng seed chưa làm tròn. Không training, inference hoặc thay đổi ngưỡng/model/protocol.

- [recomputed_metrics.csv](../../paper/review/dg20_evidence_audit/recomputed_metrics.csv)

- [baseline_three_seed_metrics.csv](../../paper/review/dg20_evidence_audit/baseline_three_seed_metrics.csv)

- [baseline_three_seed_summary.csv](../../paper/review/dg20_evidence_audit/baseline_three_seed_summary.csv)

- [paired_video_bootstrap.json](../../paper/review/dg20_evidence_audit/paired_video_bootstrap.json)

- [casia_attack_breakdown.csv](../../paper/review/dg20_evidence_audit/casia_attack_breakdown.csv)

- [runtime_whitelist.json](../../paper/review/dg20_evidence_audit/runtime_whitelist.json)

- [selected_fixed_runtime_contract.json](../../paper/review/dg20_evidence_audit/selected_fixed_runtime_contract.json)

- [baseline_seed_configs.json](../../paper/review/dg20_evidence_audit/baseline_seed_configs.json)

- [validation_results.json](../../paper/review/dg20_evidence_audit/validation_results.json)

Báo cáo đầy đủ: [03_METRIC_AND_DATA_AUDIT.md](../../paper/review/dg20_evidence_audit/03_METRIC_AND_DATA_AUDIT.md); [02_CODE_AND_METHOD_EVIDENCE.md](../../paper/review/dg20_evidence_audit/02_CODE_AND_METHOD_EVIDENCE.md); [04_COMPLEXITY_AND_SYSTEM_EVIDENCE.md](../../paper/review/dg20_evidence_audit/04_COMPLEXITY_AND_SYSTEM_EVIDENCE.md). Đường dẫn raw exact nằm trong cột `path` của hai CSV metric; checkpoint/manifest checks trong [added_seed_validation.json](../../paper/review/dg20_evidence_audit/added_seed_validation.json).

### Vị trí artifact gốc (repo-relative)

| Nhóm | Đường dẫn |
| --- | --- |
| Clean+Aux/WBST/MAST seed100 | antispoof/notebooks/final/binary_crossdomain_v2/output/18_fullscale_C_P3SF_R7SC_crop15/runs/{run_id}/ |
| CASIA ba strategy seed100 | antispoof/notebooks/final/binary_crossdomain_v2/output/casia_train_test_C_P3SF_R7SC_100k_fullscale/runs/{full_source_run_id}/ |
| CE-only100 | antispoof/notebooks/final/binary_crossdomain_v2/output/21_fullscale_CE_clean_baseline_crop15/runs/CE_CLEAN_FULL_crop15/ |
| CE-only101/102 | antispoof/notebooks/final/binary_crossdomain_v2/output/21_fullscale_CE_clean_baseline_crop15_s{seed}/runs/CE_CLEAN_FULL_crop15/ |
| Clean+Aux101/102 | antispoof/notebooks/final/binary_crossdomain_v2/output/C_FULL_crop15_s{seed}/runs/C_FULL_crop15/ |

CASIA parent folder chứa chữ100k_fullscale nhưng bảng này chỉ lấy run full-source tương ứng, không lấy các run100K. Exact keys: config.train_seed/data_seed/manifest_sha256; completion.best_epoch/locked_threshold; evaluation_freeze.checkpoint_sha256; metricJSON.auc/eer/apcer/bpcer/acer; predictions score/label/locked_threshold hoặc video_score/binary_label/threshold.

### Fingerprint của dữ liệu tổng hợp dùng để tạo tài liệu

| Audit input | SHA256 |
| --- | --- |
| recomputed_metrics.csv | 9d2212953909e36f9388ee29b182890c5d5d737b5b16107862cd292452b6b34c |
| baseline_three_seed_metrics.csv | e5ee20f41e4cab361f871da42c24e84ee03683850b3210f27cbeb562bab1e962 |
| baseline_three_seed_summary.csv | 3ce6353b65b92c263eb73a922c516c7af2e863bcebf2709b446266158944ac3c |
| paired_video_bootstrap.json | 12803959c66e6583b490e0185c8d9a8a69a08b7b3c376db362d36790bca90d91 |
| casia_attack_breakdown.csv | a7a6620c8b86434cfba643c60aa149a5230577cc93366aa6951d35916300c382 |

## 14. Những số chưa có, không điền thay

- CASIA Clean+Aux seed101/102; full completion và matching targets của WBST/MAST seed101/102.
- Hiệu ứng gate riêng, random-gate cùng rate, gate-margin-threshold sensitivity và per-band controlled gains: chưa có matched full-source evidence.
- Official independent subject-disjoint target benchmark; end-to-end identity FAR/FRR và latency có trace: chưa được xác minh trong evidence hiện tại.

Bài báo giữ author metadata và frozen seed100 evidence. File này cung cấp số liệu cho tác giả duyệt sửa; không đồng nghĩa Word/article hay source-of-truth cũ đã được chỉnh.
