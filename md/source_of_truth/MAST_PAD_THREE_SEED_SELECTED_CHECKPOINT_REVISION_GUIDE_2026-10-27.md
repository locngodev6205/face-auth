# Hướng dẫn sửa Word và paper sau ba seed — Chọn checkpoint bằng validation nguồn

**Ngày:** 27/10/2026. **Phương pháp đề xuất:** MAST-PAD. **Rule đã chốt:** source ACER thấp nhất, source AUC phá hòa, áp dụng đồng nhất cho cả bốn variant.

Tài liệu này tập trung các bảng và số liệu cần cho lượt sửa theo rule hiện tại. Nó thay thế đề nghị chọn phương pháp chính bằng mean trong bản `MAST_PAD_FINAL_METHOD_FREEZE_PROPOSAL_2026-10-27.md` ở vai trò hướng dẫn biên tập; không sửa hay ghi đè dữ liệu, scientific locks, draft và release cũ. Chưa chỉnh Word hoặc manuscript. Adaptive threshold hoàn toàn ngoài phạm vi.

## Định hướng trình bày: đóng góp rõ, kết quả cụ thể

Dẫn dắt bằng ba điểm: **huấn luyện phổ có giám sát theo margin**, **kết quả của checkpoint được chọn thống nhất trên nguồn**, và **graph suy luận gọn không thêm nhánh phổ**. Bảng mean ± SD bổ sung góc nhìn về độ nhạy theo seed. Giữ đầy đủ số liệu nhưng chỉ diễn giải những so sánh trả lời câu hỏi nghiên cứu; phần giới hạn được gom vào một đoạn Scope/Discussion.

Ưu tiên các kết quả đã được hỗ trợ: MAST-PAD có source ACER thấp nhất trong 12checkpoint; checkpoint đại diện đạt CASIA AUC 93.52%, tăng 6.01/3.39/1.29pp so với các checkpoint đại diện CE-only/Clean+Aux/WBST; trên LCC cải thiện 2.62/.74pp so hai baseline sạch và đạt mức AUC gần WBST. Toàn bộ giữ graph PAD 434,434 tham số. Đây là thông điệp trung tâm, không cần mở đầu abstract bằng phân tích giới hạn hay thứ hạng phụ.

Trong paper, dùng “các checkpoint đại diện được chọn bằng validation nguồn” tại caption và lần định nghĩa đầu. Sau đó trình bày kết quả trực tiếp, tránh lặp rule chọn model trong mọi câu. Báo cáo Word giải thích quy trình chi tiết một lần ở phần thiết lập thực nghiệm.

## 1. Rule lựa chọn và quy ước báo cáo

1. Huấn luyện mỗi variant với MODEL_SEED 100/101/102, DATA_SEED 43.
2. Trong từng run, chọn epoch đủ điều kiện có source ACER thấp nhất; source AUC dùng phá hòa. Epoch 1 của các variant phổ là warmup clean+aux, không eligible best checkpoint; hai clean controls eligible từ epoch 1, các variant phổ từ epoch 2.
3. Chọn checkpoint đại diện trong ba best checkpoint của mỗi variant theo **cùng tiêu chí nguồn**. LCC/CASIA không phải tiêu chí trong rule này.
4. Báo cáo hiệu năng của các checkpoint đại diện cùng **mean ± sample SD của đủ ba seed** và bảng từng seed. Không bỏ seed 102, không dùng target để đổi seed đại diện.
5. ACER là tiêu chí chính; AUC phá hòa. Dùng cụm “source ACER thấp nhất” để mô tả chính xác ưu thế trong lựa chọn checkpoint.

Mô tả chính sách lựa chọn và số đo được; không thêm khẳng định rule đã định trước hoặc target chưa từng được xem. Các bảng selected-model và mean-over-seeds phải phân biệt rõ.

Các cột AUC/EER/APCER/BPCER/ACER có đơn vị **%**; delta là **điểm phần trăm(pp)**. SD là sample SD(ddof 1), n 3, tính từ số chưa làm tròn. Mean AUC không phải AUC của scores gộp từ ba model. EER theo code hiện có: chỉ số đầu tiên min|FPR−FNR|, lấy trung bình FPR/FNR, không nội suy.

## 2. Lập luận: chọn checkpoint bằng ACER, đánh giá ngoại miền bằng AUC

Hai metric có vai trò khác nhau. Source ACER là lỗi cân bằng tại ngưỡng đã xác định trên CelebA Test-as-Val, dùng chọn epoch/model. Target ROC-AUC đo khả năng xếp hạng Real/Attack trên toàn ROC, không gắn với một operating threshold nguồn. Domain shift có thể thay đổi phân bố score và trade-off giữa APCER/BPCER, nên ngưỡng tốt trên nguồn không được bảo đảm tạo điểm vận hành tốt trên LCC/CASIA.

Vì vậy, AUC là metric chính hợp lý cho phân tích khả năng phân tách ngoại miền. Đồng thời giữ các rate tại **ngưỡng nguồn cố định** để phản ánh khả năng chuyển điểm vận hành. Không fit ngưỡng mới trên official target để chọn model hoặc nâng headline.

Source ACER và target AUC đánh giá hai khía cạnh bổ sung: quyết định phân loại trên nguồn và khả năng phân tách ngoại miền. Bảng operating-point metrics đi kèm giúp người đọc thấy cả chất lượng ranking và trade-off lỗi tại ngưỡng nguồn; việc thay ngưỡng riêng lẻ giữ nguyên ROC/AUC nếu scores không đổi.

**Đoạn tiếng Việt:**

> Checkpoint đại diện của mỗi biến thể được chọn theo ACER thấp nhất trên CelebA Test-as-Val, sử dụng AUC để phá hòa. ROC-AUC là chỉ số chính trên các tập ngoại miền để đánh giá khả năng phân tách Real/Attack mà không phụ thuộc vào ngưỡng vận hành đã chọn trên nguồn. APCER, BPCER và ACER tại ngưỡng nguồn cố định được báo cáo bổ sung để thể hiện khả năng chuyển điểm vận hành sang miền đích.

**Đoạn tiếng Anh:**

> Representative checkpoints are selected by the lowest ACER on CelebA Test-as-Val, with source AUC used to break ties. Target-domain ROC-AUC is the primary evaluation measure because it characterizes score discrimination independently of the source operating threshold, whose error trade-off may change under domain shift. APCER, BPCER and ACER at the frozen source threshold are reported as complementary operating-point measurements.

Nguồn giải thích metric: [scikit-learn — Decision threshold](https://scikit-learn.org/stable/modules/classification_threshold.html), [ROC definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html). Không tự thêm citation này vào bibliography paper; chúng hỗ trợ giải thích, không xác nhận protocol hiện tại là một benchmark độc lập.

## 3. Các checkpoint đại diện theo rule

| Variant | Selected seed | Best epoch | Source ACER% ↓ | Source AUC% ↑ | Ngưỡng nguồn |
| --- | --- | --- | --- | --- | --- |
| CE-only | 101 | 9 | 4.430001 | 99.202743 | 4.6982421875 |
| Clean + Aux | 102 | 9 | 4.470741 | 99.172895 | 5.0888671875 |
| WBST | 100 | 6 | 4.354846 | 99.224181 | 5.3037109375 |
| MAST-PAD | 100 | 7 | 4.345889 | 99.228027 | 4.6625976562 |

**MAST-PAD seed 100 có source ACER thấp nhất trong cả 12 best checkpoint.** Đây là checkpoint đại diện của MAST-PAD và là model được chọn nếu áp rule trên toàn bộ 12 checkpoint. Không chọn lại epoch theo target metrics.

**Cách trình bày trong bản nộp:** nhấn mạnh MAST-PAD seed100 được chọn theo source ACER và đạt source AUC 99.23%. Tiêu chí AUC phá hòa đã được định nghĩa trong phần phương pháp; không cần thêm một bảng xếp hạng12checkpoint theo AUC nguồn hoặc một đoạn riêng giải thích vì sao MAST không đứng đầu AUC. Bảng từng seed ở mục7 vẫn cung cấp đầy đủ số liệu nguồn để đối chiếu.

## 4. Bảng kết quả chính — Các checkpoint đại diện chọn bằng nguồn

| Variant | Selected seed | Source AUC% | LCC Combined image AUC% | CASIA Combined video AUC% | PAD params |
| --- | --- | --- | --- | --- | --- |
| CE-only | 101 | 99.20 | 83.41 | 87.51 | 434434 |
| Clean + Aux | 102 | 99.17 | 85.29 | 90.13 | 434434 |
| WBST | 100 | 99.22 | 86.08 | 92.23 | 434434 |
| MAST-PAD | 100 | 99.23 | 86.03 | 93.52 | 434434 |

**Caption đề nghị:** “ROC-AUC(%) của checkpoint đại diện được chọn trong ba run theo ACER thấp nhất trên CelebA Test-as-Val, AUC phá hòa. LCC sử dụng pooled images; CASIA sử dụng deduplicated reconstructed videos. Các variant có cùng ngân sách ba seed.”

Bold winner: MAST-PAD trên source/CASIA; WBST trên LCC. Các checkpoint đại diện có thể khác seed, không nói chúng cùng seed 100 hoặc cùng initial weights. Initialization được matched giữa bốn variant **trong cùng seed** của thiết kế 12run.

### Delta thay thế cho bảng selected-model

| So sánh | ΔLCC AUC(pp) | ΔCASIA AUC(pp) |
| --- | --- | --- |
| MAST-PAD − CE-only | +2.619343 | +6.011609 |
| MAST-PAD − Clean + Aux | +0.738449 | +3.387231 |
| MAST-PAD − WBST | -0.055891 | +1.293532 |

Clean+Aux−CE-only trong bảng selected: LCC+1.880894pp, CASIA+2.624378pp. Không chép các delta+.20/+2.21pp của bảng all-seed 100. Các delta mô tả hiệu năng của model được chọn theo cùng ngân sách và rule. Phạm vi so sánh strategy được giải thích một lần trong Methods/Scope.

## 5. Metric chi tiết tại ngưỡng nguồn — Selected checkpoints

| Variant | Seed | Population | AUC% | EER% | APCER% | BPCER% | ACER% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CE-only | 101 | CelebA Test-as-Val | 99.202743 | 4.463146 | 5.099430 | 3.760572 | 4.430001 |
| Clean + Aux | 102 | CelebA Test-as-Val | 99.172895 | 4.518782 | 4.637214 | 4.304269 | 4.470741 |
| WBST | 100 | CelebA Test-as-Val | 99.224181 | 4.392433 | 4.697409 | 4.012284 | 4.354846 |
| MAST-PAD | 100 | CelebA Test-as-Val | 99.228027 | 4.351873 | 4.478125 | 4.213653 | 4.345889 |
| CE-only | 101 | LCC combined | 83.407865 | 24.730511 | 7.470480 | 51.828954 | 29.649717 |
| Clean + Aux | 102 | LCC combined | 85.288759 | 23.133876 | 6.194743 | 50.850077 | 28.522410 |
| WBST | 100 | LCC combined | 86.083099 | 21.790648 | 7.725627 | 45.852653 | 26.789140 |
| MAST-PAD | 100 | LCC combined | 86.027208 | 22.055922 | 7.980775 | 46.264812 | 27.122793 |
| CE-only | 101 | CASIA combined | 87.508292 | 23.047264 | 3.358209 | 55.555556 | 29.456882 |
| Clean + Aux | 102 | CASIA combined | 90.132670 | 19.514925 | 0.373134 | 54.444444 | 27.408789 |
| WBST | 100 | CASIA combined | 92.226368 | 17.475124 | 1.492537 | 56.666667 | 29.079602 |
| MAST-PAD | 100 | CASIA combined | 93.519900 | 16.915423 | 1.865672 | 51.111111 | 26.488391 |

Ngưỡng cố định riêng của từng model ở mục 3. Không điền số từ `*_casia_calibrated` vào bảng source-frozen. EER là thống kê ROC, không phải rate tại ngưỡng nguồn. ACER=(APCER+BPCER)/2.

### Confusion matrix MAST 100 tại ngưỡng 4.66259765625

| LCC combined | Predicted Real | Predicted Attack | N |
| --- | --- | --- | --- |
| Actual Real | 1043 | 898 | 1941 |
| Actual Attack | 1345 | 15508 | 16853 |

| CASIA combined | Predicted Real | Predicted Attack | N |
| --- | --- | --- | --- |
| Actual Real | 44 | 46 | 90 |
| Actual Attack | 5 | 263 | 268 |

## 6. Mean ± sample SD qua đủ ba seed

| Variant | Source AUC% mean±SD | LCC image AUC% mean±SD | CASIA video AUC% mean±SD |
| --- | --- | --- | --- |
| CE-only | 99.1963 ± 0.0220 | 83.6818 ± 0.6267 | 89.5688 ± 1.9431 |
| Clean + Aux | 99.0817 ± 0.0874 | 85.0980 ± 0.4342 | 90.9342 ± 0.9892 |
| WBST | 99.2102 ± 0.0482 | 86.3050 ± 1.2366 | 90.9121 ± 1.5352 |
| MAST-PAD | 99.1704 ± 0.0707 | 85.9523 ± 0.2796 | 90.7850 ± 2.7158 |

Đặt bảng này gần bảng checkpoint đại diện, với caption phân biệt rõ hai loại kết quả. Qua ba seed, MAST-PAD cải thiện mean AUC so với CE-only **2.27pp trên LCC và 1.22pp trên CASIA**, đồng thời đạt LCC AUC 85.95±0.28%. Dùng bảng để thảo luận **biến thiên giữa các seed huấn luyện**; các thứ hạng mean và độ lệch chuẩn được thể hiện trực tiếp trong bảng, không cần lặp thành danh sách bất lợi ở abstract hoặc kết luận.

| Variant | Population | EER% mean±SD | APCER% mean±SD | BPCER% mean±SD | ACER% mean±SD |
| --- | --- | --- | --- | --- | --- |
| CE-only | CelebA Test-as-Val | 4.4634 ± 0.0122 | 5.0077 ± 0.1371 | 3.8696 ± 0.1362 | 4.4387 ± 0.0101 |
| Clean + Aux | CelebA Test-as-Val | 4.7015 ± 0.1965 | 5.1553 ± 0.4551 | 4.1734 ± 0.2267 | 4.6644 ± 0.2161 |
| WBST | CelebA Test-as-Val | 4.4654 ± 0.0741 | 4.7125 ± 0.1864 | 4.1751 ± 0.3174 | 4.4438 ± 0.0914 |
| MAST-PAD | CelebA Test-as-Val | 4.5859 ± 0.2738 | 4.8178 ± 0.5425 | 4.3043 ± 0.1399 | 4.5610 ± 0.2522 |
| CE-only | LCC combined | 24.4527 ± 0.2827 | 7.3538 ± 0.1015 | 52.9624 ± 2.0531 | 30.1581 ± 0.9994 |
| Clean + Aux | LCC combined | 23.2998 ± 0.2435 | 7.2134 ± 1.4763 | 48.6347 ± 1.9476 | 27.9240 ± 0.6698 |
| WBST | LCC combined | 21.8995 ± 1.2360 | 7.6485 ± 0.3593 | 45.3718 ± 2.7112 | 26.5101 ± 1.5352 |
| MAST-PAD | LCC combined | 22.3803 ± 0.4714 | 7.3874 ± 0.8829 | 48.1367 ± 2.9785 | 27.7621 ± 1.0485 |
| CE-only | CASIA combined | 19.8881 ± 3.2565 | 1.3682 ± 1.7234 | 63.7037 ± 8.9810 | 32.5359 ± 3.8505 |
| Clean + Aux | CASIA combined | 19.2067 ± 2.3391 | 0.3731 ± 0.3731 | 56.2963 ± 2.3130 | 28.3347 ± 1.0302 |
| WBST | CASIA combined | 18.8364 ± 2.0465 | 1.3682 ± 1.3104 | 59.2593 ± 2.3130 | 30.3137 ± 1.1327 |
| MAST-PAD | CASIA combined | 19.3325 ± 2.2538 | 0.9950 ± 0.9390 | 57.0370 ± 5.2509 | 29.0160 ± 2.3430 |

## 7. Số liệu từng seed và đánh dấu checkpoint đại diện

| Variant | Seed | Đại diện | Best epoch | Source ACER% | Source AUC% | LCC image AUC% | CASIA video AUC% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CE-only | 100 | — | 9 | 4.449825 | 99.214398 | 84.398794 | 89.830017 |
| CE-only | 101 | Có | 9 | 4.430001 | 99.202743 | 83.407865 | 87.508292 |
| CE-only | 102 | — | 12 | 4.436200 | 99.171879 | 83.238604 | 91.368159 |
| Clean + Aux | 100 | — | 9 | 4.624831 | 99.073553 | 84.601127 | 92.039801 |
| Clean + Aux | 101 | — | 9 | 4.897485 | 98.998677 | 85.404207 | 90.630182 |
| Clean + Aux | 102 | Có | 9 | 4.470741 | 99.172895 | 85.288759 | 90.132670 |
| WBST | 100 | Có | 6 | 4.354846 | 99.224181 | 86.083099 | 92.226368 |
| WBST | 101 | — | 9 | 4.438977 | 99.249789 | 87.637479 | 91.285240 |
| WBST | 102 | — | 11 | 4.537450 | 99.156553 | 85.194290 | 89.224710 |
| MAST-PAD | 100 | Có | 7 | 4.345889 | 99.228027 | 86.027208 | 93.519900 |
| MAST-PAD | 101 | — | 10 | 4.498618 | 99.191583 | 86.186792 | 90.746269 |
| MAST-PAD | 102 | — | 9 | 4.838598 | 99.091569 | 85.642783 | 88.088723 |

## 8. Dữ liệu, population và đơn vị cố định

| Population | Candidate/scored | Real/Attack scored | Vai trò |
| --- | --- | --- | --- |
| CelebA Train | 494405/484561 | 160073/324488 | Full usable source training |
| CelebA Test-as-Val | 67170/66379 | 19864/46515 | Chọn epoch/model/ngưỡng nguồn |
| LCC Combined images | 18827/18794 | 1941/16853 | Gộp training/dev/eval, không mean split AUC |
| CASIA Combined frames | 2660/2660 | 674/1986 | Frame population phụ |
| CASIA Combined videos | 360/358 | 90/268 | Headline CASIA, ít nhất 4valid frames |

LCC scored split counts 8283training+2947development+7564evaluation. CASIA train/test copies overlap 20subjectIDs/240videoIDs/1403frameIDs; Combined dedup(video_id,frame_index), ưu tiên test copy trước detection, không fallback sang train. Video score là mean valid-frame logit difference; ít nhất 4valid frames. Các video dưới 4frame có thể cóscore nhưng không được đưa vào AUC/rates.

Không gọi CelebA Test-as-Val là untouched test. LCC đã được theo dõi trong development; CASIA dùng video tái dựng có overlap, không verified official subject-disjoint protocol. Giữ các giới hạn này trong một đoạn ngắn, không lặp ở mọi caption.

## 9. Các thông số phương pháp không thay đổi

| Thông số | Giá trị |
| --- | --- |
| Input/crop | 80× 80BGR,float[0,1],crop requested 1.5×;geometry có thể giới hạn scale hiệu dụng |
| Seeds/init | DATA_SEED 43;MODEL_SEED 100/101/102;matched initial-state giữa 4methods trong cùngseed |
| Optimization | Batch 256;SGDlr.005,momentum.9,weight_decay.0005;max 12epochs;milestones 4/8,gamma.2;AMP |
| CE-only | CEclean,khôngaux/spectral forward hoặc optimization |
| Clean + Aux | CEclean+Laux |
| WBST | σ 1/18;0.75CEclean+0.25CEworst(allselected)+Laux |
| MAST-PAD | σ.10;0.75CEclean+0.25CEharmful+Laux;H-normalization,H 0differentiablezero,cleancoeff.75 |
| Auxiliary supervision | Spoof 11,lighting 5,attributes 40(real-only);Laux=.1CEspoof+.1CElight+BCEattrReal |
| Spectral | Centers 1/12,1/4,5/12;gainUniform.65–.90 sharedbands/channels;DCpreserved trước clamp |
| Deployment params | 434434,PAD trunk+binaryhead;khôngSCRFD/ArcFace/aux |
| Multihead params | 441658;không gọi toàn bộ là trainable CE-only |
| MACs/FLOPs | 40810892/81621784=.081621784GFLOPs;Conv/Linear-only,2FLOP/MAC |
| Raw FP 32 weights | 1737736bytes≈1.74MBdecimal;không phải ONNX bytes/RAM/FPS đã đo |

WBST và MAST khác cả sigma lẫn gate; không gọi so sánh hiện tại là gate-only ablation. Selector chọn một spectral view có aligned margin nhỏ nhất cho từng ảnh, eval/no-grad, khôi phục mode; strict detached gate. Differentiable selected-view forward vẫn chạy toàn batch ở train mode; non-harmful sample không có direct CE weight nhưng có thể ảnh hưởng BN.

MAST 100 diagnostics tại finalepoch 11: harmful 83.213465%,flip.307701%,LOW/MID/HIGH 42.645198/27.303477/30.051325%. Best checkpoint ởepoch 7. Nếu giữ paragraph 83.21%/.31%, phải ghi **seed 100, finalepoch 11**, không gọi đó là thống kê mean ba seed hay thống kê tại best checkpoint. Không thêm latency 2.22ms/67.98ms/14.7FPS khi chưa có trace phù hợp.

## 10. Hướng dẫn sửa paper theo từng phần

| File/locator | Hướng sửa |
| --- | --- |
| experimental_setup.tex:45–55 | Bốn variant×ba seeds;DATA 43;định nghĩa hai cấp chọn epoch và selectedmodel bằngACER/AUC;selectedseeds 101/102/100/100,epochs 9/9/6/7;không nói allseed 100 |
| experimental_setup.tex:99+ | Tách source ACER selection và target AUC primary;thêm rationale mục 2 và source-frozenrates;định nghĩa sampleSD/n 3 |
| results.tex:7–35,TableII | Thay all-seed 100 table bằng bảng mục 4;thêm Selectedseed column và caption rule;không để CE 100/Cclean 100 làm representative theo rulemới |
| results.tex:40–47,Fig 4 | Selected markers:LCC 83.41/85.29/86.08/86.03;CASIA 87.51/90.13/92.23/93.52;captionselectedbysource |
| Bảng/hình ba seed | Dùng mục 6–7;errorbar SD phải quanh mean, không quanh selectedscore;caption n 3,SD không phải CI |
| abstract.tex:13–17 | Dẫn bằng MAST selected AUC86.03/93.52, CASIA gains6.01/3.39/1.29pp và unchanged434434params; ghi source-selection rule và three-seed analysis gọn, không mở đầu bằng thứ hạng phụ |
| introduction.tex:45–47 | Giữ đóng góp margin-gatedtraining/compactinference;phân biệt selectedmodel vàseedvariability |
| results.tex:104–124 | Tínhlại selectedgains vàauxiliarycontrast theo mục 4;không paired-same-seed causality;thêm thảo luận mean ba seed |
| results.tex:142–153 | Ghi seed 100/finalepoch 11/bestepoch 7 cho diagnostics |
| results.tex:61,87,97,181–196 | Literature ours rows vẫn selectedMAST 86.03/93.52,ghi selectedfrom 3seeds;thay Oursusesoneseed;giữ external scores/protocols |
| results.tex:203–211 | Một đoạn scope về 3repeats,Test-as-Val,LCCpool/exposure,CASIAreconstruction,σ+gate,complexityscope |
| conclusion.tex:7–9 | Kết luận bằng đóng góp margin-gated training,selected checkpoint gains,mean improvements soCE-only và unchanged spatial graph; để chi tiết giới hạn ở một đoạn Scope |
| author/bibliography/release | Giữ metadata tác giả,IEEEformat,citation;revisionPDF/locks/release mới tách bản cũ |

### Bố trí bảng/hình đề nghị

- **BảngA:**selected checkpoints, mục 4; có seed và rule.
- **BảngB:**mean±sampleSD, mục 6; đặt gần BảngA, không ẩn để giữ câu kết luận chỉ dựa selected model.
- **BảngC:**per-seed, mục 7; full-scale appendix hoặc supplementary nếu cần.
- **BảngD:**source-frozen rates/confusion, mục 5; Word nên có đầy đủ, paper có thể dùng bảng phụ gọn.
- **Fig 4:**có thể tách selectedmodel comparison và three-seed distribution. Không vẽ meanSD quanh selected marker; hai center khác nhau.
- Giữ hình pipeline/mechanism; chỉ soát loss/head/crop đúng code. Không cần đổi kiến trúc vì rule chọnmodel thay đổi.

## 11. Hướng dẫn sửa báo cáo Word

Giữ đúng **3chương**, mục tiêu 11+20+16=47trang chính, tổng≤50; giới hạn từng chương 15/25/20. Sau format phải render và đếm trang thật; TOC cache không phải pagination đã xác minh.

- **Chương 1:**relatedwork, bài toán, đóng góp MAST; không kể lịch sử tuning.
- **Chương 2:**giữ sáu giai đoạn của hệ thống, trình bày sâu MAST, bốn objective và input/contract.
- **Chương 3:**datasets/protocol→rule selection→selected checkpoints→meanSD/per-seed→operating metrics→diagnostics/complexity→discussion→một đoạn limitations→conclusion.

LocatorP/T của audit cũ chỉ là tham chiếu khi nội dung Word còn tương ứng:

|Nội dung cũ|Sửa theo tài liệu này|
|---|---|
|Bảng bốn method toàn seed 100|Thay bằng mục 4,thêm selectedseed và tính lại gains|
|Bảng 3seed thiếu spectral runs|Đủ 12run theo mục 6–7;không dùng statuspending từREADME cũ|
|CE-only CASIA EER 23.05%|Với selected CE 101,EER 23.047264%→23.05% là đúng;CE 100EER 16.54%.Không thay mọi 23.05thành 16.54 máy móc;ghi rõseed|
|MAST CASIA rate còn trống|Seed 100 APCER 1.87%,BPCER 51.11%,ACER 26.49%;LCCEER 22.06%,CASIAEER 16.92%|
|Spoofhead 128→1|128→11;MASTloss là harmful-normalized CE|
|C/CleanAux bị gọi CE-only|Tách hai baseline;giữ tên presentation CE-only/Clean+Aux/WBST/MAST-PAD|
|meanSD gọi robustness toàn diện|Đổi thành biến thiên qua seedhuấnluyện;domaintransfer là bài đánh giá riêng|
|1.74MB ONNX/latency đã đo|Rawweights-only 1.74MB;không thêm trace chưa có|
|10K/100K/micro/severity results|Bỏ khỏi phần kết quả và phụ lục kết quả;chỉ full-source 484561|
|Hạn chế lặp nhiều nơi|Gộp một đoạn,giọng học thuật/chủ động,tránh internal-audit tone|

## 12. Nội dung gợi ý để chèn

**Rule — tiếng Việt:**

> Mỗi biến thể được huấn luyện với ba seed 100,101 và 102, sử dụng cùng phân vùng dữ liệu nguồn. Checkpoint của từng run và checkpoint đại diện của từng biến thể được chọn theo ACER thấp nhất trên CelebA Test-as-Val, với AUC làm tiêu chí phá hòa. Kết quả ngoại miền của checkpoint đại diện và thống kê trung bình, độ lệch chuẩn mẫu qua ba seed được báo cáo trên LCC Combined và CASIA Combined.

**Rule — tiếng Anh:**

> Each variant is trained with seeds 100,101 and 102 under the same source-data partition. Within each run, an eligible checkpoint is selected by minimum ACER on CelebA Test-as-Val, using source AUC to break ties. The same criterion selects a representative checkpoint from the three runs of each variant. We report its cross-domain performance alongside the mean and sample standard deviation across all three training seeds.

**Abstract — đoạn kết quả gợi ý:**

> Selected by minimum ACER on CelebA Test-as-Val, MAST-PAD achieves 86.03% AUC on pooled LCC Combined images and 93.52% on reconstructed CASIA Combined videos. It improves over the representative CE-only and clean-plus-auxiliary checkpoints by 2.62 and 0.74percentage points on LCC, and6.01 and 3.39points on CASIA. Compared with WBST, it attains similar numerical LCC AUC and improves CASIA AUC by 1.29 points. The deployed PAD graph retains 434,434 parameters, with spectral probing confined to training. Three-seed statistics further characterize training variability.

**Selected results — tiếng Việt:**

> Checkpoint MAST-PAD được chọn theo validation nguồn đạt ROC-AUC 86.03% trên LCC Combined và 93.52% trên CASIA Combined ở mức video. So với các checkpoint đại diện của CE-only và Clean+Aux, mức tăng trên LCC lần lượt là 2.62 và 0.74 điểm phần trăm; trên CASIA là 6.01 và 3.39 điểm phần trăm. So với WBST, MAST-PAD đạt kết quả gần trên LCC và cao hơn 1.29 điểm phần trăm trên CASIA, trong khi giữ nguyên graph suy luận 434,434 tham số.

**Selected results — tiếng Anh:**

> The source-selected MAST-PAD checkpoint achieves 86.03% AUC on pooled LCC Combined images and 93.52% on reconstructed CASIA Combined videos. Against the source-selected CE-only and clean-plus-auxiliary checkpoints, its gains are 2.62 and 0.74 percentage points on LCC, and 6.01 and 3.39 points on CASIA. WBST obtains 86.08% on LCC and 92.23% on CASIA under the same selection policy. All four variants retain the 434,434-parameter spatial PAD graph.

**Three-seed summary — tiếng Việt:**

> Qua ba seed, MAST-PAD đạt AUC trung bình 85.95±0.28% trên LCC và 90.78±2.72% trên CASIA, cải thiện lần lượt 2.27 và 1.22 điểm phần trăm so với CE-only. Bảng kết quả từng seed và độ lệch chuẩn bổ sung đánh giá về tính lặp lại của quá trình huấn luyện bên cạnh hiệu năng checkpoint đại diện.

**Three-seed summary — tiếng Anh:**

> Across three training seeds, MAST-PAD achieves a mean AUC of 85.95±0.28% on LCC and 90.78±2.72% on CASIA, improving over CE-only by2.27 and 1.22 percentage points, respectively. The per-seed results and sample standard deviations complement the representative-checkpoint comparison by characterizing training variability.

**Đóng góp — tiếng Việt:**

> MAST-PAD đưa can thiệp phổ vào quá trình huấn luyện theo lề quyết định của từng mẫu. Phương pháp lựa chọn biến thể phổ bất lợi nhất và tập trung giám sát vào những biến thể làm giảm lề so với ảnh sạch. Cơ chế này tạo áp lực học trên các tín hiệu nhạy phổ trong khi giữ nguyên bộ phân loại không gian gọn cho suy luận.

**Một đoạn scope — tiếng Anh:**

> The study uses three training seeds with source checkpoint selection on CelebA Test-as-Val. LCC is pooled across its native image partitions and was observed during development; CASIA comprises deduplicated reconstructed videos from overlapping extracted-image copies. WBST and MAST-PAD differ in both spectral width and gating, so their comparison concerns the configured strategies. The deployment analysis covers the PAD graph and does not measure device-level latency.

Nội dung mẫu phải được biên tập typography/LaTeX khi chèn. Không thêm statisticallysignificant/equivalent,MASTbestmean,gate-onlycausality hoặc prospectiveselectionclaim không có bằng chứng.

## 13. Metadata, provenance và kiểm tra cuối

Tái tính 132 population records× 5metric khớp JSON nguồn, tolerance 1e-12. Mean/sampleSD kiểm từ từngseed chưa làm tròn.393 deep checks khôngfailure;12 bestcheckpointSHA khớpevaluation_freeze;coreconfig vàinitialstatewithinseed khớp;132ID/labelpopulationfingerprints khớp giữa runs. Đây là offlinevalidation, không training/inference mới.

CASIAglobalprotocol còn metadata seed 100 kế thừa; per-run config là nguồn actualtrainseed. Fileprovenance lưu uploadedcheckpointpath nhưng không lưuSHA của file thực đã load, nên byteidentitylocal→uploadedCASIA chưa được chứng minh. Không phát hiện dấu hiệu cụ thể trộnseed; không gọi provenance khép kín hoàn toàn.

- [metrics_per_seed.csv](../../paper/review/three_seed_review_2026-10-27/metrics_per_seed.csv)

- [metrics_mean_sd.csv](../../paper/review/three_seed_review_2026-10-27/metrics_mean_sd.csv)

- [provenance_checks.json](../../paper/review/three_seed_review_2026-10-27/provenance_checks.json)

- [population_pairing.json](../../paper/review/three_seed_review_2026-10-27/population_pairing.json)

- [deep_artifact_checks.json](../../paper/review/three_seed_review_2026-10-27/deep_artifact_checks.json)

- [input_hashes.json](../../paper/review/three_seed_review_2026-10-27/input_hashes.json)

- [validation_results.json](../../paper/review/three_seed_review_2026-10-27/validation_results.json)

Rawpath exact ở cộtpath trong metrics_per_seed.csv. Các key quan trọng:config.train_seed/data_seed;completion.best_epoch/locked_threshold;evaluation_freeze.checkpoint_sha256;metricsJSON.auc/eer/apcer/bpcer/acer. Numerical locks/draft/release cũ giữ nguyên; khi duyệt sửa tạo phiên bản mới theo rule đã chốt.

### Checklist trước khi phát hành bản sửa

- [ ] Selectedseeds đúng 101/102/100/100; selectedepochs 9/9/6/7; ACER primary,AUC tie-break.
- [ ] Selectedmodel vàmean ba seed tách rõ; gainsselected tính lại,không lẫn all-seed 100 baselines.
- [ ] AUC%,deltapp,sampleSDddof 1,n 3,frame/videopool đúng; CE 101EER khác CE 100.
- [ ] Methodsigma/gate/loss/BN/heads đúngcode; giữ 434434PADparams.
- [ ] Figure/caption/abstract/introduction/results/literature/conclusion đồng bộ selectionpolicy.
- [ ] Word 3chapter≤50mainpages; render để kiểm bảng/công thức/crossreferences.
- [ ] IEEEauthor giữ Quang Anh Nguyen,Van Loc Ngo; affiliation University of Science and Technology – The University of Danang,Vietnam; không email.
- [ ] Compile và visuallyinspect mọi page trước release; giữ bibliography,scientific evidence và draft cũ.
