# Đánh giá và kế hoạch chỉnh sửa báo cáo Thị giác máy tính

**Mục đích:** hướng dẫn hoàn thiện báo cáo học phần Thị giác máy tính, lấy phương pháp MAST-PAD và kết quả full-scale làm trọng tâm; không áp tiêu chuẩn nghiệm thu của một bài báo chuẩn bị xuất bản.  
**Ngày rà soát:** 20/10/2026.  
**Bản Word được đọc:** `/Users/3t/Downloads/Báo cáo Thị Giác Máy Tính (1).docx`.  
**Căn cứ kỹ thuật:** mã nguồn và cấu hình hiện có trong `face_auth`, HEAD `7359b21`, bao gồm working tree chưa commit.  
**Cập nhật từ người thực hiện:** đang chạy thêm hai seed. Chưa có kết quả của hai seed này trong lần rà; tổng dự kiến là ba seed cho những cấu hình thực sự được chạy bổ sung. Hướng dẫn sửa báo cáo khi hoàn tất nằm ở Mục 10. Các số một seed và đoạn văn mẫu ở Mục 7 vẫn mô tả bằng chứng hiện có, không phải kết quả tổng hợp ba seed.  
**Sản phẩm của lần rà này:** bản đánh giá và hướng dẫn biên tập Markdown; chưa chỉnh sửa Word, code, cấu hình hoặc mô hình.

## 1. Kết luận biên tập

Báo cáo lấy **MAST-PAD làm điểm nhấn chính**: vì sao cần phương pháp, cơ chế phổ có chọn lọc theo margin hoạt động thế nào, kết quả full-scale cải thiện ra sao và vì sao giữ được graph suy luận gọn. Báo cáo vẫn mô tả đầy đủ pipeline xác thực sáu giai đoạn để người đọc hiểu hệ thống tổng thể; MAST-PAD là thành phần được giải thích sâu nhất và là đóng góp trung tâm. Cấu trúc hiện tại vẫn gần một nhật ký nghiên cứu: kể các vòng 1–7, 10–16, 17, 18, 21; giải thích nhiều quyết định trung gian; lặp lại crop và micro experiment trong đóng góp, phương pháp, thực nghiệm và kết luận. Điều đó làm mờ sản phẩm cuối cùng.

**Phạm vi kết quả cuối:** chỉ sử dụng các mô hình huấn luyện **full-scale trên 484.561 ảnh CelebA-Spoof khả dụng**, gồm CE-only, Clean + Aux, WBST và MAST-PAD. Kết quả 100K, 10K và micro screening là thử nghiệm phát triển nội bộ: **không đưa vào bảng, biểu đồ, phân tích so sánh, abstract, đóng góp, kết luận hoặc phụ lục kết quả của bản báo cáo nộp**. Không dùng chúng làm bằng chứng bổ sung để củng cố lợi thế full-scale. Artifact lịch sử vẫn giữ trong repository; không cần xóa hoặc đổi tên dữ liệu nghiên cứu.

Hướng sửa là tái cấu trúc theo câu hỏi khoa học và thiết kế cuối, không theo thứ tự thời gian. **Bỏ lịch sử và kết quả micro/100K khỏi báo cáo nộp, kể cả phụ lục kết quả.** Chỉ giữ tài liệu tái lập và phân tích bổ sung thực sự giúp hiểu kết quả cuối. Artifact nghiên cứu trong repository vẫn được bảo toàn.

Ba quyết định biên tập chính:

1. **Chỉ trình bày các phương pháp cuối với tên công bố:** CE-only, Clean + Aux, WBST và MAST-PAD. Bỏ cột biến thể severity khỏi Bảng 3.1 vì không thuộc so sánh full-scale cuối. Các mã phát triển nội bộ không dùng làm tên phương pháp trong báo cáo; xem quy tắc đặt tên tại Mục 2.3.
2. **Bỏ hàng “trọng số mẫu trong spectral loss” khỏi bảng và thay bằng “Loss phổ trên mỗi ảnh”.** Cả WBST và MAST-PAD chỉ dùng CE của **đúng một worst band trên từng ảnh**; không trung bình CE của tất cả band bị giảm margin. MAST-PAD thêm gate để dùng CE đó hoặc cho đóng góp trực tiếp bằng 0. Hệ số spectral trong total loss sau warmup là **0,25**. Các số `1/N` và `1/N_harmful` chỉ mô tả reduction giữa các ảnh trong batch, không phải trọng số chọn band và không nên đặt ở hàng này. MAST-PAD **không nhân CE với mức giảm margin**. Đồng thời sửa ký hiệu MAST-PAD từ `L_worst` thành `L_harmful`.
3. **Dùng logic `face_auth`, không lấy logic deploy ThingsBoard/hardware:** áp dụng xuyên suốt phần hệ thống. Báo cáo cần mô tả detector, tracker, PAD, alignment, embedding, matching và worker đang thực hiện trong `face_auth`. Không đưa MQTT, telemetry, rule chain, GPIO hay nghiệp vụ chấm công vào luồng xác thực nếu không thuộc phạm vi thực hiện của module.

**Chỉ số trọng tâm:** dùng ROC-AUC để so sánh khả năng phân tách; giữ thảo luận rõ về EER và các lỗi tại ngưỡng nguồn, phân biệt giới hạn mô hình với khả năng điều chỉnh điểm vận hành theo miền đích (Mục 4, P1-3).

**Thông điệp trung tâm đề xuất:** MAST-PAD khai thác biến đổi phổ trong huấn luyện để tạo thử thách cho bộ phân loại, chọn đúng một worst view trên mỗi ảnh và dùng harmful gate để quyết định có bổ sung CE phổ hay không. Trên MiniFASNetV2, phương pháp cải thiện AUC so với Clean + Aux ở cả hai target full-scale, trong khi giữ nguyên graph PAD suy luận với 434.434 tham số. Đây là điểm nhấn xuyên suốt abstract, phương pháp, kết quả và kết luận, đặt trong một pipeline xác thực khuôn mặt được mô tả đầy đủ.

**Ba trụ cột trình bày MAST-PAD:** (1) spectral views và worst-band selection theo từng ảnh; (2) harmful gate dựa trên mức giảm margin, với loss được định nghĩa rõ; (3) bằng chứng AUC full-scale trên mô hình suy luận gọn, không thêm nhánh FFT lúc inference. Mô tả gate như cơ chế thiết kế của phương pháp; số gain hiện có thuộc toàn bộ cấu hình, không quy riêng cho gate.

### 1.1. Giọng văn và mức độ chi tiết phù hợp báo cáo môn học

**Viết theo hướng tích cực, có căn cứ:** bắt đầu từ ý tưởng MAST-PAD, cơ chế huấn luyện, kiến thức thị giác đã vận dụng và kết quả full-scale đạt được. Sau đó giải thích những điểm cần hoàn thiện như các bước phát triển tiếp theo. Có thể dùng “kết quả khả quan”, “cho thấy tiềm năng”, “cải thiện so với đối chứng trong thiết lập khảo sát” khi các số liệu hỗ trợ.

Không cần biến báo cáo môn học thành một bản tự phản biện dài. Các kiểm tra chi tiết về artifact/hash, độ độc lập của dữ liệu, pairing seed và giới hạn suy luận trong bản đánh giá này chủ yếu giúp nhóm biên tập đúng; **không phải mọi cảnh báo kiểm tra đều cần chép vào Word**. Báo cáo chỉ cần một mô tả protocol rõ, số liệu chính xác và một phần “Phạm vi đánh giá và hướng hoàn thiện” ngắn.

Cách phân bổ nội dung:

- **Abstract:** bài toán PAD ngoại miền → ý tưởng MAST-PAD → cơ chế chính → kết quả full-scale và graph suy luận gọn. Không kết thúc bằng danh sách dài điểm yếu.
- **Phương pháp/thiết lập:** định nghĩa đúng loss và vai trò tập dữ liệu một lần. Không lặp caveat sau mỗi công thức hoặc bảng.
- **Kết quả:** nêu kết quả đạt được trước, so sánh cân bằng, rồi một đoạn ngắn về điểm vận hành/calibration. Giải thích kết quả chưa thuận lợi như một phát hiện kỹ thuật hữu ích, không biến nó thành đánh giá phủ định toàn bộ đề tài.
- **Phạm vi đánh giá:** khoảng 2–3 đoạn hoặc 3 ý ngắn, thường nửa trang là đủ. Giữ những thông tin thực sự ảnh hưởng cách hiểu kết quả; tránh lặp ở abstract, caption, kết quả và kết luận.
- **Kết luận:** khẳng định các mục tiêu học phần đã hoàn thành và kết quả có số liệu; hướng tiếp theo tập trung vào độ ổn định qua seed, phân tích cơ chế phổ/gate và mở rộng đánh giá ngoại miền; calibration chỉ nhắc như bước ứng dụng bổ trợ.

Đối với yêu cầu hoàn thiện bản nộp, ưu tiên **đúng số liệu, đúng thuật toán, đúng protocol, rõ bố cục**. Gate-only ablation, nhiều miền độc lập, kiểm định thống kê và benchmark nhiều thiết bị là hướng mở rộng, không phải điều kiện bắt buộc để công nhận kết quả học phần. Nếu seed hiện có còn ít, ghi số run đúng thực tế; không cần nhấn mạnh “thiếu ý nghĩa thống kê” ở mọi nơi khi báo cáo chỉ đưa nhận xét mô tả.

Viết tích cực không đòi hỏi bỏ kết quả bất lợi hay thay đổi nghĩa kỹ thuật. Vẫn sửa EER sai, giữ đúng sự đánh đổi APCER/BPCER, không gọi dữ liệu tái dựng là official benchmark và không gán tốc độ chưa đo. Những điều này có thể diễn đạt gọn trong setup và một đoạn thảo luận, không cần làm thành chuỗi cảnh báo.

### 1.2. Dung lượng và thứ tự ưu tiên nội dung

Giữ **một chương đầy đủ về pipeline sáu giai đoạn**, đủ để người đọc hiểu luồng dữ liệu, vai trò từng khối và cách phối hợp. MAST-PAD có chương phương pháp riêng và là trọng tâm phân tích thực nghiệm. Không áp giới hạn 1–2 trang cho toàn pipeline hoặc buộc hệ thống chỉ xuất hiện sau kết quả.

- Pipeline: trình bày detection, tracking, PAD, alignment, embedding và matching; đầu vào/đầu ra, điều kiện chuyển bước, temporal verdict và recognition worker ở mức cần hiểu hành vi hệ thống.
- MAST-PAD: giải thích sâu spectral views, worst-band từng ảnh, margin/gate, loss và train/inference; có ví dụ, hình thuật toán và đối chứng cuối.
- Thực nghiệm: tập trung AUC full-scale, các cải thiện đạt được và graph suy luận gọn; cập nhật nhiều seed khi hoàn tất.
- Chi tiết hạ tầng như SQL, retry, TTL cache, Docker hoặc ThingsBoard không cần chiếm dung lượng; chỉ giữ khi thực sự giúp hiểu pipeline CV.
- Calibration và metric tại ngưỡng là thảo luận hỗ trợ kết quả, không thay điểm nhấn MAST-PAD. Không cần benchmark phần cứng mở rộng để hoàn thành báo cáo học phần.

### 1.3. Chọn điểm nhấn tích cực khi trình bày kết quả

Có thể chọn lọc **thứ tự, dung lượng và ví dụ** để kể câu chuyện rõ và tích cực. Đưa các kết quả thuyết phục lên trước: MAST-PAD cải thiện AUC so với Clean + Aux trên cả hai target, tăng 1,29 điểm phần trăm so với WBST trên CASIA video Combined, và không thêm nhánh phổ vào graph suy luận. Đây là các điểm nhấn phù hợp với dữ liệu hiện có.

Bảng kết quả chính vẫn giữ đủ bốn phương pháp và hai target full-scale. Với LCC, có thể viết gọn “MAST-PAD đạt mức AUC gần WBST”, thay vì dành một đoạn dài cho chênh lệch nhỏ; số chính xác vẫn hiện trong bảng. Với EER/rates, chỉ cần một nhận xét đúng ngữ cảnh trong phần thảo luận, không lặp kết quả bất lợi ở abstract, mọi caption và kết luận. Đặt bảng metric chi tiết ở phụ lục full-scale để phần chính mạch lạc.

**Văn phong đề xuất:** “MAST-PAD cải thiện AUC so với Clean + Aux trên cả hai tập khảo sát; đạt kết quả gần WBST trên LCC và cải thiện rõ hơn trên CASIA, trong khi giữ nguyên graph suy luận gọn.” Tránh đổi thành “vượt mọi đối chứng trên cả hai tập” vì không đúng bảng số liệu.

Khi đủ seed, dùng thống kê của các run full-scale đã định thay vì chọn seed tốt nhất làm kết quả chung. Hình minh họa có thể chọn ví dụ dễ quan sát, ghi đúng vai trò minh họa; không suy tỷ lệ thành công toàn hệ thống từ vài ảnh đẹp. Cách chọn lọc này giúp làm nổi bật thành quả mà vẫn giữ bảng số liệu và kết luận nhất quán.

## 2. Căn cứ đánh giá và giới hạn xác minh

### 2.1. Thứ tự ưu tiên

| Mã | Nguồn | Nội dung đã đối chiếu |
|---|---|---|
| W | Bản Word nêu trên | Nội dung OOXML, 532 paragraph trực tiếp trong body, 22 bảng; cả text trong vùng mục lục/danh mục; 136 đối tượng công thức, 26 đối tượng drawing |
| N18 | [Notebook 18](../antispoof/notebooks/final/binary_crossdomain_v2/18_fullscale_C_P3SF_R7SC_crop15.ipynb), code cell index 1–2 | `RUNS`, `select_worst_view`, `training_terms`, `spectral_views`, `aligned_margin`; gate, reduction, warmup, sigma, DC |
| N21 | [Notebook 21](../antispoof/notebooks/final/binary_crossdomain_v2/21_fullscale_CE_clean_baseline_crop15.ipynb) và [báo cáo CE-only](source_of_truth/FULLSCALE_RESULTS_SOURCE_OF_TRUTH.md) | Định nghĩa đối chứng CE-only; số liệu báo cáo được cung cấp |
| A18 | [Artifacts full-scale 18](../antispoof/notebooks/final/binary_crossdomain_v2/output/18_fullscale_C_P3SF_R7SC_crop15) | Per-run config, metric JSON, LCC prediction CSV, training/spectral diagnostics |
| A20 | [Artifacts CASIA](../antispoof/notebooks/final/binary_crossdomain_v2/output/casia_train_test_C_P3SF_R7SC_100k_fullscale) | Source-frozen video Combined JSON và prediction CSV; population, score và calibration policy |
| R | [app.py](../app.py), [config.py](../config.py), [tracker.py](../tracking/tracker.py), [worker.py](../recognition/worker.py), `.env` và [runtime JSON hiện chọn](../antispoof/models/R7_SC_FULL_crop15_runtime_config_casia.json) | Call path realtime và cấu hình đọc được bằng import `config` trong tiến trình mới |
| H | [Notebook micro 14](../antispoof/notebooks/final/binary_crossdomain_v2/14_micro_R7_scaleaware_severity_crop15.ipynb) | Xác định severity thuộc lịch sử khảo sát, không thuộc `RUNS` full-scale 18 |

**Kiểm tra số liệu đã thực hiện:** tính lại ROC-AUC và EER từ CSV cho Clean + Aux/WBST/MAST-PAD trên LCC Combined và CASIA video Combined, tổng cộng 6 population. Chỉ dùng các dòng có score hợp lệ và video `scorable=True`. Thư mục CASIA nguồn có tên chứa cả `100k_fullscale` do cách tổ chức artifact; các số đưa vào tài liệu này chỉ lấy từ run **full-scale** tương ứng, không lấy run 100K. Kết quả khớp JSON nguồn. Ma trận nhầm lẫn MAST-PAD LCC được đếm lại tại ngưỡng nguồn đã khóa, không dựng từ tỷ lệ EER.

**Giới hạn:** chưa huấn luyện hay inference dataset mới; chưa chạy camera, cloud hoặc benchmark phần cứng. CE-only có bảng kết quả Markdown, nhưng chưa thấy output run 21 tương ứng trong thư mục output local đã kiểm; không nói đã tái tính CE-only từ prediction gốc. Không có LibreOffice/soffice trong môi trường, nên chưa render toàn bộ Word thành trang PDF/PNG. Các nhận xét bố cục cần được rà trực quan khi xuất bản; text extraction thiếu biến không tự chứng minh Equation Editor hiển thị hỏng.

### 2.2. Nguyên tắc sử dụng bản đánh giá

Tài liệu này trình bày đầy đủ kết luận, số liệu cần sửa, công thức, cấu trúc đề xuất và văn bản mẫu; không yêu cầu người đọc mở một bản đánh giá trước để hiểu hoặc thực hiện. Những dẫn nguồn còn giữ dùng để truy vết code/artifact hoặc xác nhận tài liệu khoa học. Riêng CE-only hiện có mức bằng chứng là bảng tổng hợp được cung cấp, nên vẫn phải ghi giới hạn đó.

Cấu hình hệ thống áp dụng là cấu hình đọc được từ source và runtime JSON ngày rà soát, trình bày tại Mục 5. Không xem snapshot tài liệu là bằng chứng app đang chạy hoặc là benchmark camera.

### 2.3. Tên phương pháp dùng trong báo cáo cuối

Chỉ dùng tên mô tả phương pháp; không đưa mã thử nghiệm nội bộ vào tên gọi học thuật.

| Tên thống nhất | Định nghĩa khi xuất hiện lần đầu |
|---|---|
| **CE-only** | MiniFASNetV2 huấn luyện bằng cross-entropy trên ảnh sạch, không auxiliary supervision và không can thiệp phổ |
| **Clean + Aux** | MiniFASNetV2 huấn luyện trên ảnh sạch với giám sát phụ trợ |
| **WBST** | Worst-Band Spectral Training; dùng CE của một worst view trên mỗi ảnh, kèm giám sát phụ trợ trong cấu hình đối chứng |
| **MAST-PAD** | Margin-Aware Spectral Presentation Attack Detection; dùng CE của một worst view khi harmful gate thỏa, kèm giám sát phụ trợ trong cấu hình đề xuất |

Sau lần định nghĩa đầu, dùng **WBST** và **MAST-PAD** nhất quán. Không ghép thêm mã nội bộ bằng dấu gạch chéo hoặc ngoặc; không dùng các hậu tố crop/scale của run làm tên phương pháp. Dùng **MAST-PAD** đầy đủ, tránh rút thành “MAST-PAD” tùy đoạn. “Full-scale”, crop 1,5 và sigma là thuộc tính thiết lập, đặt ở mục phương pháp/protocol hoặc caption chung.

Áp dụng cho abstract, mục tiêu/đóng góp, đề mục, công thức có tên phương pháp, hàng bảng, legend/trục biểu đồ, caption, sơ đồ/pseudocode, nhận xét và kết luận. Nếu chữ mã nội bộ đã nằm trong ảnh xuất từ notebook, cần xuất lại nhãn hình; chỉ sửa caption chưa đủ. Bảng thiết kế không cần hàng run ID. “Full-scale” là phạm vi kết quả đã chốt; không đặt thêm cột kết quả 100K hoặc một bảng đối chiếu các quy mô.

Đối với phụ lục tái lập hoặc danh mục artifact, có thể ghi tên file, checkpoint hash và run ID chính xác để truy vết; đây là metadata kỹ thuật, không phải tên phương pháp thay thế. Các đường dẫn code/artifact trong **bản đánh giá này** giữ tên file thật để mở được, không yêu cầu đổi tên notebook, run folder hoặc checkpoint. Tên seed cũng chỉ là cấu hình lần chạy, không tạo một phương pháp mới.

## 3. Sửa Bảng 3.1 và giải thích spectral loss

### 3.1. Logic thực sự trong code

Mỗi ảnh sinh ba view LOW/MID/HIGH. Ở lượt selection `eval()` và `no_grad()`, tính margin theo nhãn:

\[
 d(x)=z_{real}(x)-z_{attack}(x),\qquad
 m(x,y)=\begin{cases}d(x),&y=real\\-d(x),&y=attack.\end{cases}
\]

Nhãn code là `real=0`, `attack=1`; ký hiệu toán học phải giải thích riêng nếu dùng `+1/-1`. Margin này là margin tại điểm logit bằng nhau, **không phải khoảng cách đến ngưỡng runtime đã hiệu chuẩn**. Vì vậy không viết `m>0` đồng nghĩa được hệ thống runtime chấp nhận là người thật.

\[
 k_i^*=\arg\min_{k\in\{LOW,MID,HIGH\}}m(x_i^{(k)},y_i),\qquad
 h_i=\mathbf1[m(x_i^{(k_i^*)},y_i)<m(x_i,y_i)].
\]

**Chọn một view bất lợi nhất cho từng ảnh**, không chọn một band chung cho cả batch, không lấy trung bình CE của ba band. Gate được detach; bước training forward tính CE trên selected view, không đạo hàm qua phép `argmin` hay điều kiện gate.

Gọi `ell_i = CE(f_train(x_i^(k_i*)), y_i)`, `N` là số ảnh batch và `H=sum_i h_i`:

\[
 L_{worst}=\frac1N\sum_{i=1}^N\ell_i,
\qquad
 L_{harmful}=\begin{cases}\frac1H\sum_{i=1}^Nh_i\ell_i,&H>0\\0,&H=0.\end{cases}
\]

Code tương ứng ở `training_terms`:

```python
worst_ce = ce_each.mean()
harmful = (select_worst < select_clean).detach()
spectral_ce = ((ce_each * harmful).sum() / harmful_count
               if harmful_count else spectral_logits.sum() * 0)
```

Khi không có harmful sample, spectral loss bằng 0 có đường gradient hợp lệ. Sau warmup, clean CE **vẫn nhân 0,75**, không tự đổi lại 1,0.

### 3.2. Loss của worst band, hệ số spectral và reduction batch là ba việc khác nhau

Trong MAST-PAD, mức giảm margin chỉ được dùng để kiểm tra **có nhỏ hơn hay không**. Mẫu giảm margin 0,1 và mẫu giảm 2,0 đều có `h=1`; hệ số chuẩn hóa CE của chúng như nhau nếu ở cùng batch. Giá trị CE và gradient thực tế vẫn có thể khác nhau; “hệ số bằng nhau” không có nghĩa hai mẫu tác động bằng nhau lên cập nhật tham số.

Cần tách ba khái niệm:

- **Gate:** `h_i` bằng 0 hoặc 1, quyết định mẫu có đóng góp trực tiếp vào tổng CE phổ hay không.
- **Reduction:** chia cho `N` ở WBST hoặc `H` ở MAST-PAD.
- **Hệ số thành phần loss:** 0,75 cho clean và 0,25 cho spectral sau warmup.

Không trộn chúng thành “mẫu quan trọng được tăng trọng số theo độ khó”. MAST-PAD được kiểm tra ở đây không dùng `Delta m` làm trọng số liên tục. Với một ảnh riêng lẻ, WBST có `L_spec=ell`; MAST-PAD cũng bằng `ell` nếu harmful, ngược lại bằng 0. Đây là cách giải thích trực quan nên dùng trước công thức batch.

Ví dụ trên **một ảnh**: margin clean là `2,0`; margin LOW/MID/HIGH lần lượt là `1,5; 0,4; 1,0`. Dù cả ba band đều làm giảm margin, cả WBST và MAST-PAD chỉ dùng `CE(MID,y)` vì MID là worst band. Không cộng hoặc trung bình CE của LOW và HIGH. Nếu worst margin không thấp hơn clean margin, WBST vẫn dùng CE của worst view, còn MAST-PAD cho đóng góp spectral trực tiếp của ảnh đó bằng 0.

Chỉ sau bước chọn view/gate của từng ảnh mới tổng hợp batch: WBST lấy trung bình các CE đã chọn trên mọi ảnh; MAST-PAD lấy trung bình trên các ảnh harmful. Mẫu số là **số ảnh**, không phải số band. Để tránh hiểu sai, không gọi phép chia này là “trọng số spectral loss”; hệ số thành phần spectral trong total loss là **0,25** sau warmup.

**Lưu ý kỹ thuật nếu giữ phân tích gradient:** code forward cả batch selected views ở train mode rồi mới mask CE. Với BatchNorm, các ảnh có thể tương tác qua thống kê batch. Câu chính xác là “mẫu ngoài gate không có hạng CE phổ trực tiếp”, không khẳng định chúng bị loại hoàn toàn khỏi spectral forward, thống kê BN hoặc mọi đường gradient. Không suy ra gate giúp giảm thời gian training từ việc chỉ một số mẫu có hạng loss.

### 3.3. Bảng thay thế đề xuất

**Tên bảng:** “Các cấu hình huấn luyện trong so sánh full-scale cuối cùng”. Trong Word có thể chia thành bảng loss và bảng tham số phổ nếu khổ trang quá chật.

| Thành phần | CE-only | Clean + Aux | WBST | MAST-PAD |
|---|---|---|---|---|
| Auxiliary supervision | Không | Có, clean branch | Có, clean branch | Có, clean branch |
| View phổ | Không | Không | LOW/MID/HIGH | LOW/MID/HIGH |
| View được chọn mỗi ảnh | — | — | Margin nhỏ nhất | Margin nhỏ nhất |
| Loss phổ trên mỗi ảnh | 0 | 0 | CE của đúng một worst view | CE của đúng một worst view nếu harmful; ngược lại 0 |
| Điều kiện sử dụng CE đã chọn | — | — | Luôn dùng | `m_worst < m_clean` |
| Hệ số spectral trong total loss sau warmup | 0 | 0 | 0,25 | 0,25 |
| Loss sau warmup | `L_clean` | `L_clean + L_aux` | `0.75 L_clean + 0.25 L_worst + L_aux` | `0.75 L_clean + 0.25 L_harmful + L_aux` |
| Warmup | Không | Không | Epoch 1: clean + aux | Epoch 1: clean + aux |
| Sigma Gaussian | — | — | 0,055556 | 0,10 |

Hai phương pháp phổ dùng tâm `[0,083333; 0,25; 0,416667]`, gain lấy mẫu `[0,65; 0,90]`, crop 1,5, bảo toàn DC. Một gain cho mỗi ảnh, dùng chung cho ba band và các kênh. **WBST và MAST-PAD còn khác sigma**, vì vậy đây là so sánh hai cấu hình hoàn chỉnh; chưa phải ablation riêng bật/tắt gate.

**Bỏ hẳn hàng “trọng số mẫu trong spectral loss”.** Cách reduction batch được giải thích riêng bằng công thức ở Mục 3.1: trung bình trên toàn bộ ảnh với WBST và trên ảnh harmful với MAST-PAD. Không đưa `1/N` hoặc `1/H` vào bảng như câu trả lời cho việc chọn loss của band nào.

### 3.4. Những lỗi phương pháp khác cần sửa

- Mục 2.4.3 ghi spoof-type head `128→1` nhưng nói 11 lớp: sửa thành **128→11**; PAD `128→2`, lighting `128→5`, attributes `128→40`.
- Tên `L_worst` ở cột MAST-PAD phải là `L_harmful`; giữ nhất quán với Mục 3.4.3–3.4.4 và công thức train.
- Ghi rõ `L_aux = 0.1 CE_spoof_type + 0.1 CE_lighting + BCE_attributes_real`. Aux chỉ từ ảnh clean; attribute chỉ tính trên real.
- Bổ sung trường hợp `H=0` và epoch warmup; không trình bày loss active như đúng cho mọi epoch.
- Bộ Gaussian tạo **suy giảm biên độ ở vùng tần số**, không chia ba vùng loại trừ nhau; có overlap. DC được giữ riêng. Không chủ động thay pha trước clamp; clamp có thể thay đổi phổ của ảnh cuối.
- `m=0` là điểm hòa logit, không tự gọi là “dự đoán sai” vì còn quy tắc tie-break. Viết rõ ý nghĩa dấu và trường hợp biên.
- Không nói view phổ đo hoặc tái tạo chính xác Moiré/ánh sáng/nhòe thực. Đây là động lực thiết kế; chưa có thí nghiệm vật lý xác nhận từng band tương ứng một nguyên nhân.
- Bỏ phân tích gradient dài khỏi chính văn; một đoạn về selection detach và reduction là đủ.

## 4. Các vấn đề ưu tiên trong kết quả và diễn giải

### P0-1. EER không khớp artifact và đang dẫn đến kết luận sai

**Vị trí:** Bảng 4.1; các đoạn 4.3 diễn giải CE-only → Clean → MAST-PAD; các kết luận có nhắc giảm EER.

AUC của bảng đã bao gồm CE-only và nhìn chung đúng nguồn. Tuy nhiên EER ở một số hàng vẫn là số cũ. Đặc biệt câu “CASIA EER giảm từ 16,54% xuống 12,44%” cho MAST-PAD không đúng với video Combined hiện có. Với project convention, EER MAST-PAD là **16,9154%**, còn CE-only report ghi **16,5423%**. AUC tăng không bắt buộc EER giảm.

| Cấu hình | LCC Combined AUC | LCC EER (%) | CASIA video Combined AUC | CASIA EER (%) | Mức bằng chứng |
|---|---:|---:|---:|---:|---|
| CE-only | 0,843988 | 24,4623 | 0,898300 | 16,5423 | Báo cáo CE-only; chưa tái tính từ CSV gốc |
| Clean + Aux | 0,846011 | 23,5793 | 0,920398 | 16,7289 | CSV và JSON đối chiếu lại |
| WBST | 0,860831 | 21,7906 | 0,922264 | 17,4751 | CSV và JSON đối chiếu lại |
| MAST-PAD | 0,860272 | 22,0559 | 0,935199 | 16,9154 | CSV và JSON đối chiếu lại |

**Cách trình bày cuối:** ưu tiên AUC trong bảng headline; EER và các rates có thể ở bảng bổ sung. Chính văn chỉ cần một câu rõ rằng AUC cải thiện trong khi EER chưa cải thiện đồng đều, sau đó chuyển sang khả năng hoàn thiện điểm vận hành. Không cần nhấn đi nhấn lại kết quả bất lợi; tránh dùng “cải thiện toàn diện” khi các metric không cùng chiều.

### P0-2. Ma trận nhầm lẫn đang không phải số đếm từ prediction đã khóa

**Vị trí:** Mục 4.5.3, bảng “Ma trận nhầm lẫn của MAST-PAD trên LCC-FASD”.

Word ghi real=1.940, attack=16.854; CSV hợp lệ thực tế cho **real=1.941, attack=16.853**, tổng vẫn 18.794. Bảng Word gán hai tỷ lệ lỗi cùng 22,61%, không khớp operating point nguồn.

Nếu cần giữ một ma trận cho kết quả cuối, dùng số đếm lại sau tại **source threshold 4,66259765625**, score=`z_real-z_attack`:

| Nhãn thật | Dự đoán Real | Dự đoán Attack | Tổng |
|---|---:|---:|---:|
| Real | 1.043 | 898 | 1.941 |
| Attack | 1.345 | 15.508 | 16.853 |

Tương ứng APCER **7,9808%**, BPCER **46,2648%**, ACER **27,1228%**. Đây là điểm vận hành nguồn, **không phải ngưỡng camera CASIA-calibrated**, và cũng không phải ma trận tại EER. Có thể bỏ ma trận để rút gọn, nhưng vẫn nên nêu ngắn rằng AUC tốt hơn chưa bảo đảm tỷ lệ từ chối người thật thấp tại một ngưỡng cố định nguồn.

### P0-3. Phạm vi tập dữ liệu và tính độc lập cần chính xác hơn

**Vị trí:** 1.3, 4.1, caption các bảng/biểu đồ và 5.2.

| Population cuối | Vai trò | Số lượng cần ghi |
|---|---|---|
| CelebA official Train sau lọc | Huấn luyện | 484.561 usable / 494.405 candidates |
| CelebA full Test-as-Val | Chọn checkpoint và ngưỡng nguồn | 66.379 scored / 67.170 candidates |
| LCC Combined | Pool physical train/dev/eval, chấm điểm mức ảnh | 18.794 scored / 18.827 candidates |
| CASIA reconstructed Combined | Dedup bản ảnh; video score = mean valid-frame score | 2.660 frame; 358 scorable / 360 candidate video |

- `18,8K` LCC là số ảnh, không phải số subject như header Bảng 1.3.
- CASIA đang dùng bản ảnh màu trích xuất, tái dựng ID video từ tên file; video phải đủ ít nhất 4 frame hợp lệ. Không gọi 360 candidate của bản copy là toàn bộ dữ liệu gốc 50 người mà không phân biệt nguồn.
- CASIA copy có overlap train/test: **20 subject ID, 240 video ID, 1.403 frame ID**. Combined dedup ưu tiên test; không cộng cơ học train+test.
- Nêu rõ counts lớp tương ứng population: CASIA Combined scorable có 90 real và 268 attack; không dùng 90/270 như số đã chấm đủ 360.
- CNN chấm từng frame rồi lấy trung bình score theo video; không từ đó kết luận mô hình học chuyển động hoặc độ bền temporal. Temporal voting camera là phép khác.
- LCC không tham gia tối ưu gradient của các run cuối, nhưng đã được theo dõi trong development. Không gọi LCC là kiểm thử cuối “hoàn toàn độc lập/chưa từng nhìn thấy” chỉ vì chưa train trên nó.
- CelebA Test-as-Val không phải held-out independent test. Chữ “Test” trong tên dataset không thay đổi vai trò đã sử dụng.
- Tách **source-frozen evaluation** khỏi **CASIA train-calibrated operating points**. AUC là metric theo thứ hạng; threshold khác nhau không giải thích được khác biệt AUC nếu cùng score/population.

### P0-4. Claim latency và triển khai chưa đủ chứng cứ

**Vị trí:** Tóm tắt, 1.2, 2.2, 2.4, 3.2, 4.6 và 5.1.

Word có mô tả batch 1, 50 warmup, 1.000 frame. Mô tả này vẫn cần raw log/script có thể truy vết. Chưa xác định được bằng chứng tương ứng trong nguồn đã kiểm cho `2,22 ms`, `67,98 ms` và FPS end-to-end.

- Nếu chưa cung cấp log/script đo: bỏ các số latency khỏi abstract và đóng góp; đổi 4.6 thành **“Chi phí mô hình và minh họa tích hợp”**. Chỉ giữ thông số complexity có artifact.
- Nếu có log: ghi đúng máy đo, CPU/GPU provider, threads, model hash, input, phạm vi timer, số mặt, gallery/DB/cache, số lần đo, mean/p95; phân biệt inference PAD, camera FPS và thời gian đến quyết định danh tính.
- `67,98 ms` tương đương xấp xỉ **14,71 FPS** nếu nghịch đảo thời gian xử lý; abstract đang ghi **15,5 FPS**, còn phần khác 14,7. Đây là bất nhất số học cần sửa ngay, không cần đợi benchmark.
- Cộng latency từng stage không đủ chứng minh throughput của pipeline có cadence, queue và worker chạy bất đồng bộ.
- Không suy từ graph nhẹ thành “đã chạy realtime trên Pi/Jetson”. Trạm huấn luyện T4 không là benchmark thiết bị biên.
- Không khẳng định 1,74 MB là kích thước file ONNX/RAM thực nếu chỉ suy `434434×4 byte`. Đây là xấp xỉ footprint trọng số FP32, chưa tính graph, external data hoặc activation.
- Bỏ suy luận trọng số vừa L3 nên nhanh nếu chưa đo; không cần phần dự đoán cache CPU/NPU để hoàn thiện báo cáo CV.

### P1-1. Khẳng định nhân quả và “vượt trội” đang mạnh hơn bằng chứng

**Vị trí:** abstract, 4.3, nhận xét baseline, 5.1 và đoạn CE-only trong 5.2.

Các chênh lệch AUC chính, đơn vị **điểm phần trăm**:

| So sánh | Δ LCC | Δ CASIA video Combined |
|---|---:|---:|
| Clean+Aux − CE-only | +0,20 | +2,21 |
| WBST − Clean+Aux | +1,48 | +0,19 |
| MAST-PAD − Clean+Aux | +1,43 | +1,48 |
| MAST-PAD − WBST | −0,056 | +1,29 |
| MAST-PAD − CE-only | +1,63 | +3,69 |

Không dùng những số này để khẳng định:

- gate là nguyên nhân duy nhất tạo gain: WBST và MAST-PAD còn khác sigma;
- đã “bóc tách chính xác đóng góp độc lập và bổ trợ mạnh mẽ”: chưa có thiết kế factorial đủ các tổ hợp, chưa có nhiều seed;
- aux chứng minh “ổn định video”: bảng AUC chưa đo ổn định temporal;
- full-scale tốt hơn mọi quy mô: kết quả 100K không phải đối chứng cùng protocol chọn checkpoint;
- epoch tốt nhất sớm hơn đồng nghĩa training nhanh hơn: các bước selection/forward phổ tăng compute;
- MAST-PAD đạt SOTA hoặc tốt hơn trên mọi target: LCC Combined WBST nhỉnh hơn; CASIA EER không cùng chiều với AUC.

Viết bằng kết quả quan sát của cấu hình một seed, kèm giới hạn ở đúng nơi diễn giải. Không cần kể các vòng micro để giải thích vì sao phải thận trọng.

### P1-2. Diagnostics đang sai số và dễ bị hiểu là causal evidence

**Vị trí:** 4.5.1–4.5.2.

Log `spectral_diagnostics.csv` của cấu hình MAST-PAD full-scale cho:

| Thống kê training | Epoch 7, epoch tạo best checkpoint | Epoch 11, epoch cuối |
|---|---:|---:|
| Harmful fraction | 83,2865% | 83,2135% |
| Spectral flip rate | 0,4103% | 0,3077% |
| LOW được chọn | 45,0131% | 42,6452% |
| MID được chọn | 27,0880% | 27,3035% |
| HIGH được chọn | 27,8989% | 30,0513% |

Word ghi LOW≈31,8%, MID≈34,6%, HIGH≈33,6% không khớp các hàng này. Cần nêu epoch hoặc phép aggregate chính xác trước khi dùng số khác. Các thống kê epoch 7 là số thu trong quá trình training epoch 7, không phải lượt đánh giá tĩnh lại toàn dataset bằng frozen best checkpoint.

Trong báo cáo cuối, diagnostics là tùy chọn: giữ một hình nhỏ/đoạn ngắn nếu giúp hiểu gate, còn toàn bộ đường cong theo epoch đưa phụ lục. Không dùng harmful fraction cao để suy ra hai cơ chế tối ưu tương đương; gate/reduction/sigma vẫn khác và không có ablation kiểm soát.

### P1-3. Cần diễn giải kết quả bất lợi và vai trò của calibration

**Vị trí đề xuất trong Word:** một tiểu mục “Khả năng phân tách và điểm vận hành trên miền mới”, ngay sau bảng AUC chính. Giữ thảo luận này trong chính văn; bảng ngưỡng/metric đầy đủ có thể đặt phụ lục. AUC là chỉ số trọng tâm cho câu hỏi tổng quát hóa khả năng phân tách, nhưng không thay thế đánh giá rủi ro tại ngưỡng sử dụng thực tế.

#### Phân biệt các chỉ số trước khi giải thích

| Chỉ số | Thông tin phản ánh | Nếu chỉ đổi ngưỡng quyết định, giữ nguyên model/score/population |
|---|---|---|
| ROC-AUC | Khả năng xếp hạng real/attack trên toàn đường ROC | Không đổi |
| EER | Mức lỗi ở điểm cân bằng hai loại lỗi trên ROC, theo quy ước tính đã nêu | Không đổi |
| APCER, BPCER | Hai loại lỗi tại ngưỡng đang áp dụng | Có thể đổi, thường phải đánh đổi |
| ACER/HTER | Trung bình hai loại lỗi tại ngưỡng cụ thể theo protocol | Có thể đổi; giảm ACER chưa đồng nghĩa giảm nguy cơ lọt attack |

**Không viết “EER tệ vì chưa calibrate lại threshold”.** EER đã được tính bằng cách khảo sát đường ROC, không phải lỗi tại ngưỡng nguồn đang dùng. EER bất lợi phản ánh sự chồng lấn hoặc hình dạng phân phối score tại vùng cân bằng; đổi ngưỡng vận hành không sửa được thứ hạng score. Nếu muốn cải thiện AUC/EER, cần thay đổi có tác động đến khả năng phân tách, rồi đánh giá riêng; đó không còn là chỉ chọn lại một ngưỡng.

Ngược lại, **APCER/BPCER/ACER bất lợi tại source-frozen threshold có thể gồm cả vấn đề ngưỡng chưa phù hợp miền đích**. Chênh lệch miền có thể dịch chuyển phân phối score, trong khi model vẫn còn khả năng xếp hạng hai lớp. Đây là cách giải thích hợp lý để khảo sát, không phải bằng chứng mọi lỗi đều do threshold.

#### Đọc các kết quả hiện có

1. **LCC:** MAST-PAD có AUC **0,8603**, nhưng tại ngưỡng nguồn có APCER **7,98%**, BPCER **46,26%**, ACER **27,12%**. Điều này cho thấy điểm vận hành hiện tại từ chối nhiều người thật. EER **22,06%** cũng cho thấy phân tách chưa hoàn hảo; không thể hứa chỉ calibration là đủ để đạt mức lỗi thấp. Báo cáo source-frozen chủ động giữ ngưỡng nguồn để đánh giá khả năng chuyển ngưỡng, không phải một lỗi triển khai cần âm thầm sửa trên tập đánh giá.
2. **CASIA video Combined:** AUC **0,9352** đi kèm EER **16,92%** và source-frozen BPCER **51,11%**. AUC cao hơn CE-only nhưng EER không tốt hơn CE-only trong run được báo cáo. Đây là hai góc nhìn khác nhau của đường ROC; không kết luận cải thiện toàn diện và không dùng calibration để giải thích chênh lệch EER này.
3. **Có bằng chứng khảo sát calibration trên CASIA**, nên không viết chung rằng “tất cả tập mới đều chưa calibrate”. Bảng headline và ma trận source-frozen vẫn dùng ngưỡng nguồn; phần đánh giá bổ sung và profile camera CASIA đã có calibration bằng target train. Hai thiết lập trả lời hai câu hỏi khác nhau và phải tách tên rõ.

Minh họa trực tiếp từ artifact **MAST-PAD CASIA test, mức video**; ngưỡng target được fit trên CASIA train video:

| Policy ngưỡng | AUC | EER (%) | APCER (%) | BPCER (%) | ACER (%) |
|---|---:|---:|---:|---:|---:|
| Source-frozen | 0,925878 | 18,18 | 1,89 | 53,41 | 27,65 |
| Fit trên CASIA train video | 0,925878 | 18,18 | 20,45 | 17,05 | 18,75 |

Nguồn trực tiếp: [metric video test source-frozen](../antispoof/notebooks/final/binary_crossdomain_v2/output/casia_train_test_C_P3SF_R7SC_100k_fullscale/runs/R7_SC_FULL_crop15/test/video_metrics.json) và [metric video test sau train calibration](../antispoof/notebooks/final/binary_crossdomain_v2/output/casia_train_test_C_P3SF_R7SC_100k_fullscale/runs/R7_SC_FULL_crop15/test/video_metrics_casia_calibrated.json). Các số này đã đối chiếu JSON; đây là **test video AUC 0,925878**, không thay cho **Combined video AUC 0,935199** trong bảng chính.

ACER giảm **8,90 điểm phần trăm**, nhưng APCER tăng từ **1,89% lên 20,45%**: giảm từ chối người thật đi kèm tăng chấp nhận attack. Không mô tả đây là “bảo mật tốt hơn” chỉ dựa vào ACER. AUC/EER giữ nguyên đúng như kỳ vọng khi chỉ thay threshold. CASIA copy có overlap train/test nên đây là minh họa sự đánh đổi điểm vận hành trong dữ liệu hiện có, chưa là xác nhận độc lập chất lượng calibration cho camera mới. Không đưa threshold video này vào runtime frame.

#### Cách tập trung vào AUC mà vẫn báo cáo trung thực

- Dùng **AUC làm headline** cho so sánh khả năng phân tách ngoại miền của các cấu hình cuối. Nêu rõ AUC không phụ thuộc một threshold cố định và không phải accuracy tại ngưỡng triển khai.
- Ngay sau bảng AUC, giữ một đoạn về EER và lỗi tại source threshold. Không dùng nhãn “chỉ số phụ” để bỏ qua kết quả bất lợi hoặc nói chúng không quan trọng.
- Diễn giải AUC như bằng chứng về khả năng phân tách để tiếp tục khảo sát operating point; **không hứa calibration chắc chắn cho APCER/BPCER đạt yêu cầu**. AUC toàn cục cao vẫn có thể đi kèm hiệu năng yếu tại vùng APCER rất thấp.
- Với ứng dụng xác thực, nêu mục tiêu điểm vận hành: chẳng hạn BPCER đạt được khi ràng buộc APCER theo mức rủi ro chấp nhận. Mức yêu cầu phải được định trước theo ứng dụng, không chọn hậu nghiệm từ test để làm đẹp số.
- Tách một câu hỏi nghiên cứu **không fit target** khỏi một câu hỏi triển khai **có calibration target**. Không thay âm thầm bảng source-frozen bằng bảng calibrated rồi vẫn gọi strict cross-domain no-target-fitting.

#### Đoạn thảo luận có thể đưa thẳng vào báo cáo

> Báo cáo sử dụng ROC-AUC làm chỉ số chính để so sánh khả năng phân tách ngoại miền, vì chỉ số này không gắn với một ngưỡng quyết định cụ thể. Tuy nhiên, các kết quả tại ngưỡng cố định nguồn còn bất lợi: trên LCC Combined, MAST-PAD có AUC 0,8603 nhưng BPCER 46,26% và ACER 27,12%, cho thấy ngưỡng nguồn chưa tạo được điểm vận hành phù hợp cho miền đích. Kết quả này gợi ý cần khảo sát calibration theo miền sử dụng, đồng thời cho thấy mô hình chưa sẵn sàng đáp ứng yêu cầu xác thực chỉ từ kết quả AUC. EER 22,06% phản ánh giới hạn phân tách tại vùng cân bằng và không thể cải thiện bằng cách đơn thuần đổi ngưỡng vận hành.
>
> Trên bản sao CASIA ở mức test video, calibration bằng tập train làm ACER giảm từ 27,65% xuống 18,75%, trong khi AUC và EER không đổi. Sự giảm lỗi trung bình đi kèm APCER tăng và BPCER giảm, cho thấy calibration lựa chọn sự đánh đổi giữa bảo mật và khả dụng chứ không tự nâng khả năng phân tách. Kết quả CASIA này còn bị giới hạn bởi overlap giữa các bản chia dữ liệu. Để sử dụng ở một miền camera mới, cần chọn ngưỡng theo mục tiêu APCER/BPCER trên tập calibration riêng và xác nhận trên người/phiên độc lập; không chọn ngưỡng trên test hoặc Combined đang dùng để báo cáo kết quả cuối.

#### Đề xuất đánh giá miền sử dụng, không thay protocol nghiên cứu hiện có

Khi triển khai, thu real/attack bằng đúng camera, detector, crop và preprocessing; chia calibration và holdout theo subject/session, tránh trùng người/phiên hoặc video. Chọn ngưỡng trên calibration theo mục tiêu bảo mật đã định, rồi khóa ngưỡng và đánh giá trên holdout. Báo cáo APCER/BPCER, coverage, loại attack/ánh sáng và quyết định temporal của pipeline; không chỉ frame score hoặc balanced ACER.

Bước này là thí nghiệm triển khai tiếp theo, chưa được thực hiện trong lần chỉnh tài liệu. Không tune LCC/CASIA test hoặc Combined, không thay `.env`/runtime JSON, và không thay nội dung báo cáo source-frozen để tạo ấn tượng đã có kết quả miền mới. Khi có nhiều training seed, calibration phải theo cùng policy cho từng run và kết quả phải được tổng hợp riêng theo policy như Mục 10.

## 5. Mô tả đầy đủ pipeline, giữ MAST-PAD làm trọng tâm

**Vai trò phần này:** làm cơ sở biên tập chương pipeline đầy đủ theo logic `face_auth`. Mỗi giai đoạn cần có mục tiêu, input/output và cách phối hợp với các giai đoạn khác. Các bảng cấu hình chi tiết là căn cứ kiểm tra; lựa chọn những thông số cần để giải thích hành vi, không bắt buộc chép mọi trường vào Word.

### 5.1. Pipeline nên được trình bày thế nào?

Giữ sơ đồ sáu chức năng CV để dễ học, nhưng chỉ rõ các chức năng phối hợp qua trạng thái track và worker; không vẽ tất cả như sáu khối blocking nối tiếp trên mọi frame.

```text
Camera → detection theo lịch / LK propagation → track continuity
       → kiểm tra input PAD → crop 1.5, 80×80 → raw PAD score
       → rolling votes → REAL / SPOOF / PENDING
       → khi REAL và đến hạn: align 112×112 → gửi queue
Worker → ArcFace 512-D → cache centroid hoặc DB cosine search → kết quả
Camera → nhận kết quả còn hợp lệ với track/phiên/PAD → hiển thị
```

**Tách hai luồng input:** PAD nhận expanded crop theo contract, không dùng ảnh ArcFace align 112×112; nhận diện mới dùng 5-landmark similarity alignment. Spectral gate lúc training và rolling vote lúc camera là hai cơ chế khác nhau.

Không đưa ThingsBoard/MQTT/rule chain/hardware vào sơ đồ này. Trong Word hiện đọc chưa thấy các chuỗi ThingsBoard/MQTT; đây là quy tắc giữ phạm vi khi biên tập, không phải kết luận Word đang chứa các mục đó.

### 5.2. Các chi tiết cần sửa theo source/cấu hình hiện tại

| Hạng mục | Giá trị/cách xử lý đã đọc ngày 20/10 | Hệ quả khi sửa báo cáo |
|---|---|---|
| Tracking | IoU association + Lucas–Kanade optical flow | Bỏ `IoU + Kalman` ở phần diễn giải và bảng latency |
| Ngưỡng IoU | 0,3 | Word ghi 0,5 cần sửa nếu mô tả cấu hình hiện tại |
| Cadence | Detection 0,2 s; PAD 0,2 s; recognition 1 s | Ghi đúng lịch cấu hình; không đồng nhất cadence với FPS đo được |
| Temporal PAD | Window tối đa 5, min votes 3; spoof ratio ≥0,6; stale 3 s | Trước đủ 3 phiếu là PENDING; mẫu số ratio là số phiếu hiện có, không luôn bằng 5 |
| Model camera | MAST-PAD full-scale, best epoch 7 | Tên file/checkpoint cụ thể xem runtime artifact; không dùng mã run làm tên phương pháp |
| Threshold camera | Logit 1,6586914; P(real)≈0,8400623 | Hiệu chuẩn CASIA train frame; không dùng threshold video hoặc gọi là threshold nguồn |
| Threshold research MAST-PAD | Logit 4,66259765625 | Tách khỏi threshold demo dù cùng checkpoint |
| PAD preprocessing | BGR, 80×80, [0,1], crop 1,5; không gamma/mean/std; crop smoothing=false | Ghi contract ngắn, không đổi preprocessing để khớp lời văn |
| Matching | Cosine centroid, threshold 0,35 | Không đồng nhất với PAD threshold hoặc cache threshold |
| Recognition | Worker có queue; nhận kết quả còn hợp lệ theo track và verification generation | Bổ sung ngắn, tránh mô tả main loop gọi helper đồng bộ |

Đây là cấu hình import trong một tiến trình mới, không chứng minh app đang mở sử dụng đúng môi trường ấy và không phải kết quả camera thực nghiệm.

**Sửa đoạn voting:** nếu đã đủ 5 phiếu, quy tắc spoof ratio ≥60% tương đương ít nhất 3 spoof; ratio <60% tương đương ít nhất 3 real. Phần này đúng trong trường hợp cửa sổ đầy. Tuy nhiên cấu hình hiện tại cho ra quyết định từ 3 phiếu, nên không mô tả mọi quyết định như luôn chia cho 5. Viết ratio trên số phiếu đang có, kèm readiness và stale/reset. Ví dụ ở 3 phiếu: 2 spoof/3→SPOOF; 1 spoof/3→REAL nếu các điều kiện hợp lệ khác thỏa.

**Thời gian tích lũy:** không lấy `5×0,2=1 s` làm latency chính xác. Nếu phiếu đầu tại t=0, đủ ba phiếu lý tưởng ở t≈0,4 s, đủ năm ở t≈0,8 s; scheduling, inference và reset có thể kéo dài. Chỉ trình bày đó là ví dụ lý tưởng, không số đo người dùng cuối.

Bỏ các câu “triệt tiêu false positive”, “loại bỏ hoàn toàn false alarm”, “không làm trễ hệ thống”. Các guard và voting có mục đích giảm rủi ro nhưng tác dụng cần đo. `DETECTOR_MARGIN=5` là một guard detector; không đồng nhất với mọi kiểm tra hình học riêng trước PAD.

### 5.3. Nhận diện và hiển thị

- Worker thực hiện ArcFace rồi centroid cache/DB search; mỗi cache hit vẫn cần embedding từ ảnh mới. Không mô tả cache như bỏ toàn bộ suy luận nhận diện.
- Tách trạng thái xác thực/raw result khỏi bộ ổn định nhãn UI. Giữ tên hiển thị để giảm nhấp nháy không tạo thêm bằng chứng danh tính.
- Kết quả worker cũ không được áp cho track/phiên đã reset hoặc PAD không còn hợp lệ. Chỉ cần một đoạn giải thích nguyên tắc, không đưa TTL/retry/SQL chi tiết vào chính văn.
- Mục 2.6.2 ghi FAR `<0,001%` tại cosine 0,35: comment trong `config.py` không đủ chứng minh trị số này. Chưa xác minh được artifact tương ứng trong lần rà; bỏ claim FAR nếu không dẫn bộ pair/prediction, split calibration/test, số impostor và số false accept. Có thể ghi “ngưỡng cấu hình 0,35; chưa được kiểm chứng trên dữ liệu triển khai”.
- Không biến báo cáo CV thành báo cáo quản lý nhân sự/chấm công. Nếu cần nói DB, chỉ mô tả kho embedding và cosine search phục vụ nhận diện.

## 6. Tái cấu trúc báo cáo để tập trung vào kết quả cuối

### 6.1. Các quyết định giữ, rút, bỏ và chuyển

| Nội dung hiện tại | Hành động | Phần cuối cần giữ |
|---|---|---|
| 4.2 “Tiến trình các vòng thực nghiệm”, 4.2.1–4.2.3 | **Bỏ khỏi báo cáo nộp** | Không chuyển kết quả 100K/micro sang phụ lục; chỉ nêu setup full-scale cuối |
| 3.5 có cột biến thể severity ngoài phạm vi kết quả cuối | **Thay bảng** | Bốn đối chứng CE-only / Clean+Aux / WBST / MAST-PAD |
| 3.6 kể thay đổi crop 2,7→1,5, vòng 10–17 | **Viết lại** | Mô tả crop 1,5 và tham số phổ cuối; sigma khác nhau giữa hai cấu hình |
| 3.4.5 gradient khai triển | **Rút hoặc chuyển phụ lục** | Gate detach, normalization và H=0 |
| 1.2 “đóng góp khảo sát crop” | **Bỏ khỏi đóng góp chính** | Crop là một thành phần setup; chưa có bằng chứng cơ chế shortcut |
| 5.1 kể 10K→100K→full và crop gains | **Bỏ** | Kết luận bằng phương pháp cuối và bảng full-scale |
| Phụ lục thống kê lịch sử các vòng | **Bỏ khỏi bản nộp** | Lưu lịch sử ở repo; phụ lục báo cáo chỉ chứa cấu hình/tái lập/metric bổ sung của các run full-scale cuối |
| CE-only nằm như “vòng 21 mới bổ sung” | **Đưa vào nhóm đối chứng từ đầu** | Một phương pháp baseline bình thường trong setup và bảng kết quả |
| Đoạn CE-only trong 5.2 “hạn chế” | **Chuyển, viết lại** | Kết quả về AUC ở Chương 4; hạn chế là một seed và provenance chưa đủ |
| 4.5 harmful diagnostics | **Rút mạnh** | Tối đa một hình/đoạn, đúng epoch; phụ lục nếu dài |
| 4.6 latency chưa có log | **Bỏ số chưa truy vết** | Complexity và minh họa tích hợp; benchmark chỉ khi có dữ liệu đo |
| Nhiều bảng loại attack/metadata lặp | **Gộp** | Một minh họa attack; một bảng vai trò/count dataset |
| Survey detector/backbone không sử dụng | **Rút** | Lý do chọn SCRFD, MiniFASNetV2, ArcFace; bỏ nhận xét tốc độ/accuracy không đo |
| Literature hai bảng dài | **Rút hoặc giữ có điều kiện** | Context protocol và nguồn; không lấn át đối chứng cùng setup của nhóm |

**Phân biệt cần giữ:** đối chứng cuối không phải micro experiment. CE-only, Clean+Aux, WBST và MAST-PAD đều cần thiết để đọc kết quả. Việc bỏ lịch sử phát triển không có nghĩa chỉ giữ hàng MAST-PAD có số cao nhất.

### 6.2. Dàn ý đề xuất

**Chương 1 — Bài toán và mục tiêu**

- Presentation attack, domain shift, yêu cầu mô hình nhỏ.
- Phạm vi CV/PAD, mục tiêu và đóng góp có bằng chứng.
- Tổng quan ngắn dữ liệu và công trình liên quan; không lặp catalog attack ở các chương sau.

**Chương 2 — Pipeline xác thực khuôn mặt**

- Sơ đồ tổng thể sáu giai đoạn, luồng dữ liệu và bảng input/output.
- **Detection:** SCRFD, bbox và 5 landmarks; vai trò bộ lọc đầu vào.
- **Tracking:** liên kết track qua khung hình, IoU/LK, điều phối xử lý và lưu trạng thái.
- **PAD:** crop theo contract, raw score và temporal verdict; giới thiệu vai trò MAST-PAD, dẫn sang Chương 3 để tránh lặp thuật toán.
- **Alignment:** 5-point similarity transform, đầu vào 112×112 riêng cho nhận diện.
- **Embedding:** ArcFace 512-D và chuẩn hóa đặc trưng.
- **Matching:** cosine với centroid, ngưỡng danh tính, MATCH/UNKNOWN; recognition worker và điều kiện áp kết quả theo track/phiên/PAD.
- Giải thích đầy đủ sự phối hợp và các trạng thái chính; rút chi tiết hạ tầng/triển khai không trực tiếp phục vụ CV. Mô tả hệ thống có chiều sâu vừa đủ, trong khi MAST-PAD vẫn là phần phương pháp nổi bật nhất.

**Chương 3 — Phương pháp MAST-PAD**

- Nguyên tắc phổ chỉ trong training; backbone và aux heads.
- Tạo view phổ với tham số cuối, DC, clamp.
- Margin → worst-view từng ảnh → harmful gate → loss, warmup, H=0.
- Bảng bốn cấu hình cuối; một câu ghi hai cấu hình phổ khác sigma và được so sánh như các thiết kế hoàn chỉnh.
- Một pseudocode và một hình train/inference; không kể lịch sử thiết kế.

**Chương 4 — Thiết lập và kết quả cuối**

- Dataset/population/coverage, checkpoint selection, seeds và threshold policy.
- Bảng bốn cấu hình full-scale, LCC image AUC và CASIA reconstructed video AUC.
- Phân tích AUC: gain so với các đối chứng và xu hướng theo target; kết quả nhiều seed khi hoàn tất.
- Một đoạn ngắn giải thích EER/lỗi tại ngưỡng nguồn và vai trò calibration; chi tiết metric/threshold đưa phụ lục khi cần.
- Complexity và đặc tính giữ nguyên graph PAD suy luận; một hình kết quả tích hợp nếu có ảnh thật, liên hệ với pipeline đã mô tả ở Chương 2.
- Đối chiếu literature ngắn; phạm vi đánh giá và hướng hoàn thiện.

**Chương 5 — Kết luận và hướng phát triển**

- Phương pháp cuối, kết quả chính, lợi thế graph gọn.
- Tóm lược phạm vi: thiết lập dữ liệu và số seed thực có; tránh lặp lưu ý đã giải thích trong setup.
- Hướng nghiên cứu tiếp: hoàn tất seed bổ sung, phân tích cơ chế gate/phổ, mở rộng target. Calibration và camera chỉ nhắc ngắn ở hướng ứng dụng.

**Phụ lục chỉ khi cần:** cấu hình/manifest/checkpoint và commands tái lập của các run full-scale cuối; công thức phụ; metric theo split/threshold của chính các run đó; log benchmark có thật. Không đưa kết quả 100K, micro screening hoặc lịch sử run thất bại vào phụ lục báo cáo nộp.

## 7. Bảng headline và văn bản có thể dùng khi sửa Word

### 7.1. Bảng kết quả chính gọn

| Phương pháp | LCC Combined AUC, mức ảnh | CASIA reconstructed Combined AUC, mức video |
|---|---:|---:|
| CE-only* | 0,843988 | 0,898300 |
| Clean + Aux | 0,846011 | 0,920398 |
| WBST + Aux | **0,860831** | 0,922264 |
| MAST-PAD + Aux | 0,860272 | **0,935199** |

*CE-only lấy từ báo cáo tổng hợp được cung cấp; các run còn lại đã đối chiếu lại JSON/CSV local trong lần rà này. Cả bốn dùng cùng kiến trúc PAD suy luận; 434.434 tham số là số deployment từ complexity artifact của nhóm Clean + Aux/WBST/MAST-PAD, không phải số params toàn training model có aux.*

**Caption đề xuất:** “ROC-AUC của các cấu hình MiniFASNetV2 full-scale huấn luyện trên CelebA-Spoof. LCC Combined là pool ảnh train/dev/eval của bản dữ liệu sử dụng. CASIA Combined được dedup từ bản ảnh trích xuất và lấy trung bình score frame theo video tái dựng; không tương đương official CASIA video benchmark. Các số trong bảng là kết quả một run cho mỗi cấu hình; khi hoàn tất các seed bổ sung sẽ trình bày mean ± SD.”

Không cần lặp cột 0,434M trên mọi hàng nếu đã ghi chung caption. Không bold MAST-PAD ở cả hai target khi LCC WBST cao hơn. Không gọi chênh lệch −0,056 pp là “tương đương thống kê”; chỉ nói gần nhau về mặt trị số.

### 7.2. Đoạn mô tả loss

> Với mỗi ảnh huấn luyện, phương pháp tạo ba biến thể suy giảm phổ và chọn biến thể có margin theo nhãn nhỏ nhất. MAST-PAD chỉ đưa cross-entropy của biến thể được chọn vào thành phần phổ khi margin của nó thấp hơn margin của ảnh sạch. Loss phổ được lấy trung bình trên các mẫu thỏa điều kiện này và bằng không nếu không có mẫu nào thỏa. Phương pháp không gán trọng số liên tục theo độ giảm margin. Sau epoch warmup, mục tiêu gồm 0,75 lần loss phân loại ảnh sạch, 0,25 lần loss phổ có chọn lọc và loss phụ trợ trên ảnh sạch. Khi suy luận, mạng chỉ nhận ảnh không gian; bộ sinh phổ và gate không được sử dụng.

### 7.3. Đoạn nhận xét kết quả

> Trong thực nghiệm full-scale, MAST-PAD đạt AUC 0,8603 trên LCC Combined và 0,9352 trên CASIA reconstructed video Combined, cải thiện tương ứng 1,43 và 1,48 điểm phần trăm so với Clean + Aux. So với WBST, hai cấu hình đạt kết quả gần nhau trên LCC, còn MAST-PAD cao hơn 1,29 điểm phần trăm trên CASIA. Kết quả cho thấy tiềm năng của huấn luyện phổ có chọn lọc trong việc nâng cao khả năng phân tách ngoại miền mà vẫn giữ graph PAD suy luận gọn. Các chỉ số tại ngưỡng nguồn cho thấy calibration theo miền sử dụng là bước tiếp theo để lựa chọn sự đánh đổi phù hợp giữa chấp nhận attack và từ chối người thật; việc chọn lại ngưỡng không làm thay đổi AUC hay EER.


### 7.4. Tóm tắt đề tài đề xuất

> Báo cáo trình bày pipeline xác thực khuôn mặt gồm phát hiện, theo dõi, chống giả mạo, căn chỉnh, trích xuất đặc trưng và so khớp danh tính, với trọng tâm là MAST-PAD — chiến lược huấn luyện phổ có chọn lọc theo margin nhằm cải thiện khả năng phân tách ngoại miền của mô hình chống giả mạo MiniFASNetV2. Với mỗi ảnh, phương pháp chọn biến thể phổ bất lợi nhất và chỉ bổ sung loss phổ khi biến thể làm giảm margin so với ảnh sạch. Các thao tác phổ chỉ được sử dụng trong huấn luyện, giữ nguyên graph PAD suy luận với 434.434 tham số.
>
> Thực nghiệm full-scale dùng 484.561 ảnh huấn luyện CelebA-Spoof khả dụng, với các đối chứng CE-only, Clean+Aux và WBST. MAST-PAD đạt ROC-AUC 0,8603 trên LCC Combined ở mức ảnh và 0,9352 trên CASIA Combined ở mức video tái dựng từ ảnh trích xuất. So với Clean+Aux, mức tăng tương ứng là 1,43 và 1,48 điểm phần trăm; so với WBST, kết quả gần nhau trên LCC và cao hơn trên CASIA. Kết quả cho thấy hướng huấn luyện đề xuất cải thiện khả năng phân tách ngoại miền trong thiết lập khảo sát, đồng thời giữ mô hình PAD suy luận gọn. MAST-PAD cho thấy tiềm năng khai thác thông tin phổ trong huấn luyện mà không cần bổ sung nhánh xử lý phổ khi suy luận.

### 7.5. Đoạn “Phạm vi đánh giá và hướng hoàn thiện” đề xuất

> Đề tài được đánh giá trên các cấu hình full-scale với giao thức và số lần chạy đã trình bày trong phần thực nghiệm. CelebA Test-as-Val phục vụ chọn checkpoint và ngưỡng; LCC được theo dõi trong quá trình phát triển, còn CASIA được đánh giá từ bản ảnh trích xuất tái dựng theo video. Vì vậy, các kết quả được diễn giải trong phạm vi thiết lập này. So sánh WBST và MAST-PAD phản ánh hai cấu hình huấn luyện hoàn chỉnh, có khác biệt về độ rộng Gaussian.
>
> Hướng hoàn thiện gần là bổ sung đánh giá qua các seed đang chạy, phân tích rõ hơn vai trò của gate và các cấu hình phổ, đồng thời mở rộng đánh giá trên miền dữ liệu khác. Khi đưa mô hình vào một ứng dụng cụ thể, có thể khảo sát thêm calibration theo miền sử dụng.

Khi đủ seed, thay “bổ sung đánh giá qua các seed đang chạy” bằng mô tả kết quả nhiều seed thực có. Không cần liệt kê lại toàn bộ kiểm tra kỹ thuật của bản đánh giá trong mục này.

### 7.6. Kết luận theo định hướng học phần

> Đề tài đã xây dựng pipeline xác thực khuôn mặt sáu giai đoạn, đồng thời phát triển và đánh giá MAST-PAD như đóng góp trọng tâm về huấn luyện chống giả mạo trên MiniFASNetV2. Phương pháp kết hợp việc chọn một worst view cho từng ảnh với harmful gate, khai thác biến thể phổ trong training trong khi giữ nguyên graph PAD suy luận với 434.434 tham số. Thực nghiệm full-scale đạt AUC 0,8603 trên LCC Combined và 0,9352 trên CASIA reconstructed video Combined, cao hơn đối chứng Clean + Aux trên cả hai tập khảo sát.
>
> Qua đề tài, nhóm đã vận dụng biến đổi Fourier, margin phân loại, giám sát đa nhiệm và đánh giá ngoại miền để thiết kế một chiến lược huấn luyện có kết quả khả quan trên mô hình gọn. Hướng tiếp theo là hoàn tất đánh giá nhiều seed, phân tích sâu hơn cơ chế chọn lọc và mở rộng kiểm chứng ngoại miền. Pipeline tích hợp cho thấy cách kết nối mô hình PAD với phát hiện, theo dõi và nhận diện thành một quy trình thị giác hoàn chỉnh.

Các giá trị trên là số hiện có; thay bằng mean ± SD khi đủ seed theo Mục 10. Phần kết luận không cần lặp toàn bộ hạn chế đã nêu trước đó.

## 8. Trích dẫn, bảng hình và định dạng

### 8.1. Lỗi nguồn đã xác minh và phần chưa audit

- **AENet:** mục [1] hiện ghi “Adversarial Erasing Network for Face Anti-Spoofing”, tác giả Y. Chen… Đây không phải tên AENet trong nguồn CelebA-Spoof đang viện dẫn. Paper gốc gọi **Auxiliary Information Embedding Network**, thuộc “CelebA-Spoof: Large-Scale Face Anti-Spoofing Dataset with Rich Annotations”, Y. Zhang và cộng sự, ECCV 2020. Xem [paper gốc, Mục 4](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123570069.pdf). Sửa entry và các citation tương ứng; không gán số LCC 0,868 cho paper gốc nếu chưa tìm đúng bảng nguồn.
- **CASIA:** bản Word gắn “50 đối tượng” với “360 video” như mô tả toàn bộ database. Tách thông tin database gốc khỏi bản copy đang dùng. Có [paper gốc của tác giả](https://nlpr.ia.ac.cn/2012papers/gjhy/gh69.pdf), DOI `10.1109/ICB.2012.6199754`; endpoint PDF bị timeout khi mở trong lần này, nên không coi đã kiểm đầy đủ bảng/thông số gốc. Các count protocol của dự án lấy từ A20, không suy từ dataset gốc.
- Các số Graph Transformer/AENet LCC và MADDG/AMEL/PatchNet/GAC-FAS/SA-FAS trong Word **chưa được tái audit từng hàng literature trong lần rà này**. Không sao chép nguyên bảng cũ sang bản mới rồi gọi là đã xác minh. Khi giữ, cần exact paper/table/page, source domains, split/unit, input và loại params.
- Không coi `O&M&I→C` là cùng điều kiện với `CelebA→CASIA reconstructed Combined`. Không dùng 11,7M backbone ResNet-18 làm total params của mọi phương pháp; các số ~3,35M hoặc ~5,5M cần artifact hoặc nguồn gốc cụ thể.
- Tránh “AUC trên số tham số” như metric chuẩn hoặc bằng chứng hiệu quả triển khai. Có thể mô tả riêng AUC và số tham số đã xác minh; không suy ra latency hoặc ưu thế chất lượng cùng protocol.

### 8.2. Những lỗi còn tồn tại trong bản Word hiện đọc

- Danh mục hình/bảng vẫn còn nội dung template **Oxford Flowers 102, ConvNeXt, Streamlit** trong text lưu của Word. Xóa danh mục cũ, dùng caption đúng và rebuild fields.
- Bảng phân công nhiệm vụ còn trống; điền thông tin thực tế hoặc bỏ cột chưa có dữ liệu.
- Mục **3.2 lặp**, **3.12** sai numbering; hierarchy 3.3.2/3.3.3 và 3.6.1/3.6.2 chưa đồng bộ.
- Mục **4.4.1 “LCC” lặp** ngay trước bảng CASIA; đổi tên/cấp mục đúng.
- Caption đang dùng Heading 4/5; chuyển về Caption, để heading cho cấu trúc nội dung.
- Bảng 1.3/1.4, 2.1 và số 4.4/4.5 bị lặp/không theo thứ tự; dùng đánh số tự động, cập nhật cross-reference sau khi tái cấu trúc.
- Hình 3.2 dùng hai lần; sửa “tranning/inferance” thành “huấn luyện/suy luận”.
- Mục 4.1.1 ghi horizontal flip `p=5`; sửa **p=0,5** sau khi đối chiếu code augmentation. Đừng giữ xác suất ngoài [0,1].
- Hình 4.8 vẫn là dòng placeholder mô tả giao diện. Thay ảnh demo có thật và caption rõ vai trò hoặc bỏ.
- Nhiều bảng có nội dung/công thức trùng khi extraction; chưa kết luận lỗi hiển thị trước khi render. Rà kỹ bảng tham số, công thức `r`, `j`, margin, argmin và CE-only AUC bị trống ở 4.2.4.
- Tiêu đề Chương 3 hiện quá nghiêng về “cơ sở vật lý/quang học”, không phản ánh phần chính là thuật toán huấn luyện. Đổi thành **“Phương pháp MAST-PAD”**; động lực vật lý chỉ giữ ngắn, có nguồn.

### 8.3. Bộ hình/bảng vừa đủ cho bản cuối

Ưu tiên một hình train/inference của MAST-PAD, một hình spectral views và gate, một bảng cấu hình bốn phương pháp, một bảng population/protocol, một bảng AUC full-scale và một biểu đồ nếu thêm giá trị. Giữ một sơ đồ pipeline đầy đủ và các hình giải thích cần thiết cho sáu giai đoạn; thêm hình camera nếu có ảnh thật. Dành hình thuật toán nổi bật nhất cho MAST-PAD, tránh nhiều sơ đồ pipeline lặp nhau. Diagnostics/literature tùy dung lượng và chất lượng bằng chứng.

Bỏ các biểu đồ 10K/100K và nhiều bar chart diễn lại cùng kết quả. Không ép đủ số hình hoặc đủ trang. Công thức chính cần rõ hơn, không cần tăng số lượng công thức.

## 9. Trình tự chỉnh sửa cụ thể

### Bước 1 — Khóa nội dung nghiên cứu cuối

- [ ] Chốt bốn cấu hình cuối; bỏ biến thể severity ngoài phạm vi và các mã phát triển nội bộ khỏi tên phương pháp trong bảng, hình và chính văn.
- [ ] Thay Bảng 3.1 bằng bảng ở Mục 3.3; sửa `L_harmful`, H=0, warmup và sigma.
- [ ] Sửa Bảng 4.1 và những câu EER sai; không che sự không đồng nhất giữa AUC và EER.
- [ ] Bỏ hoặc thay ma trận nhầm lẫn bằng số đếm ở ngưỡng đã nêu.
- [ ] Thêm thảo luận kết quả bất lợi theo P1-3: AUC làm headline; phân biệt EER với lỗi tại source threshold và đánh đổi APCER/BPCER sau calibration.
- [ ] Chuẩn hóa Combined/population/unit, coverage, Test-as-Val và CASIA copy.
- [ ] Ghi nguồn CE-only đúng mức: có report; chưa đối chiếu raw local.

### Bước 2 — Xóa nhật ký phát triển khỏi narrative

- [ ] Bỏ 4.2.1–4.2.3 hiện tại và mọi bảng/biểu đồ/kết quả 100K, 10K, micro khỏi báo cáo nộp, kể cả phụ lục kết quả.
- [ ] Viết lại 3.6 thành cấu hình crop/phổ cuối; bỏ câu kể “Vòng…” và các cấu hình trung gian.
- [ ] Không dùng micro crop gains làm đóng góp chính hoặc kết luận.
- [ ] Đưa CE-only vào setup từ đầu, bỏ cách kể đây là “Vòng 21 vừa hoàn thành”.
- [ ] Chỉ giữ diagnostics của run full-scale khi cần giải thích phương pháp; bỏ phụ lục run history và không đưa số 100K vào phần thảo luận.

### Bước 3 — Hoàn thiện chương pipeline theo `face_auth`

- [ ] Mô tả đủ sáu giai đoạn và sự phối hợp trong một chương pipeline; giữ MAST-PAD là đóng góp được phân tích sâu nhất. Rút chi tiết hạ tầng không phục vụ giải thích CV.

- [ ] IoU+LK thay Kalman; IoU 0,3 nếu mô tả cấu hình hiện tại.
- [ ] Window5/min3/ratio0,6/stale3 và cadence 0,2/0,2/1,0 đúng nguồn đọc.
- [ ] Mô tả raw score → temporal verdict → nhận diện qua worker; UI stabilization riêng.
- [ ] Ghi đúng model full-scale CASIA runtime và threshold frame; không gán source-frozen operating point cho camera.
- [ ] Bỏ flow ThingsBoard/hardware nếu được thêm khi chỉnh; không mở rộng nghiệp vụ ngoài CV/PAD.
- [ ] Bỏ claim FAR hoặc latency thiếu artifact; không thay bằng số ước lượng.

### Bước 4 — Viết lại abstract, thảo luận và kết luận

- [ ] Dùng một thông điệp thống nhất: phương pháp cuối, kết quả cuối, graph suy luận gọn.
- [ ] Nêu MAST-PAD gần WBST trên LCC, hơn trên CASIA AUC; không claim gate-only hoặc SOTA.
- [ ] Không kết luận độc lập hoàn toàn của LCC hay official CASIA benchmark.
- [ ] Chọn thứ tự và ví dụ làm nổi bật thành quả; giữ đủ đối chứng/target full-scale, không chọn best seed để đại diện cho kết quả nhiều seed.
- [ ] Gom phạm vi đánh giá và hướng hoàn thiện thành phần ngắn; không lặp hạn chế ở abstract, mọi caption và kết luận. Nhấn mạnh sản phẩm, kết quả và giá trị học tập đã đạt.
- [ ] Future work ưu tiên thí nghiệm kiểm soát và dữ liệu mới; không tự chạy thêm trong bước biên tập.

### Bước 5 — Kiểm bản nộp

- [ ] Sửa AENet và kiểm các citation còn giữ bằng nguồn gốc.
- [ ] Rà toàn văn, bảng, legend/caption và chữ nằm trong ảnh: chỉ dùng tên CE-only, Clean + Aux, WBST, MAST-PAD; mã nội bộ chỉ xuất hiện trong metadata truy vết nếu cần.
- [ ] Rebuild mục lục/danh mục, loại template; sửa heading/caption/numbering.
- [ ] Rà công thức trong Word/PDF, đặc biệt trường hợp trống và công thức trong bảng.
- [ ] Kiểm đúng phép đổi `%`/fraction và `pp`; 67,98 ms không thể đồng thời là 15,5 FPS.
- [ ] Xuất PDF và rà trực quan từng trang trước khi gọi là bản cuối.

## 10. Cách cập nhật báo cáo khi có thêm hai seed

### 10.1. Trong lúc các run đang chạy

Thông tin mới là **đang chạy thêm hai seed**, chưa phải đã có ba kết quả hoàn tất. Không điền trước mean/std, không đổi câu “một seed hiện có” thành “ba seed đã đánh giá” và không dự đoán rằng chênh lệch sẽ ổn định hơn.

Nếu cần nộp trước khi tất cả run hoàn thành, ghi số run hợp lệ thực có cho từng phương pháp; phần chưa hoàn tất ghi “đang chạy/chưa có kết quả” trong tài liệu theo dõi. Không đưa kế hoạch chạy seed thành một kết quả hoặc đóng góp của báo cáo cuối. Khi đủ dữ liệu, cập nhật trực tiếp bảng tổng hợp cuối; không thêm mục kể quá trình chạy seed thứ hai/thứ ba.

Chưa có thông tin xác nhận hai seed mới áp dụng cho những phương pháp nào. Vì vậy các hướng dẫn dưới đây phân biệt hai trường hợp; không mặc định tất cả CE-only/Clean/WBST/MAST-PAD đều được chạy thêm.

### 10.2. Kiểm tính tương thích trước khi gộp

Chỉ tổng hợp các run **full-scale** cùng cấu hình phương pháp và cùng protocol cuối. Hai seed bổ sung chỉ tham gia bảng kết quả cuối khi cũng chạy đúng setup full-scale; không gộp seed 100K hoặc micro với seed full-scale. Đối chiếu config, manifest hashes, preprocessing, augmentation, initialization policy, optimizer, lịch học, warmup, checkpoint selection, threshold calibration và population/unit chấm điểm. Giữ nguyên các artifact từng run.

- Ghi đúng **seed nào thay đổi**: model initialization, thứ tự dữ liệu/augmentation, gain RNG hoặc một training seed điều khiển nhiều stream. Nếu initialization cố định còn training RNG thay đổi, viết đúng điều đó; không gọi là ba khởi tạo mô hình độc lập.
- Không đổi split/manifest chỉ để tạo “seed mới”. Nếu dữ liệu/split thực sự thay đổi, đó là một trục biến thiên khác và phải báo cáo riêng, không trộn với độ biến thiên training seed trên một split cố định.
- Giữ chính sách chọn checkpoint giống nhau; best epoch giữa các seed được phép khác. Với mỗi run, checkpoint/ngưỡng phải được chọn từ tập nguồn theo protocol, không chọn theo AUC LCC/CASIA.
- **Ngưỡng nguồn dạng số có thể khác giữa các seed** nếu cùng chính sách calibration đã định. Không ép mọi run dùng số ngưỡng của seed cũ; cũng không trộn source-frozen rates với CASIA-calibrated rates trong một mean.
- Nếu số scored/coverage khác giữa run, kiểm nguyên nhân và nêu rõ. Ưu tiên đánh giá trên cùng population và cùng localization cache đã khóa; không coi một run có nhiều mẫu dễ hơn là bằng chứng cải thiện do seed.
- Run lỗi hoặc thiếu artifact không tự biến thành một quan sát đã hoàn tất. Ghi thiếu minh bạch, kiểm nguyên nhân và không bỏ seed chỉ vì AUC thấp. Nếu cần chạy lại, lưu lý do và bảo toàn dấu vết run trước.
- CE-only vẫn cần prediction/config/checkpoint tương ứng để nâng mức xác minh từ report sang raw artifact; thêm số seed không tự giải quyết thiếu provenance của seed hiện có.

### 10.3. Bảng kết quả khi cả bốn phương pháp đều đủ ba seed

Đổi bảng headline từ **một con số của một run** sang **mean ± sample standard deviation**, ghi `n=3` rõ trong caption. Giữ số từng seed ở một bảng phụ lục ngắn hoặc file kết quả đi kèm để người đọc truy vết.

| Phương pháp | Số run hợp lệ | LCC Combined AUC, mean ± SD | CASIA reconstructed video Combined AUC, mean ± SD |
|---|---:|---|---|
| CE-only | 3, khi đủ artifact | Chờ tổng hợp | Chờ tổng hợp |
| Clean + Aux | 3, khi hoàn tất | Chờ tổng hợp | Chờ tổng hợp |
| WBST + Aux | 3, khi hoàn tất | Chờ tổng hợp | Chờ tổng hợp |
| MAST-PAD + Aux | 3, khi hoàn tất | Chờ tổng hợp | Chờ tổng hợp |

Đây là **khung dự kiến**, không phải bảng kết quả đã đo. Không ghi `n=3` vào bảng Word cuối trước khi đủ run tương ứng.

Với metric `a_s` của từng seed:

\[
 \bar a=\frac1n\sum_{s=1}^{n}a_s,\qquad
 SD=\sqrt{\frac{\sum_{s=1}^{n}(a_s-\bar a)^2}{n-1}}.
\]

Dùng sample SD (`ddof=1`), không dùng standard error thay SD mà giữ nguyên nhãn. Tính từ metric đầy đủ độ chính xác, chỉ làm tròn ở bảng xuất. Nếu AUC ở thang 0–1, cả mean và SD phải ở thang 0–1; nếu đổi sang %, đổi cả hai. Độ chênh AUC báo cáo bằng điểm phần trăm.

Tính AUC/EER/APCER/BPCER **riêng cho từng run**, rồi mới lấy mean/SD của metric đó. Không gộp prediction của ba seed thành ba lần số mẫu độc lập; không lấy trung bình score ba mô hình rồi gọi kết quả là “mean AUC”. Trung bình score tạo một ensemble, là thiết lập khác cần ghi riêng. Riêng confusion matrix nên dùng một run đại diện được xác định minh bạch, hoặc báo cáo rates theo seed; không cộng ba ma trận rồi gọi là số ảnh duy nhất của dataset.

**Caption mẫu khi đã đủ kết quả:**

> ROC-AUC trung bình ± độ lệch chuẩn mẫu của ba run với các training seed [liệt kê seed thực tế], trên cùng population đánh giá đã khóa. LCC Combined được tính ở mức ảnh; CASIA reconstructed Combined ở mức video. Mỗi run chọn checkpoint và hiệu chuẩn ngưỡng theo cùng protocol nguồn. Độ lệch chuẩn phản ánh biến thiên giữa các training seed được khảo sát, không phải khoảng bất định do lấy mẫu subject/video.

### 10.4. Nếu chỉ MAST-PAD có thêm seed hoặc các phương pháp chưa đủ cùng số run

Không trình bày bảng bốn phương pháp như tất cả đều có `n=3`. Có thể giữ bảng headline hiện có ở Mục 7 với chú thích rõ đó là so sánh các run một seed, sau đó thêm **một bảng ngắn về độ biến thiên của MAST-PAD** trên ba seed. Cách này trả lời độ ổn định của MAST-PAD, chưa trả lời độ ổn định của lợi thế MAST-PAD so với mọi baseline.

Hoặc trình bày một bảng chung có cột `n` và thống kê đúng từng hàng: `n=1` chỉ ghi trị số, SD là `—`; `n=2/3` ghi mean ± SD tương ứng. Nêu rõ số run khác nhau và giới hạn khi so sánh. Không gán `SD=0` cho phương pháp chỉ có một run.

**Câu diễn giải mẫu:**

> MAST-PAD được đánh giá thêm trên tổng cộng ba training seed nhằm khảo sát biến thiên của phương pháp. Các đối chứng [tên phương pháp] hiện chỉ có một run, nên chưa thể kết luận lợi thế giữa các phương pháp ổn định qua seed từ bảng này.

Không lấy ba kết quả MAST-PAD trừ cùng một baseline một seed rồi gọi đó là ba so sánh ghép cặp độc lập. Các run một phía chưa phản ánh biến thiên của baseline.

### 10.5. So sánh chênh lệch khi các phương pháp có seed ghép cặp hợp lệ

Nếu MAST-PAD và baseline dùng cùng bộ seed và thiết kế hỗ trợ pairing, tính mỗi cặp:

\[
 \Delta_s=100\left(AUC_{MAST-PAD,s}-AUC_{baseline,s}
ight).
\]

Báo cáo `mean(Delta_s) ± SD(Delta_s)` theo điểm phần trăm, kèm các delta từng seed và số cặp dương/tổng số cặp. Cùng một nhãn seed là điều kiện truy vết, chưa tự bảo đảm hai run được kiểm soát tương ứng; kiểm chính sách initialization và các nguồn RNG. Nếu không có pairing hợp lệ, trình bày thống kê riêng từng nhóm và không đặt tên là phân tích paired.

Không thay SD của delta bằng hiệu SD của hai phương pháp. Không kết luận “không khác biệt” chỉ vì mean±SD overlap, cũng không kết luận “có ý nghĩa thống kê” chỉ vì error bar không overlap hoặc MAST-PAD thắng cả ba seed. Với ba seed, bằng chứng về độ biến thiên vẫn hạn chế; báo cáo mô tả thận trọng thường phù hợp hơn một kiểm định có lực thấp. Thêm seed không cô lập gate vì sigma WBST/MAST-PAD vẫn khác.

Nếu vẽ biểu đồ, error bar phải ghi rõ là SD giữa training seed. Không gọi đó là 95% CI hoặc CI theo ảnh/video. Nếu sau này tính uncertainty theo dữ liệu, cần tôn trọng cụm subject/video và phân biệt với uncertainty theo training seed.

### 10.6. Những phần phải viết lại trong Word sau khi đủ seed

| Vị trí | Cách cập nhật |
|---|---|
| Abstract và đóng góp | Thay hai AUC một seed bằng mean ± SD và số run thực tế. Chỉ nói “ổn định qua các seed đã khảo sát” nếu số liệu hỗ trợ; không mặc định dùng từ “vượt trội” |
| Thiết lập thực nghiệm | Liệt kê seed thực tế và các RNG thay đổi/cố định, số run mỗi phương pháp, chính sách chọn checkpoint/calibration cho từng run |
| Bảng 4.1 và biểu đồ chính | Cập nhật từ cùng một nguồn tổng hợp; giữ population/unit/protocol. Highlight theo mean nếu cần, không dùng best seed |
| Đoạn so sánh gain | Tính lại gain từ kết quả tổng hợp. Khi có paired design, thêm mean/SD delta và số seed cùng chiều; không giữ nguyên +1,43/+1,48/+1,29 pp như kết quả ba seed |
| Bảng EER/operating points | Tổng hợp riêng từng metric/threshold policy. Kiểm lại kết luận vì AUC và EER có thể không cùng chiều |
| Diagnostics | Ghi seed/run/epoch của hình. Nếu chỉ dùng seed cũ để minh họa, nói rõ; không gọi đó là hành vi trung bình của ba seed |
| Complexity và runtime demo | Graph params không cần nhân theo số seed. Ghi checkpoint cụ thể đang demo; không lấy AUC mean ba seed gán cho một checkpoint camera |
| Phạm vi đánh giá 5.2 | Ghi số seed thực tế; cập nhật một đoạn ngắn về phạm vi và hướng hoàn thiện. Không lặp lại danh sách hạn chế đã giải thích trong setup |
| Future work | Khi đã đủ seed, bỏ cách viết “chưa thực hiện multi-seed”; chuyển ưu tiên sang gate-only ablation, target độc lập và benchmark. Nếu muốn thêm seed nữa, nói rõ mục đích còn thiếu |
| Kết luận | Dùng mean và độ biến thiên, kết luận theo từng target; không kể lịch sử seed/run mới |

**Mẫu abstract/kết luận để điền sau khi có kết quả:**

> Trên [n] training seed với protocol cố định, MAST-PAD đạt ROC-AUC trung bình ± SD là [mean ± SD] trên LCC Combined mức ảnh và [mean ± SD] trên CASIA reconstructed Combined mức video. So với [baseline], chênh lệch [thống kê phù hợp] là [giá trị] điểm phần trăm. Kết quả [cùng chiều ở k/n seed hoặc mô tả thực tế]. Graph PAD suy luận vẫn có 434.434 tham số. Các kết quả cho thấy [nhận xét được số liệu hỗ trợ] trong thiết lập khảo sát và là cơ sở để tiếp tục hoàn thiện calibration, đánh giá hệ thống trên miền sử dụng.

Các ngoặc vuông là chỗ chờ dữ liệu, không được giữ trong bản nộp. Nếu một số baseline vẫn chỉ có một seed, dùng câu mẫu ở Mục 10.4 thay cho mô tả so sánh đồng đều nhiều seed.

### 10.7. Checklist trước khi thay bảng một seed

- [ ] Xác nhận phương pháp nào thực sự có thêm hai seed full-scale và số run hoàn tất của từng phương pháp; loại run 100K/micro khỏi bảng tổng hợp cuối.
- [ ] Kiểm protocol/config/manifest; xác định rõ các nguồn ngẫu nhiên thay đổi.
- [ ] Thu metric và provenance từng run, không chọn best seed theo target.
- [ ] Giữ per-seed values, dùng mean và sample SD từ full precision.
- [ ] Ghép cặp chỉ khi hợp lệ; thống kê delta riêng cho từng target.
- [ ] Đồng bộ abstract, bảng, chart, caption, diễn giải, hạn chế và kết luận.
- [ ] Giữ đúng worst-band/gate/loss: thêm seed không thay công thức phương pháp.
- [ ] Không gán AUC tổng hợp nhiều run cho một model camera hoặc ensemble chưa đánh giá.

## 11. Tiêu chí hoàn thành báo cáo học phần

Bản báo cáo học phần đạt mục tiêu khi làm nổi bật MAST-PAD, kiến thức đã vận dụng và kết quả full-scale đạt được. Người đọc cần trả lời được: vì sao đề xuất MAST-PAD, phương pháp chọn view và tính loss thế nào, đối chứng là gì, mô hình được đánh giá trên population nào, kết quả cuối tăng/giảm ra sao và giới hạn ở đâu. Người đọc không cần biết nhóm đã thử bao nhiêu vòng để hiểu các câu trả lời này.

**Không cần chạy lại training, đổi model, chỉnh threshold hoặc áp logic hardware để thực hiện các sửa đổi đã xác định.** Những việc cần thêm bằng chứng — gate-only ablation, hai seed bổ sung đang chạy, benchmark camera/edge, artifact CE-only đầy đủ — phải được ghi đúng trạng thái; sau khi các seed hoàn tất, cập nhật đồng bộ theo Mục 10. Với bản nộp hiện tại, ưu tiên sửa nội dung, bỏ claim chưa có chứng cứ và trình bày kết quả cuối một cách nhất quán.
