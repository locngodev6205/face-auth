# Adaptive threshold PAD trên miền dữ liệu do nhóm tự thu thập — source of truth

Ngày xác minh: **27/10/2026**. Phạm vi: cấu hình Face Auth thực sự được nạp từ `.env` tại thời điểm xác minh, công thức inference, tập ảnh đích của nhóm và replay cùng ảnh giữa ngưỡng cứng gốc với pipeline adaptive. Đây là hiệu chỉnh phục vụ vận hành sản phẩm, tách khỏi kết quả đánh giá nghiên cứu trên CelebA-Spoof/LCC-FASD/CASIA-FASD.

**Cấu hình đang chọn là `R7_SC_FULL_pi_final_conservative_runtime.json`, crop yêu cầu 1.7.** Tên `conservative` là tên profile khi được chọn trên subset có frame gốc; không có nghĩa 0 fake lọt trên toàn bộ tập. Replay toàn bộ tập hợp lệ cho **10 fake lọt/491 fake và 29 real bị từ chối/645 real**. Không thay bằng số liệu của profile `pi_all1136_balanced` trong example.

## 1. Cấu hình được chốt và thứ tự ưu tiên

Các dòng hoạt động trong `face_auth/.env`:

```dotenv
PAD_ENABLED=true
PAD_ADAPTIVE_ENABLED=true
PAD_ADAPTIVE_CONFIG_PATH=antispoof/models/R7_SC_FULL_pi_final_conservative_runtime.json
PAD_RUNTIME_CONFIG_PATH=antispoof/models/R7_SC_FULL_crop15_runtime_config_casia.json
PAD_THRESHOLD=
PAD_THRESHOLD_LOGIT=
PAD_BBOX_EXPANSION_FACTOR=
FORCE_GAMMA=false
PAD_CROP_SMOOTHING=false
```

Khi `PAD_ADAPTIVE_ENABLED=true`, [config.py](../../config.py) dùng `PAD_ADAPTIVE_CONFIG_PATH` làm runtime profile và bỏ qua override threshold/crop/gamma thủ công. Vì vậy dòng `PAD_RUNTIME_CONFIG_PATH=..._casia.json` không phải profile đang điều khiển PAD ở snapshot này. Profile adaptive phải có object `adaptive_threshold`; lỗi nạp không được âm thầm chuyển sang cấu hình khác.

Baseline trong tài liệu này **đúng theo yêu cầu** là [R7_SC_FULL_crop15_runtime_config.json](../../antispoof/models/R7_SC_FULL_crop15_runtime_config.json), không phải file `_casia.json`, không phải các profile local80/local98/local100/local842 và không phải ngưỡng 0 hay xác suất 0.5.

| Thành phần | Ngưỡng cứng gốc | Pipeline adaptive đang hoạt động |
|---|---|---|
| Neural model | R7_SC_FULL_crop15.onnx | Cùng file ONNX, cùng trọng số |
| Đầu vào model | BGR, 80×80, float [0,1], mean/std null | Giữ nguyên hợp đồng tensor |
| Crop | MiniFASNet train-v1, yêu cầu 1.5 | Cùng cơ chế crop, yêu cầu 1.7 |
| Lighting bổ sung | Không | `wb_exposure_gamma` |
| Ngưỡng logit | Cố định `4.66259765625` | Base `-1.5668816359970292` + offset theo từng crop |
| Gamma flag cũ `apply_gamma` | false | false; gamma của lighting profile vẫn được áp dụng |
| Model forwards | 1/crop | 1/crop; không multi-crop/ensemble |

Đổi crop và lighting có thể đổi cả logit do model sinh ra. Vì thế so sánh chính là **pipeline gốc với pipeline vận hành adaptive**, không chỉ đổi một con số threshold trên cùng logit. Mục 8 bổ sung đối chứng dùng cùng logit của pipeline mới để tách ảnh hưởng offset.

## 2. Thuật toán quyết định theo từng crop

### 2.1. Điểm PAD

Với hai logits `z_real`, `z_fake`:

```text
d = z_real - z_fake
REAL  nếu d >= t(x)
SPOOF nếu d <  t(x)
```

Dấu bằng thuộc REAL theo code. So sánh trực tiếp hai logits chỉ tương đương trường hợp `t=0`. `sigmoid(d)` bằng xác suất softmax lớp real cho head hai lớp, nhưng không mặc nhiên là xác suất đã được hiệu chỉnh trên miền camera đích.

### 2.2. Bốn thống kê RAW trước xử lý sáng

Từ crop BGR uint8 80×80, dùng `cv2.COLOR_BGR2YCrCb`:

- `F`: trung bình Y toàn crop.
- `C`: trung bình Y trong vùng `[16:64,16:64]`, tức phần giữa 48×48, bằng 60% mỗi chiều.
- `Cb`, `Cr`: trung bình hai kênh màu tương ứng trong cùng vùng giữa.

**Thứ tự kênh OpenCV là Y, Cr, Cb**, nên Cb lấy channel 2, Cr lấy channel 1. Các thống kê đo trên ảnh RAW trước white balance/gamma, không đo trên ảnh đã sửa sáng. Vùng giữa là hình chữ nhật cố định, không phải mask da hoặc vùng mặt được tách riêng.

### 2.3. Công thức adaptive threshold hiện tại

```text
u(x) = 4.0 × (C - 110)/50
     + 10.5 × (C - F)/40
     + 10.5 × (Cb - 128)/20
     + 6.0 × (Cr - 128)/20
     - 4.0

offset(x) = clip(u(x), -8.0, +8.0)
t(x)      = -1.5668816359970292 + offset(x)
m(x)      = d(x) - t(x)
REAL iff m(x) >= 0
```

Ý nghĩa các thành phần:

| Thành phần | Tác dụng khi các biến còn lại giữ nguyên |
|---|---|
| `(C-110)/50`, hệ số 4.0 | Trung tâm sáng hơn làm tăng ngưỡng, khó đạt REAL hơn. |
| `(C-F)/40`, hệ số 10.5 | Mặt/vùng giữa sáng hơn nền toàn crop làm tăng ngưỡng; chênh sáng có vai trò riêng với độ sáng tuyệt đối. |
| `(Cb-128)/20`, hệ số 10.5 | Điều chỉnh theo thành phần sắc xanh–vàng của dữ liệu mục tiêu. |
| `(Cr-128)/20`, hệ số 6.0 | Điều chỉnh theo thành phần sắc đỏ–lục của dữ liệu mục tiêu. |
| Bias -4.0 | Dịch toàn bộ offset; ở `C=F=110`, `Cb=Cr=128`, offset là -4, không phải 0. |
| Clip ±8.0 | Chặn mức điều chỉnh; không để ngưỡng tăng/giảm vô hạn. |

Ngưỡng nằm trong **[-9.56688163599703; 6.43311836400297]**. `predictor_threshold_probability=0.17266139367426664` trong JSON là sigmoid của **base**, không phải một ngưỡng xác suất cố định áp dụng cho mọi ảnh khi adaptive bật.

Ví dụ chỉ để minh họa công thức, không phải ảnh benchmark:

- `F=100, C=110, Cb=Cr=128`: offset = -1.375, threshold ≈ -2.941882. Điểm `d=-2` được nhận REAL.
- `F=C=180, Cb=Cr=128`: offset = +1.6, threshold ≈ +0.033118. Cùng `d=-2` bị từ chối.

Các hệ số màu được hiệu chỉnh từ tập đích. Không diễn giải chúng thành quy tắc phổ quát “ảnh ám xanh là fake” hoặc “một màu da cụ thể là real”. Ánh sáng/cân bằng trắng của camera mới có thể làm lệch quy tắc này. Cũng không suy ra rằng mọi màn hình đều sáng hơn người thật.

### 2.4. White balance và gamma trước model

Sau khi đo ngưỡng từ RAW, [LightingTransform](../../antispoof/lighting.py) xử lý ảnh để đưa vào model:

1. Gray-world white balance trên trung bình ba kênh BGR toàn crop: `gain_k = clip(mean_all_channels / max(mean_k,1), 0.7,1.3)`; `blend=1.0` áp dụng toàn bộ gain giới hạn.
2. Sau white balance, đo Y vùng giữa là `C_wb` và tính:

   ```text
   gamma = clip(log(80/255) / log(clip(C_wb,1,254)/255), 0.7,1.5)
   ```

3. Gamma làm tròn hai chữ số thập phân, dùng LUT cache: `I_out = uint8(clip((I_wb/255)^gamma × 255,0,255))`.
4. Preprocess tensor BGR [0,1] và chạy đúng một lần ONNX; quyết định dùng threshold đã đo ở bước RAW.

Đây là gamma có giới hạn, không phải ép chính xác Y đầu ra bằng 80. Lighting profile hiện tại chạy cho mọi crop, không phải chỉ bật khi ảnh tối. `apply_gamma=false`/`FORCE_GAMMA=false` tắt cơ chế gamma cũ; không tắt gamma bên trong `wb_exposure_gamma`.

Các phép bổ sung là trung bình trên crop 80×80, arithmetic, clip và LUT; không có neural model phụ, CLAHE, blur hoặc ba lượt PAD trong profile này. Chưa có số đo overhead hoặc FPS trên Pi để gán một mức tăng/giảm cụ thể.

## 3. Crop, biên khung hình và temporal voting

[preprocess.py](../../antispoof/preprocess.py) giới hạn scale:

```text
used_scale = min(requested_scale, (frame_h-1)/bbox_h, (frame_w-1)/bbox_w)
```

Nếu ROI mở rộng vượt mép, ROI được dịch vào trong frame; khuôn mặt có thể lệch tâm. Không padding hoặc tạo thêm ngữ cảnh. Crop được resize về 80×80. Với ảnh cũ không có frame gốc, replay chỉ dùng tối đa phần crop đã lưu và hệ số 1.5, không crop rộng 1.7 giả hoặc nhân mở rộng hai lần.

Vùng trung tâm cố định để đo C/Cb/Cr có thể chứa nền khi ROI bị dịch. Có dữ liệu crop giới hạn/dịch trong replay, nhưng chưa có protocol riêng bao phủ có kiểm soát mọi góc/khoảng cách.

Gate hình học giữ nguyên: detector confidence tối thiểu 0.5, bbox/landmarks hữu hạn, đủ 5 landmarks nằm bên trong mặt và bbox không sát mép frame theo margin. Ảnh không đạt gate không được gán là SPOOF để tính các chỉ số model; ghi riêng là không đủ điều kiện.

Quyết định một crop phía trên được đưa vào tracker: tối thiểu 3 vote, cửa sổ 5, SPOOF khi tỷ lệ vote SPOOF trong cửa sổ hiện tại >=0.6; PAD interval 0.2 giây, stale timeout 3 giây, crop smoothing false. Đây là tầng ổn định temporal riêng. Các chỉ số trong tài liệu là **từng crop**, không phải tỷ lệ thành công của toàn lượt điểm danh, video hoặc mỗi người.

## 4. Bộ dữ liệu đích do nhóm tự thu thập

Đây là tập local `face_auth/pad_captures`, thu qua camera trong quá trình thử nghiệm hệ thống, bao gồm batch người dùng xác nhận thu trực tiếp trên Pi. Có ảnh người thật và các mẫu tấn công bằng ảnh/màn hình nhóm chuẩn bị. Metadata chỉ có nhãn binary real/fake, không đủ phân loại chắc chắn từng mẫu thành print/screen hoặc thống kê từng thiết bị tấn công.

[app_capture.py](../../app_capture.py) và [pad_capture.py](../../pad_capture.py) lưu crop native, ảnh đầu vào model và — với schema mới — frame gốc chưa vẽ overlay cùng bbox/landmarks/cấu hình. Nhãn do nhóm gán trong `labels.json`; thông tin kỹ thuật ở `metadata.json`. Kết quả model tại lúc capture chỉ là diagnostic, không dùng làm ground truth.

Snapshot đã kiểm tra lại có:

| Phạm vi | Tổng | Real | Fake |
|---|---:|---:|---:|
| Toàn bộ ảnh có nhãn | 1.139 | 646 | 493 |
| Không qua gate hình học, lưu riêng | 3 | 1 | 2 |
| **Đủ điều kiện replay** | **1.136** | **645** | **491** |
| Có frame gốc | 682 | 426 | 256 |
| Chỉ còn crop đã lưu | 454 | 219 | 235 |
| Batch Pi mới hợp lệ | 294 | 40 | 254 |
| Dữ liệu cũ hợp lệ | 842 | 605 | 237 |

- 29 session trên tập hợp lệ; 1.136 SHA-256 crop native phân biệt. Khác file/hash không đồng nghĩa quan sát độc lập: nhiều ảnh cùng người/cảnh/phiên có tương quan.
- Batch Pi mới gồm 5 session: 1 phiên real, 4 phiên fake. Vì thế 254 ảnh fake Pi không phải 254 kiểu tấn công độc lập.
- Timestamp metadata trải từ 21/10/2026 06:38:20 UTC đến 22/10/2026 03:18:06 UTC (13:38:20 ngày 21 đến 10:18:06 ngày 22 theo Việt Nam). Đây là thời gian ghi trong metadata, không kiểm chứng đồng hồ thiết bị độc lập.
- Metadata frame của toàn bộ ảnh hợp lệ ghi 640×480 BGR; native crop có kích thước khác nhau. Chỉ 682 mẫu thực sự còn file frame gốc để dựng lại crop rộng.
- `subject_id` chỉ có 4 giá trị ghi chú, có giá trị `unspecified`, và được dùng lẫn cho phiên/chế độ camera. Không thể từ đó kết luận có đúng 4 người độc lập hoặc đã chia subject-disjoint.
- Không có train/validation/test độc lập cho phiên hiệu chỉnh này. Toàn bộ tập đích được phép dùng calibrate và chọn profile; replay ngày 27/10 chỉ xác nhận lại, không tối ưu thêm.
- Ba ảnh bị loại gồm hai `face_too_close_to_frame_edge` và một `landmark_at_face_edge`. Coverage hợp lệ 1.136/1.139 ≈99.74%; đây là coverage của gate, không phải accuracy PAD.

### Phân bố ánh sáng theo metadata

Giữ nguyên tên condition để không tự sửa dữ liệu gốc. `side_ligh_left` và `side_light_left` là hai cách ghi; `blacklight_lamp` là nhãn nguyên bản, không tự suy ra nguồn UV hay đổi thành backlight. Các condition không bảo đảm có cả real và fake.

| Condition | Real | Fake | Tổng |
|---|---:|---:|---:|
| `blacklight_lamp` | 41 | 95 | 136 |
| `dim_room` | 40 | 0 | 40 |
| `mixed` | 44 | 36 | 80 |
| `normal` | 0 | 123 | 123 |
| `room_normal` | 174 | 173 | 347 |
| `side_ligh_left` | 42 | 0 | 42 |
| `side_light_left` | 95 | 0 | 95 |
| `side_light_right` | 74 | 0 | 74 |
| `side_top_light` | 34 | 0 | 34 |
| `top_light` | 43 | 0 | 43 |
| `unspecified` | 18 | 2 | 20 |
| `very_dark` | 40 | 62 | 102 |

Các nhóm bao phủ sáng phòng thông thường, phòng tối/rất tối, đèn bên trái/phải, đèn phía trên, phối hợp và điều kiện chưa xác định theo ghi chú người thu. Không có phép đo lux hoặc calibration màu để coi chúng là các mức sáng định lượng chuẩn hóa.

## 5. Protocol so sánh

1. Giữ nguyên labels/metadata, thứ tự ảnh, gate hình học và ONNX model hash. Không thêm ảnh augmentation vào bảng kết quả chính.
2. Baseline: nạp đúng JSON gốc `.json`, tắt adaptive, xóa override threshold/crop, dùng crop yêu cầu 1.5, không lighting bổ sung và ngưỡng 4.66259765625.
3. Adaptive: nạp profile đang hoạt động, crop yêu cầu 1.7, WB/gamma và threshold `raw_color` như trên. Không hạ/tăng threshold để làm đẹp số liệu.
4. Cả hai chạy `AntiSpoofPredictor.predict_crops` thực trên CPU, cùng 1.136 ảnh và cùng thứ tự. Có frame thì dựng lại production crop; thiếu frame thì dùng vùng lưu trữ thực tế tối đa cho phép.
5. Đo per-image confusion, phân nhóm cùng ảnh và ROC-AUC. Không áp temporal votes vào một tập crop tĩnh không có protocol video liên tục.
6. Kiểm tra snapshot nhãn/metadata và SHA-256 frame trước/sau; `.env` không thay đổi. Đây là offline replay, không mở camera, không gửi điểm danh hoặc MQTT.

Quy ước **real là lớp dương**: TP = real được nhận; FN/FR = real bị từ chối; FP/FA = fake bị nhận real; TN = fake bị chặn. Không đảo tên false accept với false reject.

```text
APCER = FA / N_fake                 BPCER = FR / N_real
ACER  = (APCER + BPCER) / 2
Accuracy = (TP + TN) / N
Balanced accuracy = (Recall_real + Specificity_fake) / 2 = 1 - ACER
Precision_real = TP / (TP + FA)     Recall_real = TP / N_real
Specificity_fake = TN / N_fake
F1_real = 2TP / (2TP + FA + FR)
```

APCER ở đây gộp mọi fake, chưa có nhãn PAI để tính worst-case APCER theo từng loại vật mang tấn công. Precision/accuracy còn phụ thuộc tỷ lệ real/fake của tập. Không suy diễn tỷ lệ này thành tần suất vận hành ngoài thực tế.

## 6. Kết quả chính trên toàn bộ 1.136 ảnh hợp lệ

| Chỉ số | Ngưỡng cứng gốc | Adaptive đang dùng |
|---|---:|---:|
| Real được nhận (TP) | 99 | 616 |
| Real bị từ chối (FR/FN) | 546 | 29 |
| Fake lọt (FA/FP) | 7 | 10 |
| Fake bị chặn (TN) | 484 | 481 |
| APCER ↓ | 1.43% | 2.04% |
| BPCER ↓ | 84.65% | 4.50% |
| ACER ↓ | 43.04% | 3.27% |
| Accuracy ↑ | 51.32% | 96.57% |
| Balanced accuracy ↑ | 56.96% | 96.73% |
| Precision real ↑ | 93.40% | 98.40% |
| Recall real ↑ | 15.35% | 95.50% |
| Specificity fake ↑ | 98.57% | 97.96% |
| F1 real ↑ | 26.36% | 96.93% |
| ROC-AUC của raw logit `d` ↑ | 0.915777 | 0.784089 |
| ROC-AUC của decision margin `d-t(x)` ↑ | 0.915777 | 0.986694 |

ACER giảm từ 43.04% xuống 3.27%, tức **39.77 điểm phần trăm**; BPCER giảm **80.16 điểm phần trăm**. Đổi lại APCER tăng **0.61 điểm phần trăm**, tương ứng fake lọt tăng ròng từ 7 lên 10. Không mô tả kết quả này là đồng thời giảm mọi loại lỗi.

So từng ảnh: adaptive cứu được **518 real** mà baseline từ chối, nhưng từ chối mới **1 real**; chặn được **7 fake** baseline từng cho qua, đồng thời cho qua mới **10 fake**. Hai profile không nhất thiết sai trên cùng các mẫu.

### Đọc ROC-AUC đúng cách

- Với threshold hằng số, trừ threshold không đổi thứ hạng, nên AUC của `d-t` bằng AUC của `d`.
- Adaptive offset phụ thuộc ảnh nên `m=d-t(x)` có thể đổi thứ hạng. AUC margin 0.986694 là của **hệ thống model + thống kê ảnh + profile đã calibrate trên tập đích**, không phải AUC riêng của neural model.
- Raw-logit AUC của pipeline mới là 0.784089, thấp hơn raw-logit AUC baseline 0.915777. Do crop/lighting khác nhau, model score ranking cũng đổi. Lợi ích vận hành ở đây không chứng minh bản thân neural model đã cải thiện.
- AUC đo khả năng xếp hạng qua nhiều operating point, không phải “accuracy tối ưu” hay bảo đảm ngưỡng triển khai tốt. Ngưỡng gốc có AUC raw khá cao nhưng BPCER rất lớn ở operating point 4.6626 trên miền camera này.
- Không thay các số AUC local này vào bảng CelebA/LCC/CASIA hoặc kết quả paper/source-frozen.

## 7. So sánh trên cùng các nhóm dữ liệu

FA/FR ở mỗi dòng dùng cùng mẫu số giữa hai phương án.

| Nhóm | N real | N fake | Fixed FA/FR | Adaptive FA/FR | Fixed ACER | Adaptive ACER |
|---|---:|---:|---:|---:|---:|---:|
| Có frame gốc | 426 | 256 | 1/345 | 0/28 | 40.69% | 3.29% |
| Chỉ crop đã lưu | 219 | 235 | 6/201 | 10/1 | 47.17% | 2.36% |
| Pi mới | 40 | 254 | 0/40 | 0/0 | 50.00% | 0.00% |
| Tập cũ | 605 | 237 | 7/506 | 10/29 | 43.29% | 4.51% |

**0 FA/28 FR** trong metadata profile ban đầu chỉ áp dụng cho 682 ảnh có frame gốc. Khi dùng đầy đủ tập, 454 ảnh thiếu frame đóng góp thêm **10 FA/1 FR**; không được báo 0 FA cho toàn tập. Pi mới 0/0 cũng không chứng minh mọi camera/người/kiểu tấn công đều được xử lý đúng.

### Phân rã theo condition

`FA/FR` = fake lọt/real bị từ chối; nhóm không có real hoặc fake vẫn ghi số đếm, không suy diễn thành ACER hợp lệ cho hai lớp.

| Condition | N real/fake | Fixed FA/FR | Adaptive FA/FR |
|---|---:|---:|---:|
| `blacklight_lamp` | 41/95 | 0/41 | 3/0 |
| `dim_room` | 40/0 | 0/40 | 0/0 |
| `mixed` | 44/36 | 6/27 | 6/1 |
| `normal` | 0/123 | 0/0 | 1/0 |
| `room_normal` | 174/173 | 0/173 | 0/0 |
| `side_ligh_left` | 42/0 | 0/19 | 0/0 |
| `side_light_left` | 95/0 | 0/67 | 0/0 |
| `side_light_right` | 74/0 | 0/73 | 0/2 |
| `side_top_light` | 34/0 | 0/22 | 0/10 |
| `top_light` | 43/0 | 0/35 | 0/16 |
| `unspecified` | 18/2 | 1/9 | 0/0 |
| `very_dark` | 40/62 | 0/40 | 0/0 |

Lỗi real còn lại tập trung ở `top_light` (16) và `side_top_light` (10), thêm `side_light_right` (2), `mixed` (1). Fake lọt tập trung `mixed` (6), `blacklight_lamp` (3), `normal` (1). Điều kiện sáng chưa cân bằng giữa hai lớp; có thể tồn tại confounding theo session/đối tượng/thiết bị. Chưa kết luận hiệu quả theo mỗi loại đèn từ số mẫu đơn lẻ.

## 8. Đối chứng cùng preprocessing để hiểu vai trò adaptive offset

Giữ đúng crop 1.7/WB/gamma và cùng raw logits của pipeline mới, chỉ thay cách quyết định sau inference. Không fit thêm threshold nào trong lần lập tài liệu này.

| Quyết định trên cùng logits mới | FA | FR | APCER | BPCER | ACER |
|---|---:|---:|---:|---:|---:|
| Ngưỡng cố định nguồn 4.66259765625 | 3 | 528 | 0.61% | 81.86% | 41.24% |
| Chỉ base cố định -1.566881636, bỏ offset | 126 | 231 | 25.66% | 35.81% | 30.74% |
| Base + adaptive offset đang dùng | 10 | 29 | 2.04% | 4.50% | 3.27% |

Đối chứng cho thấy chỉ hạ threshold về base hoặc chỉ thêm lighting với threshold nguồn không tái tạo được cân bằng lỗi của profile hiện tại. Đây không phải phép so với **ngưỡng cứng tốt nhất được tối ưu lại trên miền đích**; chưa có bằng chứng adaptive luôn vượt mọi cách calibrate ngưỡng hằng số. Chỉ số vẫn là in-sample trên dữ liệu đã dùng chọn hệ số.

Trên 1.136 ảnh, threshold thực tế có min=-9.566882, Q1=-9.566882, median=-2.603147, Q3=5.703434, max=6.433118. Có 299 ảnh chạm mức clip dưới và 245 ảnh chạm mức clip trên. Offset không phải chỉ là một dao động nhỏ quanh base; đây là hiệu chỉnh có tác động đáng kể lên operating point.

## 9. Kết luận áp dụng và giới hạn

- Giữ profile đang được chọn trong `.env` làm mô tả vận hành tại snapshot này. Adaptive giúp giảm từ chối người thật rõ rệt trên tập local, đổi lại số fake lọt tăng nhẹ so với threshold source-frozen chỉ định.
- Hệ thống vẫn có 10 fake lọt trên toàn tập. Tên conservative hoặc kết quả Pi mới 0/0 không phải cam kết an toàn tổng quát.
- Không có holdout; ảnh/session có tương quan, ánh sáng hai lớp không cân bằng và subject/device metadata chưa chuẩn hóa. Không báo confidence interval kiểu coi mọi ảnh là mẫu độc lập, không coi đây là kết quả cross-domain nghiên cứu.
- Crop rộng có thể bị cap/dịch; ảnh cũ thiếu full frame phải giới hạn ở vùng đã lưu. Không giả lập mở rộng để làm đầy mẫu số.
- Replay không đánh giá temporal attack success, chống replay video, FPS camera, SPI TFT, MQTT, recognition hoặc điểm danh đầu-cuối. Không suy ra 95.50% recall ảnh thành tỷ lệ điểm danh thực tế.
- Những stress test ánh sáng/bbox ở phiên hiệu chỉnh trước còn có fake lọt. Tài liệu này không mở rộng grid search, không chọn lại profile và không thay đổi `.env`/model/code pipeline.

## 10. Nguồn, hash và tái lập

### Nguồn triển khai

- [Runtime gốc source-frozen](../../antispoof/models/R7_SC_FULL_crop15_runtime_config.json).
- [Profile adaptive đang chọn](../../antispoof/models/R7_SC_FULL_pi_final_conservative_runtime.json).
- [Config và thứ tự ưu tiên](../../config.py).
- [Thống kê RAW, lighting và threshold adapter](../../antispoof/lighting.py).
- [Predictor: đo threshold trước lighting, sau đó inference](../../antispoof/predictor.py).
- [Crop và giới hạn biên](../../antispoof/preprocess.py).
- [Gate hình học và cập nhật vote](../../vision_pipeline.py), [tracker](../../tracking/tracker.py).
- [Capture metadata/nhãn](../../pad_capture.py), [capture entrypoint](../../app_capture.py).

### Snapshot đã xác minh

- Git snapshot tại lúc lập tài liệu: `edf6ff31dabc04790e42445b1a5f9342174809fa`; không thay đổi code inference trong lần này.

- Model SHA-256: `7b3008a040e155f7b96fca74f52dd38cc6296d88fee7a9424e653203c367a253`.
- Baseline JSON SHA-256: `192c0fcb399985ba67bf7b727489183f10174e828b13c54d3bd571886fb0ebaa`.
- Adaptive JSON SHA-256: `393d80c0355b320f6e6f2ab5dc35ee316cd503bc64695755d1e135f8dcdda0d1`.
- `.env` SHA-256: `df538c87273def7d9fd8abd53f7992ca2c0ab61f5fda5fefecec71f14f1012d1`; chỉ ghi hash và các key PAD cần thiết, không sao chép token/credential.
- Evaluator SHA-256: `65b9e2a3014e35005317fa3622ca17ff595b71c6a930372be15839a9a283dc0b`.
- Môi trường replay: Python 3.11.7, OpenCV 4.13.0, ONNX Runtime 1.24.4, CPUExecutionProvider.

### Artifact local của lần lập tài liệu

- [summary.json](../../output/pad_domain/2026-10-27_adaptive_source_of_truth/summary.json): metrics, paired changes, phân nhóm condition và threshold quantiles.
- [dataset.json](../../output/pad_domain/2026-10-27_adaptive_source_of_truth/dataset.json): session counts, nhãn, coverage và exclusions.
- [fixed_predictions.csv](../../output/pad_domain/2026-10-27_adaptive_source_of_truth/fixed_predictions.csv), [adaptive_predictions.csv](../../output/pad_domain/2026-10-27_adaptive_source_of_truth/adaptive_predictions.csv): quyết định theo từng ảnh.
- [Fixed provenance](../../output/pad_domain/2026-10-27_adaptive_source_of_truth/fixed_provenance.json), [adaptive provenance](../../output/pad_domain/2026-10-27_adaptive_source_of_truth/adaptive_provenance.json): profile/model/source hashes, snapshot nhãn/metadata, hash các frame sử dụng, effective config và phiên bản môi trường.
- [Evaluator](../../output/pad_domain/2026-10-27_adaptive_source_of_truth/evaluate.py), [metrics script](../../output/pad_domain/2026-10-27_adaptive_source_of_truth/summarize.py): replay và tổng hợp; script chỉ suy luận trên ảnh đã lưu, không fitting.

Các artifact và tập ảnh local nằm trong đường dẫn gitignored theo quy ước dự án, không mặc định có trong fresh clone. Bảng/công thức/hash trong tài liệu này tự chứa thông tin để review; muốn tái chạy phải có bundle dữ liệu và các helper calibration local tương ứng. Không coi link local là dữ liệu đã được public.

Lệnh đã chạy từ repo root:

```bash
python3 face_auth/output/pad_domain/2026-10-27_adaptive_source_of_truth/evaluate.py
python3 face_auth/output/pad_domain/2026-10-27_adaptive_source_of_truth/summarize.py
```

Evaluator chạy hai subprocess riêng để tránh cache module `config`: baseline tắt adaptive/nạp đúng JSON gốc và xóa override threshold/crop, adaptive dùng đúng profile hiện tại; sau đó xác nhận hash dữ liệu và `.env` không đổi. Các số liệu được tổng hợp từ CSV, không chép lại từ comment `.env` hoặc metadata calibration của subset cũ.
