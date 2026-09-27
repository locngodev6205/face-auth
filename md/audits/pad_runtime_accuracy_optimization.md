# PAD runtime: phân tích input và các hướng tối ưu

Ngày kiểm tra: 2026-09-27. Phạm vi: working tree hiện tại, spatial E1 đang active; không chạy camera, training hoặc chỉnh calibration/runtime chính.

## Kết luận

Tăng spoof vote ratio không sửa được trường hợp phần lớn raw scores đã nghiêng về SPOOF. Runtime có thể cải thiện độ ổn định của đầu vào, tránh kết luận trên input thiếu chất lượng và giảm reset không cần thiết. Chưa có bằng chứng rằng một thay đổi runtime cụ thể sẽ khắc phục toàn bộ lỗi ở xa; cần kiểm chứng cả bona-fide và print/replay.

Chưa xác nhận lỗi RGB, normalization, công thức score hay ONNX batching trên các input đã thử. Ưu tiên kiểm tra **crop/bbox và chất lượng ảnh thực tế**, sau đó **continuity của track**; chưa nên tiếp tục đổi ratio, gamma hoặc threshold để tìm kết quả xanh hơn.

## Đã kiểm tra

- `app.py`: capture, cadence detector/PAD, batching, diagnostics, raw frame/display frame, recognition orchestration.
- `detection/detector.py`, implementation SCRFD đang cài đặt: resize detector và mapping bbox về ảnh gốc.
- `tracking/tracker.py`: LK propagation, bbox/landmarks, redetection, reset, smoothing và stale guard.
- `antispoof/preprocess.py`, `antispoof/predictor.py`, model loader/config và active JSON/.env.
- Training path E1/E2, đối chiếu hai commit cũ trong `runtime_old_commits_comparison.md`, log camera và phân tích size trong `output/pad_distance_analysis.json`.
- Offline sensitivity trên ba ảnh gallery, chín biến thể mỗi ảnh; kết quả trong `output/pad_input_sensitivity.json`.

## Pipeline thực tế

| Bước | Hiện tại | Điểm cần chú ý |
|---|---|---|
| Capture | .env yêu cầu 640×480, 30 FPS, buffer 1; force resolution hiện là true | Đây là yêu cầu backend, phải xem actual mode. Exposure/focus/white balance và tuổi frame chưa được đo |
| Detection | SCRFD input 640×640; resize giữ tỷ lệ, padding; bbox map về frame gốc; confidence 0.5, min side 60 px | Bbox integer; detector size khác camera size. Tăng camera resolution không tự tăng detector input nhưng có thể giữ thêm chi tiết trong crop PAD |
| Tracking | LK trên bốn góc bbox và tâm; status/finite checks; scale/geometry guards | Góc bbox có thể nằm trên background; code bỏ qua LK error, chưa có forward-backward check. Status hợp lệ chưa chứng minh bbox bám đúng mặt |
| PAD scheduling | Detector và PAD interval đều 0.1 s, nhưng lịch riêng; PAD đo từ lần inference hoàn tất | PAD có thể nhận bbox DETECTOR hoặc TRACKER. Không bảo đảm mỗi vote dùng một bbox detector mới |
| Crop | Square side = max(width,height) × 1.55; lấy từ raw frame; reflect padding nếu vượt biên | Dịch tâm/thay scale thay đổi phần mặt/background/padding. PAD không dùng ArcFace landmark alignment |
| Tensor | RGB, resize 224 giữ tỷ lệ; AREA khi giảm, LANCZOS4 khi tăng; reflect letterbox; /255; mean/std; float32 CHW | Contract đang khớp recipe E1 đã kiểm tra. Resize không khôi phục chi tiết texture mất khi đứng xa |
| Inference | Active MobileNetV3-Small v5.3 ONNX; batch các track đến hạn | Trên máy này chạy CPU. Chưa có phép so kết quả giữa edge/provider khác |
| Score | d = real_logit − logsumexp(spoof_logits); p_real = sigmoid(d); REAL khi d ≥ −0.4650222063 | Tương đương ngưỡng softmax 0.3857950953. p_real là model score, không phải xác suất đúng đã được kiểm chứng trên camera |
| Temporal | Window 5, min votes 5, spoof ratio .6 trong .env; stale 3 s | Binary votes bỏ độ lớn score. Ratio .8 chỉ đổi từ 3/5 sang 4/5 SPOOF; 5/5 SPOOF vẫn giữ kết luận |
| Continuity | Flow mất thật hoặc detector unmatched reset PAD/identity; unsafe geometry được redetect cùng frame | Một LK failure vẫn có thể làm mất window dù detector ngay sau đó match. Đây là vấn đề ổn định/pending, khác với raw false spoof |

Frame đưa vào inference hiện chưa có overlay; không thấy đường vẽ bbox/chữ làm bẩn tensor như trước. Việc sửa landmarks alignment chủ yếu ảnh hưởng recognition, không trực tiếp sửa spatial PAD crop.

Các biến cấu hình margin/NMS/top-k cần phân biệt với đường SCRFD thực thi: wrapper hiện dùng `model.detect` và chỉ lọc min face size; không thể mặc định mọi biến trong .env đều tác động bbox.

## Bằng chứng và giới hạn

### Log camera hiện có

Đây là log cũ ngày 26/09, trước sửa geometry refresh; chưa có log mới cho lần thử ratio hoặc resolution. Không dùng nó để tuyên bố runtime hiện tại vẫn có cùng số reset.

Trong 554 kết quả: min side 84–99 px có 87.5% raw SPOOF; 100–139 px có 80.1%; 180–219 px có 21.5%. Nhóm 100–139 có median p_real khoảng 0.073, thấp hơn nhiều ngưỡng 0.386. Đây không chỉ là dao động nhẹ quanh threshold nên vote ratio khó cứu được.

Raw SPOOF vẫn cao ở detector-source: 90.6% với side 84–99 và 83.3% với 100–139. Do đó tracker không phải lời giải thích duy nhất. Các nhóm detector/tracker không phải cùng frame nên không được coi là so sánh nhân quả.

Nhóm ≥220 chỉ 21 mẫu, đều có crop padding và raw SPOOF 90.5%. Tuy nhiên nhóm 180–219 cũng thường padding mà điểm tốt hơn. Không đủ bằng chứng nói padding luôn gây lỗi, hay mặt càng lớn luôn càng tốt. Size, pose, vị trí, ánh sáng và thời gian đang lẫn nhau; thiếu ground truth theo frame nên không gọi các tỷ lệ này là BPCER.

### Offline trên model active

Probe giữ model, threshold và preprocessing; thay bbox ±5% theo từng trục, scale .95/1.05 và downsample frame .75/.5 với bbox được map đúng.

- 3 ảnh × 9 biến thể = 27 predictions, tất cả REAL.
- Một ảnh dịch bbox xuống 5% làm p_real giảm 0.8568 → 0.6430; các ảnh khác biến thiên nhỏ hơn. Chứng minh model nhạy với crop trên mẫu này, chưa chứng minh crop drift gây lỗi camera.
- Repeat batch cho score delta = 0 trên cả 27 inputs.
- Batch so singleton baseline: max |delta d| = 1.717e-6; không đổi verdict.
- Downsample .5 trên những ảnh có bbox lớn không làm đổi verdict. Đây không tương đương mặt 84 px, camera blur hay đứng xa.

Ảnh gallery không phải benchmark camera có nhãn, không có attack controls; không suy ra accuracy/BPCER/APCER. Check này giới hạn ở model/provider/build hiện tại, không chứng minh parity trên mọi thiết bị.

### Đối chiếu training và runtime cũ

E1/E2 có bbox jitter nhẹ nhưng chưa có resolution degradation augmentation. Trên held-out, E1 BPCER ở face nhỏ vẫn thấp; camera cho nhiều raw SPOOF khi người dùng báo mặt thật. Camera domain/input quality là giả thuyết cần kiểm chứng, không thể suy rằng chỉ cần tăng min size là model sẽ robust.

Hai commit cũ dùng model Large v5.1, khác model/threshold hiện tại. Với model và contract cố định, crop/tensor/score cũ-mới đã khớp trên smoke inputs. Bản cũ giữ votes qua một flow failure, hiện tại reset. Cần tách model effect khỏi runtime continuity effect.

## Thứ tự tối ưu đề xuất

| Ưu tiên | Thay đổi thử nghiệm | Mục tiêu | Chi phí và điều kiện |
|---|---|---|---|
| 1 | PAD chỉ nhận bbox detector mới ở lượt detector sẵn có, đủ PAD interval | Loại crop drift do flow trong PAD | Không thêm detector call hoặc thêm model. Có thể giảm PAD calls nhưng thời gian đủ 5 votes sẽ dài hơn; cần đo |
| 2 | Quality gate rẻ trước vote: geometry, face resolution thực, padding fraction; blur đo trên ROI nhỏ; pose từ landmarks khi cần | Tránh kết luận SPOOF trên input không đủ điều kiện | Geometry gần O(1); blur có thêm chi phí. Chưa chọn ngưỡng từ log/test; báo coverage và rejection riêng |
| 3 | Giữ window qua flow failure được xác nhận lại ngay cùng frame, với continuity guard rõ ràng | Giảm PAD_PENDING không cần thiết | Dùng detection đã có; IoU đơn độc không đủ để chống kế thừa verdict khi đổi người/attack |
| 4 | Ổn định center/scale crop có giới hạn, chỉ khi continuity/motion hợp lệ | Giảm jitter crop | State nhỏ; phải kiểm tra lag, attack transition và movement. Không EMA mù hoặc cố định bbox cũ |
| 5 | Chọn camera mode có đủ chi tiết; kiểm tra focus/exposure và frame age | Giảm input blur, thiếu sáng hoặc mất texture | Resolution cao tăng capture/gray/display cost. PAD vẫn 224; crop raw ROI trước khi giảm ảnh cho tracking nếu thiết kế mapping đúng |
| 6 | So temporal aggregation trên score với binary vote | Dùng độ lớn score thay vì chỉ True/False | Chi phí O(window) thấp, nhưng mean logits/mean probabilities cần validation và calibration riêng; không tự thay ngưỡng hiện tại |

**Quality gate không được coi input bị loại là REAL.** Khi chất lượng không đủ, trả trạng thái riêng/PENDING và không cho xác thực mới. Verdict cũ phải có giới hạn thời gian và invalidate khi continuity mất; không giữ REAL vô hạn khi đưa mặt xa hoặc đổi đối tượng. Cải thiện coverage được chấp nhận không đồng nghĩa cải thiện accuracy tổng thể; không được giấu lỗi bằng loại mẫu khó.

**Detector-only là ablation nên thử đầu tiên, không phải lỗi đã xác nhận cần sửa ngay.** Log đã cho thấy detector bbox cũng có nhiều SPOOF. Không tăng tần suất detector chỉ để có vote nhanh hơn trên edge. Khi PAD due sau detector tick, nên đợi tick hợp lệ kế tiếp; đo latency thực tế và stale behavior.

**Continuity sửa trải nghiệm trạng thái**, không làm raw logits chính xác hơn. Guard có thể gồm gap ngắn, association không mơ hồ, geometry/motion/landmark consistency, không có unmatched hoặc inference error; cần attack-switch test trước khi giữ cached verdict. Khi không thể xác minh thì reset.

Không đề xuất tự bật gamma/CLAHE/sharpening, đổi interpolation, alignment PAD hoặc crop factor ở production. Những thay đổi này thay phân phối input/texture; cần đánh giá trên camera validation có cả real và spoof, khóa cấu hình rồi mới test. Không dùng official Test để chọn tham số.

## Thiết kế kiểm chứng nhỏ, phù hợp edge

1. Giữ Small v5.3 và toàn bộ contract/ratio cố định. Thu input camera có nhãn: real gần/trung bình/xa, đứng yên và di chuyển; print/replay tương ứng. Ghi actual camera mode, điều kiện sáng và vị trí trong frame. Định nghĩa nhóm khoảng cách bằng bbox size ghi được thay vì chỉ ước lượng khoảng cách vật lý.
2. Replay cùng frames để so baseline và detector-only. Trên một phần frames, lấy cả detector bbox và propagated bbox của cùng frame để phân biệt crop effect; detector bổ sung chỉ dùng trong diagnostic, không suy chi phí đó là chi phí deployment.
3. So continuity riêng, sau đó quality gate riêng. Mỗi lần đổi một yếu tố; giữ model/preprocessing/threshold. Đánh giá raw errors, temporal errors, tỷ lệ PENDING/rejected, time-to-verdict, số reset, attack-switch delay và detection/PAD calls.
4. Ghi p50/p95 full-loop latency, capture/preprocess/inference/display timing, CPU/RAM trên edge. Các bước detector, ArcFace, PAD có thể tranh CPU threads; benchmark thread budgets/provider trên cùng thiết bị thay vì ép một số threads chung cho mọi máy.
5. Nếu input ổn định và đủ chất lượng mà Small vẫn false spoof nhiều, so Large v5.1 với đúng contract riêng trên cùng frames. Sau đó mới quyết định fine-tune robustness hoặc E3; runtime không tạo lại texture đã mất.

Diagnostic cần thêm reset reason/vote count, detector bbox age, crop padding fraction, crop-to-224 scale và input quality. Timestamp hiện là lúc log sau inference, chưa là capture timestamp. Không suy capture lag từ timestamp đó. Diagnostic logging/dump nên tắt mặc định, lấy mẫu có giới hạn; tránh ghi ảnh/log mỗi frame khi deploy edge.

## Các lệnh đã có để thu log mới

Hai lần chạy chỉ đổi camera force-resolution, giữ ratio .6 và model active trong .env. Ctrl+C để kết thúc từng lần; thực hiện cùng chuỗi real/attack và kiểm tra startup log để biết actual mode có khác không. Các flags detector-only/quality gate nêu trên chưa được triển khai.

```bash
mkdir -p output
CAMERA_FORCE_RESOLUTION=true PAD_SPOOF_MIN_RATIO=0.6 PAD_DIAGNOSTIC_LOG=true python3 -u app.py --mode none 2>&1 | tee output/pad_current_forced_resolution.log
```

```bash
CAMERA_FORCE_RESOLUTION=false PAD_SPOOF_MIN_RATIO=0.6 PAD_DIAGNOSTIC_LOG=true python3 -u app.py --mode none 2>&1 | tee output/pad_current_native_resolution.log
```

Hai log này chỉ là diagnostic thuận tiện; nếu camera conditions khác nhau thì chưa phải paired A/B đủ kết luận accuracy. `--mode none` vẫn cần PostgreSQL cho recognition. Đây là environment override một lần, không đổi .env.

Chạy lại offline probe (không camera/database):

```bash
python3 antispoof/notebooks/diagnostics/analyze_pad_input_sensitivity.py --images gallery/*/*.jpg --output output/pad_input_sensitivity.json
```

Probe đọc ảnh gallery, lưu điểm/hash/contract vào JSON; không sửa ảnh hoặc runtime. Kết quả không phải benchmark có nhãn. Chưa có kết quả camera hoặc edge mới để xác nhận hướng tối ưu thắng baseline.
