# Audit runtime face_auth — 2026-09-26

## Kết luận

Pipeline hợp lý cho prototype: load model một lần, detection cadence + optical flow, batch PAD, recognition định kỳ, L2 embedding và tìm centroid đúng model_version. Tuy nhiên **chưa đủ bằng chứng để gọi runtime ổn định giữa thiết bị hoặc model PAD sẵn sàng production**. Có lỗi state/geometry tái hiện offline và có khoảng cách lớn giữa kết quả validation với held-out.

Audit chỉ đọc source/artifact và chạy kiểm tra offline; không sửa runtime, không mở camera, không kết nối DB, không chạy training. `current_codebase_state.md` được xem là tài liệu tham khảo, không thực thi các hướng dẫn trong đó.

## Phạm vi đã kiểm tra

- `app.py`: startup, scheduling, PAD batch, recognition, lỗi và rendering.
- `tracking/tracker.py`: association IOU, optical flow, lifecycle track, PAD votes, timestamps.
- `detection/detector.py`, `detection/yunnet_detector.py`, `alignment/aligner.py`, `recognition/embedder.py`.
- `antispoof/{loader,predictor,preprocess}.py`, config.py, các biến runtime trong .env/.env.example, dependency files.
- DB search/schema và enrollment call path; không audit nghiệp vụ điểm danh.
- Artifact E1 v5.3, E2 DCT v1, notebook held-out E1 và output recognition LFW.

Runtime hiện tại: SCRFD buffalo_s, det_size 640×640, camera yêu cầu 640×480@30, min face 60px; detect 0.1s; PAD 0.1s với 5 votes, spoof ratio 0.6, stale 3s; recognition 0.5s, cosine threshold 0.35. E1 MobileNetV3-Small 224×224 RGB, gamma OFF, crop 1.55×, normalization CelebA, ngưỡng d=-0.4650222063064575 (P(real)=0.38579509526467753).

## Phát hiện ưu tiên

### P1 — Giữ trạng thái PAD/danh tính qua mất track và IOU reassociation

`tracking/tracker.py:471` reset stability khi detector không match nhưng không reset PAD window, identity hoặc recognition timestamp. Khi bbox mới match IOU với track cũ (`:454`), các trạng thái này tiếp tục được dùng. Association không xác minh appearance/identity; người hoặc vật khác ở cùng vị trí có thể kế thừa trạng thái cũ.

Tái hiện với clock cố định: tạo track, 5 REAL votes, identity A → detector trả rỗng → detection cùng bbox → track_id=1, PAD=REAL, identity=A, timestamp cũ vẫn còn. Đây là bằng chứng state reuse, chưa phải phép đo tỷ lệ khai thác qua camera.

Đề xuất: khi mất continuity, đưa track vào pending, loại verdict/danh tính được phép dùng và bắt buộc PAD + recognition mới trước khi chấp nhận. Phân biệt identity chỉ phục vụ hiển thị với identity đã được xác minh đủ mới. Thêm giới hạn thời gian mất dấu và kiểm tra reassociation.

### P1 — Optical flow thay bbox nhưng không biến đổi tương ứng landmark

`tracking/tracker.py:326` lấy bbox từ 4 góc flow, nhưng `:376` chỉ cộng median dx/dy vào landmark. Scale nhỏ hơn ngưỡng redetect vẫn được coi là tracking an toàn; PAD và recognition có thể chạy ngay cả khi status là needs_redetection (`app.py:463`, `:511`).

Mock flow scale 1.2× quanh tâm: bbox (30,30,130,130) thành (20,20,140,140), landmark đầu vẫn (55,60), trong khi biến đổi đúng là (50,56); status=tracking. Rotation/pose cũng không được mô hình hóa bởi cập nhật landmark này.

Đề xuất: propagate landmark bằng cùng transform được kiểm chứng, hoặc chỉ recognition trên detector landmarks mới. Redetect ngay trong frame trước inference khi propagation không an toàn. Flow hiện dùng góc bbox và tâm, thiếu forward-backward consistency/error gate; status LK thành công không chứng minh điểm vẫn thuộc khuôn mặt.

### P1 — Lỗi inference có thể tiếp tục dùng kết quả thành công cũ

`antispoof/predictor.py:304` trả [] khi lỗi; `app.py:473` bỏ qua crop lỗi. PAD cũ tiếp tục hợp lệ đến stale timeout 3s. `app.py:159` bắt lỗi recognition nhưng không invalidate identity/stability hoặc đặt retry backoff. Alignment trả None cũng không cập nhật timestamp.

Tái hiện recognition exception: identity=A, stability=3, last_recognition_time=100 vẫn giữ nguyên sau lượt lỗi tại t=101. Các frame tiếp theo có thể retry liên tục và HUD vẫn giữ kết quả cũ. Đường propagation lost (`tracking/tracker.py:307`, `:315`) không reset stability như đường detector unmatched; offline check xác nhận stability=3 khi status=lost.

Đề xuất: lỗi/geometry không hợp lệ phải chuyển verdict dùng cho quyết định sang pending; identity có tuổi tối đa; reset stability nhất quán ở mọi đường lost; log lỗi có rate limit và retry có backoff. Không coi lỗi inference là spoof đo được; ghi riêng ERROR/PENDING.

### P1 — Rendering làm bẩn input recognition của các mặt phía sau

Trong `app.py:511–564`, mỗi face được recognition rồi draw ngay lên cùng frame. Recognition face tiếp theo đọc frame đã có bbox/text. PAD batch chạy trước draw nên không bị cùng lỗi này. Khi các vùng overlap, kết quả phụ thuộc thứ tự track, geometry và lịch recognition.

Offline check dùng fallback crop: vẽ một track đè vùng face tiếp theo làm thay đổi 7.070 giá trị kênh trong ảnh recognition 112×112. Chưa đo mức thay đổi embedding/identity.

Đề xuất: inference đọc raw frame bất biến; rendering làm trên bản copy sau khi xử lý tất cả face.

### P2 — Thời gian tính bằng giây chưa làm runtime độc lập FPS

- Detector dùng monotonic (`app.py:432`), PAD/recognition/stale dùng time.time (`tracking/tracker.py:82–153`). Thay đổi wall clock có thể làm timer nhảy, khác với detection.
- PAD interval kiểm tra từ lúc completion; inference và phần việc còn lại làm cadence thực tế dài hơn cấu hình. Không có cơ chế bảo đảm 5 PAD calls trong một recognition cycle.
- 5 votes là 5 mẫu, không phải cửa sổ thời gian cố định. Ở 5 FPS, ngay cả khi PAD chạy mọi frame, 5 mẫu cần ít nhất khoảng 0.8s từ mẫu đầu đến mẫu cuối; ở 30 FPS với interval 0.1s cần ít nhất khoảng 0.4s và thực tế có thể lâu hơn.
- Với 5 REAL votes cũ, cần 3 SPOOF votes mới để đổi verdict: độ trễ này phụ thuộc FPS/cadence. Không được suy APCER video từ majority vote nếu chưa đo trên chuỗi video; lỗi liên tiếp có thể tương quan.
- max_missing_frames=10 là số lần update mất dấu, không phải timeout giây. Khi không có track hoặc có track lost, needs_redetection buộc detect mỗi frame, bỏ qua cadence 0.1s.

Đề xuất: monotonic cho timer nội bộ; track TTL theo thời gian; ghi timestamp acquisition/inference/vote và cadence thực tế; định nghĩa rõ giới hạn tuổi mẫu, thời gian pending và response-to-spoof. Đánh giá temporal policy trước khi đổi contract.

### P2 — Camera/preprocessing không bảo đảm cùng input giữa thiết bị

App in requested/actual camera properties nhưng không ép về kích thước chuẩn hoặc reject mode không hỗ trợ. Buffer size có thể bị backend bỏ qua; capture và inference đồng bộ nên hàng đợi ảnh cũ có thể tăng latency dù FPS overlay nhìn hợp lý. Exposure, white balance, sharpening, compression và field of view không được ghi thành deployment profile.

Min face=60px tạo điều kiện lọc khác nếu resolution/FOV khác. YunNet xử lý toàn bộ resolution thực tế (`detection/yunnet_detector.py:139`); input_size 320 chỉ là kích thước khởi tạo. SCRFD resize nội bộ 640. SCRFD không áp margin/out-of-frame policy tương đương YunNet và không nhận NMS config từ app. Optical flow cũng có thể đưa bbox xuống dưới min face detector mà vẫn chạy PAD.

Đề xuất: profile camera rõ ràng, ghi actual frame shape/FPS và tuổi frame; nếu dùng capture thread latest-frame thì bỏ frame cũ có kiểm soát. Thống nhất geometry/quality gates trước mọi model. Không chỉ resize rồi mặc nhiên coi hai camera là tương đương.

### P2 — Provider/dependency/model contract chưa khóa cho deployment

SCRFD và ArcFace hard-code CPUExecutionProvider; MODEL_CTX_ID không tự bật GPU. PAD auto chọn CUDA rồi CPU, bỏ CoreML khi CPU hiện diện. Trên môi trường audit, ORT có CoreML/Azure/CPU nhưng PAD chạy CPU. Không cấu hình thread budget hoặc GPU device rõ ràng; cần đo oversubscription trước khi kết luận là bottleneck.

requirements.txt không pin versions. Model pack name và EMBEDDING_MODEL_VERSION=buffalo_s không chứng minh cùng hash recognition/detection trên hai máy. Không có startup verification hash model và preprocessing contract; runtime JSON không chứa hash. CLI --pad-model thay file nhưng app vẫn truyền threshold/color/mean/std/crop của model hiện tại, có thể dùng sai contract cho MiniFASNet/E2. Predictor còn suy profile theo filename/kích thước và mặc định class 0=real. ONNX metadata chỉ giúp size/channel, không xác nhận label order hay output là logits.

Đề xuất: deployment manifest gồm hash ONNX/sidecar, contract, dependency lock, provider/device/thread settings; validate startup thay vì heuristic. Khi chọn model khác phải chọn contract riêng. So sánh logits, embedding cosine và flips quanh threshold trên input cố định giữa providers; không đòi bitwise parity cho mọi backend.

### P2 — Hiệu năng hiện chưa được đo đủ để quyết định tối ưu tiếp

Pipeline đơn luồng gồm capture, detection/flow, PAD, từng ArcFace + DB query, draw/UI. Pool DB tái sử dụng connection nhưng không khiến query bất đồng bộ. Nhiều mặt đến hạn cùng lúc tạo burst latency. DB search exact distance + sort, schema chưa có index cho centroid/model_version hoặc vector ANN; với gallery nhỏ có thể hoàn toàn phù hợp, cần EXPLAIN/benchmark trước khi đổi search protocol.

recognition_ms cộng cả orchestration, lượt không recognition và thao tác DB khác, nhưng chia cho recognition_calls; pad_ms đo batch nhưng pad_calls đếm số crops, crop ngoài vùng timing. Vì vậy các mean này không phải latency mỗi invocation của model và khó so thiết bị. Có FPS end-to-end đúng cách nhưng chưa có p50/p95, capture latency, queue age, DB time riêng, số face mỗi batch hay first-verdict latency.

Đề xuất thứ tự: sửa correctness → instrumentation → baseline trên thiết bị → tối ưu thread/provider/detection cadence/capture → batching recognition hoặc cache gallery nếu số liệu cho thấy cần. Không giảm resolution, quantize hoặc thay model/threshold chỉ để tăng FPS mà chưa đánh giá lại PAD/recognition.

## Kết quả model và provenance

Áp dụng đối chiếu provenance theo `reproducibility-guard`: hash file E1 đang chạy khớp test summary:

`f644427b7b4351093a3cfac64f6ddedafd11c4a05e81fda7e5e156dea4a5c96e`.

Contract active khớp best_meta/runtime JSON: 224, RGB, mean/std, gamma OFF, crop 1.55 và threshold. E1 ONNX hỗ trợ dynamic batch, output 3 logits. Đã so hàm crop runtime với hàm AST trích từ held-out notebook ở 3 bbox (giữa ảnh, mép trên/trái, mép dưới/phải): bằng nhau chính xác. Kiểm tra batch 1 so batch 2 trên cùng ảnh synthetic: chênh pad_score tối đa 1.83e-7, chỉ kiểm tra smoke một input CPU; không phải chứng nhận parity giữa thiết bị.

Tính lại trực tiếp từ prediction CSV:

| Model | N | Spoof bị nhận REAL | REAL bị nhận SPOOF | APCER | BPCER | ACER | AUC trong summary |
|---|---:|---:|---:|---:|---:|---:|---:|
| E1 MobileNetV3-Small | 10.000 | 1.330/7.034 | 12/2.966 | 18,908% | 0,405% | 9,656% | 0,98143 |
| E2 DCT Tiny CNN | 10.000 | 2.883/7.034 | 282/2.966 | 40,987% | 9,508% | 25,247% | 0,83657 |

E1 best_meta ghi validation ACER=0,3333%, test ACER=9,6564%. Đây là khoảng cách cần phân tích theo subject/attack/crop source; chưa thể quy nguyên nhân cho overfit, domain shift hoặc leakage chỉ từ số tổng hợp. Test protocol ghi threshold khóa từ Validation; audit này không kiểm tra toàn bộ quy trình tạo split/training từ dữ liệu gốc. Các số inference 2,468ms/image E1 và 0,568ms/image E2 trong summary là số từ evaluation artifact, không phải latency webcam trên máy hiện tại hay edge.

Held-out là preliminary 10k, detector-conditioned crops có fallback SCRFD sizes/CelebA bbox; runtime chỉ SCRFD640 hoặc YunNet, filter 60px, có tracker và temporal votes. Do đó không tương đương end-to-end camera authentication. E2 hiện đã có implementation và artifact, không còn chỉ PLANNED như tài liệu trạng thái.

Recognition: output notebook LFW tìm threshold mean=0.31, individual=0.33, trong khi runtime dùng 0.35. Người dùng xác nhận 0.35 là ngưỡng được chủ động điều chỉnh/làm tròn lên; giữ nguyên lựa chọn này. Không gắn FAR/FRR của threshold notebook cho 0.35; nếu cần số đo camera/gallery thực tế thì calibration riêng chỉ dùng Validation. Enrollment luôn dùng SCRFD trong khi app có thể chọn YunNet, nên cần đánh giá ảnh hưởng landmark khác detector.

## Bằng chứng kiểm tra và giới hạn

- `python3 -m unittest discover -s tests -p 'test_*.py' -v`: 17/17 pass, khoảng 0.05s phần test; chủ yếu unit/mock, không chứng minh runtime camera/backend.
- Offline deterministic reproductions: PAD/identity reuse khi reacquire; flow lost giữ stability; bbox scale không scale landmarks; recognition exception giữ cache; overlay ảnh hưởng input mặt sau.
- Real ONNX smoke: E1 load thành công CPU, dynamic batch 1/2 trả đủ kết quả; SHA256 khớp held-out.
- Tính lại confusion counts/APCER/BPCER/ACER của E1/E2 từ CSV, khớp summary.
- Môi trường audit: Python 3.11, OpenCV 4.13.0, NumPy 2.4.4, ONNX Runtime 1.24.4. Không benchmark camera, detector/ArcFace đầy đủ, GPU hay edge, không thử DB thật.

## Protocol kiểm tra trên từng thiết bị

Không chạy camera/hardware trong audit này. Dùng cùng model hashes, config, gallery và cùng chuỗi video đầu vào cố định để so thiết bị trước; camera thật cần cùng cảnh/ánh sáng/khoảng cách và ghi actual properties. Chạy ít nhất các tình huống: 0/1/nhiều mặt; đứng yên/di chuyển/scale; che mặt rồi trở lại; đổi người/ảnh ở cùng vị trí; real→print/replay→real; DB timeout và PAD inference error. Failure injection làm offline, không cần gây lỗi thiết bị thật.

Lệnh camera hiện có (từ repo root, mode none để quan sát):

```bash
PAD_DIAGNOSTIC_LOG=true python3 app.py --mode none > /tmp/face_auth_runtime_diagnostic.log 2>&1
```

Chạy cùng thời lượng 60s mỗi tình huống, thoát bằng q để có runtime summary. Trả lại log, OS/CPU/RAM, camera model/backend, dependency versions, model hashes, tình huống đã chạy và recording có timestamp nếu thu được. Diagnostic logging có overhead; chạy thêm lần PAD_DIAGNOSTIC_LOG=false để so tốc độ. Log hiện có lỗi dùng biến detection còn sót từ vòng crop khi ghi bbox_source (`app.py:496`), nên không coi trường này đáng tin cho nhiều mặt trước khi sửa.

Sau khi bổ sung instrumentation và replay input, thu: p50/p95 loop/capture/detection/flow/PAD/ArcFace/DB; actual FPS, queue age/dropped frames; PAD_PENDING duration, spoof-response latency, ID switches, false accept/reject trên video; logits/embedding trên bộ input cố định. App hiện chưa có replay CLI nên không đưa một lệnh replay chưa tồn tại.

Ưu tiên sửa P1 trước, rồi đo. Không thay threshold/test split/preprocessing trong quá trình tối ưu mà thiếu experiment spec và calibration độc lập.

## Bản sửa sau audit

- Mất detector match hoặc optical flow mất dấu: reset PAD, identity và stability; reacquire phải xác minh lại. Bật lại PAD cũng yêu cầu xác minh mới. Nếu flow chỉ yêu cầu chỉnh geometry, app redetect trước inference; detector match ngay cùng frame giữ votes, detector không match vẫn reset.
- Propagate landmark bằng similarity transform từ optical flow để giữ scale/rotation; flow thiếu góc hoặc geometry không an toàn không được đưa vào inference. App redetect ngay cùng frame.
- Crop/PAD batch lỗi, thiếu kết quả hoặc score không hữu hạn: đưa track liên quan về pending; recognition lỗi/không có embedding cũng bỏ identity cũ. Retry theo interval hiện có để tránh chạy lỗi mỗi frame.
- PAD/recognition/stale dùng monotonic clock; timestamp DB/cooldown tiếp tục dùng wall clock.
- Render trên display_frame riêng; models luôn đọc raw frame. Diagnostic dùng đúng bbox_source của từng track và sigmoid ổn định số học.
- CLI không cho đổi sang file PAD khác nhưng tái sử dụng contract hiện tại. Đổi model cần chọn runtime config/model tương ứng trước khi chạy. Đây là guard có chủ ý; không tự suy contract cho file khác.
- Giữ recognition threshold 0.35, PAD model/preprocessing/threshold, vote policy và dataset protocol.
- Validation sau patch: 29/29 tests pass (12 regression tests mới); syntax và git diff --check pass. Smoke với E1 ONNX thật trên 2 crops CPU trả đủ 2 kết quả và cả hai track vẫn PAD_PENDING sau vote đầu, đúng policy 5 votes. Không mở camera hoặc DB thật.

Các phát hiện phía trên mô tả trạng thái trước patch. Những giới hạn về camera/provider, temporal policy phụ thuộc tốc độ lấy mẫu, chất lượng model và benchmark thiết bị vẫn cần đo thực tế; patch không khẳng định đã giải quyết chúng.

### Điều chỉnh sau diagnostic camera

Log người dùng cung cấp: 554 PAD results trong 54,63s, cùng track_id=1; 303 PAD_PENDING, 141 SPOOF, 110 REAL; 13 lần từ verdict sang pending. Log chưa ghi reset reason nên không quy mọi transition cho cùng nguyên nhân. Có nhiều bbox vượt mép dưới 480px, phù hợp đường flow yêu cầu detector refresh. Patch đầu reset votes cả khi cần refresh geometry; đã sửa để detector xác nhận lại ngay cùng frame có thể giữ votes. Không thay PAD interval 0.1s, vote threshold hay model threshold. Regression test bao phủ thiếu góc flow, scale lớn, bbox vượt mép và cả detector match/unmatched; 30/30 tests pass. Cần chạy camera lại để đo mức giảm pending thực tế.
