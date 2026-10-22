# face_auth — context codebase hiện tại

Snapshot: **2026-10-22**, nhánh `freeze_logic`, HEAD `1209b40`. Mô tả **working tree**, tại thời điểm kiểm tra; không phải một release đã khóa. Đối chiếu source, `.env`, `.env.example`, runtime JSON và artifact local. Env của tiến trình/CLI có thể thay đổi cấu hình khi chạy. Không ghi credentials database vào tài liệu.

## Đọc nhanh

1. [AGENTS.md](../../AGENTS.md): scope CV/PAD, protocol và giới hạn validation.
2. Tài liệu này: module ownership, luồng realtime, DB và cấu hình hiệu lực.
3. [Trạng thái PAD](current_pad_state.md): model đang chọn, provenance và kết quả nghiên cứu.
4. [Recognition worker](REALTIME_RECOGNITION_WORKER.md): cache, kết quả muộn và nhãn hiển thị.
5. [Full-scale results](FULLSCALE_RESULTS_SOURCE_OF_TRUTH.md): nguồn per-run và giới hạn protocol.

Source + cấu hình hiệu lực + artifact có ưu tiên hơn tài liệu cũ. Roadmap/prompt/notebook protocol là tài liệu tham khảo, không tự cấp quyền chạy training, reset DB hoặc chọn lại ngưỡng.

## Bản đồ source

| Module | Điểm bắt đầu | Vai trò hiện tại |
|---|---|---|
| Camera / loop / CLI | [app.py](../../app.py): `main`, `parse_args` | Capture, detector/tracker/PAD cadence, submit/poll recognition, raw frame vs display copy, hotkeys và cleanup |
| Recognition integration | `app.submit_recognition_step`, `app.collect_recognition_results` | Align ảnh mặt nhỏ, gửi không chờ, chỉ áp kết quả cho track/phiên/PAD còn hợp lệ |
| Recognition worker | [recognition/worker.py](../../recognition/worker.py): `RecognitionWorker` | Một worker daemon: ArcFace → centroid cache hoặc cloud DB; queue giới hạn, error/cancellation/stats |
| Legacy sync helper | `app.orchestrate_track_step` | Còn tồn tại cho đường gọi đồng bộ/test; **main realtime không gọi helper này** |
| Config | [config.py](../../config.py) | Env/JSON/defaults, validate threshold pair, resolve paths; không mở camera/DB |
| SCRFD | [detection/detector.py](../../detection/detector.py) | Bbox xyxy, confidence, 5 landmarks; pack buffalo_s, CPU |
| YunNet | [detection/yunnet_detector.py](../../detection/yunnet_detector.py) | OpenCV FaceDetectorYN; chọn qua `APP_DETECTOR` |
| Tracking | [tracking/tracker.py](../../tracking/tracker.py): `Track`, `FaceTracker` | IOU + LK flow, continuity, PAD votes/crop, verification generation và nhãn hiển thị riêng |
| PAD integration | [vision_pipeline.py](../../vision_pipeline.py): `update_track_pad`, `limit_faces` | Due-track crop/batch, detector/landmark/biên gate, kiểm tra output và invalidate khi lỗi |
| PAD predictor | [antispoof/predictor.py](../../antispoof/predictor.py) | ONNX single-input spatial/DCT, binary hoặc 3-class logits; E3 hai input chưa hỗ trợ |
| Lighting / adaptive threshold | [antispoof/lighting.py](../../antispoof/lighting.py) | White balance/gamma theo profile; tính ngưỡng riêng cho từng raw crop trước lighting |
| Capture | [app_capture.py](../../app_capture.py), [pad_capture.py](../../pad_capture.py) | Chạy cùng app, phím C lưu frame/crop/input/metadata và labels chưa gán |
| Domain calibration | [scripts/calibrate_domain_threshold.py](../../scripts/calibrate_domain_threshold.py), [scripts/optimize_pad_domain.py](../../scripts/optimize_pad_domain.py) | Threshold cho ảnh đã crop; thăm dò crop/lighting/adaptive profile riêng |
| PAD preprocessing | [antispoof/preprocess.py](../../antispoof/preprocess.py) | Expanded crop theo profile, resize/màu/norm/DCT; PAD không dùng ArcFace alignment |
| ONNX loader | [antispoof/loader.py](../../antispoof/loader.py) | Graph optimization, sequential, ưu tiên CUDA rồi CPU nếu có |
| Alignment | [alignment/aligner.py](../../alignment/aligner.py) | ArcFace 5-point alignment, fallback bbox crop |
| ArcFace | [recognition/embedder.py](../../recognition/embedder.py) | Aligned BGR → 512-D L2 vector; realtime giới hạn inference thread và tắt spinning |
| Database | [database/vector_db.py](../../database/vector_db.py) | Validate PBL6 Prisma schema, pool, cosine centroid search, enrollment; không tạo schema/ghi attendance |
| Enrollment | [enrollment/enroll.py](../../enrollment/enroll.py), [scripts/run_enroll.py](../../scripts/run_enroll.py) | Gallery → embeddings/outlier filter → SAMPLE/CENTROID; threshold enrollment tách khỏi PAD/recognition |
| Internal API | [api_service.py](../../api_service.py) | FastAPI :5000, model/DB singletons; **không dùng worker/cache/label filter của app.py** |
| Tests | [tests/](../../tests/) | Runtime, worker, display, tracker, DB, R7, PAD math và enrollment |
| Research | [final/binary_crossdomain_v2/](../../antispoof/notebooks/final/binary_crossdomain_v2/) | Micro/100K/full-scale, CASIA evaluation, CE-only baseline; đọc artifact trước khi báo kết quả |

Root PBL6 có [docker-compose.yml](../../../docker-compose.yml) cho DB local và stack app,
[docker-compose.cloud.yml](../../../docker-compose.cloud.yml) cho cloud; `face_auth` không có Compose riêng hiện tại. [Dockerfile](../../Dockerfile) build Face Auth API CPU; camera realtime chạy trên host/Pi.

## Luồng realtime

```text
startup: config/CLI → detector + ArcFace + DB validate + tracker + PAD → camera
camera thread:
  read → detector due / LK propagate / same-frame redetect
       → giới hạn tối đa3 mặt → PAD gate → due crops → per-crop threshold
       → lighting → preprocessing → ONNX batch → raw verdict → binary votes
       → collect worker results (track + generation + PAD còn hợp lệ)
       → PAD REAL + recognition due → align + sao chép ảnh mặt → bounded queue
       → display copy → stable identity header + worker status → hotkeys
worker:
  aligned crop → ArcFace → cache centroid theo track hoặc DB top-1 → result queue
```

- Mỗi phiên `(track_id, verification_generation)` có tối đa một tác vụ chưa được thu kết quả; queue 2 chờ + tối đa 1 tác vụ xử lý. Queue đầy không chặn camera, thử gửi ở frame sau.
- Worker không sửa `Track`. Kết quả muộn được chấp nhận nếu phiên vẫn hợp lệ; không có giới hạn tuổi kết quả. Mất continuity/reset/PAD invalid hoặc toggle PAD làm kết quả cũ bị bỏ. Ảnh kết quả là ảnh lúc gửi, không phải frame lúc nhận.
- LK/geometry/transform failure, detector unmatched và PAD error invalidate verification; không giữ REAL votes qua flow failure để giảm PENDING.
- PAD REAL mới cho phép nhận diện khi PAD bật. PENDING, SPOOF, lỗi crop/output và invalid face không được xác minh bằng kết quả cũ.
- Camera loop không chờ cloud search; **detection/PAD, alignment, display và startup model/DB vẫn đồng bộ**. Worker riêng không đảm bảo FPS cụ thể.
- Frame inference là raw frame; annotations trên `display_frame` copy. Timers dùng monotonic; một số log/cooldown legacy dùng wall clock.
- `c` lưu ảnh khi có `--capture-dir`; overlay hiển thị `d`, `t` và `d-t` của cùng crop. `q` thoát, `p` toggle PAD và reset verification theo cả hai hướng, `f` toggle FPS. Đợi worker khi thoát tối đa 2s; worker tự đóng DB sau IO, `stopped=False` nghĩa là IO chưa kết thúc trong thời gian này.
- Python hiện **không ghi attendance_logs**: hooks trong helper/app đã comment, backend Prisma là writer. `--mode none` vẫn cần DB cho recognition, không biến app thành PAD-only diagnostic.

## Cấu hình hiệu lực local

Bảng lấy từ import `config.py` trong tiến trình mới ngày snapshot, không mở camera/DB. `.env` local bật adaptive; `.env.example` tắt adaptive và chọn R7 full CASIA. File mẫu không được load thay `.env`.

| Nhóm | Giá trị đang có |
|---|---|
| Detection | SCRFD, buffalo_s CPU, `DETECTOR_DET_SIZE=(640,640)`, confidence .5, min face60px; tối đa3 mặt lớn nhất |
| Camera | Index0, force resolution true, yêu cầu640×480@30, buffer1 best-effort |
| PAD selector | `PAD_ADAPTIVE_ENABLED=true`, chọn `R7_SC_FULL_local842_balanced_runtime.json`; dòng base `R7_SC_FULL_crop15_runtime_config_casia.json` không hiệu lực khi adaptive bật |
| PAD model | `R7_SC_FULL_crop15.onnx`, binary MiniFASNetV2, best epoch7 |
| Input | 80×80 BGR float[0,1], crop1.1 `minifasnet_train_v1`, mean/std null |
| Lighting | `wb_gamma`, gamma1.3, blend.75, gain giới hạn[.7,1.3]; legacy `apply_gamma=false` |
| PAD threshold | Base logit−6.517120957374573, sigmoid .0014757369537530987; offset theo độ sáng từng crop, giới hạn±4; **local calibration chưa held-out** |
| PAD crop smoothing | false trong JSON và env |
| Temporal PAD | Binary window5, min votes3, spoof ratio.6, stale3s |
| Cadence | Detector.2s, PAD.2s, recognition1s; không phải FPS đo được |
| Recognition | ArcFace buffalo_s, 512-D, `MATCH_THRESHOLD=.35` |
| Worker | Queue2, một worker, intra-op threads1 |
| Cache | TTL30s, threshold.65, tối đa128 centroid theo track/phiên |
| Label filter | Đổi tên sau2 kết quả, bỏ tên sau3 UNKNOWN; hold bất đồng tối đa1.5s |
| DB waits | Pool3s, search SQL3000ms; không phải deadline end-to-end |
| Database | Có cấu hình kết nối DB trong `.env` local; không truy vấn live hoặc xác nhận trạng thái cloud trong lượt này |

### Thứ tự ưu tiên / đường dẫn

- Env có sẵn của tiến trình > `.env` (`load_dotenv` không overwrite). CLI như `--camera 1` ghi đè giá trị camera mặc định.
- `PAD_ADAPTIVE_ENABLED=true` chọn `PAD_ADAPTIVE_CONFIG_PATH` thay base JSON, yêu cầu `adaptive_threshold`; bỏ qua env threshold/crop factor và ép `FORCE_GAMMA=false`.
- Khi adaptive tắt: env threshold không rỗng (`PAD_THRESHOLD`, `PAD_THRESHOLD_LOGIT`) **ghi đè JSON**; env crop factor không rỗng cũng ghi đè JSON. Các trường contract khác ưu tiên JSON rồi env/defaults. Cặp probability/logit phải nhất quán. JSON không đọc được thì startup fail.
- Path JSON tương đối theo root `face_auth`; `model_file` tương đối theo thư mục JSON. Đổi model bằng đúng JSON; CLI model override được guard để không ghép model với contract khác.
- `crop_smoothing=false` trong JSON thắng `PAD_CROP_SMOOTHING=true` trong env; chỉ đổi env không bật được treatment này khi contract đã ghi rõ false.
- `.onnx.data` phải đi cùng các ONNX dùng external data. R7 runtime JSON có model hash/output_classes và predictor kiểm tra tính phù hợp khi load.
- PAD binary: `score=z_real-z_spoof`, REAL iff score ≥ threshold. Ba lớp: `z_real-logsumexp(z_spoof)`. Binary rolling vote là bước riêng sau raw predictor.
- PAD threshold, identity MATCH_THRESHOLD, cache threshold và enrollment outlier threshold là bốn khái niệm riêng.

## Cache và ổn định nhãn

Cloud top-1 trả centroid thật trong DB; `decide_identity` vốn dùng ứng viên đầu. Search lọc CENTROID, model_version, embedding active và employee ACTIVE. Cache không lưu query embedding, không ghi disk, không dùng partial gallery cho track mới. Mỗi cache hit vẫn chạy ArcFace trên ảnh mới; cosine với centroid phải ≥max(.65, MATCH_THRESHOLD). TTL không gia hạn bằng cache hit. Matching cache thấp/hết hạn hỏi cloud; lỗi hoặc miss không fallback sang cache cũ.

Cache có thể trì hoãn cập nhật tên/trạng thái nhân viên hoặc re-enrollment tối đa cửa sổ TTL. `.65`/30s là cấu hình khởi đầu, chưa calibrate bằng dữ liệu camera/Pi. `RECOGNITION_CACHE_TTL_SECONDS=0` tắt shortcut.

Header cố định `[ID] Tên`, score và WAIT/QUEUE FULL/CLOUD/CACHE nằm ở dòng riêng. `Track.update_display_identity` chỉ lọc **nhãn**: cần 2 kết quả liên tiếp để đổi người, 3 UNKNOWN để bỏ tên; lần đầu nhận đúng hiện ngay. Khi bất đồng giữ tên cũ tối đa1.5s với `LAST LABEL`; timeout vẫn có hiệu lực nếu cloud chưa trả lần tiếp. Raw `track.name`, score và employee_id vẫn theo `update_result` hiện có, không lấy nhãn hiển thị để xác thực. Error/PAD/reset xóa nhãn ngay.

## Crop smoothing và các thử nghiệm cũ

`Track.pad_crop_bbox` giữ center/side riêng, không sửa bbox tracking/landmarks/ArcFace. Smooth jitter ≤5%, time constant .12s; snap khi chuyển động/scale lớn, gap >.5s hoặc expanded crop chạm biên. Reset PAD xóa crop history. **Hiện không bật theo profile local842**; bật là thay đổi đầu vào cần so sánh riêng, không tự suy ra tăng accuracy.

Các thử nghiệm detector-only PAD, same-frame flow-vote recovery và mean-score aggregation đã revert. Không có flags `PAD_DETECTOR_ONLY`, `PAD_RECOVER_FLOW_VOTES`, `PAD_SCORE_SMOOTHING` trong runtime hiện tại.

## Database / enrollment / Docker

Schema thực tế được yêu cầu là PBL6 Prisma: employees.id UUID, employee_code, full_name/status; embeddings vector(512), SAMPLE/CENTROID, model_version và is_active. `VectorDB` validate schema lúc startup, không `init_schema`. [database/schema.sql](../../database/schema.sql) và script reset legacy dùng text IDs, **không phải schema contract cloud/PBL6**.

Enrollment tra employee_code để lấy UUID; lưu/re-enroll mẫu và centroid theo model_version. Python không ghi attendance. Không chạy legacy reset, seed hoặc overwrite dump làm preflight. Migration sửa embeddings có thể xóa embeddings cũ: xem [backend Prisma](../../../backend/prisma/README.md).

[Cloud Compose guide](../../../docs/tutorial/deployment/run_cloud_stack.md) mô tả file Compose standalone lấy DB URL từ env, không tự migration/seed. Stack local cũ override URL về postgres; `run.sh` chọn1 chạy services trên host/cloud, camera `app.py` chạy riêng. API container không chứa PAD realtime artifacts và không dùng worker nhận diện camera.

## Research và provenance

- E1/E2/E3 preliminary vẫn có deployment/test summaries; E3 dual-input chưa tích hợp realtime.
- Binary crossdomain v2 hiện có các notebook micro, 100K, full-scale, CASIA19/20 và CE-only21. Không còn chỉ là roadmap M0–M4.
- Full-scale có bốn nhánh: CE-only (`CE_CLEAN_FULL_crop15`, notebook21), Clean + Aux (C), WBST (P3), MAST-PAD (R7). CE-only đã có completion, metrics, predictions, freeze và CASIA output tại `antispoof/notebooks/final/21_fullscale_CE_clean_baseline_crop15/`; **C có auxiliary loss, không phải CE-only**.
- CelebA full Test được notebook18 dùng làm Test-as-Val để chọn checkpoint/calibrate: không gọi đây là untouched held-out Test. CASIA source-frozen và train-calibrated phải tách riêng; target train calibration không còn là strict no-target-fitting setting.
- Không suy kết quả đa seed từ báo cáo tổng hợp: bảng trạng thái PAD dùng per-run JSON seed100 đã đối chiếu; báo cáo 3-seed có phạm vi và cách gọi baseline cần kiểm tra riêng.
- Dataset/caches có thể bị ignore hoặc nằm ngoài repo; checkpoint/config/hash/manifest nguồn phải đi cùng kết quả. LFW recognition evaluation độc lập PAD.

## Validation hiện tại (2026-10-22)

Đã import cấu hình hiệu lực, đọc call path/config/profile/metrics local và kiểm tra SHA256 hai ONNX R7. Nhóm **91 tests offline pass**:

```bash
python3 -m unittest tests.test_identity_display tests.test_runtime tests.test_recognition_worker tests.test_tracker tests.test_vector_db_logic tests.test_enrollment_math tests.test_pad_capture tests.test_capture_domain_faces tests.test_calibrate_domain_threshold tests.test_pad_domain_optimization tests.test_pad_shadow_policy tests.test_pad_target_transforms
# Tắt adaptive để test đúng R7 source contract trong tiến trình riêng:
PAD_ADAPTIVE_ENABLED=false PAD_RUNTIME_CONFIG_PATH=antispoof/models/R7_SC_100K_crop15_runtime_config.json python3 -m unittest tests.test_r7_runtime
```

**4 tests R7 source-contract pass** với override trên. Chi tiết và giới hạn tại [trạng thái PAD](current_pad_state.md). Không chạy toàn bộ suite, training/Kaggle, camera, live cloud hay benchmark Pi trong lượt cập nhật tài liệu. `recognition_submit_mean_ms` là overhead camera-side; latency worker gồm queue/inference/DB.

Lệnh camera để người dùng kiểm tra trên thiết bị, chưa chạy ở đây:

```bash
python3 scripts/list_cameras.py
python3 app.py --camera 0 --mode none
python3 app_capture.py --camera 0 --mode none --subject person01 --condition backlight
```

Capture cần PAD model và DB như app chính. Dùng subject/session riêng cho calibration và held-out; trả lại log cấu hình, metadata/labels, thời gian tới verdict và lỗi theo điều kiện sáng trước khi kết luận chất lượng realtime.
