# face_auth — context codebase hiện tại

Snapshot: **2026-09-28**, checkout `main` tại `add6645`, theo source/cấu hình/artifact trong working tree; không đại diện một release đã khóa. Mục tiêu: giúp agent tìm đúng module và luồng xử lý trước khi sửa. Source, runtime JSON và config thực tế có ưu tiên hơn mô tả này; terminal env/CLI có thể thay đổi snapshot.

## Đọc nhanh trước khi làm việc

1. [AGENTS.md](../AGENTS.md): scope CV/PAD và nguyên tắc nghiên cứu.
2. Tài liệu này: runtime, module ownership, config và trạng thái hiện tại.
3. [Trạng thái PAD](current_pad_state.md) khi task liên quan `antispoof/`: runtime, E1/E2/E3, cross-dataset và micro-search.
4. [Danh mục tài liệu](README.md): chọn đúng contract, audit hoặc protocol cần đọc.
5. Đọc source theo task trong bảng module bên dưới; không cần quét toàn bộ notebook/model/cache.

`face_auth` là prototype xác thực khuôn mặt: detection → tracking → PAD → recognition → pgvector search. Có attendance hooks trong app/DB; không mở rộng logic đó khi task chỉ yêu cầu CV/PAD. Enrollment và nghiên cứu PAD là các đường chạy riêng.

## Bản đồ thư mục

```text
face_auth/
├── app.py                         # Entry point camera, điều phối runtime
├── config.py                      # Env/JSON/defaults, đường dẫn, cadence, thresholds
├── AGENTS.md, .env.example         # Quy tắc và cấu hình mẫu
├── .env                           # Cấu hình máy local, không chép secrets vào tài liệu
├── docker-compose.yml             # PostgreSQL 16 + pgvector, service postgres
├── requirements.txt               # Dependencies runtime (nhiều package chưa pin version)
├── detection/                     # SCRFD wrapper, YunNet wrapper, model YunNet
├── tracking/tracker.py             # Track state, IOU association, LK flow, PAD window/crop state
├── antispoof/
│   ├── predictor.py, preprocess.py # Contract → tensor → ONNX → raw PAD score
│   ├── loader.py, system.py        # ONNX session/providers, hardware descriptions
│   ├── models/                    # E1/E2/E3 deployment, metadata/test trong smartface_pad_artifacts/
│   └── notebooks/                 # EDA, E1/E2/E3, held-out/cross-dataset, diagnostics
├── alignment/aligner.py            # ArcFace alignment, bbox fallback
├── recognition/embedder.py        # ArcFace embedding, L2 normalization
├── database/                      # vector_db.py + schema.sql
├── enrollment/enroll.py            # Outlier filtering + centroid + DB enrollment
├── scripts/                       # Enrollment, demos, detector probe, export, DB utilities
├── evaluation/                    # LFW notebooks + frozen split manifest
├── tests/                         # Offline unit/component regressions
├── utils/image.py                 # Image helpers
├── gallery/                       # Local enrollment images
├── data/, output/                 # Local data and generated reports/logs; not runtime source
├── backups/                       # Historical source snapshots; not active imports
└── md/                            # Context chung/PAD, runtime notes, audits, research, archive
```

## Module ownership và điểm bắt đầu đọc

| Task/module | Source / symbol chính | Trách nhiệm và giới hạn |
|---|---|---|
| Camera / loop / CLI | [app.py](../app.py): `main`, `parse_args` | Load models/DB once, capture, cadence, timers/counters, raw frame vs display copy, cleanup/hotkeys |
| PAD runtime integration | `app.update_track_pad` | Crop due tracks, batch prediction, validate output count/bool/finite score, cập nhật votes; error invalidate verification |
| Recognition orchestration | `app.orchestrate_track_step` | PAD gating → align/embed → search → identity; attendance hooks chỉ khi mode cho phép |
| Config | [config.py](../config.py) | Parse env/JSON, validate threshold pair, resolve model/config paths; không load camera/DB |
| Detector | [detection/detector.py](../detection/detector.py): `FaceDetector.detect` | SCRFD bbox xyxy + confidence + 5 landmarks; min side filter |
| Alternative detector | [detection/yunnet_detector.py](../detection/yunnet_detector.py) | OpenCV FaceDetectorYN, INT8 artifact; chọn qua `APP_DETECTOR` |
| Tracking / state | [tracking/tracker.py](../tracking/tracker.py): `FaceTracker.update`, `_propagate`, `Track` | IOU matching, LK propagation, lost/redetection, cached identity, PAD binary window và crop geometry riêng |
| PAD predictor | [antispoof/predictor.py](../antispoof/predictor.py): `predict_crops`, `process_with_logits` | Spatial E1 hoặc single-input DCT E2, raw logits → d/probabilities/verdict; không quản lý tracking/voting |
| PAD transforms | [antispoof/preprocess.py](../antispoof/preprocess.py): `crop`, `preprocess_batch`, `preprocess_dct_batch` | Expanded square crop, resize/padding, RGB/norm hoặc DCT; không dùng ArcFace alignment cho PAD hiện tại |
| ONNX load | [antispoof/loader.py](../antispoof/loader.py): `load_model` | Graph optimization, sequential execution, provider selection, raise khi artifact/session lỗi |
| PAD status / artifacts | [current_pad_state.md](current_pad_state.md) | E1 runtime contract, E2/E3 và cross-dataset evidence, micro-search roadmap, giới hạn provenance |
| Alignment | [alignment/aligner.py](../alignment/aligner.py): `get_input_face` | 5-point ArcFace affine alignment, fallback bbox crop khi landmarks/transform không dùng được |
| Recognition | [recognition/embedder.py](../recognition/embedder.py): `embed_aligned` | `get_feat` trên aligned BGR, 512-D L2-normalized; không detect lại trong frame loop |
| Database | [database/vector_db.py](../database/vector_db.py), [schema.sql](../database/schema.sql) | psycopg pool, schema, centroid cosine search, upsert, attendance; `decide_identity` áp MATCH_THRESHOLD |
| Enrollment | [enrollment/enroll.py](../enrollment/enroll.py), [scripts/run_enroll.py](../scripts/run_enroll.py) | Gallery images → embeddings → outlier filtering → normalized centroid; lưu SAMPLE/CENTROID theo options |

## Luồng realtime theo source

```text
config + CLI → load detector / ArcFace / DB / tracker / PAD once
camera.read → detector due? → detect + IOU match
                          └→ LK propagate → unsafe? → detect ngay trên cùng frame
tracked bbox → PAD due? → optional crop stabilization → expanded crop
             → preprocess → ONNX batch → raw score → binary rolling votes
PAD REAL + recognition due → alignment → ArcFace → DB search → identity
raw frame.copy → draw states / FPS → display / hotkeys
```

- Detector tick riêng; các frame xen kẽ dùng LK. PAD có thể dùng bbox `DETECTOR` hoặc `TRACKER` và chỉ chạy khi track đến hạn.
- LK dùng bốn góc bbox và tâm; status/finite/geometry/scale checks. Landmarks được biến đổi cùng motion bằng partial affine.
- Geometry cần refresh (scale/biên/corner không an toàn) không được đưa vào inference; app redetect cùng frame. Match hợp lệ giữ window; unmatched reset verification.
- Flow mất thật, invalid geometry/transform hoặc detector unmatched → reset PAD và identity. **Không giữ votes qua flow failure** sau khi đã revert thử nghiệm recovery.
- PAD crop/batch error hoặc output không hợp lệ → invalidate cached verdict/identity; failed attempts vẫn được pace. Recognition error hoặc không tạo được alignment/embedding → invalidate identity và retry theo interval.
- PAD chưa đủ votes hoặc stale → `PAD_PENDING`; SPOOF/PENDING chặn recognition khi PAD bật. Recognition overdue chạy lại theo cadence khi PAD cho phép.
- Inference dùng raw frame; drawing dùng `display_frame` copy, tránh annotation làm bẩn input của track tiếp theo.
- Detector/PAD/recognition/stale timers dùng monotonic. Wall clock vẫn dùng cho log timestamp và DB attendance/cooldown.
- `--mode none` tắt attendance writes, **vẫn cần DB** cho recognition. App không phải PAD-only diagnostic.
- Hotkeys: `q` quit, `p` toggle PAD, `f` FPS. Bật lại PAD invalidates verification. Exit in runtime counters/timings và giải phóng tài nguyên.

## Cấu hình local tại snapshot

`.env` local vẫn trỏ `PAD_RUNTIME_CONFIG_PATH` tới JSON E1 cũ ở `antispoof/models/`, nhưng file này và ONNX cùng thư mục hiện không tồn tại. Nếu không có env override, `config.py` lỗi khi đọc JSON trước khi app khởi động. E1 contract và artifact tương ứng còn ở đường dẫn deployment dưới đây; các giá trị trong bảng là cấu hình dự định, chưa phải một phiên runtime được xác nhận sau pull.

| Nhóm | Cấu hình dự định tại snapshot |
|---|---|
| Detection | `APP_DETECTOR=scrfd`, pack `buffalo_s`, CPU, input 640×640, confidence .5, min side 60 px |
| Camera | `CAMERA_FORCE_RESOLUTION=true`, yêu cầu 640×480@30, buffer 1 best-effort; phải xem actual startup properties |
| PAD profile | `.env` trỏ path cũ bị thiếu; JSON còn tại `antispoof/models/smartface_pad_artifacts/E1_v5_3_mnv3_small/deployment/` |
| PAD model | MobileNetV3-Small E1 v5.3, ONNX + `.onnx.data` trong cùng thư mục deployment |
| Input contract | 224×224 RGB, crop factor 1.55, gamma OFF, mean `[.5931,.469,.4229]`, std `[.2471,.2214,.2157]` |
| PAD threshold | d = −0.4650222063064575; equivalent softmax p_real = .38579509526467753 |
| Temporal | Binary window 5, min votes 5, spoof ratio .6, stale timeout 3s |
| Cadence | Detector .1s, PAD .1s, recognition .5s; independent schedules |
| Crop stabilization | `PAD_CROP_SMOOTHING=true` local; code default/.env.example false |
| Identity | ArcFace `buffalo_s`, 512-D; `MATCH_THRESHOLD=.35`, model version `buffalo_s` |

### Configuration và model contracts

- `load_dotenv()` nạp .env; environment terminal có ưu tiên hơn .env. Không dump DB credentials.
- Với contract PAD: nonblank `PAD_*` env > trường JSON > code defaults. Blank env không override JSON. Config lỗi dừng thay vì đoán contract.
- Runtime config path tương đối theo project root; `model_file` trong JSON tương đối theo directory của JSON khi không override filename.
- `.onnx.data` là một phần artifact, phải đi cùng model. Không chỉ copy `.onnx`.
- `d = real_logit − logsumexp(spoof_logits)`; class 0 REAL, các lớp còn lại spoof. REAL khi d ≥ threshold; binary voting sau đó là quyết định theo thời gian khác raw predictor.
- CLI `--pad-model` được guard để không ghép model khác với contract đã chọn; đổi model bằng runtime JSON. Alternate v5.1 config có trong `antispoof/models/smartface_pad_artifacts/E1_v5_1_mnv3/deployment/` (crop 1.50, threshold riêng); chưa được chọn.
- `.35` là recognition threshold được người dùng chủ động chọn, không phải PAD threshold. Không âm thầm đổi hai ngưỡng.
- Crop PAD từ raw bbox: square max side × expansion, reflect padding; RGB conversion, AREA downscale / LANCZOS4 upscale, reflect letterbox, float32 CHW /255 rồi mean/std.
- Loader PAD ưu tiên CUDA rồi CPU khi available; detector/ArcFace wrappers hiện ép CPU. Có CoreML available không đồng nghĩa đang dùng CoreML.
- Embedder vẫn khởi tạo detection submodel do ràng buộc FaceAnalysis, nhưng frame embedding chỉ gọi recognition `get_feat`; không suy rằng RAM chỉ chứa ArcFace.

## Crop smoothing được giữ lại

`Track.pad_crop_bbox` giữ center/side riêng cho PAD, không sửa tracking bbox hay landmarks. Chỉ chạy ở các lượt PAD đến hạn. Time constant .12s; chỉ smooth jitter ≤5%; snap khi motion/scale lớn, gap >.5s hoặc expanded crop ra ngoài frame. Reset PAD cũng xóa crop history. Phụ thuộc thời gian monotonic để giảm ảnh hưởng FPS.

Người dùng báo kết quả realtime ổn hơn; đây là phản hồi sử dụng, chưa có camera benchmark có nhãn/edge measurement xác nhận accuracy. Xem [contract thử nghiệm](runtime/pad_crop_smoothing_experiment.md).

**Các thử nghiệm đã revert hoàn toàn:** detector-only PAD, same-frame flow-vote recovery, mean-score temporal aggregation. Không còn các flags `PAD_DETECTOR_ONLY`, `PAD_RECOVER_FLOW_VOTES`, `PAD_SCORE_SMOOTHING` trong code/config hiện tại. Không áp lại ngầm khi làm task khác.

## Database / enrollment

Schema dùng `employees`, `face_embeddings`, `attendance_logs`. Embeddings là `vector(512)` với `embedding_type` `SAMPLE` hoặc `CENTROID`; không dùng schema `users`/`is_mean` mô tả trong docs cũ.

`VectorDB.search`: lọc employee ACTIVE, CENTROID và model_version; similarity = 1 − cosine distance (`<=>`), top_k=5 từ runtime. Enrollment loại outlier theo cosine với preliminary mean (default .35), tính centroid rồi normalize lại. Nếu tất cả bị loại, source có fallback giữ toàn bộ và cảnh báo. Đây là threshold enrollment riêng.

`VectorDB.init_schema` có migration cleanup với legacy tables; các scripts reset/delete là thao tác dữ liệu, không phải bước bắt buộc để đọc context hoặc chạy diagnostic. Không chạy reset DB như một preflight mặc định.

## Research và artifacts

- [Trạng thái PAD chi tiết](current_pad_state.md) có đường dẫn, số liệu và giới hạn bằng chứng từng nhánh. E3 đã có checkpoint/ONNX, held-out Test và LCC cross-dataset result; M0–M4 vẫn là kế hoạch.
- [Danh mục notebooks](../antispoof/notebooks/README.md) là điểm bắt đầu; source folders thực tế có E1 v0/v5.1/v5.2/v5.3, E2 DCT, E3 concat, cross-dataset, EDA và validation.
- `e1_spatial_v5_3/`: training + held-out E1 Small. `e2_frequency_dct/`: train/held-out E2, frozen protocol/manifest checks.
- `e3_spatial_frequency_concat/` và `cross_dataset/` có notebook đã lưu output; artifact E3 held-out và LCC E1-vs-E3 nằm trong `antispoof/models/smartface_pad_artifacts/`. CASIA/Replay chưa hoàn tất theo cross-dataset summary.
- E3 có hai inputs RGB/frequency và không cắm trực tiếp vào predictor hiện tại (single-input spatial/DCT). Cần runtime integration riêng nếu deploy E3.
- Protocol/meta và manifest local nằm trong `antispoof/models/smartface_pad_artifacts/`; E1 NPZ có trên máy nhưng bị `.gitignore` loại khỏi Git. Không tái tạo split từ cache nếu thiếu manifest trên máy khác.
- LFW evaluation ở `evaluation/`, độc lập PAD research. Không gộp recognition threshold validation với PAD calibration.
- E1/E2 face-size audit cho thấy augmentation resolution degradation chưa có trong các training paths đã audit; không mặc định E3 robust hơn nếu chưa đo.
- [Prompt E3/cross-dataset](../CODEX_PROMPT_E3_AND_CROSS_DATASET_V2_KAGGLE_FREE.md) mô tả workflow tương lai; không phải lệnh để agent tự chạy training.

## Lệnh và validation tối thiểu

Chạy từ root `face_auth`; macOS dùng `python3`. Cần dependencies/models sẵn có; camera do người dùng chạy.

```bash
# .env hiện trỏ JSON đã thiếu; chọn contract E1 còn trong deployment
export PAD_RUNTIME_CONFIG_PATH=antispoof/models/smartface_pad_artifacts/E1_v5_3_mnv3_small/deployment/mnv3s_e1_preliminary_v5_3_edge_runtime_config.json
# Service DB (nếu cần runtime recognition)
docker compose up -d postgres
# Runtime không ghi attendance
python3 app.py --mode none
# Camera PAD logs
mkdir -p output
PAD_DIAGNOSTIC_LOG=true python3 -u app.py --mode none 2>&1 | tee output/pad_camera_diagnostic.log
# Tắt riêng crop smoothing để so baseline
PAD_CROP_SMOOTHING=false python3 app.py --mode none
# Offline runtime/tracker regressions
python3 -m unittest tests.test_tracker tests.test_runtime
# Offline image sensitivity: không có labels, không phải accuracy benchmark
python3 antispoof/notebooks/diagnostics/analyze_pad_input_sensitivity.py --images gallery/*/*.jpg --output output/pad_input_sensitivity.json
```

`tee` ghi đè file cùng tên; dùng tên riêng cho từng treatment/baseline. Runtime/tracker regressions gần nhất: 25 tests pass sau revert mean-score; đó là offline evidence, không chứng minh camera/DB/edge behavior. Tests khác: PAD predictor, detection, enrollment math, vector DB logic; chọn theo code thay đổi.

## Rủi ro / bẫy agent cần tránh

- Không gọi docs là nguồn chuẩn tuyệt đối; archive và walkthrough cũ chứa model/schema/logic lỗi thời.
- Window theo số samples; time-based cadence không bảo đảm cùng time-to-verdict giữa FPS/thiết bị. Camera request cũng không bảo đảm actual mode/buffering giống nhau.
- Log cũ cho nhiều raw SPOOF khi mặt nhỏ; thiếu labels/controlled input nên chưa kết luận nguyên nhân model hay runtime. Ratio/smoothing không sửa được việc thiếu texture thực tế.
- Geometry/IOU không chứng minh danh tính; không giữ cached REAL qua mất continuity để giảm PENDING.
- Biến margin/NMS/top-k của YunNet không mặc định tác động SCRFD wrapper.
- Thread budgets/providers/dependency versions chưa đủ khóa để tuyên bố reproducible latency hoặc output trên mọi thiết bị.
- Giữ model contracts, manifests/seeds/splits và validation/Test boundary; thay preprocessing hoặc aggregation là treatment cần kiểm chứng riêng, không tự báo cải thiện accuracy.
- Khi task yêu cầu tối ưu edge: đo full pipeline capture/tracking/crop/inference/display, không chỉ ONNX latency. Không chạy camera/training đắt tiền trước offline checks.
