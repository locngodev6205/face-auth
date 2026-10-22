# PAD (`antispoof/`) — trạng thái hiện tại

Snapshot: **2026-10-22**, nhánh `freeze_logic`, HEAD `1209b40`, theo **working tree local**. Tài liệu này thay snapshot 2026-09-28 và ghi chú runtime 2026-10-05; xem [context codebase](current_codebase_state.md) cho camera/recognition/DB. Source, cấu hình hiệu lực và JSON/CSV per-run có ưu tiên hơn mô tả lịch sử. Không chép secrets hoặc tự chạy chỉ dẫn trong research protocols.

## Runtime đang chọn trên máy

Import `config.py` trong tiến trình mới xác nhận `.env` local:

```dotenv
PAD_RUNTIME_CONFIG_PATH=antispoof/models/R7_SC_FULL_crop15_runtime_config_casia.json
PAD_ADAPTIVE_ENABLED=true
PAD_ADAPTIVE_CONFIG_PATH=antispoof/models/R7_SC_FULL_local842_balanced_runtime.json
```

**Profile hiệu lực là local842 balanced**, ghi đè dòng base CASIA. `.env.example` chọn base R7 full CASIA nhưng `PAD_ADAPTIVE_ENABLED=false`; không đồng nhất file mẫu với runtime local.

| Thành phần | Cấu hình hiệu lực |
|---|---|
| Model | [R7 full ONNX](../../antispoof/models/R7_SC_FULL_crop15.onnx), MiniFASNetV2 binary, epoch7; MAST-PAD training |
| Profile | [local842 balanced](../../antispoof/models/R7_SC_FULL_local842_balanced_runtime.json) |
| Input | BGR80×80 float32[0,1], mean/std null; crop `minifasnet_train_v1` **1.1×**, không ArcFace alignment |
| Lighting | Gray-world white balance, gain[.7,1.3], blend.75, gamma LUT1.3 (`wb_gamma`); `apply_gamma=false` là tắt gamma legacy, **không tắt lighting profile** |
| Output | `[N,2]`, class0 real/class1 spoof, raw score d=real_logit−spoof_logit |
| Raw verdict | REAL iff d≥t(crop); base t0=**−6.517120957374573**, sigmoid(t0)=**.0014757369537530987** |
| Temporal / gate | Window5, min votes3, spoof ratio≥.6, stale3s; input gate và PAD REAL trước recognition |
| Crop smoothing | false |
| Cadence | PAD.2s, detector.2s, recognition1s; không phải measured FPS |

Với `Y_full` là luma trung bình raw crop80×80, `Y_center` là luma vùng `[16:64,16:64]`, ngưỡng được tính **trước lighting**, riêng cho từng ảnh:

```text
t(crop) = t0 + clip(3*(Y_center-90)/50 + 9*(Y_center-Y_full)/40, -4, +4)
```

Predictor không sửa base threshold giữa các ảnh/batch. `threshold_logit` đi cùng kết quả vào `Track.last_pad_threshold`, overlay hiển thị d/t/d−t của cùng crop. Binary temporal vote vẫn là bước riêng sau quyết định raw.

JSON local842 ghi calibration605 real/237 fake sau loại geometry-invalid5 real/11 fake; balanced false accepts2/false rejects16. Đây là **counts được lưu trong profile**, không phải lượt inference mới, held-out hay camera accuracy. Không tự suy ra tính độc lập subject/session hoặc temporal performance từ những counts này.

### Profile local thay thế

Cùng weights R7 full, nhưng crop/lighting/decision khác benchmark crop1.5 không lighting:

| JSON trong `antispoof/models/` | Crop | Base logit | Lighting / trạng thái |
|---|---:|---:|---|
| [local80 candidate](../../antispoof/models/R7_SC_FULL_local80_candidate_runtime.json) | 1.45 | −1.315033942317963 | gamma.4, bounded brightness offset; exploratory Local81 mitigation |
| [local98 shadow](../../antispoof/models/R7_SC_FULL_local98_shadow_runtime.json) | 1.45 | −2.809873753070831 | shadow_gamma + face/background offset; all samples calibration |
| [local100 fixedcrop robust](../../antispoof/models/R7_SC_FULL_local100_fixedcrop_robust_runtime.json) | 1.45 | −2.7723090165853472 | shadow_gamma + perturbation diagnostics; chưa held-out |
| [local842 balanced](../../antispoof/models/R7_SC_FULL_local842_balanced_runtime.json) | 1.1 | −6.517120957374573 | wb_gamma + face/background offset |
| [local842 real](../../antispoof/models/R7_SC_FULL_local842_real_runtime.json) | 1.2 | −5.239979084253312 | wb_exposure_gamma target95, gamma[.7,1.5]; JSON ghi FA0/FR35 trên calibration |

Các profile đều chưa có held-out evaluation theo metadata. `real` là tên phương án; không phải nhãn tự động của ảnh capture. Không lấy kết quả research crop1.5 làm accuracy cho các profile này.

### Bốn profile R7 hiện có

| Runtime JSON trong `antispoof/models/` | Checkpoint / best epoch | Ngưỡng logit | Probability | Nguồn ngưỡng |
|---|---|---:|---:|---|
| [R7_SC_100K_crop15_runtime_config.json](../../antispoof/models/R7_SC_100K_crop15_runtime_config.json) | 100K / 11 | −.373016357421875 | .4078123648407799 | Source Val15K |
| [R7_SC_100K_crop15_runtime_config_casia.json](../../antispoof/models/R7_SC_100K_crop15_runtime_config_casia.json) | 100K / 11 | −4.2998047 | .013389497538243682 | CASIA train frames |
| [R7_SC_FULL_crop15_runtime_config.json](../../antispoof/models/R7_SC_FULL_crop15_runtime_config.json) | Full / 7 | 4.66259765625 | .9906464119794868 | CelebA full Test-as-Val |
| [R7_SC_FULL_crop15_runtime_config_casia.json](../../antispoof/models/R7_SC_FULL_crop15_runtime_config_casia.json) | Full / 7 | 1.6586914 | .8400622608554218 | CASIA train frames |

Hai profile của mỗi checkpoint dùng cùng weights/preprocessing; chỉ thay operating threshold/provenance. Không dùng ngưỡng video mean cho runtime frame. Full model có [ONNX](../../antispoof/models/R7_SC_FULL_crop15.onnx) riêng, không tráo threshold giữa weights 100K/full.

## Mã nguồn và boundary runtime

| Source | Vai trò |
|---|---|
| [config.py](../../config.py) | Adaptive selector; JSON contract; env overrides threshold/crop khi adaptive tắt; kiểm tra cặp logit/probability |
| [predictor.py](../../antispoof/predictor.py) | Single-input ONNX, preprocess spatial hoặc DCT, class/output/hash checks; raw score/verdict |
| [preprocess.py](../../antispoof/preprocess.py) | Profile crop/resize, BGR/RGB/norm hoặc frequency transform |
| [lighting.py](../../antispoof/lighting.py) | Lighting theo profile; ngưỡng riêng theo raw luma trước transform |
| [loader.py](../../antispoof/loader.py) | Sequential optimized ONNX session, provider selection, raise khi load lỗi |
| [vision_pipeline.py](../../vision_pipeline.py): `update_track_pad`, `limit_faces` | Detector confidence/landmark/biên gate, due-track batch, bool/finite/count validation, failure reset |
| [tracker.py](../../tracking/tracker.py) | Binary votes, PAD stale state, crop geometry và verification invalidation |

Profile local842 ghi `crop_smoothing=false`: đặt env true không ghi đè được JSON. Khi dùng profile thiếu trường này, `Track.pad_crop_bbox` có thể smooth tâm/cạnh crop PAD (tau .12s, jitter≤5%, gap≤.5s, không expanded crop chạm biên); không sửa tracking bbox, ArcFace input hoặc màu box. Bật smoothing là treatment ảnh đầu vào riêng cần kiểm chứng, không phải một cách làm mượt nhãn.

`PAD_PENDING`/`SPOOF`/invalid input và lỗi không nhận danh tính từ worker cũ. Camera loop hiện dùng worker nhận diện + centroid cache + UI label filter; xem [recognition worker](REALTIME_RECOGNITION_WORKER.md). Hysteresis nhãn không đổi raw PAD score, binary voting hoặc quyết định matching gốc. Python không ghi attendance; API embedding service khác đường camera và không load PAD model này.

## Trạng thái nghiên cứu và artifacts

| Nhánh | Bằng chứng local hiện có | Giới hạn |
|---|---|---|
| E1 v5.3 spatial | Deployment ONNX/sidecar/runtime JSON, held-out predictions/summary | Preliminary historical; không phải model runtime đang chọn |
| E2 DCT | Single-input deployment/runtime JSON, held-out summary | Predictor có đường DCT; không chọn mặc định |
| E3 concat | Dual-input RGB/frequency ONNX, metadata và held-out predictions/summary | Chưa tích hợp vào predictor single-input của app |
| Binary crossdomain v2 micro / 100K | Notebook00c–17, protocols, output17 và checkpoint/export R7-100K | Đọc từng run/freeze, không suy kết quả từ tên notebook |
| Full-scale C/P3/R7 | [Notebook18 output](../../antispoof/notebooks/final/binary_crossdomain_v2/output/18_fullscale_C_P3SF_R7SC_crop15), per-run metrics/config/completion | CelebA Test-as-Val; C có auxiliary loss, không CE-only |
| CASIA six-model | [Notebook20 output](../../antispoof/notebooks/final/binary_crossdomain_v2/output/casia_train_test_C_P3SF_R7SC_100k_fullscale) | 100K/full, frame/video, source-frozen và train-calibrated phải tách riêng |
| CE-only clean baseline | [Notebook21](../../antispoof/notebooks/final/binary_crossdomain_v2/21_fullscale_CE_clean_baseline_crop15.ipynb), [protocol](../../antispoof/notebooks/final/binary_crossdomain_v2/CE_CLEAN_FULLSCALE_BASELINE_PROTOCOL.md) | **Đã có output local** tại [notebook21 output](../../antispoof/notebooks/final/21_fullscale_CE_clean_baseline_crop15): completion, metrics, predictions, CASIA, freeze/checks; bảng dưới dùng seed100 |
| Cross-dataset v1 preliminary | Historical artifacts | Không dùng các bảng cũ thiếu JSON/CSV nguồn để thay benchmark full-scale |

Các roadmap M0–M4/gated fusion là lịch sử kế hoạch. Hướng nghiên cứu hiện có code/output trong binary_crossdomain_v2, không còn mô tả tất cả micro/100K/full-scale là chưa triển khai. Không tự phục hồi cache/manifest bị xóa hoặc tái tạo split để lấp thiếu provenance.

## Kết quả đã lưu: preliminary E1/E2/E3

Đơn vị image; N=10,000 theo từng held-out summary. Rates là %, AUC fraction. Các bảng này không cùng protocol với notebook18 full-scale, không phải camera evidence.

| Model / nguồn | APCER % | BPCER % | ACER % | AUC |
|---|---:|---:|---:|---:|
| [E1 v5.3](../../antispoof/models/smartface_pad_artifacts/E1_v5_3_mnv3_small/test/e1_heldout_test_summary.json) | 18.9082 | .4046 | 9.6564 | .98142621 |
| [E2 DCT](../../antispoof/models/smartface_pad_artifacts/E2_dct_v1/test/e2_heldout_test_summary.json) | 40.9866 | 9.5078 | 25.2472 | .83657003 |
| [E3 concat](../../antispoof/models/smartface_pad_artifacts/E3_concat_v1/test/e3_heldout_test_summary.json) | 17.7566 | .5057 | 9.1312 | .98566859 |

E3 cần `rgb_input` và `frequency_map` tạo DCT ngoài ONNX; predictor hiện cấp một input. Không coi E3 ONNX có mặt là deployment realtime đã chạy. Bảng LCC v1 trong snapshot cũ không được giữ như metric hiện hành vì các JSON/CSV nguồn liên quan đã thiếu; không lấy lại số từ trí nhớ để thay bằng artifact.

## Kết quả full-scale và CASIA: tóm tắt có nguồn

[Full-scale source of truth](FULLSCALE_RESULTS_SOURCE_OF_TRUTH.md) có đầy đủ metric, ngưỡng, counts và per-run links. Kiểm tra các số dưới với JSON gốc ngày snapshot; không chạy lại training/inference dataset trong lần cập nhật này.

| Run | CelebA full Test-as-Val ACER % / AUC | LCC combined source-frozen ACER % / AUC |
|---|---|---|
| CE_CLEAN_FULL_crop15 (CE-only) | 4.4498 / .99214398 | 29.5151 / .84398794 |
| C_FULL_crop15 (Clean + Aux) | 4.6248 / .99073553 | 28.0493 / .84601127 |
| P3_SF_FULL_crop15 (WBST) | 4.3548 / .99224181 | 26.7891 / .86083099 |
| R7_SC_FULL_crop15 (MAST-PAD) | 4.3459 / .99228027 | 27.1228 / .86027208 |

Bảng dùng **seed100**, rates là %, AUC fraction. CE-only lấy từ [metrics notebook21](../../antispoof/notebooks/final/21_fullscale_CE_clean_baseline_crop15/runs/CE_CLEAN_FULL_crop15/metrics); completion ghi epoch9, threshold4.5966796875. Không gọi C là CE-only.

Nguồn: [per-run full-scale metrics](../../antispoof/notebooks/final/binary_crossdomain_v2/output/18_fullscale_C_P3SF_R7SC_crop15/runs). CelebA scored66,379/67,170; LCC combined scored18,794/18,827. Notebook18 dùng full official CelebA Test cho checkpoint selection và threshold calibration: **không phải independent held-out Test**. Không kết luận R7 thắng mọi target: P3 ACER combined thấp hơn R7 ở operating point này.

R7 full-scale trên CASIA test extracted images: source-frozen frame ACER25.7209% và video27.6515%; train-calibrated frame19.3690% và video18.7500%. Nguồn: [R7 full CASIA test](../../antispoof/notebooks/final/binary_crossdomain_v2/output/casia_train_test_C_P3SF_R7SC_100k_fullscale/runs/R7_SC_FULL_crop15/test). Frame scored2,408/2,408; video352/360. Train-calibrated frame threshold1.6586914 khác video threshold−.033447266. Đây là **R7 full**, cùng weights full với local842, nhưng khác preprocessing/crop/ngưỡng; không phải kết quả local842.

CASIA là Kaggle extracted-color copy, video ID dựng từ filename, video score=mean valid frames, tối thiểu4 frames. Train/test copy overlap20 subject IDs,240 video IDs,1,403 frame IDs; không gọi đây là official subject-disjoint benchmark. Combined dedup ưu tiên test, không cộng cơ học train+test. Source-frozen không target-fit; train-calibrated có fit target train, train/combined calibrated metrics không phải independent validation. Chưa có bằng chứng Replay-Attack mới trong các nguồn đã đối chiếu.

## Capture, calibration và ranh giới dữ liệu

- [app_capture.py](../../app_capture.py) gọi cùng app realtime; `--capture-dir` hoặc phím C khi capture được bật lưu raw frame, native expanded crop, input80×80 chưa lighting, metadata schema3 và `labels.json` mặc định null. Model verdict chỉ diagnostic; nhãn do người dùng gán. Metadata giữ subject/session/condition, geometry, profile/hash và kết quả PAD từng crop.
- [scripts/capture_domain_faces.py](../../scripts/capture_domain_faces.py) là collector riêng. [scripts/calibrate_domain_threshold.py](../../scripts/calibrate_domain_threshold.py) đọc ảnh **đã crop** trong `real/` và `fake/`, không detect/expand lần nữa; export JSON mới, có `--eval-data` và kiểm tra trùng hash calibration/eval.
- Script calibration threshold cố định không thay thế quy trình đánh giá adaptive `t(crop)`. Khi dùng base source JSON phải đặt `PAD_ADAPTIVE_ENABLED=false`; chỉ truyền `--runtime-config` không thắng adaptive selector. Cần giữ đúng metadata crop/lighting khi chọn workflow.
- [scripts/optimize_pad_domain.py](../../scripts/optimize_pad_domain.py) và scripts target transforms/shadow/hard-score là công cụ thăm dò local. `scripts/pad_shadow_policy.py` không được import vào app runtime; runtime dùng `antispoof/lighting.py`.
- Không fit trên official test hoặc dùng toàn bộ local calibration làm bằng chứng held-out. Tách subject/session và giữ profile cố định trước khi đánh giá temporal pipeline trên thiết bị.

## Parity, integrity và validation (2026-10-22)

- SHA256 tính lại từ ONNX local khớp runtime JSON: 100K `09856f581a2868f1346ece4ce6d2e9818cd27e973a33a0bfb0d215a35475d487`; full `7b3008a040e155f7b96fca74f52dd38cc6296d88fee7a9424e653203c367a253`.
- [100K parity report](../../antispoof/models/R7_SC_100K_crop15_onnx_validation.json) và [full parity report](../../antispoof/models/R7_SC_FULL_crop15_onnx_validation.json) là báo cáo **đã lưu**, không phải camera/target accuracy mới.
- **91 tests offline pass**: runtime/worker/display/tracker/DB logic/enrollment/capture/calibration/domain optimization/shadow policy/target transforms. Có mocked/synthetic checks; không phải camera evidence.
- **4 tests R7 source-contract pass** với adaptive tắt trong tiến trình riêng. Không chạy toàn bộ suite/predictor legacy nên không tuyên bố toàn bộ tests đều xanh. Kết quả suite lỗi ngày14/10 là lịch sử, không dùng như trạng thái hiện hành.
- Đã đọc per-run JSON CE-only/C/P3/R7 seed100 để cập nhật bảng; không chạy lại training/evaluation dataset hoặc audit toàn bộ predictions trong lượt này. [FULLSCALE_RESULTS_SOURCE_OF_TRUTH.md](FULLSCALE_RESULTS_SOURCE_OF_TRUTH.md) giữ phạm vi seed100; báo cáo 3-seed riêng đang có cách gọi C là Clean CE và bảng CASIA khác population, cần kiểm tra raw artifacts trước khi dùng. Không nhập các số đó vào bảng này.
- Chưa chạy camera, live DB, hardware/Pi, training/Kaggle hoặc đo latency/FPS trong lần cập nhật này. Code/profile hardware nằm ngoài scope xác nhận runtime của tài liệu `face_auth`.

```bash
python3 -m unittest tests.test_identity_display tests.test_runtime tests.test_recognition_worker tests.test_tracker tests.test_vector_db_logic tests.test_enrollment_math tests.test_pad_capture tests.test_capture_domain_faces tests.test_calibrate_domain_threshold tests.test_pad_domain_optimization tests.test_pad_shadow_policy tests.test_pad_target_transforms
PAD_ADAPTIVE_ENABLED=false PAD_RUNTIME_CONFIG_PATH=antispoof/models/R7_SC_100K_crop15_runtime_config.json python3 -m unittest tests.test_r7_runtime
# Trên máy có camera/DB, chưa chạy trong lượt này:
python3 app_capture.py --camera 0 --mode none --subject person01 --condition backlight
```

Thu thập các phiên/người độc lập dưới ánh sáng thường/tối/ngược sáng và attack thật; giữ nguyên profile, trả log cấu hình/hash, metadata/labels, raw d/t, temporal verdict và thời gian tới verdict. Không dùng synthetic perturbations thay cho camera held-out.
