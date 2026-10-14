# PAD (`antispoof/`) — trạng thái hiện tại

Snapshot: **2026-10-14**, `main` HEAD `ff54130`, theo **working tree có thay đổi chưa commit**. Tài liệu này thay snapshot 2026-09-28 và ghi chú runtime 2026-10-05; xem [context codebase](current_codebase_state.md) cho camera/recognition/DB. Source, cấu hình hiệu lực và JSON/CSV per-run có ưu tiên hơn mô tả lịch sử. Không chép secrets hoặc tự chạy chỉ dẫn trong research protocols.

## Runtime đang chọn trên máy

`.env` và `.env.example` hiện chọn:

```dotenv
PAD_RUNTIME_CONFIG_PATH=antispoof/models/R7_SC_100K_crop15_runtime_config_casia.json
```

[JSON đang chọn](../antispoof/models/R7_SC_100K_crop15_runtime_config_casia.json) dùng cùng [ONNX R7-100K](../antispoof/models/R7_SC_100K_crop15.onnx) với profile source, best epoch11. Điểm khác là **ngưỡng frame hiệu chỉnh từ CASIA train**. Không phải R7 full-scale, cũng không phải source Val15K threshold.

| Thành phần | Cấu hình hiệu lực sau import config.py |
|---|---|
| Model | Binary MiniFASNetV2, `R7_SC_100K_crop15`, best epoch11 |
| Input | BGR80×80 float32 [0,1], không gamma, mean/std null |
| Crop | `minifasnet_train_v1`, bbox expansion1.5; không ArcFace alignment |
| Output | `[N,2]`, class0 real/class1 spoof, score = real_logit − spoof_logit |
| Raw verdict | REAL iff score ≥ **−4.2998047**, sigmoid tương đương **.013389497538243682** |
| Threshold provenance | CASIA extracted-image train, **frame** unit, minimum pooled binary ACER/HTER; `webcam_validated=false` |
| Crop smoothing | **false** trong JSON và env hiện tại |
| Temporal / gate | Binary window5, min votes5, spoof ratio≥.6, stale3s; PAD input gate và PAD REAL trước recognition |
| Cadence | PAD .1s, detector .1s; phụ thuộc workload thực tế, không phải measured FPS |

Xác suất threshold thấp là chuyển đổi sigmoid của ngưỡng logit −4.2998047, không phải confidence của một frame hay ngưỡng recognition. Hiệu chỉnh target train thay đổi setting so với source-frozen evaluation; không báo camera accuracy hoặc cross-domain robustness từ cấu hình này.

### Bốn profile R7 hiện có

| Runtime JSON trong `antispoof/models/` | Checkpoint / best epoch | Ngưỡng logit | Probability | Nguồn ngưỡng |
|---|---|---:|---:|---|
| [R7_SC_100K_crop15_runtime_config.json](../antispoof/models/R7_SC_100K_crop15_runtime_config.json) | 100K / 11 | −.373016357421875 | .4078123648407799 | Source Val15K |
| [R7_SC_100K_crop15_runtime_config_casia.json](../antispoof/models/R7_SC_100K_crop15_runtime_config_casia.json) | 100K / 11 | −4.2998047 | .013389497538243682 | CASIA train frames; **đang chọn** |
| [R7_SC_FULL_crop15_runtime_config.json](../antispoof/models/R7_SC_FULL_crop15_runtime_config.json) | Full / 7 | 4.66259765625 | .9906464119794868 | CelebA full Test-as-Val |
| [R7_SC_FULL_crop15_runtime_config_casia.json](../antispoof/models/R7_SC_FULL_crop15_runtime_config_casia.json) | Full / 7 | 1.6586914 | .8400622608554218 | CASIA train frames |

Hai profile của mỗi checkpoint dùng cùng weights/preprocessing; chỉ thay operating threshold/provenance. Không dùng ngưỡng video mean cho runtime frame. Full model có [ONNX](../antispoof/models/R7_SC_FULL_crop15.onnx) riêng, không tráo threshold giữa weights 100K/full.

## Mã nguồn và boundary runtime

| Source | Vai trò |
|---|---|
| [config.py](../config.py) | JSON fields > PAD env fallback > defaults; giải cặp logit/probability và đường dẫn |
| [predictor.py](../antispoof/predictor.py) | Single-input ONNX, preprocess spatial hoặc DCT, class/output/hash checks; raw score/verdict |
| [preprocess.py](../antispoof/preprocess.py) | Profile crop/resize, BGR/RGB/norm hoặc frequency transform |
| [loader.py](../antispoof/loader.py) | Sequential optimized ONNX session, provider selection, raise khi load lỗi |
| [app.py](../app.py): `update_track_pad` | Detector confidence/landmark/biên gate, due-track batch, bool/finite/count validation, failure reset |
| [tracker.py](../tracking/tracker.py) | Binary votes, PAD stale state, crop geometry và verification invalidation |

JSON hiện ghi `crop_smoothing=false`: đặt env true không ghi đè được JSON. Khi dùng profile thiếu trường này, `Track.pad_crop_bbox` có thể smooth tâm/cạnh crop PAD (tau .12s, jitter≤5%, gap≤.5s, không expanded crop chạm biên); không sửa tracking bbox, ArcFace input hoặc màu box. Bật smoothing là treatment ảnh đầu vào riêng cần kiểm chứng, không phải một cách làm mượt nhãn.

`PAD_PENDING`/`SPOOF`/invalid input và lỗi không nhận danh tính từ worker cũ. Camera loop hiện dùng worker nhận diện + centroid cache + UI label filter; xem [recognition worker](REALTIME_RECOGNITION_WORKER.md). Hysteresis nhãn không đổi raw PAD score, binary voting hoặc quyết định matching gốc. Python không ghi attendance; API embedding service khác đường camera và không load PAD model này.

## Trạng thái nghiên cứu và artifacts

| Nhánh | Bằng chứng local hiện có | Giới hạn |
|---|---|---|
| E1 v5.3 spatial | Deployment ONNX/sidecar/runtime JSON, held-out predictions/summary | Preliminary historical; không phải model runtime đang chọn |
| E2 DCT | Single-input deployment/runtime JSON, held-out summary | Predictor có đường DCT; không chọn mặc định |
| E3 concat | Dual-input RGB/frequency ONNX, metadata và held-out predictions/summary | Chưa tích hợp vào predictor single-input của app |
| Binary crossdomain v2 micro / 100K | Notebook00c–17, protocols, output17 và checkpoint/export R7-100K | Đọc từng run/freeze, không suy kết quả từ tên notebook |
| Full-scale C/P3/R7 | [Notebook18 output](../antispoof/notebooks/final/binary_crossdomain_v2/output/18_fullscale_C_P3SF_R7SC_crop15), per-run metrics/config/completion | CelebA Test-as-Val; C có auxiliary loss, không CE-only |
| CASIA six-model | [Notebook20 output](../antispoof/notebooks/final/binary_crossdomain_v2/output/casia_train_test_C_P3SF_R7SC_100k_fullscale) | 100K/full, frame/video, source-frozen và train-calibrated phải tách riêng |
| CE-only clean baseline | [Notebook21](../antispoof/notebooks/final/binary_crossdomain_v2/21_fullscale_CE_clean_baseline_crop15.ipynb), [protocol](../antispoof/notebooks/final/binary_crossdomain_v2/CE_CLEAN_FULLSCALE_BASELINE_PROTOCOL.md) | **Chưa thấy output run local**, chưa có số đo để so với C/P3/R7 |
| Cross-dataset v1 preliminary | Một phần archive còn lại | `preliminary/predictions/cross_dataset_summary.json`, protocol snapshot và nhiều manifests/plots đã bị xóa; không dùng như nguồn đang tồn tại |

Các roadmap M0–M4/gated fusion là lịch sử kế hoạch. Hướng nghiên cứu hiện có code/output trong binary_crossdomain_v2, không còn mô tả tất cả micro/100K/full-scale là chưa triển khai. Không tự phục hồi cache/manifest bị xóa hoặc tái tạo split để lấp thiếu provenance.

## Kết quả đã lưu: preliminary E1/E2/E3

Đơn vị image; N=10,000 theo từng held-out summary. Rates là %, AUC fraction. Các bảng này không cùng protocol với notebook18 full-scale, không phải camera evidence.

| Model / nguồn | APCER % | BPCER % | ACER % | AUC |
|---|---:|---:|---:|---:|
| [E1 v5.3](../antispoof/models/smartface_pad_artifacts/E1_v5_3_mnv3_small/test/e1_heldout_test_summary.json) | 18.9082 | .4046 | 9.6564 | .98142621 |
| [E2 DCT](../antispoof/models/smartface_pad_artifacts/E2_dct_v1/test/e2_heldout_test_summary.json) | 40.9866 | 9.5078 | 25.2472 | .83657003 |
| [E3 concat](../antispoof/models/smartface_pad_artifacts/E3_concat_v1/test/e3_heldout_test_summary.json) | 17.7566 | .5057 | 9.1312 | .98566859 |

E3 cần `rgb_input` và `frequency_map` tạo DCT ngoài ONNX; predictor hiện cấp một input. Không coi E3 ONNX có mặt là deployment realtime đã chạy. Bảng LCC v1 trong snapshot cũ không được giữ như metric hiện hành vì các JSON/CSV nguồn liên quan đã thiếu; không lấy lại số từ trí nhớ để thay bằng artifact.

## Kết quả full-scale và CASIA: tóm tắt có nguồn

[Full-scale source of truth](FULLSCALE_MODEL_RESULTS_SOURCE_OF_TRUTH.md) có đầy đủ metric, ngưỡng, counts và per-run links. Kiểm tra các số dưới với JSON gốc ngày snapshot; không chạy lại training/inference dataset trong lần cập nhật này.

| Run | CelebA full Test-as-Val ACER % / AUC | LCC combined source-frozen ACER % / AUC |
|---|---|---|
| C_FULL_crop15 | 4.6248 / .99073553 | 28.0493 / .84601127 |
| P3_SF_FULL_crop15 | 4.3548 / .99224181 | 26.7891 / .86083099 |
| R7_SC_FULL_crop15 | 4.3459 / .99228027 | 27.1228 / .86027208 |

Nguồn: [per-run full-scale metrics](../antispoof/notebooks/final/binary_crossdomain_v2/output/18_fullscale_C_P3SF_R7SC_crop15/runs). CelebA scored66,379/67,170; LCC combined scored18,794/18,827. Notebook18 dùng full official CelebA Test cho checkpoint selection và threshold calibration: **không phải independent held-out Test**. Không kết luận R7 thắng mọi target: P3 ACER combined thấp hơn R7 ở operating point này.

R7 full-scale trên CASIA test extracted images: source-frozen frame ACER25.7209% và video27.6515%; train-calibrated frame19.3690% và video18.7500%. Nguồn: [R7 full CASIA test](../antispoof/notebooks/final/binary_crossdomain_v2/output/casia_train_test_C_P3SF_R7SC_100k_fullscale/runs/R7_SC_FULL_crop15/test). Frame scored2,408/2,408; video352/360. Train-calibrated frame threshold1.6586914 khác video threshold−.033447266. Đây là **R7 full**, không phải model100K runtime hiện chọn.

CASIA là Kaggle extracted-color copy, video ID dựng từ filename, video score=mean valid frames, tối thiểu4 frames. Train/test copy overlap20 subject IDs,240 video IDs,1,403 frame IDs; không gọi đây là official subject-disjoint benchmark. Combined dedup ưu tiên test, không cộng cơ học train+test. Source-frozen không target-fit; train-calibrated có fit target train, train/combined calibrated metrics không phải independent validation. Chưa có bằng chứng Replay-Attack mới trong các nguồn đã đối chiếu.

## Parity, integrity và validation

- SHA256 hai ONNX R7 khớp cả hai JSON tương ứng ngày2026-10-14: 100K `09856f581a2868f1346ece4ce6d2e9818cd27e973a33a0bfb0d215a35475d487`; full `7b3008a040e155f7b96fca74f52dd38cc6296d88fee7a9424e653203c367a253`.
- [100K parity report](../antispoof/models/R7_SC_100K_crop15_onnx_validation.json) và [full parity report](../antispoof/models/R7_SC_FULL_crop15_onnx_validation.json) lưu status passed, CPU, batch1/3/8. Đây là **artifact parity đã lưu**, không phải một lượt camera/target accuracy mới.
- Ngày cập nhật: **56 tests runtime/worker/display/tracker/DB/enrollment pass**; **4 tests R7 source pass** với env override phù hợp (gồm crop/preprocessing/ONNX đối chiếu notebook). Các kết quả này không bao gồm các predictor tests legacy bị thiếu model.
- Chạy gộp 8 module với config CASIA hiện tại: **64 tests, 1 failure + 4 errors**. R7 test cố định source threshold−.373016357421875 nên không khớp −4.2998047. Bốn predictor errors tham chiếu model legacy thiếu (`E1_v5_1_mnv3/...best.onnx`, `minifasnet_v2se/...best_model_quantized.onnx`). Không sửa runtime/test trong task cập nhật tài liệu.
- Không chạy training/Kaggle, webcam, live cloud DB hay benchmark Pi4 trong lần này. Không dùng số test pass hoặc ONNX parity để tuyên bố camera accuracy, security hoặc edge FPS.

```bash
# Từ face_auth: test runtime/worker/label/tracker/DB/enrollment, không camera/cloud
python3 -m unittest tests.test_identity_display tests.test_runtime tests.test_recognition_worker tests.test_tracker tests.test_vector_db_logic tests.test_enrollment_math
# Test R7 source contract trong tiến trình riêng; .env không bị sửa
PAD_RUNTIME_CONFIG_PATH=antispoof/models/R7_SC_100K_crop15_runtime_config.json python3 -m unittest tests.test_r7_runtime
```

Protocol/model manifests, checkpoint hashes, completion và evaluation_freeze phải đi cùng mọi kết quả dùng trong paper/slide. Không chọn lại ngưỡng bằng target test/combined hoặc dùng kết quả notebook làm bằng chứng cho preprocessing/crop khác ở camera.
