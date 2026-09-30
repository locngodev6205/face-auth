# PAD (`antispoof/`) — trạng thái hiện tại

Snapshot: **2026-09-28**, checkout `main` tại `add6645`, có thêm các tài liệu research chưa được commit. Tài liệu này đối chiếu mã, cấu hình và artifact đang có trên máy; [context toàn repo](current_codebase_state.md) mô tả detection, tracking và recognition xung quanh PAD. Env của tiến trình, thiết bị và dữ liệu bên ngoài có thể thay đổi trạng thái khi chạy.

## Trạng thái các nhánh

| Nhánh | Bằng chứng trong checkout | Trạng thái |
|---|---|---|
| E1 v5.3 MobileNetV3-Small | ONNX + sidecar, runtime JSON, weights, metadata, manifest local, held-out summary | Model mà `.env` muốn chọn cho app; **đường dẫn JSON trong `.env` hiện không tồn tại** |
| E2 v1 frequency-only DCT | Notebook, ONNX + sidecar, weights, runtime JSON, held-out summary | Nhánh nghiên cứu đã có kết quả; predictor hỗ trợ tensor DCT đơn input, app không chọn mặc định |
| E3 v1 spatial + frequency concat | Notebook đã lưu output, ONNX, weights, run config, held-out predictions/summary | Đã có kết quả preliminary; model hai input **chưa tương thích app/predictor runtime** |
| Cross-dataset v1 | Manifest, detector cache, predictions, protocol snapshot và summary E1/E3 trên LCC-FASD | Đã có LCC image-level stress test; `primary_targets_complete=false`, CASIA/Replay chưa có kết quả trong summary |
| M0–M4 micro-search, gated fusion | [Roadmap](research/pad_micro_experiment_roadmap_v1_updated.md) và [thiết kế](research/pad_spatial_frequency_architecture_v3_microsearch.md) | Kế hoạch tiếp theo; chưa thấy notebook/manifest/kết quả micro-search trong `antispoof/` |

Hai tài liệu research trên cung cấp giả thuyết và bước thực nghiệm dự kiến. Các mệnh đề về kết quả hiện tại bên dưới lấy từ artifact và source trong checkout, không lấy từ chỉ dẫn thực hiện trong roadmap.

## Mã nguồn và luồng PAD

| File | Vai trò |
|---|---|
| [`predictor.py`](../antispoof/predictor.py) | `AntiSpoofPredictor` chạy batch ONNX một input; chọn spatial RGB hoặc E2 DCT, tính score/verdict từ logits. `predict_frame` nhận bbox **xyxy**. |
| [`preprocess.py`](../antispoof/preprocess.py) | Crop mặc định dạng vuông với `BORDER_REFLECT_101`; `minifasnet_train_v1` dùng hình học crop của notebook MiniFASNet. Spatial preprocess resize/letterbox theo profile; frequency preprocess tạo luminance → global DCT → signed-log → z-score từng mẫu. |
| [`loader.py`](../antispoof/loader.py) | Tạo ONNX Runtime session, tối ưu graph, chạy tuần tự; ưu tiên CUDA rồi CPU nếu có, báo lỗi nếu load thất bại. |
| [`system.py`](../antispoof/system.py) | Thông tin CPU/GPU/provider để chẩn đoán; không đưa ra verdict. |
| [`models/smartface_pad_artifacts/`](../antispoof/models/smartface_pad_artifacts/README.md) | Deployment, metadata, protocol và đánh giá của E1/E2/E3; có thêm cross-dataset artifacts. |
| [`notebooks/`](../antispoof/notebooks/README.md) | EDA, training/evaluation E1–E3, cross-dataset và diagnostics. |

Ngoài thư mục `antispoof/`, [`config.py`](../config.py) chọn contract; [`app.update_track_pad`](../app.py) quyết định track nào đến hạn, crop/batch và xác thực output; [`Track`](../tracking/tracker.py) giữ vote, timeout và hình học crop. PAD `REAL` mới cho phép recognition khi PAD bật.

## Runtime E1: contract và tình trạng cấu hình local

- `.env` local đặt `PAD_ENABLED=true`, `PAD_CROP_SMOOTHING=true` và `PAD_RUNTIME_CONFIG_PATH=antispoof/models/mnv3s_e1_preliminary_v5_3_edge_runtime_config.json`. **File JSON này hiện vắng mặt**, cùng với bản ONNX E1 tại `antispoof/models/`. Khi path này được chọn, `config.py` lỗi khi đọc JSON trước khi app vào camera loop; vì thế bảng này mô tả model được chọn theo ý định cấu hình, chưa xác nhận một phiên chạy hiện tại.
- Cặp [runtime JSON E1](../antispoof/models/smartface_pad_artifacts/E1_v5_3_mnv3_small/deployment/mnv3s_e1_preliminary_v5_3_edge_runtime_config.json) và [ONNX E1](../antispoof/models/smartface_pad_artifacts/E1_v5_3_mnv3_small/deployment/mnv3s_e1_preliminary_v5_3_edge_best.onnx) cùng sidecar `.onnx.data` vẫn có trong `smartface_pad_artifacts/E1_v5_3_mnv3_small/deployment/`. Trỏ `PAD_RUNTIME_CONFIG_PATH` tới JSON này sẽ làm `model_file` resolve tương đối đúng thư mục deployment; cần kiểm tra startup thực tế sau khi chọn đường dẫn.
- Contract E1: crop bbox xyxy thành hình vuông, mở rộng **1.55×**; ảnh 224×224 RGB, gamma **off**, mean `[0.5931, 0.4690, 0.4229]`, std `[0.2471, 0.2214, 0.2157]`. Spatial preprocess dùng resize giữ tỉ lệ, reflect letterbox, float32 CHW /255 rồi mean/std; PAD không dùng ArcFace alignment. SHA256 ONNX E1 trong deployment khớp summary: `f644427b7b4351093a3cfac64f6ddedafd11c4a05e81fda7e5e156dea4a5c96e`.
- Predictor tính `d = real_logit − logsumexp(spoof_logits)` với class 0 là REAL. Raw verdict REAL nếu `d ≥ -0.4650222063064575`, tương đương `P(real) ≥ 0.38579509526467753`. Đây là ngưỡng PAD, tách khỏi recognition threshold `.35`. Trường JSON ưu tiên hơn env `PAD_*`; `config.py` kiểm tra cặp ngưỡng xác suất/logit; CLI `--pad-model` phải khớp contract đã chọn.
- App gom các track đến hạn thành batch, kiểm tra đủ kết quả, `is_real` là bool và score hữu hạn trước khi ghi vote. Lỗi crop/inference/output làm mất PAD và identity cache. `Track` giữ 5 binary verdict gần nhất, cần 5 vote trước khi rời `PAD_PENDING`; spoof ratio ≥0.6 cho `SPOOF`, dữ liệu quá 3 giây thành stale. `PAD_PENDING`/`SPOOF` chặn recognition. Cadence local dự kiến: detector/PAD 0.1 giây, recognition 0.5 giây; chưa phải FPS hoặc thời gian ra verdict đã đo.
- Crop smoothing local chỉ tác động tâm/cạnh crop PAD khi jitter ≤5% (time constant 0.12 giây); snap khi chuyển động/scale lớn, khoảng trống >0.5 giây hoặc crop chạm biên. Code default và `.env.example` là `false`. Nó không sửa tracking bbox, model contract hoặc binary voting; xem [ghi chú thử nghiệm](runtime/pad_crop_smoothing_experiment.md).

## Kết quả preliminary đã lưu

Các bảng sau là số liệu trong artifact, không phải kết quả camera/edge. E1/E2/E3 dùng split fingerprints CelebA-Spoof tương ứng; ngưỡng E3 được khóa từ Validation, `test_threshold_tuning=false`. Số liệu PAD điều kiện theo ảnh có mặt được detector/crop xử lý thành công.

| Model | Held-out CelebA-Spoof (`N=10,000`) | APCER | BPCER | ACER | AUC |
|---|---:|---:|---:|---:|---:|
| [E1 spatial](../antispoof/models/smartface_pad_artifacts/E1_v5_3_mnv3_small/test/e1_heldout_test_summary.json) | preliminary | 18.91% | 0.40% | 9.66% | 0.9814 |
| [E2 DCT](../antispoof/models/smartface_pad_artifacts/E2_dct_v1/test/e2_heldout_test_summary.json) | preliminary | 40.99% | 9.51% | 25.25% | 0.8366 |
| [E3 concat](../antispoof/models/smartface_pad_artifacts/E3_concat_v1/test/e3_heldout_test_summary.json) | preliminary | 17.76% | 0.51% | 9.13% | 0.9857 |

E3 có [run config](../antispoof/models/smartface_pad_artifacts/E3_concat_v1/metadata/e3_concat_preliminary_v1_run_config.json) với seed 42, khởi tạo độc lập, cùng split fingerprints E1; [test protocol snapshot](../antispoof/models/smartface_pad_artifacts/E3_concat_v1/test/e3_test_protocol_snapshot.json), [per-image predictions](../antispoof/models/smartface_pad_artifacts/E3_concat_v1/test/e3_test_predictions.csv) và notebook đã lưu output. SHA256 ONNX E3 khớp runtime JSON/test snapshot: `ffa4d93059d616a3cbaeba9eac30ad65f8d9372207cfd2864b11421fd6b129f0`. E3 nhận `rgb_input` 3×224×224 và `frequency_map` 1×224×224 tạo bằng DCT bên ngoài ONNX; [`predictor.py`](../antispoof/predictor.py) chỉ cấp một input nên chưa chạy E3 trong app. [Host efficiency](../antispoof/models/smartface_pad_artifacts/E3_concat_v1/metadata/e3_concat_preliminary_v1_efficiency.json) đo E3 `total_pad` p50 8.85 ms và E1 5.88 ms trên cùng host CPU; không suy ra FPS camera/edge.

| LCC-FASD image-level, cùng 3,762 ảnh có detection | APCER | BPCER | HTER (= ACER ở protocol này) | AUC |
|---|---:|---:|---:|---:|
| E1 | 40.44% | 22.62% | 31.53% | 0.7533 |
| E3 | 35.34% | 26.48% | 30.91% | 0.7443 |

[Cross-dataset summary](../antispoof/models/smartface_pad_artifacts/cross_dataset_v1/preliminary/predictions/cross_dataset_summary.json) và [protocol snapshot](../antispoof/models/smartface_pad_artifacts/cross_dataset_v1/preliminary/predictions/cross_dataset_protocol_snapshot.json) ghi 3,766 candidate, loại 4 ảnh không xử lý được; ngưỡng khóa từ CelebA Validation, không tune/fine-tune trên LCC. E3 giảm HTER khoảng **0.62 điểm phần trăm** nhưng AUC giảm khoảng **0.90 điểm phần trăm** so với E1; đây là so sánh mô tả, chưa chứng minh lợi ích cross-domain ổn định. CASIA-FASD và Replay-Attack chưa được mount/chưa có kết quả trong summary; `primary_targets_complete=false`.

## Provenance và điểm cần nhớ

- [Manifest NPZ E1](../antispoof/models/smartface_pad_artifacts/E1_v5_3_mnv3_small/data_protocol/celeba_spoof_preliminary_v5_3_edge_mnv3_small_seed42.npz) hiện có trên máy; SHA256 `7799486ebcae0babe09e061579ac05db7c1dc61e10746be0755bcddc10b2df9e` khớp E3 run/test snapshot. Quy tắc `*.npz` trong `.gitignore` loại file này khỏi Git, nên checkout mới cần mang đúng manifest và kiểm fingerprint; bbox cache không thể thay thế split gốc.
- E3 và cross-dataset có model/config/summaries/predictions, nhưng bộ ảnh nguồn nằm ngoài repo và metadata không khóa đầy đủ một Git revision đã chạy. Vì vậy evidence có thể truy theo artifact local, còn việc tái chạy end-to-end cần dữ liệu nguồn và môi trường tương ứng. Không dùng Test hoặc target set để chọn lại model/ngưỡng.
- [`antispoof/README.md`](../antispoof/README.md) còn bảng/ví dụ MiniFASNet legacy (128×128, INT8, default quantized model, bbox xywh) trộn với ghi chú E1. Khi kiểm tra runtime dùng source và runtime JSON; xem thêm [runtime config guide](runtime/pad_runtime_config.md) và [audit](audits/runtime_pipeline_audit_2026-09-26.md).
