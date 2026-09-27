# E1/E2 face-size audit và hướng thử E3

## Quyết định đề xuất

Giữ E1 v5.3 MobileNetV3-Small và E2 DCT v1 làm baseline nghiên cứu đã khóa; thử E3 MobileNetV3-Small + DCT concat với cùng data/preprocessing để trả lời frequency có bổ sung thông tin hay không. Không gọi baseline là production-ready hoặc coi kết quả ổn khi đứng gần là bằng chứng robust ở xa. Không sửa riêng augmentation E3 nếu muốn quy cải thiện cho kiến trúc fusion.

Đã đọc source code cells của:

- `antispoof/notebooks/e1_spatial_v5_3/e1_v5_3_training.ipynb`.
- `antispoof/notebooks/e2_frequency_dct/e2_frequency_only_dct_train.ipynb`.
- `E2_frequency_only_dct_protocol.md`, E1 bbox cache và E1/E2 held-out prediction CSV.

Không chạy notebook/training hoặc thay runtime. Chưa thấy notebook train E3 hiện có trong các file đã tìm.

## Notebook hiện xử lý face size đến đâu?

| Cơ chế | E1 | E2 | Ý nghĩa |
|---|---|---|---|
| Train min side 48px | Cell 3 constants, cell 5 lọc train | Cell 4 dùng train_keys manifest E1 đã lọc | Lọc input cực nhỏ; không phải augmentation robustness |
| Val/Test giữ mặt nhỏ | Có | Tái sử dụng manifest E1 | Tránh che giấu điểm yếu bằng loại mẫu test |
| Resize về 224, giữ tỷ lệ, reflect padding | Cell 3 preprocessing | Cell 6 resize trước DCT | Chuẩn hóa tensor; không khôi phục chi tiết bị mất |
| BBox jitter p=0.2 | Scale 0.95–1.05, translate ±5% | Giống E1, trước DCT | Tolerate crop/geometry sai nhẹ, không mô phỏng camera distance rộng |
| Appearance augmentation | Flip p=0.5, brightness/contrast ±0.1 p=0.3 | Giống E1 | Biến thiên appearance nhẹ |
| Random resolution degradation/downsample-upsample | Không thấy trong data path | Không thấy trong data path | Chưa chủ động mô phỏng mất resolution do mặt xa |
| Sampling theo face-size strata | Không thấy; lấy mẫu theo class | Giữ manifest E1 | Chưa bảo đảm coverage cân bằng từng face-size/class |
| Metric APCER/BPCER theo face size | Training/evaluation chủ yếu tổng hợp | Tương tự | Size được lưu trong cache nhưng chưa dùng để báo cáo metric theo bins |

E1 detector fallback 640/480/320 thay kích thước detector input để tìm bbox; không phải augmentation giảm độ phân giải ảnh vào PAD. E2 thực hiện augmentation trên RGB crop trước resize/luminance/DCT và dùng cache/split của E1, có cross-check fingerprint khi cung cấp run config.

## Đối chiếu held-out theo size

Join `key` từ `e1_test_predictions.csv` và `e2_test_predictions.csv` với `face_min_side` trong E1 bbox cache. Cả hai CSV có 10.000 dòng, không thiếu cache key. Dùng prediction đã có tại threshold khóa từ validation; không tune threshold bằng test. Face size là pixel của ảnh dataset gốc, không phải khoảng cách vật lý hoặc thông số camera chung giữa dataset/runtime.

| Min side | REAL / SPOOF | E1 BPCER | E1 APCER | E2 BPCER | E2 APCER |
|---|---:|---:|---:|---:|---:|
| <48 | 46 / 10 | 4.348% | 50.000% | 13.043% | 90.000% |
| 48–99 | 803 / 489 | 0.249% | 42.536% | 3.113% | 75.460% |
| 100–139 | 808 / 1427 | 0.124% | 29.082% | 6.931% | 56.272% |
| 140–179 | 600 / 1410 | 0.333% | 16.667% | 8.167% | 41.206% |
| ≥180 | 709 / 3698 | 0.705% | 12.628% | 20.592% | 30.314% |

Các bins được tạo để audit sau khi đã có kết quả, là phân tích exploratory; số mẫu <48 rất ít. Attack/subject composition giữa bins không được kiểm soát nên chưa quy mọi khác biệt cho resolution. E1 lỗi ở face nhỏ trên dataset nổi bật ở APCER; camera log lại cho thấy raw SPOOF cao khi người dùng báo mặt thật. Chưa thể gộp hai hiện tượng hoặc suy camera BPCER từ held-out. E2 độc lập yếu hơn không chứng minh fusion E3 sẽ thất bại hoặc thành công.

## Spec định hướng cho E3 trước khi training

- Câu hỏi: thêm frequency branch có cải thiện PAD so với E1 spatial-only trong cùng protocol và chi phí edge có chấp nhận được không?
- Hypothesis cần kiểm chứng: feature DCT bổ sung cho spatial feature; không mặc định robust hơn khi face nhỏ.
- Treatment: E3 thêm đúng `FrequencyBranch` E2 và concat vào feature spatial E1. Giữ MobileNetV3-Small, cùng input crop, label contract và score definition.
- Controls: cùng manifest/cache/fingerprints, train filtering, RGB augmentation, preprocessing 224/gamma OFF, calibration/evaluation policy và runtime version. Mỗi nhánh phải đọc cùng RGB crop đã augmentation; không jitter/degrade độc lập hai nhánh.
- Initialization đề xuất cho so sánh đầu: spatial dùng cùng ImageNet initialization như E1, frequency random initialization; không âm thầm warm-start E1/E2 checkpoints. Warm-start có thể là ablation riêng sau đó.
- Training budget đề xuất: recipe E1 làm điểm xuất phát, tối đa 24 epochs, batch 128, seed 42 cho preliminary, cùng early stopping/validation selection; LR mới cho frequency/head phải ghi rõ trước run. Cần E3 experiment spec hoàn chỉnh, config và preflight trước lệnh chạy. Một seed chỉ cho kết luận preliminary; xác nhận kiến trúc cần multi-seed đối xứng E1/E3.
- Evaluation: checkpoint và threshold chọn trên Validation; Test chỉ đánh giá threshold đã khóa. Báo cáo APCER/BPCER/ACER/AUC tổng thể và theo face size, attack, crop source, với mẫu số từng nhóm. Không lấy ngưỡng E1 dùng trực tiếp cho E3 nếu score distribution khác.
- Edge: đo batch-1 latency/RAM/model size cho cùng thiết bị/provider/thread budget; tính cả luminance+DCT, fusion và inference, không chỉ ONNX model-only latency. Phân biệt benchmark offline với full camera pipeline.
- Artifacts: training config/seed/init/hash/code revision, manifest/cache fingerprint, history, best checkpoint/meta, ONNX/contract, predictions có key/face size/source, size breakdown và latency report.
- Stages: deterministic contract/shape/gradient checks → short smoke → preliminary training → locked held-out evaluation → camera near/far real và print/replay trong protocol riêng. Không chạy camera/hardware tự động.
- Success/stopping: quyết định dựa trên trade-off error/latency và các strata đã định trước; không yêu cầu E3 phải thắng và không thay test threshold sau khi xem số.

Nếu tiếp tục nghiên cứu robustness khoảng cách, mở protocol phiên bản mới: resolution degradation train-only với range/probability định trước, áp dụng cùng cho E1/E2/E3 trước các branch transforms, calibration lại trên Validation. Chạy E1+augmentation làm control để tách hiệu quả augmentation khỏi fusion. Baseline cũ vẫn được lưu nguyên trạng. Không áp blur/JPEG mạnh hoặc augmentation phá texture tần số khi chưa có ablation.
