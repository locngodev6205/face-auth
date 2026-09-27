# E3 spatial + frequency CONCAT v1

## Câu hỏi và đối chứng

DCT có bổ sung thông tin cho MobileNetV3-Small, đặc biệt khi đổi dataset, với chi phí triển khai hợp lý không? E1 spatial-only là đối chứng chính; E2 frequency-only là chẩn đoán đã đóng băng. Không giả định E3 thắng. Không thay đổi E1/E2, không tạo E4 trong bước này.

Nguồn chuẩn là notebook E1 `../e1_spatial_v5_3/e1_v5_3_training.ipynb`, E2 `../e2_frequency_dct/e2_frequency_only_dct_train.ipynb`, các notebook held-out tương ứng và metadata của run đã đóng băng. Thiết kế cũ trong `md/` không thay thế implementation thực thi. Hash hai notebook nguồn được ghi vào run config.

## Protocol dữ liệu đã xác nhận

- Seed 42, input 224×224, gamma OFF; lớp 0 Real, 1 Physical Spoof, 2 Digital Spoof.
- Đọc chính xác `celeba_spoof_<mode>_v5_3_edge_mnv3_small_seed42.npz` và `celeba_scrfd_bbox_cache_v5_3_<mode>_seed42.json` cùng run config E1. Không resample, không chạy SCRFD lại trên CelebA.
- SCRFD crop 1.55×; annotation fallback 1.50×; square crop và REFLECT_101. Resize LANCZOS4 khi tăng kích thước, AREA khi giảm; letterbox REFLECT_101.
- TRAIN min-face 48 px và loại geometry nghi ngờ đã nằm trong `train_keys`; Val/Test giữ mặt nhỏ.
- Spatial mean `[0.5931,0.4690,0.4229]`, std `[0.2471,0.2214,0.2157]`.
- Kiểm tra schema v5.3, policy, key/subject disjointness, trùng key, split/cache fingerprint, số lượng và class counts. Sai khác làm notebook dừng.

Preliminary class counts từ output notebook E1/E2 đã thực thi:

| Split | Real | Physical | Digital | Tổng |
|---|---:|---:|---:|---:|
| Train | 32389 | 45761 | 20723 | 98873 |
| Val | 4980 | 6957 | 3179 | 15116 |
| Test | 2966 | 5284 | 1750 | 10000 |

Fingerprint split theo SHA256 của sorted keys nối newline:

```text
train 4ecbcee78758366dd454c0e446444f6c74c82ac5c4d1c937730382b34d4e25f8
val   4560d12efe0bb840c0584a38bda12097e2048f3066d89177cda62f8d931edf91
test  394d6a5ba98cde1cfbda3b9c41c7824262c4c5bc648883b93581a9585400a26c
cache 4ac3763a6c566c976de11dda1cbfdb8426f2df73d95347c7fe2c1b9629ab9b85
```

Cache fingerprint dùng union requested TRAIN + Val + Test theo đúng E1, bao gồm record TRAIN bị lọc. Official phải có artifact official thật và class-count reference đã đóng băng; không dùng counts preliminary cho official.

## Kiến trúc và khởi tạo

```text
Một RGB face crop sau một bbox jitter và một augmentation
 ├─ E1 normalize → MobileNetV3-Small ImageNet → GAP → 576→256
 │                → BN → Hardswish → Dropout .2 → spatial 256-D
 └─ resize 224 → RGB/255 → luminance .299R+.587G+.114B → cv2.dct
                  → sign(C)*log1p(abs(C)) → z-score full map → exact E2 CNN
Concat(256,64) → Linear 320→128 → Hardswish → Dropout .2 → Linear 128→3
```

DCT float32, std population, denominator `max(std,1e-6)`, không mask. `FrequencyBranch` được chép nguyên implementation E2: Conv 1→32 3×3 stride 2 + BN/ReLU; depthwise/pointwise 32→32 stride 2 và 32→64 stride 2 (BN/ReLU); GAP; Flatten; Linear 64→64 + ReLU. Nhánh này có 8,480 tham số.

`INIT_MODE="independent"` là chế độ duy nhất được triển khai trong v1. Backbone ImageNet như E1; projection, frequency, fusion random. Không nạp task-trained E1/E2 PAD weights. Warm-start là ablation riêng sau này, cần tên artifact/protocol khác.

Spatial và frequency nhận cùng ảnh sau augmentation, không có jitter/brightness riêng cho từng nhánh. HorizontalFlip .5; brightness/contrast ±.10 với p=.30; bbox jitter p=.20, scale .95–1.05, translate ±.05. Không có downsample-upsample augmentation. Ablation độ phân giải sau này phải có E1 cùng augmentation làm control.

## Huấn luyện và khóa threshold

AdamW; batch 128; backbone LR 1e-5; tất cả layer mới LR 1e-4; weight decay 1e-4; tối đa 24 epoch; warmup 2 epoch, start factor .2; cosine min factor .1. CrossEntropy label smoothing .10, weights `[1,1,1]`; AMP train; clip gradient 5 sau unscale. Validation FP32 như E1. Seed 42 nhưng strict determinism OFF và cuDNN benchmark ON như run E1; không tuyên bố bitwise determinism.

Checkpoint chọn theo Validation ACER, tie-break lower APCER như E1. Patience 4, min delta 1e-4 để reset patience, dừng sớm nhất sau epoch 10. Calibration dùng nguyên hàm E1 tìm min ACER, tie APCER rồi BPCER, giữ logit range tương ứng probability `[1e-6,1-1e-6]`.

`d = real_logit - logsumexp(spoof logits)`, Real iff `d >= threshold`. Lưu cả logit threshold và sigmoid threshold. Không có Test loader trong notebook train. Held-out Test chỉ chạy sau freeze; không đổi model/protocol sau khi nhìn Test.

## Artifact và triển khai

Prefix `e3_concat_<mode>_v1`: `_best.pth`, `_checkpoint_last.pth`, `_best_meta.json`, `_run_config.json`, `_training_history.json`, `_training_history.png`, `_best.onnx`, `_runtime_config.json`, `_efficiency.json`; một ZIP chứa thư mục run. Không ghi đè preliminary bằng official.

ONNX float32 hai input `rgb_input [N,3,224,224]`, `frequency_map [N,1,224,224]`, output logits `[N,3]`. DCT nằm ngoài graph. Export kiểm tra batch 1/2 và crop Validation thật, bắt buộc max absolute logit difference <1e-4. Runtime config ghi DCT/RGB contract, model hash và threshold. Predictor hiện tại trong app chỉ hỗ trợ một input: đổi `PAD_RUNTIME_CONFIG_PATH` chưa đủ để chạy E3. Tích hợp app là công việc riêng.

Efficiency ghi tổng/spatial/frequency/fusion params, ONNX bytes, latency p50/p95/mean của crop+resize+spatial, luminance+DCT, model-only, total PAD. Với E1 artifact mounted, đo cùng ảnh, host CPU provider, thread budget 2 intra/1 inter, batch 1, warmup 10, timed samples 100. Đo RSS bằng psutil trước và sau từng vòng, sampled peak whole-process; không coi là isolated model peak. Loại disk I/O/detector khỏi PAD timing. Đây là host diagnostic, chưa phải Raspberry Pi/camera benchmark.

## Đánh giá và giới hạn

Held-out CSV có key, subject, attack code, bbox source, native face_min_side, kích thước ảnh, crop factor, logits, d/p_real, binary/3-class correctness; giữ alias tên cột E1/E2. Bins cố định `<48`, `48–99`, `100–139`, `140–179`, `>=180`; áp dụng một threshold toàn cục. Thiếu Real/Attack thì metric undefined lưu null/NaN, không zero. Không suy luận kích thước mặt gây lỗi nếu chưa kiểm soát composition.

Cross-dataset dùng LCC-FASD/CASIA-FASD với threshold CelebA Validation. Cải thiện nhiều target hỗ trợ giả thuyết bổ trợ; chỉ cải thiện source có thể phản ánh shortcut; tương đương với overhead nhỏ là inconclusive. Nếu E3 giảm cross-domain, kiểm tra branch/domain sensitivity trước E4. E4 chỉ hợp lý sau evidence và một kế hoạch ablation mới. Notebook chưa được chạy Kaggle nên chưa có kết quả nghiên cứu.

Checkpoint cuối hỗ trợ tiếp tục tại ranh giới epoch trên Kaggle, bao gồm best state/meta, optimizer/scheduler/scaler và RNG/loader state; config/environment phải khớp. Không có resume giữa epoch và không tuyên bố bitwise replay. Chỉ load checkpoint tin cậy do chính run này tạo.
