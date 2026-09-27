# Chạy E3 trên Kaggle

## 1. Chuẩn bị

Upload `antispoof_e3_spatial_frequency_concat_v1.ipynb` bằng Import Notebook. Chọn GPU và bật Internet để cài dependencies/nạp ImageNet pretrained. Không chạy các notebook E1/E2 để tạo lại protocol.

Add Input gồm:

1. CelebA-Spoof cùng distribution với E1, có `Data/` và `metas/intra_test/train_label.json`, `test_label.json`.
2. E1 frozen NPZ `celeba_spoof_preliminary_v5_3_edge_mnv3_small_seed42.npz`.
3. Cache `celeba_scrfd_bbox_cache_v5_3_preliminary_seed42.json`.
4. `mnv3s_e1_preliminary_v5_3_edge_run_config.json` để verify fingerprints.

Independent E3 không cần E1/E2 PAD weights. Có thể attach E1 ONNX + `.onnx.data` nếu có, runtime config và best_meta để đo efficiency E1/E3 cùng host. E2 weights không được load.

## 2. Cấu hình và preflight

- `E1_RUN_MODE="preliminary"`.
- `E1_ARTIFACT_DIR=""`: recursive exact-name discovery dưới `/kaggle/input`; nếu nhiều bản trùng tên, đặt thư mục E1 cụ thể, có thể chứa các thư mục con data_protocol/metadata/deployment.
- `DATA_ROOT`: đường dẫn thư mục chứa `Data/` và `metas/`, lấy từ sidebar Kaggle.
- Giữ `INIT_MODE="independent"`, `NUM_WORKERS=4` (có thể giảm workers khi Kaggle thiếu RAM).

Run từ đầu đến cell preflight. Mong đợi `Frozen E1 preflight PASS`, Train 98,873 / Val 15,116 / Test 10,000, class counts và fingerprints khớp protocol. Notebook dừng nếu cache/split/mode khác. Không sửa fingerprint để vượt guard.

Official: thay mode và attach toàn bộ artifact E1 official thật. Nếu run_config không chứa `class_counts`, đặt `FROZEN_CLASS_COUNTS_PATH` đến JSON xuất từ E1 official đã đóng băng, dạng `{"train":{"0":...,"1":...,"2":...},"val":{...},"test":{...}}`. Không tự tạo reference từ metadata đang kiểm tra. Official artifact chưa được xác minh ở workspace này.

## 3. Huấn luyện và xuất model

Run các cell còn lại. Mô hình ImageNet có thể cần download ở lần đầu; notebook không download dataset. Train tối đa 24 epoch, early stop theo Validation, không đánh giá Test. Chờ cell ONNX parity PASS và efficiency hoàn tất. Không bỏ qua parity.

Output trong `/kaggle/working/e3_concat_<mode>_v1/` và ZIP cùng prefix:

```text
e3_concat_<mode>_v1_best.pth
e3_concat_<mode>_v1_checkpoint_last.pth
e3_concat_<mode>_v1_best_meta.json
e3_concat_<mode>_v1_run_config.json
e3_concat_<mode>_v1_training_history.json
e3_concat_<mode>_v1_training_history.png
e3_concat_<mode>_v1_best.onnx
e3_concat_<mode>_v1_runtime_config.json
e3_concat_<mode>_v1_efficiency.json
```

Tải ZIP về ngay sau run; lưu notebook đã executed và Kaggle dataset/kernel version. E1 efficiency chưa có nếu không attach model E1; vẫn phải đo sau trước khi kết luận overhead. RSS/CPU timings không thay thế Raspberry Pi đo thật.

## 4. Held-out Test sau freeze

Import `antispoof_e3_heldout_test_v1.ipynb`. Attach CelebA + E1 NPZ/cache/run config như trên, và E3 ONNX/runtime config/best_meta vừa xuất. Đặt `E3_ARTIFACT_DIR` hoặc để blank discovery. Run All. Không sửa threshold; runtime và best_meta phải khớp.

Download `e3_<mode>_heldout_test_results.zip`: summary, protocol snapshot, per-image predictions, attack breakdown, face-size breakdown, confusion matrix, score distribution và ROC bổ sung. Giữ CSV để paired analysis theo key với E1/E2.

## 5. Lưu lâu dài

```text
pad_experiments/
  E1_v5_3/<mode>/{training,data_protocol,test}/
  E2_dct_v1/<mode>/{training,test}/
  E3_concat_v1/<mode>/{training,test}/
  cross_dataset_v1/<mode>/{manifests,detector_cache,predictions,plots}/
```

Giữ NPZ/cache, executed notebooks, metadata và hash cùng model. Tách preliminary/official. ONNX E3 cần hai input và external DCT; app hiện tại chưa hỗ trợ contract này.

## Tiếp tục khi Kaggle hết thời gian

Download `_checkpoint_last.pth`, attach vào session mới và đặt `CHECKPOINT_INPUT_PATH` tới checkpoint của chính bạn. Notebook xác nhận config/environment bằng đúng run trước, khôi phục model/optimizer/scheduler/scaler, best weights/meta và RNG/loader state, rồi tiếp tục epoch kế tiếp. Không đổi mode, batch, seed hoặc protocol khi resume. Khôi phục chỉ ở ranh giới epoch; không hỗ trợ resume giữa epoch. Khi run trước đã đạt early stop/tối đa epoch, chỉ xuất lại artifact. Vì strict determinism OFF và augmentation/worker library có RNG nội bộ, đây không phải cam kết bitwise replay qua mọi environment.
