# PAD Runtime Config — ghi chú phát triển

Tài liệu này mô tả cách đóng gói và dùng runtime config cho model Presentation Attack Detection (PAD), để model, tiền xử lý và ngưỡng quyết định luôn đi cùng một hợp đồng runtime.

## Cách chọn cấu hình

`PAD_RUNTIME_CONFIG_PATH` là biến môi trường tùy chọn. Đường dẫn tương đối được tính từ thư mục gốc dự án. Ví dụ:

```dotenv
PAD_RUNTIME_CONFIG_PATH=antispoof/models/mnv3s_e1_preliminary_v5_3_edge_runtime_config.json
```

Để trống, hoặc đặt `None`/`null`, sẽ bỏ qua JSON và dùng các biến `PAD_*` cùng giá trị mặc định trong `config.py`.

Thứ tự ưu tiên cho từng thiết lập là:

1. Biến môi trường `PAD_*` được đặt giá trị.
2. Trường tương ứng trong runtime JSON.
3. Giá trị mặc định trong `config.py` hoặc predictor.

Biến `PAD_*` để trống không ghi đè JSON. Nếu file JSON được chỉ định nhưng không đọc được hoặc JSON không phải object, ứng dụng dừng với lỗi cấu hình để tránh chạy model bằng contract không xác định.

## Schema runtime JSON

Ví dụ dưới đây dùng model v5.3 edge hiện có:

```json
{
  "protocol_version": "spatial_e1_mnv3_small_v5_3_edge_final",
  "model_file": "mnv3s_e1_preliminary_v5_3_edge_best.onnx",
  "model_img_size": 224,
  "bbox_expansion_factor": 1.55,
  "color_order": "RGB",
  "mean": [0.5931, 0.469, 0.4229],
  "std": [0.2471, 0.2214, 0.2157],
  "apply_gamma": false,
  "calibrated_logit_threshold": -0.4650222063064575,
  "predictor_threshold_probability": 0.38579509526467753
}
```

| Trường | Kiểu | Ý nghĩa và yêu cầu |
| --- | --- | --- |
| `protocol_version` | string | Nhãn phiên bản contract để người phát triển truy vết. Loader hiện tại không dùng trường này để thay đổi hành vi. |
| `model_file` | string | Tên hoặc đường dẫn model ONNX. Đường dẫn tương đối được tính từ thư mục chứa JSON. |
| `model_img_size` | integer | Cạnh ảnh vuông đầu vào, phải lớn hơn 0. Nếu ONNX có kích thước đầu vào cố định, predictor đọc kích thước đó từ graph; giá trị JSON cần khớp với graph. |
| `bbox_expansion_factor` | number | Hệ số mở rộng bbox trước khi crop mặt. |
| `color_order` | `RGB` hoặc `BGR` | Thứ tự màu model mong đợi. Ảnh đầu vào từ OpenCV là BGR; preprocessing đổi sang RGB khi cần. |
| `mean`, `std` | 3 số mỗi mảng | Tham số chuẩn hóa theo thứ tự kênh của tensor model, trên ảnh đã scale về `[0, 1]`. Cần khai báo cả hai hoặc bỏ cả hai. |
| `apply_gamma` | boolean | Bật/tắt adaptive gamma trước resize và chuẩn hóa. |
| `calibrated_logit_threshold` | number | Ngưỡng trên `real_logit - logsumexp(spoof_logits)`. |
| `predictor_threshold_probability` | number trong `(0, 1)` | Cách biểu diễn xác suất của cùng ngưỡng quyết định. Phải nhất quán với logit threshold theo `log(p / (1-p))`. |

Các giá trị màu, normalization, gamma, crop và threshold phải lấy từ runtime/evaluation contract của chính checkpoint đó. Không suy ra threshold từ tên file và không hiệu chỉnh bằng official test set.

## Biến môi trường ghi đè

Các trường runtime có thể ghi đè qua:

| JSON | Biến môi trường |
| --- | --- |
| `model_file` | `PAD_MODEL_FILENAME` |
| `model_img_size` | `PAD_MODEL_IMG_SIZE` |
| `bbox_expansion_factor` | `PAD_BBOX_EXPANSION_FACTOR` |
| `color_order` | `PAD_COLOR_ORDER` |
| `mean` | `PAD_MEAN` |
| `std` | `PAD_STD` |
| `apply_gamma` | `PAD_GAMMA_ENABLED` |
| `predictor_threshold_probability` | `PAD_THRESHOLD` |
| `calibrated_logit_threshold` | `PAD_THRESHOLD_LOGIT` |

`PAD_MEAN` và `PAD_STD` nhận ba số phân cách bằng dấu phẩy, ví dụ `PAD_MEAN=0.5931,0.469,0.4229`. Nếu chỉ đặt một trong `PAD_THRESHOLD` hoặc `PAD_THRESHOLD_LOGIT`, giá trị còn lại được suy ra; nếu đặt cả hai thì chúng phải nhất quán.

Khi `PAD_MODEL_FILENAME` được đặt tường minh, đường dẫn tương đối của nó được tính từ `antispoof/models/`. Nếu biến này để trống và JSON có `model_file`, đường dẫn tương đối được tính từ thư mục chứa JSON.

## Đặt các artifact cạnh nhau

Đặt runtime JSON cạnh ONNX và external data của ONNX:

```text
antispoof/models/
├── <model>.onnx
├── <model>.onnx.data        # nếu graph dùng external tensor data
└── <model>_runtime_config.json
```

ONNX Runtime cần tìm được sidecar `.onnx.data` theo đường dẫn lưu trong graph. Khi chuyển hoặc đổi tên model, cần chuyển/cập nhật cả sidecar và `model_file` trong JSON. Nên lưu checksum SHA-256 của ONNX và sidecar trong manifest hoặc hồ sơ experiment đi kèm.

## Checklist khi thêm checkpoint

1. Giữ lại checkpoint, ONNX, sidecar, runtime JSON và metadata export/calibration cùng một phiên bản.
2. Ghi chính xác input size, thứ tự màu, mean/std, gamma, crop factor và ngưỡng vào JSON; đối chiếu input/output của ONNX với contract.
3. Trỏ `PAD_RUNTIME_CONFIG_PATH` tới JSON. Để trống các biến ghi đè nếu muốn JSON là nguồn runtime.
4. Xác nhận startup in đúng đường dẫn model, ONNX provider, input shape và threshold. Chạy kiểm tra offline trên dữ liệu phù hợp trước khi thử camera.
5. Ghi rõ dữ liệu dùng để chọn/calibrate threshold; giữ tách biệt train, validation và test.
6. Khi thêm trường mới, cập nhật đồng bộ loader trong `config.py`, truyền tham số qua `app.py`, predictor và công cụ chẩn đoán; cập nhật tài liệu này.

## Nơi áp dụng trong code

- `config.py` đọc JSON tùy chọn, xử lý đường dẫn và hợp nhất giá trị theo thứ tự ưu tiên.
- `app.py` truyền model path và contract đã hợp nhất vào `AntiSpoofPredictor`.
- `antispoof/predictor.py` áp dụng kích thước đầu vào, crop, normalization, gamma, thứ tự màu và ngưỡng.
- `antispoof/notebooks/diagnostics/diagnose_pad_runtime.py` dùng cùng contract để tránh diagnostic vô tình kiểm tra model bằng preprocessing hoặc threshold cũ.
