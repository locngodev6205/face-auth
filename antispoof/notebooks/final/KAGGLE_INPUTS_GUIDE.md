# Kaggle Inputs cho các notebook MiniFASNet final

Tài liệu này ghi lại các input mà năm notebook hiện có trong thư mục `final/` đang đọc theo cấu hình trong notebook. Mỗi notebook Kaggle chạy trong một môi trường riêng: dữ liệu ở `/kaggle/working` của notebook trước không tự xuất hiện trong notebook sau.

## Các input dùng chung

| Dataset / tài nguyên | Đường dẫn notebook đang cấu hình | Notebook dùng | Ghi chú |
|---|---|---|---|
| CelebA-Spoof | `/kaggle/input/datasets/attentionlayer241/celeba-spoof-for-face-antispoofing/CelebA_Spoof_/CelebA_Spoof` | 00, 00b, 01, 02, 03 | Cần giữ nguyên thư mục ảnh và `metas/intra_test/train_label.json`, `metas/intra_test/test_label.json`. |
| LCC-FASD | `/kaggle/input/datasets/aleksandrpikul222/lcc-fasd/LCC_FASD` | 00, 00b, 01, 02, 03 | Cần ảnh LCC. Notebook 00 còn đọc `CLIENT_TEST.txt` và `IMPOSTER_TEST.txt` ngay trong thư mục này nếu không dùng được cache Stage-0. |
| MiniFASNet resources | `/kaggle/input/datasets/maithanhphuong/minifasnet/` | 00b, 01, 02, 03 | `MiniFASNet.py` là file bắt buộc cho cả bốn notebook. `2.7_80x80_MiniFASNetV2.pth` bắt buộc cho 00b và 03; 01/02 khởi tạo random và không nạp checkpoint này. |
| SCRFD-500M | `/kaggle/input/datasets/maithanhphuong/scrfd-model/det_500m.onnx` | 00 | Notebook 00 xác thực SHA256 trước khi chạy. Không thay bằng model khác. |
| Stage-0 micro bundle | `/kaggle/input/datasets/maithanhphuong/face-auth-stage0-micro-v1` | 00, tùy chọn | Chỉ để tái sử dụng manifest/cache LCC đã được xác thực. Không thay cho resource bundle final do Notebook 00 tạo. |
| Final resource bundle từ Notebook 00 | Mặc định sau giải nén: `/kaggle/working/minifasnet_final_resources_v1` | 00b, 01, 02, 03 | Đây là input bắt buộc cho các notebook sau; cần chuyển output qua Kaggle Dataset hoặc giải nén lại ở từng notebook mới. |

Trong Kaggle, mở **Add Input** và gắn đúng dataset. Sau khi gắn, dùng **Copy Input Path** để xác nhận mount. Nếu Kaggle hiển thị đường dẫn khác với đường dẫn mặc định ở bảng trên, sửa biến tương ứng trong cell cấu hình đầu notebook. Tên file checkpoint dùng ký tự `x` ASCII: `2.7_80x80_MiniFASNetV2.pth`.

## Input cho từng notebook

### 00 — `00_prepare_minifasnet_final_resources.ipynb`

**Bắt buộc gắn:**

- CelebA-Spoof, tại `CELEBA_ROOT` như bảng trên.
- LCC-FASD, tại `LCC_FASD_ROOT`.
- Dataset SCRFD có file `det_500m.onnx`, tại `SCRFD_MODEL_PATH`.

**Tùy chọn:**

- `face-auth-stage0-micro-v1`. Notebook kiểm tra `micro_resource_fingerprints.json`, `lcc_external_dev_manifest.csv` và `lcc_scrfd_cache_micro_v1.json` ở thư mục gốc hoặc thư mục con `stage0/`. Nếu hash và detector khớp, notebook tái sử dụng các tài nguyên LCC này.
- Nếu không gắn được Stage-0 bundle, notebook dựng lại phần LCC từ SCRFD và yêu cầu `CLIENT_TEST.txt`, `IMPOSTER_TEST.txt` nằm trực tiếp trong `LCC_FASD_ROOT`.

Stage-0 chỉ phục vụ nhánh cache LCC; notebook vẫn cần model SCRFD chính xác để tạo cache CelebA. Cell preflight yêu cầu `cv2`, `numpy`, `pandas`, `sklearn`, `insightface` đúng phiên bản `2.0` và `onnxruntime` có `CPUExecutionProvider`; SCRFD phải có SHA256 đã đóng băng trong notebook. Nếu preflight báo thiếu hoặc không khớp, dừng và sửa môi trường/input trước khi chạy các cell tạo cache.

Notebook xuất thư mục `/kaggle/working/minifasnet_final_resources_v1` và file `/kaggle/working/minifasnet_final_resources_v1.zip`. Giữ nguyên toàn bộ bundle, gồm manifests, bbox caches, config, provenance và fingerprints.

### 00b — `00b_minifasnetv2_5k_hparam_tuning.ipynb`

**Bắt buộc gắn:**

- Final resource bundle được tạo từ Notebook 00.
- CelebA-Spoof và LCC-FASD.
- Dataset MiniFASNet có cả `MiniFASNet.py` và `2.7_80x80_MiniFASNetV2.pth`.

**Không cần gắn:** SCRFD hoặc Stage-0 bundle. Notebook này dùng bbox cache trong final resource bundle và không tạo lại cache.

Notebook ghi kết quả vào `/kaggle/working/minifasnetv2_5k_hparam_tuning/` và tạo ZIP cùng tên trong `/kaggle/working`. Chọn GPU để chạy tuning; code tự chuyển sang CPU nếu không có CUDA nhưng việc huấn luyện sẽ chậm hơn đáng kể.

### 01 — `01_minifasnetv2_scale20k_randominit_vanilla_vs_csmr.ipynb`

**Bắt buộc gắn:**

- Final resource bundle từ Notebook 00.
- CelebA-Spoof và LCC-FASD.
- Dataset MiniFASNet có file `MiniFASNet.py`.
- Môi trường có `onnx` và `onnxruntime` cho cell export/kiểm tra ONNX.

Notebook khởi tạo từ random; file `.pth` không được nạp và không bắt buộc theo call path hiện tại, dù biến đường dẫn checkpoint vẫn có trong cell cấu hình. SCRFD và Stage-0 không cần cho notebook này. GPU được khuyến nghị; code tự chọn CPU nếu CUDA không có.

### 02 — `02_minifasnetv2_final100k_randominit_vanilla_vs_csmr.ipynb`

**Bắt buộc gắn:**

- Final resource bundle từ Notebook 00.
- CelebA-Spoof và LCC-FASD.
- Dataset MiniFASNet có file `MiniFASNet.py`.
- `onnx` và `onnxruntime` để preflight, export và kiểm tra parity.

Notebook khởi tạo random nên không cần nạp checkpoint `.pth`. Không cần SCRFD hoặc Stage-0. Dùng GPU để huấn luyện; preflight ONNX chạy trước phần đọc tài nguyên lớn và training.

### 03 — `03_minifasnetv2_final100k_pretrained_vanilla_vs_csmr.ipynb`

**Bắt buộc gắn:**

- Final resource bundle từ Notebook 00.
- CelebA-Spoof và LCC-FASD.
- Dataset MiniFASNet có cả `MiniFASNet.py` và checkpoint `2.7_80x80_MiniFASNetV2.pth`.
- `onnx` và `onnxruntime` để preflight, export và kiểm tra parity.

Notebook nạp checkpoint pretrained; nếu thiếu file hoặc kiến trúc/trọng số không khớp, setup sẽ dừng. Không cần SCRFD hoặc Stage-0. Dùng GPU để huấn luyện.

## Chuyển final resource bundle sang notebook Kaggle khác

Notebook 00 lưu bundle trong `/kaggle/working`; notebook Kaggle độc lập không tự nhìn thấy thư mục đó. Sau khi Notebook 00 chạy xong:

1. Tải `minifasnet_final_resources_v1.zip` từ Output.
2. Tạo/version một Kaggle Dataset từ file ZIP đó rồi gắn dataset vào 00b, 01, 02 và 03.
3. Trong mỗi notebook sau, giải nén ZIP vào `/kaggle/working` trước cell cấu hình, rồi giữ `FINAL_RESOURCE_DIR` trỏ đến thư mục được giải nén:

```python
from pathlib import Path
from zipfile import ZipFile

bundle_zip = Path("/kaggle/input/datasets/<owner>/<dataset>/minifasnet_final_resources_v1.zip")
with ZipFile(bundle_zip) as archive:
    archive.extractall("/kaggle/working")

FINAL_RESOURCE_DIR = "/kaggle/working/minifasnet_final_resources_v1"
assert Path(FINAL_RESOURCE_DIR, "config.json").is_file()
```

Thay `<owner>/<dataset>` bằng đường dẫn thực tế lấy từ **Copy Input Path**. ZIP do Notebook 00 tạo có thư mục gốc `minifasnet_final_resources_v1/`. Nếu dataset chứa sẵn thư mục đã giải nén, có thể đặt `FINAL_RESOURCE_DIR` trực tiếp tới thư mục đó và bỏ bước giải nén.

## Thứ tự chạy và các lưu ý về input

1. Chạy Notebook 00 để đóng băng manifests, bbox caches, fingerprints và provenance.
2. Chuyển bundle của Notebook 00 thành input cho các notebook sau.
3. Chạy Notebook 00b nếu cần hoàn thành tuning; notebook này chỉ cần input list của chính nó ở trên.
4. Chạy Notebook 01, sau đó 02 và/hoặc 03 với đúng bundle và dữ liệu đã gắn.

Notebook 00b xuất `selection/frozen_hparams.json`, nhưng mã nguồn hiện tại của 01/02/03 không tự đọc file này. Nếu muốn dùng các giá trị đã tuning cho lần chạy sau, cần đối chiếu report rồi cập nhật các hằng số cấu hình tương ứng trong notebook trước khi chạy; chỉ gắn ZIP của 00b làm input sẽ không tự áp dụng hyperparameters.

Notebook 01/02/03 dùng ảnh CelebA và LCC cùng manifests/cache trong bundle; không cần chạy lại SCRFD. Kết quả được lưu trong `/kaggle/working` của từng notebook. Hãy tải ZIP output tương ứng sau khi chạy xong nếu cần lưu hoặc chuyển tiếp.
