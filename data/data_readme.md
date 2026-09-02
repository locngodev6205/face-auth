# 📁 Hướng Dẫn Thiết Lập Dataset (LFW Dataset Guide)

Tài liệu này hướng dẫn cách tải, giải nén và tổ chức dữ liệu **Labeled Faces in the Wild (LFW)** phục vụ cho việc benchmark, đánh giá hiệu năng (evaluation) và demo hệ thống Face Authentication.

> [!NOTE]
> Hai thư mục `data/lfw/` và `data/lfw_metadata/` đã được cấu hình trong `.gitignore` để tránh commit dữ liệu dung lượng lớn lên Git repository. Do đó, khi thiết lập môi trường mới, bạn cần tải và tổ chức dữ liệu theo hướng dẫn này.

---

## 1. 📌 Tổng Quan Dữ Liệu

- **Dataset:** Labeled Faces in the Wild (LFW)
- **Nguồn:** [Kaggle — Jessica Li: LFW Dataset](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset/versions/4?resource=download)
- **Kích thước:**
  - File nén `.zip`: ~115 MB
  - Sau khi giải nén: ~140 MB – 180 MB
  - Tổng số định danh (identities): **5.749 người**
  - Tổng số ảnh khuôn mặt: **13.233 ảnh** (`.jpg`)
- **Mục đích sử dụng trong dự án:**
  - Calibration ngưỡng nhận diện (`MATCH_THRESHOLD` trong `config.py`).
  - Đánh giá kích thước Gallery tối ưu (`RECOMMENDED_GALLERY_SIZE`).
  - Chạy benchmark định danh và xác thực (`evaluation/lfw_identity_benchmark.ipynb`).
  - Demo đăng ký khuôn mặt giả lập (`evaluation/lfw_demo_enroll.ipynb`).

---

## 2. 📥 Hướng Dẫn Tải Dữ Liệu

Bạn có thể tải dữ liệu theo một trong hai cách dưới đây:

### Cách 1: Tải trực tiếp qua trình duyệt (Khuyên dùng)
1. Truy cập liên kết:
   🔗 [Kaggle — LFW Dataset (Version 4)](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset/versions/4?resource=download)
2. Nhấn nút **Download** (yêu cầu đăng nhập tài khoản Kaggle) để tải file `archive.zip` (hoặc `lfw-dataset.zip`).

### Cách 2: Tải qua Kaggle CLI (Dành cho Terminal / Server)
```bash
# Đảm bảo đã thiết lập Kaggle API key tại ~/.kaggle/kaggle.json
kaggle datasets download -d jessicali9530/lfw-dataset -p face_auth/data/
```

---

## 3. 📂 Cấu Trúc Thư Mục Chuẩn Sau Khi Thiết Lập

Sau khi tải và giải nén, thư mục `face_auth/data/` cần tuân theo cấu trúc chuẩn:

```text
face_auth/data/
├── data_readme.md                  # File hướng dẫn này
│
├── lfw/                            # Thư mục chứa ảnh khuôn mặt phân theo người
│   ├── Aaron_Eckhart/
│   │   └── Aaron_Eckhart_0001.jpg
│   ├── Aaron_Peirsol/
│   │   ├── Aaron_Peirsol_0001.jpg
│   │   └── Aaron_Peirsol_0002.jpg
│   ├── Mick_Jagger/
│   │   ├── Mick_Jagger_0001.jpg
│   │   └── Mick_Jagger_0002.jpg
│   └── ... (5.749 thư mục con)
│
└── lfw_metadata/                   # Thư mục chứa 10 file CSV metadata
    ├── lfw_allnames.csv
    ├── lfw_readme.csv
    ├── matchpairsDevTest.csv
    ├── matchpairsDevTrain.csv
    ├── mismatchpairsDevTest.csv
    ├── mismatchpairsDevTrain.csv
    ├── pairs.csv
    ├── people.csv
    ├── peopleDevTest.csv
    └── peopleDevTrain.csv
```

> [!IMPORTANT]
> - Thư mục ảnh phải được đặt tên chính xác là `lfw` (chứa trực tiếp các thư mục tên người, tránh trường hợp lồng nhau như `lfw/lfw/...` hay `lfw-deepfunneled/lfw-deepfunneled/...`).
> - Thư mục metadata phải được đặt tên chính xác là `lfw_metadata` và chứa trực tiếp các file `.csv`.

---

## 4. 🛠️ Hướng Dẫn Giải Nén & Sắp Xếp Tự Động (Bash Script)

Nếu bạn đặt file tải về `lfw-dataset.zip` (hoặc `archive.zip`) vào `face_auth/data/`, bạn có thể chạy các lệnh sau trong terminal:

```bash
cd face_auth/data

# 1. Giải nén vào thư mục tạm
unzip -q lfw-dataset.zip -d temp_lfw/

# 2. Tạo thư mục metadata và di chuyển tất cả file CSV
mkdir -p lfw_metadata
mv temp_lfw/*.csv lfw_metadata/

# 3. Chuẩn hóa thư mục ảnh thành 'lfw'
if [ -d "temp_lfw/lfw-deepfunneled/lfw-deepfunneled" ]; then
    mv temp_lfw/lfw-deepfunneled/lfw-deepfunneled lfw
elif [ -d "temp_lfw/lfw/lfw" ]; then
    mv temp_lfw/lfw/lfw lfw
elif [ -d "temp_lfw/lfw" ]; then
    mv temp_lfw/lfw lfw
fi

# 4. Xóa thư mục tạm và file zip sau khi hoàn tất
rm -rf temp_lfw lfw-dataset.zip
```

---

## 5. ✅ Kiểm Tra Tính Toàn Vẹn (Verification)

Sau khi hoàn tất việc giải nén và sắp xếp, hãy kiểm tra lại tính toàn vẹn của dữ liệu:

### Kiểm tra nhanh bằng Terminal:
```bash
# Đứng tại thư mục face_auth/
# 1. Kiểm tra số lượng người (kỳ vọng: ~5.749 thư mục)
find data/lfw -mindepth 1 -maxdepth 1 -type d | wc -l

# 2. Kiểm tra tổng số file ảnh (kỳ vọng: 13.233 ảnh .jpg)
find data/lfw -type f -name "*.jpg" | wc -l

# 3. Kiểm tra số lượng file metadata CSV (kỳ vọng: 10 files)
ls -1 data/lfw_metadata/*.csv | wc -l
```

### Hoặc kiểm tra bằng Python:
```python
import os

lfw_dir = "data/lfw"
meta_dir = "data/lfw_metadata"

assert os.path.isdir(lfw_dir), f"❌ Thiếu thư mục: {lfw_dir}"
assert os.path.isdir(meta_dir), f"❌ Thiếu thư mục: {meta_dir}"

identities = [d for d in os.listdir(lfw_dir) if not d.startswith(".")]
csv_files = [f for f in os.listdir(meta_dir) if f.endswith(".csv")]

print(f"✅ LFW Dataset sẵn sàng: {len(identities)} identities | {len(csv_files)} CSV metadata files.")
```

---

## 6. 🚀 Các Module Liên Quan Trong Dự Án

Dữ liệu sau khi thiết lập sẽ được các module và notebook sau sử dụng:

| Module / File | Vai trò |
| :--- | :--- |
| `evaluation/lfw_identity_benchmark.ipynb` | Đánh giá độ chính xác nhận diện, calibrate ngưỡng `MATCH_THRESHOLD` và phân tích lỗi (FAR, FRR, EER). |
| `evaluation/lfw_demo_enroll.ipynb` | Tạo danh tính mẫu, trích xuất embedding và đăng ký hàng loạt vào CSDL pgvector để kiểm thử. |
| `config.py` | Áp dụng các tham số tối ưu thu được từ benchmark LFW (`RECOMMENDED_GALLERY_SIZE = 3`, `MATCH_THRESHOLD = 0.35`). |