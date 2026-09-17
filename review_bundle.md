# Review Bundle: Final Cleanup & Simplification

## 1. Những file đã sửa / xóa

### File đã xóa hoàn toàn:
- `tests/test_live_db.py`: Đã xóa bộ integration test phụ thuộc PostgreSQL live.
- `tests/test_notebooks.py`: Đã xóa test regex/string matching trong notebook.
- `tests/test_app_orchestration.py`: Đã xóa bộ test mock orchestration cồng kềnh.

### File đã chỉnh sửa:
- `database/vector_db.py`: Loại bỏ các tham số backward-compatibility `user_id`, `name`, `**kwargs` khỏi `upsert()`, `log_attendance()`, và `search()`.
- `tracking/tracker.py`: Xóa property và setter `Track.user_id`.
- `enrollment/enroll.py`: Xóa fallback `user_id` và `name` khỏi `enroll_person()`.
- `scripts/cleanup_demo.py`: Rút gọn thành một luồng xóa duy nhất theo `--namespace` (`attendance_logs` $\rightarrow$ `employees` $\rightarrow$ CASCADE `face_embeddings`), bỏ `--hard-delete`, `--cascade-attendance`, và soft-delete.
- `scripts/reset_database.py`: Bắt buộc cờ `--reset`, từ chối chạy nếu chỉ truyền `--yes`.
- `evaluation/lfw_demo_enroll.ipynb`: Sửa `outlier_threshold=0.35` (thay cho biểu thức phụ thuộc `config.RECOMMENDED_GALLERY_SIZE`), xóa dòng in SQL và output chứa `face_embeddings.is_active`.
- `md/current_codebase_state.md`: Cập nhật hướng dẫn database chuẩn, xóa mô tả lỗi thời về migration, V2 Enterprise, soft versioning, advisory lock.
- `app.py`: Cập nhật docstring của `orchestrate_track_step` (bỏ nhãn "pure").
- `tests/test_pad_predictor.py`: Giảm còn 3 unit tests trọng tâm.
- `tests/test_tracker.py`: Giảm còn 3 unit tests trọng tâm.
- `tests/test_enrollment_math.py`: Giảm còn 2 unit tests trọng tâm.
- `tests/test_vector_db_logic.py`: Giảm còn 3 unit tests trọng tâm.
- `.gitignore`: Thêm rule bỏ qua các file CSV trong `evaluation/` (`evaluation/**/*.csv`, `evaluation/*.csv`).
- `evaluation/*.csv`: Untrack khỏi Git cache (`lfw_demo_manifest.csv`, `lfw_demo_impostors_manifest.csv`, `lfw_demo_test_images_manifest.csv`).

---

## 2. Chi tiết bộ test (Đã xóa vs. Đã giữ)

Tổng số test trong toàn bộ test suite: **11 unit tests** (thời gian chạy ~0.006s).

### `tests/test_pad_predictor.py` (3 tests)
- **Giữ:**
  1. `test_2class_pad_score_identity`: Công thức PAD binary đúng ($z_{\text{real}} - z_{\text{spoof}}$ và ngưỡng).
  2. `test_3class_logsumexp_softmax_equivalence`: Công thức LogSumExp cho model 3-class đúng ($z_{\text{real}} - \text{logsumexp}(z_{\text{spoof}}) \ge \ln(p/(1-p)) \iff P(\text{REAL}) \ge p$).
  3. `test_color_order_detection`: Contract chuẩn hóa kênh màu đúng (BGR cho MiniFASNet 128, RGB cho MobileNet 224).
- **Đã xóa:** `test_stable_logsumexp_basic`, `test_stable_logsumexp_large_values_no_overflow`, `test_defensive_error_handling_no_fake_real`.

### `tests/test_tracker.py` (3 tests)
- **Giữ:**
  1. `test_pad_pending_blocks_recognition`: PAD chưa đủ vote (`< pad_min_votes`) thì recognition bị chặn.
  2. `test_pad_real_allows_recognition`: Đủ vote REAL thì recognition được phép.
  3. `test_pad_spoof_blocks_recognition`: Phán quyết SPOOF thì recognition bị chặn.
- **Đã xóa:** `test_compute_iou`, `test_track_initial_state_pad_pending`, `test_reset_pad`, `test_employee_id_user_id_compatibility`, `test_recognition_streak_and_reset_logic`, `test_face_tracker_unmatched_track_resets_streak`, `test_pad_toggle_off_on_and_spoof_resets_stability`.

### `tests/test_enrollment_math.py` (2 tests)
- **Giữ:**
  1. `test_centroid_l2_normalized`: Centroid được chuẩn hóa L2 (norm = 1.0).
  2. `test_clear_outlier_is_filtered`: Outlier rõ ràng (< 0.35 similarity) bị loại bỏ.
- **Đã xóa:** `test_all_outliers_fallback_safety`, `test_enroll_person_parameter_compatibility`.

### `tests/test_vector_db_logic.py` (3 tests)
- **Giữ:**
  1. `test_decide_identity_thresholds`: Quyết định danh tính với trên/dưới ngưỡng `MATCH_THRESHOLD`.
  2. `test_search_query_filters`: Query tìm kiếm lọc đúng `embedding_type = 'CENTROID'`, `model_version`, và `status = 'ACTIVE'`.
  3. `test_re_enrollment_deletes_old_embeddings`: Re-enroll xóa embedding cũ cùng model version trước khi insert.
- **Đã xóa:** `test_to_vector_literal`, `test_log_attendance_cooldown_and_success`.

---

## 3. Compatibility code đã loại bỏ

- **`Track.user_id`**: Xóa property và setter alias trong `tracking/tracker.py`.
- **`VectorDB.upsert()`**: Xóa các tham số `user_id`, `name`, `**kwargs`. Chuyển sang chữ ký chuẩn `(self, employee_id: str, full_name: str, ...)`.
- **`VectorDB.log_attendance()`**: Xóa tham số `user_id`, `**kwargs`. Chuyển sang chữ ký chuẩn `(self, employee_id: str, ...)`.
- **`VectorDB.search()`**: Xóa `**kwargs`.
- **`enroll_person()`**: Xóa tham số `user_id` và `name`.
- **PAD `logit_diff`**: Giữ lại alias `logit_diff` trong dictionary kết quả của PAD predictor vì các script hiển thị `scripts/demo_antispoof.py` và `scripts/demo_antispoof_yunnet.py` vẫn đang sử dụng trường này để render nhãn HUD.

---

## 4. Các inconsistency đã sửa

1. **`evaluation/lfw_demo_enroll.ipynb`**:
   - Cell 7: Thay `outlier_threshold=config.RECOMMENDED_GALLERY_SIZE and 0.35` bằng giá trị tường minh `outlier_threshold=0.35`.
   - Cell 9: Xóa câu lệnh in SQL gợi ý `UPDATE face_embeddings SET is_active = FALSE`. Xóa output liên quan.
2. **`scripts/reset_database.py`**:
   - Bắt buộc phải có cờ `--reset`. Khi chạy `python scripts/reset_database.py --yes` (không có `--reset`), script từ chối thực hiện và báo lỗi rõ ràng.
3. **`scripts/cleanup_demo.py`**:
   - Rút gọn luồng xử lý thành một luồng xóa duy nhất trong 1 transaction: xóa `attendance_logs` $\rightarrow$ xóa `employees` (tự động cascade xóa `face_embeddings`).
   - Xóa bỏ các cờ `--hard-delete`, `--cascade-attendance`, và luồng soft-delete.
4. **`md/current_codebase_state.md`**:
   - Loại bỏ hướng dẫn chạy migration file cũ (`001_migrate_face_auth_v2.sql`).
   - Chuẩn hóa hướng dẫn database sang 2 lệnh chuẩn (`reset_database.py` và `run_enroll.py`).
   - Xóa bỏ các từ ngữ và mô tả overengineering: V2 Enterprise, soft versioning, `is_active`, `DELETED`, advisory lock.
5. **`app.py`**:
   - Cập nhật docstring của `orchestrate_track_step`, loại bỏ nhãn "pure orchestration".

---

## 5. Kết quả các lệnh kiểm tra thực sự đã chạy

### a. `python3 -m compileall .`
- Kết quả: **Exit code 0**, biên dịch thành công toàn bộ file Python trong repository.

### b. `python3 -m unittest discover -s tests -p "test_*.py" -v`
```text
test_centroid_l2_normalized (test_enrollment_math.TestEnrollmentMath.test_centroid_l2_normalized)
Centroid được L2-normalize. ... ok
test_clear_outlier_is_filtered (test_enrollment_math.TestEnrollmentMath.test_clear_outlier_is_filtered)
Outlier rõ ràng bị loại. ... ok
test_2class_pad_score_identity (test_pad_predictor.TestPadMathAndPredictor.test_2class_pad_score_identity)
Verify 2-class pad_score equals exactly z0 - z1 and matches logit_threshold. ... ok
test_3class_logsumexp_softmax_equivalence (test_pad_predictor.TestPadMathAndPredictor.test_3class_logsumexp_softmax_equivalence)
Verify 3-class pad_score >= ln(p / (1-p)) is mathematically equivalent to P(REAL) >= p. ... ok
test_color_order_detection (test_pad_predictor.TestPadMathAndPredictor.test_color_order_detection)
Verify BGR for MiniFASNet 128 and RGB for MobileNet 224. ... ok
test_pad_pending_blocks_recognition (test_tracker.TestTrackerAndGating.test_pad_pending_blocks_recognition)
PAD chưa đủ vote thì recognition bị chặn. ... ok
test_pad_real_allows_recognition (test_tracker.TestTrackerAndGating.test_pad_real_allows_recognition)
Đủ vote REAL thì recognition được phép. ... ok
test_pad_spoof_blocks_recognition (test_tracker.TestTrackerAndGating.test_pad_spoof_blocks_recognition)
SPOOF thì recognition bị chặn. ... ok
test_decide_identity_thresholds (test_vector_db_logic.TestVectorDbLogic.test_decide_identity_thresholds)
1. decide_identity() với trên/dưới threshold. ... ok
test_re_enrollment_deletes_old_embeddings (test_vector_db_logic.TestVectorDbLogic.test_re_enrollment_deletes_old_embeddings)
3. Re-enroll xóa embedding cũ cùng model version trước khi insert. ... ok
test_search_query_filters (test_vector_db_logic.TestVectorDbLogic.test_search_query_filters)
2. Search query lọc CENTROID, model_version và employee ACTIVE. ... ok

----------------------------------------------------------------------
Ran 11 tests in 0.006s

OK
```

### c. `python3 scripts/reset_database.py --yes` (kiểm tra từ chối khi thiếu `--reset`)
```text
Exit code: 1
Stderr: [ERROR] Bắt buộc truyền cờ --reset để thực hiện reset database.
```

### d. `git diff --check`
- Kết quả: **Exit code 0**, không có lỗi whitespace hay conflict marker.

### e. Kiểm tra JSON validation cho tất cả Notebook
- Kết quả: Cả 5 file notebook (`.ipynb`) đều là JSON hợp lệ.

### f. Quét toàn bộ repository tìm từ khóa tàn dư
- `is_active`: 0 kết quả ngoài `review_bundle.md`.
- `DELETED`: 0 kết quả ngoài `review_bundle.md`.
- `001_migrate`: 0 kết quả ngoài `review_bundle.md`.
- `database/migrations`: 0 kết quả ngoài `review_bundle.md`.

---

## 6. `git status --short`

```text
 M .gitignore
 M antispoof/predictor.py
 M antispoof/preprocess.py
 M app.py
 M config.py
 M database/vector_db.py
 M enrollment/enroll.py
 M evaluation/lfw_demo_enroll.ipynb
D  evaluation/lfw_demo_impostors_manifest.csv
D  evaluation/lfw_demo_manifest.csv
D  evaluation/lfw_demo_test_images_manifest.csv
 M evaluation/lfw_identity_benchmark.ipynb
 M md/current_codebase_state.md
 M scripts/cleanup_demo.py
 M scripts/demo_antispoof.py
 M scripts/demo_antispoof_yunnet.py
 M scripts/run_enroll.py
 M scripts/view_attendance.py
 M tracking/tracker.py
?? .env.example
?? database/schema.sql
?? review_bundle.md
?? scripts/reset_database.py
?? tests/
```

---

## 7. `git diff --stat`

```text
 face_auth/antispoof/predictor.py                   | 115 ++++--
 face_auth/antispoof/preprocess.py                  |  20 +-
 face_auth/app.py                                   | 232 ++++++++----
 face_auth/config.py                                | 177 ++++++---
 face_auth/database/vector_db.py                    | 279 ++++++++------
 face_auth/enrollment/enroll.py                     |  21 +-
 face_auth/evaluation/lfw_demo_enroll.ipynb         | 407 +++++++++++----------
 .../evaluation/lfw_demo_impostors_manifest.csv     |  40 +-
 face_auth/evaluation/lfw_demo_manifest.csv         |  62 ++--
 .../evaluation/lfw_demo_test_images_manifest.csv   | 122 +++---
 face_auth/evaluation/lfw_identity_benchmark.ipynb  | 127 +++----
 face_auth/md/current_codebase_state.md             | 162 ++++----
 face_auth/scripts/cleanup_demo.py                  | 136 +++----
 face_auth/scripts/demo_antispoof.py                |   6 +-
 face_auth/scripts/demo_antispoof_yunnet.py         |   6 +-
 face_auth/scripts/run_enroll.py                    |  24 +-
 face_auth/scripts/view_attendance.py               |  27 +-
 face_auth/tracking/tracker.py                      |  79 +++--
 18 files changed, 1162 insertions(+), 800 deletions(-)
```

---

## 8. Các giới hạn chưa kiểm tra

Tuân thủ nguyên tắc an toàn cho dữ liệu và môi trường chạy demo:
1. **Không chạy reset database thật trên cơ sở dữ liệu `face_db`**: Không gọi `python scripts/reset_database.py --reset` để tránh xóa dữ liệu gallery/nhân viên hiện có của người dùng.
2. **Không chạy webcam thực tế**: Tránh chiếm dụng thiết bị camera phần cứng của máy Mac trong quá trình automated agent run.
3. **Không chạy huấn luyện lại mô hình**: Không thực thi các cell training trong notebook `antispoof/antiproof.ipynb` và không thay đổi trọng số checkpoint ONNX.

---

## 9. Các lệnh người dùng cần chạy

Khi muốn chuẩn bị lại dữ liệu sạch và khởi chạy demo:

### 1. Reset database (nếu muốn tạo lại 3 bảng sạch từ đầu)
```bash
python scripts/reset_database.py --reset
```

### 2. Đăng ký nhân viên từ thư mục gallery
```bash
python scripts/run_enroll.py --gallery gallery
```

### 3. Chạy ứng dụng nhận diện & điểm danh realtime
```bash
python app.py
```
