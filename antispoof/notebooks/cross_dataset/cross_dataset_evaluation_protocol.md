# E1 vs E3: cross-dataset zero-shot v1

Chạy sau khi E3 đã đóng băng. Câu hỏi là frequency có bổ trợ spatial dưới domain shift không. E1 và E3 cùng source CelebA-Spoof; E3 independent initialization; cùng source split/cache. Đây là **Project cross-dataset / zero-shot stress test**, không tự động là official benchmark.

## Target và provenance

Primary LCC-FASD (ảnh), CASIA-FASD (video/session); Replay-Attack optional khi có bản được phép sử dụng. Notebook không download dataset. Không fine-tune, adaptation, target calibration, target checkpoint selection hoặc target hyperparameter tuning. E2 diagnostic được bỏ khỏi implementation v1 để tập trung E1/E3.

Mỗi distribution phải khai báo identity/version/source/license/partition và label mapping. CASIA-FASD không được nhầm CASIA-SURF. Kaggle mount không tự chứng minh nguồn/quyền sử dụng. Nếu chưa xác minh, preflight dừng trước detector/PAD inference, yêu cầu xác nhận nguồn và quyền bằng flag cấu hình `provenance_confirmed`. Snapshot lưu root, config và metadata license JSON nếu có. Đây là xác nhận thủ công của người chạy; notebook không chứng nhận giấy phép.

## Adapter và sample manifest

Ảnh: một ảnh là một sample, không tạo video_id. LCC mặc định mapping `evaluation/real`→0 và `evaluation/spoof`→1; người chạy phải sửa theo layout thực tế, hoặc dùng CSV. Chọn evaluation partition khi distribution có train/development/evaluation. Không pool train/dev vào target mặc định.

Video CSV: `path,label,partition,video_id`. Extracted-frame CSV thêm `frame_index` integer, mapping video/session phải rõ. Label 0 Real / 1 Attack; optional subject_id/attack_subtype. Một video ID phải có nhãn duy nhất; duplicate paths/frame IDs làm fail. Nhãn không được suy đoán từ số thứ tự video.

`FRAMES_PER_VIDEO=16`, seed 42, uniform `np.linspace(0,n-1,min(16,n),dtype=int)` sau khi biết frame count; không dùng performance để chọn frame. Extracted frames sort frame_index rồi lấy uniformly đến 16. Lưu resolved manifest, file SHA256, unit manifest và exact sampled-frame manifest. Không đọc được frame count thì dừng và yêu cầu mapping extracted frames, không đoán sampling.

## External SCRFD và common valid set

Mount `det_500m.onnx`; không tự download. CPU SCRFD confidence .5, primary 640×640, fallback 480 và 320, crop 1.55×. Chọn face lớn nhất hợp lệ, tie confidence rồi bbox coordinates. Ghi n_faces, ambiguous multi-face, native minimum side, ảnh width/height, bbox, confidence, size detector dùng, status. Không CelebA annotation fallback.

Cache gắn SHA256 detector, hash sampled manifest (bao gồm hash source files) và policy/environment versions. Mismatch làm fail. READ_ERROR, NO_VALID_FACE, PREPROCESS_ERROR giữ trong cache/coverage. Hai model đánh giá cùng các candidate frame có preprocessing thành công. Model inference error dừng toàn bộ run; không âm thầm loại riêng một model.

Coverage gồm overall, từng dataset và Real/Attack, multi-face rate, failure counts; lưu frame coverage và unit coverage. Zero-valid-frame video là NO_VALID_FACE, không score/prediction; giữ trong manifest, loại khỏi conditional PAD metrics. `MIN_VALID_FRAMES_PER_VIDEO=1` cố định: video có ít frame vẫn dùng, flag low_coverage nếu số valid thấp hơn sampled candidates. Không đổi minimum sau khi nhìn kết quả.

## Score và metrics

Spatial dùng E1 RGB normalize, gamma OFF. E3 từ cùng crop lấy E1 spatial input + exact E2 DCT input (full-map signed-log/z-score, no mask).

Score `d=real_logit-logsumexp(spoof logits)`; image apply threshold trực tiếp. Video lấy arithmetic mean d của valid sampled frames rồi apply threshold. Mỗi model dùng threshold riêng khóa trên CelebA Validation; không sweep target thresholds, không EER calibration.

Primary AUC/HTER theo image cho image datasets, video/session cho video datasets. `HTER=(FAR+FRR)/2`; FAR là Attack→Real, FRR là Real→Attack; bằng ACER với binary definition này. Supplementary APCER/BPCER/ACER/Accuracy; tất cả rates trong `[0,1]`, không phải phần trăm. Thiếu lớp hoặc sample thì metric undefined null trong JSON/NaN trong CSV. AUC dùng Real positive và d cao nghĩa Real.

ΔAUC/ΔHTER E3−E1 là descriptive, không significance claim. In-domain summary optional: chỉ tính degradation delta khi metadata xác nhận cùng image evaluation unit và cùng metric units. CelebA image và CASIA video mean phải trình bày cạnh nhau kèm caveat, không coi là pure domain delta.

Face-size bins `<48`, `48–99`, `100–139`, `140–179`, `>=180`, threshold global. Với video đây chỉ là frame diagnostics, không sample video độc lập. Native pixels không biểu diễn camera distance chung giữa các dataset. Controlled near/far camera experiment chưa thực hiện.

## Outputs và diễn giải

ZIP giữ summary, protocol snapshot, model comparison, face-size breakdown, per-image/per-frame/per-video predictions, resolved/sampled/unit manifests, external cache, coverage CSVs, ROC và score distributions. Prefix run folder/ZIP chứa mode; archive convenience tên `cross_dataset_e1_vs_e3_results.zip` phải lưu dưới thư mục mode riêng.

Nếu E3 cải thiện nhiều target, evidence hỗ trợ complementary robust cues. Nếu chỉ source cải thiện, có thể source shortcut. Tương đương cùng overhead nhỏ là inconclusive. Cross-domain giảm cần kiểm tra domain sensitivity trước E4. Không điều chỉnh target preprocessing/aggregation từ kết quả. Kaggle CPU inference timing chỉ diagnostic; matched efficiency comparison nằm trong notebook E3, Raspberry Pi là experiment khác.
