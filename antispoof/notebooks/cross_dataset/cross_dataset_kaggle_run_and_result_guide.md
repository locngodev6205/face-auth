# Chạy cross-dataset E1 vs E3 trên Kaggle

## Input và thứ tự chạy

1. Huấn luyện/freeze E3; chạy held-out E1/E3 riêng nếu cần in-domain reference.
2. Import `antispoof_cross_dataset_e1_vs_e3_v1.ipynb`.
3. Attach E1 `mnv3s_e1_<mode>_v5_3_edge_best.onnx`, `.onnx.data` sidecar nếu có, runtime_config, best_meta và run_config.
4. Attach E3 `e3_concat_<mode>_v1_best.onnx`, runtime_config và best_meta cùng mode. Checkpoints/E2 weights không cần.
5. Attach target LCC-FASD/CASIA-FASD đã kiểm tra nguồn/quyền, và SCRFD `det_500m.onnx`. Detector phải là frozen SCRFD-500M tương ứng, không thay bằng model quantized khác.
6. Bật Internet cho dependencies; evaluation dùng CPU ORT 2 intra/1 inter threads để inference hoàn tất đồng bộ. Dataset/model detector không tự download.

## Cấu hình distribution và labels

Đặt `E1_RUN_MODE`, `E1_ARTIFACT_DIR`, `E3_ARTIFACT_DIR` (blank discovery được hỗ trợ cho artifact model), `LCC_FASD_ROOT`, `CASIA_FASD_ROOT`, `SCRFD_MODEL_PATH`. Optional `REPLAY_ATTACK_ROOT`; blank tắt dataset. Chạy đủ cả hai primary targets trước khi gọi comparison hoàn chỉnh.

Trong `DATASET_CONFIG`, điền version, source URL/identifier, license, partition chính xác. Kiểm tra nguồn/permission rồi mới đặt `provenance_confirmed=True`. Có thể thêm path các metadata JSON vào `LICENSE_METADATA_PATHS`. Việc đặt flag là xác nhận của bạn, không phải chứng nhận tự động.

LCC image default folder mapping là **ví dụ có thể chỉnh**, `evaluation/real` và `evaluation/spoof`; không giả định mọi Kaggle copy dùng cây thư mục này. Nếu path là `LCC_FASD_evaluation/real`, đổi mapping tương ứng. Chỉ map folder thuộc evaluation partition đã xác minh. Hoặc set manifest_csv:

```csv
path,label,partition,subject_id,attack_subtype
evaluation/real/a.jpg,0,evaluation,s01,real
evaluation/spoof/b.jpg,1,evaluation,s02,print
```

CASIA video yêu cầu manifest_csv, kind `video`, partition `test` nếu distribution đúng test. Ví dụ cấu trúc CSV (đây là schema, không phải mapping nhãn CASIA thực tế):

```csv
path,label,partition,video_id,subject_id,attack_subtype
test/subject_a/live.avi,0,test,subject_a_live,subject_a,real
test/subject_a/attack.avi,1,test,subject_a_attack,subject_a,replay
```

Nếu đã trích frame, kind `extracted_frames`, thêm `frame_index` đúng thứ tự gốc:

```csv
path,label,partition,video_id,frame_index
frames/live/frame_0001.jpg,0,test,live_session,1
frames/attack/frame_0001.jpg,1,test,attack_session,1
```

Không suy ra nhãn CASIA từ tên số mà chưa có metadata nguồn. Không gộp CASIA-SURF vào CASIA-FASD. CSV path relative target root; manifest có thể được mount bằng một Kaggle Input riêng. Không tạo video_id cho LCC image-only.

## Preflight, cache và inference

Run theo cell order. Manifest audit in dataset, mounted path, Real/Attack, subtype, number images/videos, candidate frames. Kiểm tra các số này trước khi chạy detector. Không hiểu nhãn thì sửa mapping dựa metadata, không dựa predictions.

Sampling cố định uniform tối đa 16 frame/video, min valid=1; không đổi vì performance. Detector cache tạo trong output. Run sau set `EXTERNAL_CACHE_INPUT_PATH` đến cache mounted cũ; source file hashes, manifest, detector hash, versions và policy phải khớp. Không xóa failures để nâng coverage.

Hai model dùng cùng valid frames, own locked source threshold. Output giữ zero-valid-face videos, flag low coverage. Inference lỗi sẽ fail, không xuất kết luận paired từ run chưa hoàn tất.

Optional `E1_INDOMAIN_SUMMARY`, `E3_INDOMAIN_SUMMARY`: summaries với AUC/ACER rates và evaluation unit metadata; notebook không đoán units thiếu metadata. Image/video comparison có caveat.

## Download và lưu trữ

Output folder `/kaggle/working/cross_dataset_<mode>_v1/`, mode ZIP `cross_dataset_<mode>_v1_results.zip`; convenience ZIP `cross_dataset_e1_vs_e3_results.zip` chứa cùng kết quả.

```text
cross_dataset_summary.json
cross_dataset_protocol_snapshot.json
cross_dataset_model_comparison.csv
cross_dataset_face_size_breakdown.csv
lcc_fasd_image_predictions.csv
casia_fasd_video_predictions.csv
casia_fasd_frame_predictions.csv
external_dataset_manifest.csv / .json
sampled_frame_manifest.csv
evaluation_unit_manifest.csv
external_scrfd_cache.json
frame_coverage.csv
unit_coverage_and_predictions.csv
coverage_by_class.csv
roc_lcc_fasd.png / roc_casia_fasd.png
score_distribution_lcc_fasd.png / score_distribution_casia_fasd.png
```

Ngoài ra có per-model plots, optional Replay frame/video predictions, optional in_domain_comparison.csv. File cho dataset chưa bật sẽ không có; `primary_targets_complete` trong summary báo đủ hai primary chưa.

Lưu `pad_experiments/cross_dataset_v1/<mode>/` gồm `manifests/`, `detector_cache/`, `predictions/`, `plots/`, ZIP gốc, executed notebook và Kaggle dataset/kernel version. Giữ source model metadata, sidecar ONNX, fingerprints. Archive E1/E2/E3 song song như hướng dẫn E3. Không ghi đè modes hoặc distributions. Các CSV/cache này là evidence dùng lại cho E4/paper sau này; chưa có claim E3 thắng trước khi chạy.
