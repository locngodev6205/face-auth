# R7 harmful-gate ablation — Báo cáo thay đổi notebook

Ngày:28/10/2026. Nguồn: `/Users/3t/Downloads/notebook039491a34a.ipynb`. Bản gốc giữ nguyên; bản mới là [22_fullscale_R7_harmful_gate_ablation_crop15.ipynb](22_fullscale_R7_harmful_gate_ablation_crop15.ipynb).

## Câu hỏi và thiết kế

So sánh harmful-gated R7 với một variant tính CE trên mọi selected worst view, giữ sigma0.10. Mục tiêu là đo hướng và độ lớn của tác động gate policy cùng phép reduction H-versus-B; không giả định gate phải cải thiện kết quả.

| Cấu hình | R7 gốc | Ablation mới |
|---|---|---|
| Run ID | R7_SC_FULL_crop15 | R7_SC_NOGATE_FULL_crop15 |
| Method dispatch | HARMFUL | WORST |
| Gaussian sigma |0.10|0.10|
| Spectral loss |sum(h_i CE_i)/H; H=0 cho differentiable zero|mean_B CE_i của toàn bộ selected views|
| Active total loss |.75CEclean+.25CEharmful+Laux|.75CEclean+.25CEworst+Laux|
| Forward selected views |Toàn batch,train mode|Toàn batch,train mode|

Các yếu tố giữ nguyên: source usable484561, full Test-as-Val selection, DATA_SEED43, init cùng seed, crop1.5,80×80BGR[0,1], centers[1/12,1/4,5/12], gainUniform.65–.90 dùng chung các band/channel, FFT/DC/clamp, augmentation, clean-only auxiliary losses, SGD/batch256/max12/LRmilestones4/8, warmup1epoch, patience và sourceACER/AUC checkpoint rule. WBST/P3 cũ vẫn giữ sigma1/18.

## Mặc định và cách dùng

- Chạy riêng variant mới: `RUN_FILTER=[ABLATION_RUN_ID]`, mặc định seed101 giống notebook người dùng gửi.
- Đổi `TRAIN_SEED` thành100/101/102 để chạy từng repeat. Guard chấp nhận đúng ba seed này, giữ DATA_SEED43. Output root tự đổi theo seed: `/kaggle/working/22_fullscale_R7_harmful_gate_ablation_crop15_s{TRAIN_SEED}`.
- Config và resume vẫn kiểm equality, không nhận state của runID/config khác. Run ZIP và overall ZIP có seed trong tên, không trỏ về ZIP lịch sử.
- Muốn train lại matched pair trong thư mục mới: `RUN_FILTER=['R7_SC_FULL_crop15',ABLATION_RUN_ID]`. Đây là tùy chọn, không phải mặc định. `RUN_FILTER=None` sẽ chạy cả bốn run đã đăng ký.
- Không tự import historical R7 results. Khi dùng R7 cũ làm comparator, đối chiếu seed/initSHA/manifests/source/backbone/dependencies và recipe. Ghi rõ khác biệt môi trường Kaggle/Colab nếu có.
- Notebook giữ đánh giá source/LCC vốn có. CASIA đã được tích hợp trong cập nhật28/10 bên dưới, chạy sau source freeze; chưa chạy trên dữ liệu thật trong phiên chỉnh sửa.

## Thay đổi đã thực hiện

1. Thêm run mới bằng bản sao spec R7, chỉ đổi `method` sang WORST để tái sử dụng nhánh all-selected CE đã có.
2. Thêm fairness assertion yêu cầu spec ablation bằng spec R7 ngoài trường method; kiểm tra seed, cùng manifest/init hash và crop.
3. Thêm metadata `ablation` cho config run mới; export `r7_gate_ablation_protocol.json` cùng evaluation protocol, ghi đầy đủ giả thuyết/controls/metrics.
4. Mở rộng báo cáo/ZIP/pairwise delta `R7_minus_R7_NOGATE`. Delta chỉ được tính nếu cả hai run hoàn tất trong cùng output root, còn lại pending.
5. Thêm sanity checks no-gate: normalize theo toàn B, có gradient cho mọi selected sample khi H=0 hoặc H subset, full-batch train forward và BN updates giống recipe.
6. Xóa toàn bộ saved outputs/execution counts khỏi bản mới. Notebook có thể chạy sau khi sửa đường dẫn Kaggle cần thiết; không mang số đo cũ sang thành kết quả ablation.

`harmful_count/fraction=0` ở method WORST là counters không kích hoạt harmful loss, không phải bằng chứng mọi perturbation đều không giảm margin. Sử dụng worst CE,margin drop,flip rate và band frequencies để đọc diagnostics. Ghi chú này có trong protocol và markdown notebook.

## Kiểm tra đã chạy

```bash
python3 -m unittest tests.test_r7_gate_ablation_notebook -v
```

Kết quả:4/4test pass. Kiểm tra cấu hình/root cho cả3seed; notebook cells compile; clean outputs; invalid filter/spec/seed bị chặn; CPU synthetic tests về loss/gradient/BN/selector/RNG/patience đều pass. Kiểm tra export bằng metadata tổng hợp giả lập xác nhận chỉ run mới complete, comparator pending và ZIP tách seed. Không chạy toàn notebook, không đọc dữ liệu Kaggle, không train, không infer model thật, không chạy GPU/CASIA. `nbformat` không có trong môi trường; notebook được kiểm JSON/schema cơ bản và compile từng code cell.

Đã đối chiếu AST: `training_terms`, `select_worst_view`, `radial_masks`, `spectral_views`, `auxiliary_loss`, `run_one`, `metrics`, `threshold_from_val`, `init_model` giữ nguyên. Tức là body huấn luyện, optimizer/checkpoint selection, preprocessing phổ và metric không được viết lại để thêm variant.

Trước khi chạy GPU: gắn đúng resource bundle, pretrained model/source và dataset; giữ batch256/AMP/recipe. Khi hoàn tất, kiểm `config.json`, `completion.json`, `evaluation_freeze.json`, checkpointSHA, predictions/metrics, init/manifests SHA; so sánh cùng seed, sau đó aggregate mean±sampleSD cho100/101/102. Không tune trên target.

Source SHA256: `100a858c09a3739203c1417d39a04acab203cee5a699526bcaa44db2032f685f`.

Bản notebook mới và bài test là thay đổi code; không thay artifact thực nghiệm, paper, Word hoặc notebook nguồn. Chưa commit/push.

## Cập nhật28/10/2026 — Tích hợp CASIA như notebook21

Đã chuyển evaluator CASIA từ `21_fullscale_CE_clean_baseline_crop15.ipynb` sang notebook22, tổng cộng12cells. Notebook21 và file Downloads gốc giữ nguyên. Các body training, `run_one`, `run_config`, source-selection, spectral generator, loss và fairness contract không đổi; việc thêm evaluation không đổi training config để có thể tiếp tục source run đã lưu trước đó.

### Luồng chạy mới

1. Cell cấu hình: giữ mặc định chỉ run `R7_SC_NOGATE_FULL_crop15`, seed101; thêm mounts CASIA/SCRFD từ notebook21.
2. Preflight: kiểm GPU và SCRFD CUDA, thư mục ảnh và dependency trước khi bắt đầu training tốn thời gian. Dependency setup tương tự notebook21, có thể cần Kaggle Internet khi chưa cài đủ package.
3. Source training → source/LCC evaluation → export source results, giữ nguyên rule ACER/AUC trên Test-as-Val.
4. Reload đúng `best.pth` và kiểm SHA256 với `evaluation_freeze.json`; đánh giá CASIA bằng checkpoint đã khóa, không chọn checkpoint trên CASIA.
5. CASIA outputs được đưa vào cả per-run và overall ZIP. Nếu CASIA lỗi, source exports và các kết quả target đã ghi vẫn được giữ để kiểm tra/rerun.

### Cấu hình mặc định

- `RUN_CASIA=True` bật đánh giá tích hợp. Đặt False trước khi chạy nếu muốn source/LCC-only.
- `CASIA_ROOT='/kaggle/input/datasets/minhnh2107/casiafasd'`.
- `SCRFD_MODEL_PATH='/kaggle/input/datasets/locngodev/scrfd-model/det_500m.onnx'`.
- `CALIBRATE_CASIA_THRESHOLD=True` giống notebook21: lưu source-frozen results và thêm diagnostics ngưỡng cố định fit trên **CASIA train**. Đặt False trước lần CASIA đầu để chỉ chạy source-frozen; không có thành phần adaptive threshold.
- `MIN_VALID_FRAMES=4`, detector sizes640/480/320, confidence.5, crop1.5,80×80BGR[0,1], batch256; giữ nguyên protocol notebook21.

Train/test/combined đều có frame/video predictions và metrics. Combined dedup(video_id,frame_index), ưu tiên test copy trước detection; video score là mean valid-frame logit difference. Không fallback sang train khi test copy lỗi detection. Mức video yêu cầu ít nhất4validframes. Source-frozen rates và optional train-only threshold results được lưu riêng; đổi ngưỡng không thay AUC/EER trên cùng score/population.

### Output và resume

`runs/<run_id>/casia/` có:

- `evaluation_protocol.json`, `model_provenance.json`, `config.json`, `completion.json`.
- `train/`, `test/`, `combined/`: `frame_predictions.csv`, `video_predictions.csv`, `frame_metrics.json`, `video_metrics.json`.
- Summary CSV/JSON, ROC figures, detection/manifests/coverage, report Markdown và `casia.zip` ở run folder.
- Nếu calibration bật: calibration JSON theo frame/video, train threshold curves, `*_metrics_casia_calibrated.json`, `*_predictions_casia_calibrated.csv`, summary calibrated riêng.

Marker `runs/<run_id>/casia_completion.json` chỉ ghi sau khi evaluator hoàn tất và đủ các output bắt buộc. Marker gắn với checkpoint SHA, source-config/freeze SHA, seed, epoch, source threshold, SCRFD SHA, mounts, detector settings và calibration flag. Chỉ bỏ qua khi contract khớp và các file cần thiết còn đủ; khác checkpoint/settings hoặc mất file sẽ báo lỗi. Mục đích là tránh dùng nhầm kết quả cũ, không thay đổi training completion marker.

Cache `shared_crops_bgr_uint8.npy` và detection `.partial.csv` không vào ZIP, giống notebook21. Các metadata/prediction/metric cần tái kiểm được giữ. Trong training `ablation` metadata cũ, cụm “CASIA separately” mô tả evaluation tách khỏi source-selection; runner hiện đã tích hợp bước này sau source freeze.

### Kiểm tra cho cập nhật

```bash
MPLCONFIGDIR=/private/tmp/mast_r7_casia_mpl python3 -m unittest tests.test_r7_gate_ablation_notebook tests.test_r7_casia_notebook -v
```

**7/7tests pass.** Bốn test ablation/source-export cũ tiếp tục pass; ba test CASIA mới kiểm:

- Preflight/evaluator được định nghĩa trước runner, CASIA gọi sau source training.
- Synthetic frame detection/dedup, preferred test failure không fallback train, video mean và threshold4frame; output không đủframe đểprediction trống hợp lệ.
- Separate source-frozen/calibrated metrics, train-only calibration, AUC/EER invariant; checkpoint SHA/seed/run label đúng, resume skip khi khớp và chặn settings/checkpoint đổi hoặc output thiếu.
- ZIP chứa target metrics và bỏ crop cache lớn.

Không cài dependency qua pip, không gọi detector/PAD thật, không đọc dataset CASIA/Kaggle và không train/GPU trong các test. Synthetic tests dùng ảnh màu đồng nhất, fake detector và toy CPU scores. Test không chứng minh SCRFD CUDA/dependency hoạt động trên Kaggle; notebook preflight sẽ kiểm điều đó khi người dùng chạy.

## Đối chiếu trực tiếp notebook20 —28/10/2026

**Kết luận:** phần CASIA của notebook22 đồng bộ với notebook20 về protocol và tính toán. Không cần sửa toán học hoặc quy tắc population. Đã thêm regression tests và ghi checksum notebook20 trong notebook22 metadata để lưu nguồn đối chiếu.

| Thành phần | Kết quả đối chiếu |
|---|---|
| Filename/video-code mapping | EXPECTED_CODES và FILENAME_RE trùng AST; cùng cross-check real/fake label |
| Input manifest | `read_split`, `combined_unique`, `video_manifest` trùng AST; cùng sort và test-copy precedence |
| Detector/crop | SCRFD640/480/320,confidence.5;largest valid box;`detect_boxes` và `mini_crop_bgr` trùng AST |
| Input/network wrapper |80×80BGR[0,1],batch256;`PADNet` class trùng AST; model.eval/inference_mode,AMP theo source config |
| Frame score | logitReal−logitAttack sau chuyển logits float32; cùng source threshold và >= decision |
| Video/coverage | mean valid-frame score,minimum4frames; unscorable videos giữ trong coverage nhưng không vào metrics |
| AUC/EER/rates | `metrics` và `safe_metrics` trùng AST,bao gồm single-class null và coverage denominator |
| Optional train-only calibration | `optimize_casia_threshold` trùng AST;fit trên train riêng frame/video,giữ source-frozen files |
| Dataset overlap/protocol | cùng extracted-color train/test/combined,không official subject-disjoint claim |

Có8function definitions và2mapping/regex assignments trùng AST; PADNet wrapper cũng trùng.15backbone definitions được embed trong notebook20 trùng AST với `antispoof/models/MiniFASNet.py` hiện có, SHA256 `e498c4ec5e1ddfaba62b941a126c19d65aa564999f3309661fe43ee8bf38acd7`. Notebook22 nạp MiniFASNet từ Kaggle mount giống training; kiểm `initialization_audit.json.source_sha256` khi chạy thật để đảm bảo đúng bản source. Kiểm tra local không tự chứng minh file được gắn trên Kaggle giống byte-for-byte.

### Regression test đầu-cuối

Chạy nguyên các evaluation cells6–10 của notebook20 và evaluator notebook22 trên cùng fixture tổng hợp. Detector/weights loader được thay bằng fake detector/toy model; không dùng dataset hay model nghiên cứu. Kiểm hai chế độ calibration False/True:

- MetricsJSON và predictionsCSV của frame/video tại train/test/combined khớp hoàn toàn.
- Với calibration bật, cả calibrated metricsJSON và predictionsCSV cũng khớp.
- Detection metadata,Combined manifest và video coverage khớp.
- Fixture chứa overlap,preferred test-copy detection failure và một video không đủ4frame,để kiểm các trường hợp ảnh hưởng population thực sự.

**9/9tests pass** với lệnh unittest đã ghi ở phần cập nhật phía trên. Các test mới: `test_notebook20_algorithm_ast_alignment`, `test_notebook20_end_to_end_parity_on_same_synthetic_inputs`. Không chạy lại training,inference model thật hoặc CUDA.

### Các khác biệt có chủ đích

1. Notebook20 là evaluator độc lập nhiều checkpoint từ input mounts; notebook22 đánh giá model vừa train bằng exact local best/freeze. Notebook22 có thêm checkpoint/config/freeze SHA và completion settings guard.
2. Notebook20 ghi `<output>/runs/<run_id>/<split>/`; notebook22 ghi `<output>/runs/<run_id>/casia/<split>/`. Các đơn vị,score và metric không thay đổi.
3. Notebook20 có global SEED100 và per-run train_seed trong metadata; notebook22 dùng actual TRAIN_SEED của run. Trong eval-mode với cùng weights/input đây không phải thay đổi model-selection protocol; không kết luận bitwise CUDA parity chỉ từ test CPU.
4. Logging cadence notebook20 là5batch,notebook22 kế thừa25batch; không ảnh hưởng batching256 hoặc metric.
5. Registry RUN_IDS notebook20 hiện chưa có `R7_SC_NOGATE_FULL_crop15`. Đánh giá tích hợp notebook22 đã hỗ trợ đúng ID này; nếu sau này chạy bằng notebook20 độc lập thì phải bổ sung run discovery/label cho ID mới. Không sửa notebook20 trong lượt này.

Notebook20 SHA256 tham chiếu: `c3ad47409fb4abeabd16347c74ec1f2be9a1be59a35956b8a7ac51d1da47d04f`.
