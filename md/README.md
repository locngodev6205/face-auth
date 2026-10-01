# Danh mục tài liệu face_auth

Đọc [current_codebase_state.md](current_codebase_state.md) trước: context ngắn về source/runtime/module ownership và config tại snapshot. Đọc [AGENTS.md](../AGENTS.md) cho working rules. Source/config thực tế có ưu tiên hơn tài liệu; ghi chú lịch sử không phải yêu cầu thực thi.

```text
md/
├── current_codebase_state.md     # Điểm vào cho agent; cập nhật khi behavior thay đổi
├── README.md                     # Index này
├── runtime/                      # Contract/cách vận hành hoặc thử runtime
├── audits/                       # Evidence, regression và analysis có scope/date
├── research/                     # Thiết kế PAD, literature references
└── archive/                      # Prototype/walkthrough/context cũ, không active truth
```

## Runtime

- [PAD runtime config](runtime/pad_runtime_config.md): JSON/env/default precedence và artifact contract.
- [Crop smoothing experiment](runtime/pad_crop_smoothing_experiment.md): feature được giữ, guards, lệnh chạy/tắt và giới hạn evidence.

## Audits

- [Runtime pipeline audit 2026-09-26](audits/runtime_pipeline_audit_2026-09-26.md): lỗi runtime và offline regressions; xem status fixes trong nội dung.
- [Old commits comparison](audits/runtime_old_commits_comparison.md): b70172a/74b2785, model contracts và cached-state differences.
- [PAD runtime accuracy optimization](audits/pad_runtime_accuracy_optimization.md): preprocessing/crop/input sensitivity, log evidence và proposals; proposals không có nghĩa đã triển khai. Context chính ghi trạng thái mới nhất.
- [E1/E2 face-size audit](audits/e1_e2_face_size_audit.md): augmentation/size strata và giới hạn diễn giải; phần “chưa thấy E3” là observation của thời điểm audit, E3 notebooks hiện đã tồn tại.

## Research

- [Spatial/frequency architecture v4](research/pad_spatial_frequency_architecture_v4_CSMR.md): Thiết kế & thảo luận kiến trúc kết hợp đa miền không gian - tần số.
- [Frequency-domain references v4](research/pad_frequency_domain_references_v4_CSMR.md): Tổng hợp tài liệu học thuật về miền tần số trong chống giả mạo khuôn mặt.
- [PAD Paper Roadmap & Design Plan v4](research/PAD_PAPER_ROADMAP_AND_DESIGN_PLAN_v4_CSMR.md): Kế hoạch nghiên cứu và bài báo khoa học.
- [PAD Experiment Roadmap v2](research/pad_experiment_roadmap_v2_CSMR.md): Lộ trình thực nghiệm đánh giá mô hình.
- [Notebook index](../antispoof/notebooks/README.md): Danh mục notebooks huấn luyện & held-out test.

## Archive

- [Previous context 2026-09-24](archive/current_codebase_state_2026-09-24.md): giữ mô tả cũ để truy vết, có dữ kiện đã lỗi thời.
- [Prototype v1](archive/face_recognition_prototype_v1.md) và [prototype v2](archive/face_recognition_prototype_v2.md): lịch sử thiết kế.
- [Execution-flow walkthrough](archive/app_execution_flow_walkthrough.md): giải thích dài của phiên bản cũ; tra symbol trong source trước khi áp dụng.
- [Q&A notes](archive/note.md): ghi chú học tập/lý giải; không phải specification đã kiểm chứng.

## Cách duy trì context

Khi runtime/module/config thay đổi, cập nhật context chính và tài liệu runtime liên quan. Audit cũ giữ scope/date và evidence, không viết lại thành kết quả mới. Kết quả training/evaluation phải gắn với protocol/run/artifacts riêng; không chép notebook outputs chưa chạy thành current state. Không đưa credentials hay ảnh gallery vào context.
