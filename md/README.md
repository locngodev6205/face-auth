# Danh mục tài liệu face_auth

Đọc [trạng thái codebase](source_of_truth/current_codebase_state.md) để nắm source, runtime và module ownership. Với lượt sửa báo cáo MAST-PAD hiện tại, bắt đầu từ [số liệu đã xác minh](source_of_truth/MAST_PAD_REVISION_NUMBERS_SOURCE_OF_TRUTH.md). Đọc [AGENTS.md](../AGENTS.md) cho working rules. Source/config và artifact thực tế có ưu tiên hơn tài liệu; ghi chú lịch sử không phải yêu cầu thực thi.

```text
md/
├── README.md                     # Index này
├── source_of_truth/              # Trạng thái hiện tại và kết quả đã đối chiếu
├── runtime/                      # Contract/cách vận hành hoặc thử runtime
├── research/                     # Thiết kế PAD, literature references
├── plan/                         # Đánh giá và đề nghị chỉnh sửa tài liệu
└── archive/                      # Prototype/walkthrough/context cũ, không active truth
```

## Source of truth

- [Số liệu cho lượt sửa MAST-PAD](source_of_truth/MAST_PAD_REVISION_NUMBERS_SOURCE_OF_TRUTH.md): bảng bốn phương pháp seed100, baseline ba seed đã xác minh, mean ± sample SD, EER và các rate tại ngưỡng nguồn, confusion matrix, complexity và danh sách số cần sửa. Đây là điểm vào cho lượt sửa sau DG20 ngày 22/10/2026.
- [Kết quả huấn luyện nguồn đầy đủ](source_of_truth/FULLSCALE_RESULTS_SOURCE_OF_TRUTH.md): số liệu chi tiết và provenance cho CE-only, Clean + Aux, WBST và MAST-PAD.
- [Tổng hợp ba seed](source_of_truth/FULLSCALE_3SEEDS_SUMMARY_REPORT.md): bảng tổng hợp trước lượt audit. Khi dùng để sửa báo cáo, đối chiếu file số liệu đã xác minh ở trên; đặc biệt phân biệt CASIA frame/video và quy ước SD.
- [Trạng thái codebase](source_of_truth/current_codebase_state.md): snapshot source/runtime/module ownership.
- [Trạng thái PAD](source_of_truth/current_pad_state.md): snapshot mô hình và luồng PAD.
- [Recognition worker](source_of_truth/REALTIME_RECOGNITION_WORKER.md): thiết kế và hành vi nhận diện bất đồng bộ.
- [Kế hoạch tiếp nối slide MAST-PAD](source_of_truth/mast_pad_existing_deck_continuation_plan_FULLSCALE.md): tài liệu phục vụ trình bày, không thay thế kết quả thực nghiệm.

`source_of_truth/` và README này được mở theo dõi Git bằng ngoại lệ trong `face_auth/.gitignore`. Các thư mục tài liệu khác tiếp tục chịu quy tắc ignore hiện có; những tệp đã được Git theo dõi vẫn được theo dõi.

## Runtime

- [PAD runtime config](runtime/pad_runtime_config.md): JSON/env/default precedence và artifact contract.
- [Crop smoothing experiment](runtime/pad_crop_smoothing_experiment.md): feature được giữ, guards, lệnh chạy/tắt và giới hạn evidence.
- [Database unification proposal](runtime/database_unification_proposal.md): đề xuất hợp nhất dữ liệu; không phải lệnh thực thi hoặc bằng chứng đã triển khai.

## Đánh giá và kế hoạch chỉnh sửa

- [Đánh giá báo cáo Word ngày 20/10/2026](plan/MAST_PAD_DANH_GIA_VA_KE_HOACH_CHINH_SUA_BAO_CAO_WORD_FINAL_2026-10-20.md): đề nghị biên tập ở thời điểm đánh giá; số liệu và nhận xét cần đối chiếu với nguồn mới nhất.
- Hồ sơ DG20 audit ngày 22/10/2026 nằm tại `paper/review/dg20_evidence_audit/` trên máy. Thư mục `paper/` vẫn bị ignore; số liệu cần cho lượt sửa đã được tổng hợp vào `source_of_truth/MAST_PAD_REVISION_NUMBERS_SOURCE_OF_TRUTH.md`.

## Research

- [Spatial/frequency architecture v6](research/pad_spatial_frequency_architecture_v6_FULLSCALE_MAST_PAD.md): thiết kế MAST-PAD và kiến trúc PAD.
- [Frequency-domain references v6](research/pad_frequency_domain_references_v6_FULLSCALE_MAST_PAD.md): tài liệu học thuật về miền tần số trong chống giả mạo khuôn mặt.
- [PAD Paper Roadmap & Design Plan v7](research/PAD_PAPER_ROADMAP_AND_DESIGN_PLAN_v7_FULLSCALE_MAST_PAD.md): kế hoạch nghiên cứu và bài báo.
- [PAD Experiment Roadmap v5](research/pad_experiment_roadmap_v5_FULLSCALE_MAST_PAD.md): lộ trình thực nghiệm.
- [Notebook index](../antispoof/notebooks/README.md): Danh mục notebooks huấn luyện & held-out test.

## Archive

- [Previous context 2026-09-24](archive/current_codebase_state_2026-09-24.md): giữ mô tả cũ để truy vết, có dữ kiện đã lỗi thời.
- [Execution-flow walkthrough](archive/app_execution_flow_walkthrough.md): giải thích dài của phiên bản cũ; tra symbol trong source trước khi áp dụng.

## Cách duy trì context

Khi runtime/module/config thay đổi, cập nhật snapshot tương ứng trong `source_of_truth/` và tài liệu runtime liên quan. Ghi ngày, phạm vi và nguồn kiểm chứng. Audit cũ giữ scope/date và evidence, không viết lại thành kết quả mới. Kết quả training/evaluation phải gắn với protocol/run/artifacts riêng; phân biệt seed100 với aggregate ba seed, frame với video, AUC/EER với rate tại ngưỡng cố định. Không chép notebook outputs chưa chạy thành current state. Không đưa credentials hay ảnh gallery vào context.
