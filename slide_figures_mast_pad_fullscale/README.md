# MAST-PAD full-scale: material cho slide và paper

[Xem nhanh toàn bộ hình](preview_contact_sheet.png) · [Nguồn số liệu, caption và tài liệu tham khảo](figure_manifest.md)

| Hình | Nội dung | File |
|---|---|---|
| 1 | Main: full-scale AUC trên LCC Combined và CASIA Combined | [PNG](fig_main_ablation_two_target_auc.png) · [PDF](fig_main_ablation_two_target_auc.pdf) |
| 2 | LCC AUC và số tham số, tham khảo literature | [PNG](fig_lcc_auc_vs_params_context.png) · [PDF](fig_lcc_auc_vs_params_context.pdf) |
| 3 | CASIA AUC trong bối cảnh literature, có nhãn protocol | [PNG](fig_casia_auc_literature_context.png) · [PDF](fig_casia_auc_literature_context.pdf) |
| 4 | Luồng phương pháp MAST-PAD | [PNG](fig_mast_pad_method_flow.png) · [PDF](fig_mast_pad_method_flow.pdf) · [SVG](fig_mast_pad_method_flow.svg) |
| 5 | Diagnostics của run full-scale, best epoch 7 | [PNG](fig_mast_pad_training_diagnostics.png) · [PDF](fig_mast_pad_training_diagnostics.pdf) |
| 6 | Đường cong source-selection full-scale | [PNG](fig_training_curves_fullscale.png) · [PDF](fig_training_curves_fullscale.pdf) |
| 8 | Hình 100K hỗ trợ, dùng cho backup slide / appendix | [PNG](fig_supporting_100k_lcc_auc.png) · [PDF](fig_supporting_100k_lcc_auc.pdf) |

PNG 300 DPI dùng cho slide, PDF vector dùng cho paper, SVG để chỉnh sơ đồ. Số liệu lấy từ artifacts của từng run, giữ độ chính xác trước khi làm tròn trên hình.

CASIA trong hình chính là **video-level AUC**, LCC là **image-level AUC**. Các so sánh literature chỉ cung cấp bối cảnh do source domains và protocol khác nhau. Chi tiết kiểm tra nguồn và caveat trong manifest.

Figure 7 chưa tạo vì thiếu crop nguồn CelebA Real/Attack phù hợp: [ghi chú bổ sung dữ liệu](missing_frequency_examples_note.md).

Tái tạo: `python3 slide_figures_mast_pad_fullscale/generate_figures.py`. Không train, chạy model, đổi threshold hay sửa kết quả thí nghiệm.
