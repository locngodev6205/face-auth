# Material cho slide và paper MAST-PAD

[Xem nhanh toàn bộ hình](preview_contact_sheet.png) · [Nguồn dữ liệu, số liệu chính xác và caption](figure_manifest.md)

| Figure | Nội dung | File |
|---|---|---|
| 1 | Main: Clean / WBST / MAST-PAD, LCC Combined AUC | [PNG](fig_main_lcc_combined_auc.png) · [PDF](fig_main_lcc_combined_auc.pdf) |
| 2 | AUC và số tham số; các công trình tham khảo có protocol khác | [PNG](fig_lcc_auc_vs_params_context.png) · [PDF](fig_lcc_auc_vs_params_context.pdf) |
| 3 | Luồng train và inference MAST-PAD | [PNG](fig_mast_pad_method_flow.png) · [PDF](fig_mast_pad_method_flow.pdf) · [SVG](fig_mast_pad_method_flow.svg) |
| 4 | Harmful fraction, margin drop, flip rate và phân bố band | [PNG](fig_mast_pad_training_diagnostics.png) · [PDF](fig_mast_pad_training_diagnostics.pdf) |
| 5 | Main: Validation AUC và ACER, đánh dấu checkpoint được chọn | [PNG](fig_training_curves_val_auc_acer.png) · [PDF](fig_training_curves_val_auc_acer.pdf) |
| 7 | Kết quả full-scale bổ trợ trên LCC Evaluation / Combined | [PNG](fig_fullscale_scaleup_summary.png) · [PDF](fig_fullscale_scaleup_summary.pdf) |

PNG 300 DPI dùng cho slide; PDF vector dùng cho paper; SVG dùng để chỉnh sơ đồ. Tên phương pháp và caption trên hình dùng tiếng Anh.

Figure 6 chưa tạo vì thiếu ảnh nguồn Real/Attack từ Train100K. [Ghi chú dữ liệu cần bổ sung](missing_frequency_examples_note.md).

Số liệu thực nghiệm đọc từ từng run; số liệu tham khảo của Figure 2 lấy theo prompt. Kết quả 100K là kết quả chính, một seed. Full-scale dùng Test-as-Val và protocol khác, được thể hiện riêng.

Tái tạo: `python3 slide_figures_mast_pad_100k/generate_figures.py`. Script chỉ đọc artifacts hiện có và ghi file vào thư mục material.
