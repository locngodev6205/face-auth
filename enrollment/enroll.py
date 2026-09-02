import numpy as np


def build_identity_embedding(embeddings, outlier_threshold=0.35):
    """
    Trung bình cộng các embedding (đã L2-normalize từng cái) thành một vector
    đại diện cho identity. Trước khi trung bình, loại các ảnh có similarity
    quá thấp so với mean sơ bộ — thường là dấu hiệu detect sai mặt, mặt bị
    che, góc nghiêng quá, ảnh mờ... để tránh kéo lệch vector đại diện.

    Trả về (identity_embedding, warnings, valid_embeddings) — warnings là list string để caller
    tự log/hiển thị, không raise exception trừ khi hoàn toàn không có dữ liệu.
    valid_embeddings là list các vector cá nhân đã vượt qua màng lọc outlier.
    """
    matrix = np.stack(embeddings).astype(np.float32)

    prelim_mean = matrix.mean(axis=0)
    prelim_norm = np.linalg.norm(prelim_mean)
    if prelim_norm == 0:
        raise ValueError("Invalid mean embedding")
    prelim_mean = prelim_mean / prelim_norm

    # Các embedding đầu vào đã được L2-normalize (embedder.py normalize rồi),
    # nên dot product ở đây chính là cosine similarity.
    similarities = matrix @ prelim_mean
    keep_mask = similarities >= outlier_threshold

    warnings = []
    if not np.any(keep_mask):
        warnings.append(
            f"Tất cả {len(embeddings)} ảnh có similarity thấp so với mean sơ bộ "
            f"(min={similarities.min():.2f}, threshold={outlier_threshold}). "
            "Giữ nguyên toàn bộ để không mất dữ liệu, nhưng nên kiểm tra lại "
            "chất lượng ảnh gallery cho identity này."
        )
        keep_mask = np.ones_like(keep_mask, dtype=bool)
    elif not np.all(keep_mask):
        n_dropped = int((~keep_mask).sum())
        warnings.append(
            f"Loại {n_dropped}/{len(embeddings)} ảnh nghi ngờ outlier "
            f"(similarity < {outlier_threshold} so với mean sơ bộ)."
        )

    final_matrix = matrix[keep_mask]
    mean_embedding = final_matrix.mean(axis=0)
    norm = np.linalg.norm(mean_embedding)
    if norm == 0:
        raise ValueError("Invalid mean embedding after outlier filtering")

    valid_embeddings = [emb for i, emb in enumerate(embeddings) if keep_mask[i]]
    return mean_embedding / norm, warnings, valid_embeddings


def enroll_person(db, user_id, name, embeddings, overwrite=True, outlier_threshold=0.35, save_individuals=True):
    if not embeddings:
        raise ValueError("No embeddings provided for enrollment")

    identity_embedding, warnings, valid_embeddings = build_identity_embedding(embeddings, outlier_threshold)
    person_id, is_new, is_ignored = db.upsert(user_id, name, valid_embeddings, identity_embedding, overwrite=overwrite, save_individuals=save_individuals)
    return person_id, is_new, is_ignored, warnings