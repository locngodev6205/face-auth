"""Unit tests for VectorDB logic, identity decisions, and search/re-enroll queries."""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from database.vector_db import VectorDB, decide_identity


class TestVectorDbLogic(unittest.TestCase):

    def test_decide_identity_thresholds(self):
        """1. decide_identity() với trên/dưới threshold."""
        # Empty rows -> UNKNOWN
        self.assertEqual(decide_identity([], threshold=0.35), (None, "UNKNOWN", 0.0))

        # Dưới threshold -> UNKNOWN
        rows_under = [("NV001", "Nguyen Van A", 0.34)]
        self.assertEqual(decide_identity(rows_under, threshold=0.35), (None, "UNKNOWN", 0.34))

        # Bằng hoặc trên threshold -> trả về danh tính
        rows_exact = [("NV001", "Nguyen Van A", 0.35)]
        self.assertEqual(decide_identity(rows_exact, threshold=0.35), ("NV001", "Nguyen Van A", 0.35))

        rows_above = [("NV001", "Nguyen Van A", 0.88), ("NV002", "Tran Van B", 0.40)]
        self.assertEqual(decide_identity(rows_above, threshold=0.35), ("NV001", "Nguyen Van A", 0.88))

    @patch("database.vector_db.ConnectionPool")
    def test_search_query_filters(self, mock_pool_cls):
        """2. Search query lọc CENTROID, model_version và employee ACTIVE."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_pool.connection.return_value.__enter__.return_value = mock_conn
        mock_conn.execute.return_value.fetchall.return_value = [("NV001", "Nguyen Van A", 0.9)]
        mock_pool_cls.return_value = mock_pool

        db = VectorDB("fake_conninfo")

        dummy_emb = np.zeros(512, dtype=np.float32)
        results = db.search(dummy_emb, top_k=3, model_version="buffalo_s")

        self.assertEqual(len(results), 1)
        search_call = None
        for call in mock_conn.execute.call_args_list:
            sql = call[0][0]
            if "SELECT" in sql and "face_embeddings" in sql and "<=>" in sql:
                search_call = call
                break

        self.assertIsNotNone(search_call, "Search query was not executed")
        sql, params = search_call[0][0], search_call[0][1]
        self.assertIn("f.embedding_type = 'CENTROID'", sql)
        self.assertIn("f.model_version = %s", sql)
        self.assertIn("e.status = 'ACTIVE'", sql)
        self.assertEqual(params[1], "buffalo_s")
        self.assertEqual(params[3], 3)

    @patch("database.vector_db.ConnectionPool")
    def test_re_enrollment_deletes_old_embeddings(self, mock_pool_cls):
        """3. Re-enroll xóa embedding cũ cùng model version trước khi insert."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_pool.connection.return_value.__enter__.return_value = mock_conn
        # 1. Lookup UUID: ('uuid-1234',), 2. Check existing: (1,)
        mock_conn.execute.return_value.fetchone.side_effect = [("uuid-1234",), (1,)]
        mock_pool_cls.return_value = mock_pool

        db = VectorDB("fake_conninfo")

        dummy_sample = np.zeros(512, dtype=np.float32)
        dummy_centroid = np.zeros(512, dtype=np.float32)

        emp_code, is_new, is_ignored = db.upsert(
            employee_code="NV-001",
            individual_embeddings=[dummy_sample],
            mean_embedding=dummy_centroid,
            overwrite=True,
            model_version="arcface_v1",
        )

        self.assertEqual(emp_code, "NV-001")
        self.assertFalse(is_new)
        self.assertFalse(is_ignored)

        # Verify DELETE was called with employee UUID and model_version
        delete_called = False
        for call in mock_conn.execute.call_args_list:
            sql = call[0][0]
            if "DELETE FROM face_embeddings" in sql and "employee_id = %s" in sql:
                delete_called = True
                self.assertEqual(call[0][1], ("uuid-1234", "arcface_v1"))

        self.assertTrue(delete_called, "Must delete old embeddings for the same employee and model_version")

    @patch("database.vector_db.ConnectionPool")
    def test_upsert_employee_not_found(self, mock_pool_cls):
        """4. Upsert ném ValueError nếu employee_code không tồn tại trong DB."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_pool.connection.return_value.__enter__.return_value = mock_conn
        mock_conn.execute.return_value.fetchone.return_value = None
        mock_pool_cls.return_value = mock_pool

        db = VectorDB("fake_conninfo")
        dummy_centroid = np.zeros(512, dtype=np.float32)

        with self.assertRaises(ValueError):
            db.upsert(
                employee_code="NV-999",
                mean_embedding=dummy_centroid,
                model_version="arcface_v1",
            )


if __name__ == "__main__":
    unittest.main()
