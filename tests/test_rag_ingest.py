"""
Testes para src.rag_ingest: get_env, upload_bytes_to_gcs (mock), ingest_chunks_to_vector_search,
ingest_pdfs_from_bytes (deduplicação), _file_hash, manifesto (mocks).
"""
import pytest


class TestGetEnv:
    """Testes para get_env."""

    def test_get_env_returns_stripped_values(self, monkeypatch):
        monkeypatch.setattr("src.rag_ingest.os.getenv", lambda k, default="": {
            "GOOGLE_CLOUD_PROJECT": "  proj  ",
            "GOOGLE_CLOUD_LOCATION": " us-central1 ",
            "GOOGLE_CLOUD_STORAGE_BUCKET": " bucket ",
            "VECTOR_SEARCH_INDEX_NAME": " idx ",
            "VECTOR_SEARCH_INDEX_ENDPOINT_NAME": " ep ",
        }.get(k, default or ""))
        from src.rag_ingest import get_env
        project, location, bucket, index_name, index_endpoint, repo = get_env()
        assert project == "proj"
        assert location == "us-central1"
        assert bucket == "bucket"
        assert index_name == "idx"
        assert index_endpoint == "ep"
        assert "rag" in repo or "agente" in repo or len(repo) > 0

    def test_get_env_empty_index_returns_none_when_empty_string(self, monkeypatch):
        monkeypatch.setattr("src.rag_ingest.os.getenv", lambda k, default="": "" if "VECTOR" in k else ("us-east1" if "LOCATION" in k else "x"))
        from src.rag_ingest import get_env
        _, _, _, index_name, index_endpoint, _ = get_env()
        assert index_name is None
        assert index_endpoint is None


class TestIngestChunksToVectorSearch:
    """Testes para ingest_chunks_to_vector_search com mocks."""

    def test_ingest_requires_index_name(self, monkeypatch):
        monkeypatch.setattr("src.rag_ingest.get_env", lambda: ("p", "l", "b", None, "ep", "/repo"))
        from src.rag_ingest import ingest_chunks_to_vector_search
        with pytest.raises(ValueError, match="VECTOR_SEARCH_INDEX_NAME"):
            ingest_chunks_to_vector_search([("id1", "text1")])

    def test_ingest_calls_embed_and_upsert_and_save(self, monkeypatch):
        """Com mocks, verifica que embed_texts, index.upsert_datapoints e save_chunks são chamados."""
        import google.cloud.aiplatform as ap
        mock_index = type("Index", (), {})()
        upsert_calls = []
        mock_index.upsert_datapoints = lambda datapoints: upsert_calls.append(datapoints)
        save_calls = []
        def fake_save(mapping, merge=True):
            save_calls.append((mapping, merge))
        def fake_embed(texts, dimension=768):
            return [[0.1] * dimension for _ in texts]
        monkeypatch.setattr("src.rag_ingest.get_env", lambda: ("proj", "loc", "bucket", "projects/p/loc/indexes/123", "ep", "/repo"))
        monkeypatch.setattr(ap, "init", lambda **kw: None)
        monkeypatch.setattr(ap, "MatchingEngineIndex", lambda index_name: mock_index)
        monkeypatch.setattr("src.embedding_gemini.embed_texts", fake_embed)
        monkeypatch.setattr("src.chunk_store.save_chunks", fake_save)
        from src.rag_ingest import ingest_chunks_to_vector_search
        chunks = [
            ("id1", {"text": "t1", "metadata": {}}),
            ("id2", {"text": "t2", "metadata": {}}),
        ]
        n = ingest_chunks_to_vector_search(chunks)
        assert n == 2
        assert len(upsert_calls) >= 1
        assert len(save_calls) == 1
        saved = save_calls[0][0]
        assert saved["id1"]["text"] == "t1"
        assert saved["id2"]["text"] == "t2"


class TestFileHash:
    """Testes para _file_hash."""

    def test_returns_sha256_hex(self):
        from src.rag_ingest import _file_hash
        h = _file_hash(b"hello")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_same_content_same_hash(self):
        from src.rag_ingest import _file_hash
        assert _file_hash(b"x") == _file_hash(b"x")
        assert _file_hash(b"x") != _file_hash(b"y")


class TestIngestPdfsFromBytesDeduplication:
    """Testes para ingest_pdfs_from_bytes com deduplicação e force."""

    def test_returns_tuple_count_and_skipped(self, monkeypatch):
        manifest = {}
        def load_manifest(bucket, project=None):
            return manifest
        def save_manifest(m, bucket, project=None):
            pass  # caller already updated manifest (same dict)
        monkeypatch.setattr("src.rag_ingest.get_env", lambda: ("proj", "loc", "bucket", "projects/p/loc/indexes/i", "ep", "/repo"))
        monkeypatch.setattr("src.rag_ingest._load_manifest", load_manifest)
        monkeypatch.setattr("src.rag_ingest._save_manifest", save_manifest)
        monkeypatch.setattr("src.rag_ingest._chunk_pdf_files", lambda f: [("doc_0_abc123def456", {"text": "t", "metadata": {}})])
        import google.cloud.aiplatform as ap
        mock_index = type("Index", (), {})()
        mock_index.upsert_datapoints = lambda datapoints: None
        monkeypatch.setattr(ap, "init", lambda **kw: None)
        monkeypatch.setattr(ap, "MatchingEngineIndex", lambda index_name: mock_index)
        monkeypatch.setattr("src.embedding_gemini.embed_texts", lambda texts, dimension=768: [[0.1] * 768 for _ in texts])
        monkeypatch.setattr("src.chunk_store.save_chunks", lambda mapping, merge=True: None)
        from src.rag_ingest import ingest_pdfs_from_bytes
        count, skipped = ingest_pdfs_from_bytes([("a.pdf", b"pdf content")], force=False)
        assert count == 1
        assert skipped == []
        assert "a.pdf" in manifest
        assert manifest["a.pdf"]["sha256"] == __import__("hashlib").sha256(b"pdf content").hexdigest()

    def test_skips_when_same_hash_and_not_force(self, monkeypatch):
        h = __import__("hashlib").sha256(b"same").hexdigest()
        manifest = {"doc.pdf": {"sha256": h, "ingested_at": "2026-01-01T00:00:00Z", "chunk_count": 1, "chunk_ids": ["x"]}}
        def load_manifest(bucket, project=None):
            return manifest
        monkeypatch.setattr("src.rag_ingest.get_env", lambda: ("proj", "loc", "bucket", "idx", "ep", "/repo"))
        monkeypatch.setattr("src.rag_ingest._load_manifest", load_manifest)
        from src.rag_ingest import ingest_pdfs_from_bytes
        count, skipped = ingest_pdfs_from_bytes([("doc.pdf", b"same")], force=False)
        assert count == 0
        assert "doc.pdf" in skipped
