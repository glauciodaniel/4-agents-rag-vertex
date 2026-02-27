"""
Testes para src.rag_ingest: get_env, upload_bytes_to_gcs (mock), ingest_chunks_to_vector_search (mocks).
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
        n = ingest_chunks_to_vector_search([("id1", "t1"), ("id2", "t2")])
        assert n == 2
        assert len(upsert_calls) >= 1
        assert len(save_calls) == 1
        assert save_calls[0][0] == {"id1": "t1", "id2": "t2"}
