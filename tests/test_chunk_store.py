"""
Testes para src.chunk_store: get_chunk_store_path, load_chunks, save_chunks (com mocks GCS).
"""
import json
import pytest


class TestGetChunkStorePath:
    """Testes para get_chunk_store_path."""

    def test_returns_empty_without_bucket(self, monkeypatch):
        def env(k, default=""):
            return "" if k == "GOOGLE_CLOUD_STORAGE_BUCKET" else (default or "")
        monkeypatch.setattr("src.chunk_store.os.getenv", env)
        from src.chunk_store import get_chunk_store_path
        assert get_chunk_store_path() == ""

    def test_returns_gs_uri_with_bucket(self, monkeypatch):
        monkeypatch.setattr("src.chunk_store.os.getenv", lambda k, default="": "my-bucket" if k == "GOOGLE_CLOUD_STORAGE_BUCKET" else (default or ""))
        from src.chunk_store import get_chunk_store_path, CHUNKS_OBJECT
        path = get_chunk_store_path()
        assert path == f"gs://my-bucket/{CHUNKS_OBJECT}"


class TestLoadChunks:
    """Testes para load_chunks com blob mockado."""

    def test_load_empty_when_blob_not_exists(self, monkeypatch):
        mock_bucket = type("Bucket", (), {})()
        mock_blob = type("Blob", (), {})()
        mock_blob.exists = lambda: False
        mock_bucket.blob = lambda path: mock_blob
        monkeypatch.setattr("src.chunk_store.os.getenv", lambda k, default="": "b" if k == "GOOGLE_CLOUD_STORAGE_BUCKET" else (default or ""))
        monkeypatch.setattr("src.chunk_store._bucket_and_client", lambda: (mock_bucket, None))
        from src.chunk_store import load_chunks
        assert load_chunks() == {}

    def test_load_returns_dict_when_blob_exists(self, monkeypatch):
        data = {"id1": "texto 1", "id2": "texto 2"}
        mock_bucket = type("Bucket", (), {})()
        mock_blob = type("Blob", (), {})()
        mock_blob.exists = lambda: True
        mock_blob.download_as_string = lambda: json.dumps(data).encode()
        mock_bucket.blob = lambda path: mock_blob
        monkeypatch.setattr("src.chunk_store.os.getenv", lambda k, default="": "b" if k == "GOOGLE_CLOUD_STORAGE_BUCKET" else (default or ""))
        monkeypatch.setattr("src.chunk_store._bucket_and_client", lambda: (mock_bucket, None))
        from src.chunk_store import load_chunks
        assert load_chunks() == data


class TestSaveChunks:
    """Testes para save_chunks com blob mockado."""

    def test_save_uploads_json(self, monkeypatch):
        uploaded = []

        def capture_upload(content, **kwargs):
            uploaded.append((content, kwargs))

        mock_bucket = type("Bucket", (), {})()
        mock_blob = type("Blob", (), {})()
        mock_blob.upload_from_string = capture_upload
        mock_bucket.blob = lambda path: mock_blob
        monkeypatch.setattr("src.chunk_store.os.getenv", lambda k, default="": "b" if k == "GOOGLE_CLOUD_STORAGE_BUCKET" else (default or ""))
        monkeypatch.setattr("src.chunk_store._bucket_and_client", lambda: (mock_bucket, None))
        monkeypatch.setattr("src.chunk_store.load_chunks", lambda: {})
        from src.chunk_store import save_chunks
        save_chunks({"k1": "v1"}, merge=True)
        assert len(uploaded) == 1
        payload = json.loads(uploaded[0][0])
        assert payload.get("k1") == "v1"
