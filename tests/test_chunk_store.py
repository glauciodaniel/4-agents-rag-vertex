"""
Testes para src.chunk_store: get_chunk_store_path, load_chunks, save_chunks,
load_chunks_by_ids, sharding e retrocompatibilidade (com mocks GCS).
"""
import json
import pytest


def _make_bucket_with_state(state: dict):
    """Bucket mock: state[path] = {"exists": bool, "data": dict ou str}."""
    class MockBlob:
        def __init__(self, path):
            self.path = path
            self._state = state

        def exists(self):
            return self._state.get(self.path, {}).get("exists", False)

        def download_as_string(self):
            data = self._state.get(self.path, {}).get("data", {})
            if isinstance(data, dict):
                return json.dumps(data).encode()
            return json.dumps(data).encode() if isinstance(data, str) else data

        def upload_from_string(self, content, **kwargs):
            self._state[self.path] = {"exists": True, "data": json.loads(content)}

    class MockBucket:
        def __init__(self, s):
            self._state = s
        def blob(self, path):
            b = MockBlob(path)
            b._state = self._state
            return b
    return MockBucket(state)


class TestChunkIdToSourceId:
    """Testes para _chunk_id_to_source_id."""

    def test_parses_standard_format(self):
        from src.chunk_store import _chunk_id_to_source_id
        assert _chunk_id_to_source_id("doc_0_abc123def456") == "doc"
        assert _chunk_id_to_source_id("relatorio_anual_3_a1b2c3d4e5f6") == "relatorio_anual"

    def test_returns_chunk_id_when_not_standard(self):
        from src.chunk_store import _chunk_id_to_source_id
        assert _chunk_id_to_source_id("simple_id") == "simple_id"
        assert _chunk_id_to_source_id("only_two_parts_here") == "only_two_parts_here"


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
        monkeypatch.setattr("src.chunk_store._sharded_index_exists", lambda b: False)
        from src.chunk_store import load_chunks
        result = load_chunks()
        assert result["id1"]["text"] == "texto 1"
        assert result["id2"]["text"] == "texto 2"


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
        # save_chunks grava shard(s) e índice; valores podem ser str ou dict com text/metadata
        payloads = [json.loads(u[0]) for u in uploaded]
        def has_k1_v1(p):
            v = p.get("k1")
            return v == "v1" or (isinstance(v, dict) and v.get("text") == "v1")
        assert any(has_k1_v1(p) for p in payloads)


class TestLoadChunksByIds:
    """Testes para load_chunks_by_ids."""

    def test_returns_empty_when_no_ids(self, monkeypatch):
        monkeypatch.setattr("src.chunk_store.os.getenv", lambda k, default="": "b" if k == "GOOGLE_CLOUD_STORAGE_BUCKET" else (default or ""))
        state = {}
        monkeypatch.setattr("src.chunk_store._bucket_and_client", lambda: (_make_bucket_with_state(state), None))
        from src.chunk_store import load_chunks_by_ids
        assert load_chunks_by_ids([]) == {}

    def test_load_by_ids_from_legacy(self, monkeypatch):
        state = {
            "vector-search-index/chunks.json": {"exists": True, "data": {"id1": "t1", "id2": "t2", "id3": "t3"}},
        }
        monkeypatch.setattr("src.chunk_store.os.getenv", lambda k, default="": "b" if k == "GOOGLE_CLOUD_STORAGE_BUCKET" else (default or ""))
        monkeypatch.setattr("src.chunk_store._bucket_and_client", lambda: (_make_bucket_with_state(state), None))
        monkeypatch.setattr("src.chunk_store._sharded_index_exists", lambda b: False)
        from src.chunk_store import load_chunks_by_ids
        result = load_chunks_by_ids(["id1", "id3"])
        assert result["id1"]["text"] == "t1"
        assert result["id3"]["text"] == "t3"

    def test_load_by_ids_from_shards(self, monkeypatch):
        from src.chunk_store import CHUNKS_DIR, CHUNKS_INDEX_OBJECT
        state = {
            CHUNKS_INDEX_OBJECT: {
                "exists": True,
                "data": {"doc_a_0_abc123def456": "doc_a", "doc_a_1_def456abc789": "doc_a", "doc_b_0_111222333444": "doc_b"},
            },
            f"{CHUNKS_DIR}/doc_a.json": {"exists": True, "data": {"doc_a_0_abc123def456": "texto a0", "doc_a_1_def456abc789": "texto a1"}},
            f"{CHUNKS_DIR}/doc_b.json": {"exists": True, "data": {"doc_b_0_111222333444": "texto b0"}},
        }
        monkeypatch.setattr("src.chunk_store.os.getenv", lambda k, default="": "b" if k == "GOOGLE_CLOUD_STORAGE_BUCKET" else (default or ""))
        monkeypatch.setattr("src.chunk_store._bucket_and_client", lambda: (_make_bucket_with_state(state), None))
        monkeypatch.setattr("src.chunk_store._sharded_index_exists", lambda b: True)
        from src.chunk_store import load_chunks_by_ids
        result = load_chunks_by_ids(["doc_a_0_abc123def456", "doc_b_0_111222333444"])
        assert result["doc_a_0_abc123def456"]["text"] == "texto a0"
        assert result["doc_b_0_111222333444"]["text"] == "texto b0"


class TestSaveChunksSharding:
    """Testes para save_chunks em modo sharding."""

    def test_save_creates_index_and_shards(self, monkeypatch):
        from src.chunk_store import CHUNKS_DIR, CHUNKS_INDEX_OBJECT
        state = {}
        bucket = _make_bucket_with_state(state)
        monkeypatch.setattr("src.chunk_store.os.getenv", lambda k, default="": "b" if k == "GOOGLE_CLOUD_STORAGE_BUCKET" else (default or ""))
        monkeypatch.setattr("src.chunk_store._bucket_and_client", lambda: (bucket, None))
        from src.chunk_store import save_chunks
        save_chunks({"doc_0_abc123def456": "text1", "doc_1_def456abc789": "text2"}, merge=False)
        assert CHUNKS_INDEX_OBJECT in state
        assert state[CHUNKS_INDEX_OBJECT]["data"] == {"doc_0_abc123def456": "doc", "doc_1_def456abc789": "doc"}
        assert f"{CHUNKS_DIR}/doc.json" in state
        doc_shard = state[f"{CHUNKS_DIR}/doc.json"]["data"]
        assert doc_shard["doc_0_abc123def456"]["text"] == "text1"
        assert doc_shard["doc_1_def456abc789"]["text"] == "text2"
