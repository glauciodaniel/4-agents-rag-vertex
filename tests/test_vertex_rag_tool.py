"""
Testes para src.tools.vertex_rag_tool: vertex_rag_retrieval com mocks.
"""
import pytest


class TestVertexRagRetrieval:
    """Testes para vertex_rag_retrieval."""

    def test_returns_error_when_project_missing(self, monkeypatch):
        monkeypatch.setattr("src.tools.vertex_rag_tool.os.getenv", lambda k, default=None: None)
        from src.tools.vertex_rag_tool import vertex_rag_retrieval
        out = vertex_rag_retrieval("query")
        assert "GOOGLE_CLOUD_PROJECT" in out
        assert "Erro" in out or "não configurado" in out

    def test_returns_error_when_endpoint_missing(self, monkeypatch):
        def env(k, default=None):
            if k == "GOOGLE_CLOUD_PROJECT":
                return "proj"
            if k == "GOOGLE_CLOUD_LOCATION":
                return "us-east1"
            if k == "VECTOR_SEARCH_INDEX_ENDPOINT_NAME":
                return ""
            return default or ""
        monkeypatch.setattr("src.tools.vertex_rag_tool.os.getenv", env)
        from src.tools.vertex_rag_tool import vertex_rag_retrieval
        out = vertex_rag_retrieval("query")
        assert "VECTOR_SEARCH_INDEX_ENDPOINT" in out or "não configurado" in out

    def test_returns_context_when_mocks_ok(self, monkeypatch):
        """Com embed, find_neighbors e load_chunks_by_ids mockados, retorna string com contextos."""
        import google.cloud.aiplatform as ap
        fake_chunks = {
            "dp1": {"text": "Texto do chunk 1", "metadata": {"source_filename": "doc.pdf", "page_numbers": [1]}},
            "dp2": {"text": "Texto do chunk 2", "metadata": {}},
        }
        fake_neighbors = [[type("N", (), {"id": "dp1"})(), type("N", (), {"id": "dp2"})()]]
        mock_endpoint = type("Endpoint", (), {})()
        mock_endpoint.find_neighbors = lambda deployed_index_id, queries, num_neighbors: fake_neighbors

        monkeypatch.setattr("src.tools.vertex_rag_tool.os.getenv", lambda k, default=None: {
            "GOOGLE_CLOUD_PROJECT": "proj",
            "GOOGLE_CLOUD_LOCATION": "us-east1",
            "VECTOR_SEARCH_INDEX_ENDPOINT_NAME": "projects/p/locations/l/indexEndpoints/e",
            "VECTOR_SEARCH_DEPLOYED_INDEX_ID": "rag_pdf_deployed",
        }.get(k, default or ""))
        monkeypatch.setattr("src.embedding_gemini.embed_texts", lambda texts, dimension=768, task_type="retrieval_query": [[0.1] * dimension])
        monkeypatch.setattr("src.chunk_store.load_chunks_by_ids", lambda ids: {i: fake_chunks[i] for i in ids if i in fake_chunks})
        monkeypatch.setattr(ap, "init", lambda **kw: None)
        monkeypatch.setattr(ap, "MatchingEngineIndexEndpoint", lambda index_endpoint_name: mock_endpoint)
        from src.tools.vertex_rag_tool import vertex_rag_retrieval
        out = vertex_rag_retrieval("query", top_k=2)
        assert "Texto do chunk 1" in out
        assert "Texto do chunk 2" in out
        assert "Fonte" in out or "doc.pdf" in out or "dp1" in out or "dp2" in out
