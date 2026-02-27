"""
Testes para src.embedding_gemini: embed_texts com cliente mockado.
"""
import pytest


class TestEmbedTexts:
    """Testes para embed_texts (com mock do client)."""

    def test_empty_texts_returns_empty(self):
        from src.embedding_gemini import embed_texts
        assert embed_texts([]) == []

    def test_embed_texts_with_mock_client(self, monkeypatch):
        """Com client mockado, retorna vetores fake."""
        fake_vectors = [[0.1] * 768, [0.2] * 768]

        class FakeEmbedding:
            def __init__(self, values):
                self.values = values

        class FakeResponse:
            def __init__(self, vectors):
                self.embeddings = [FakeEmbedding(v) for v in vectors]

        class FakeModels:
            def embed_content(self, model, contents, config):
                n = len(contents) if isinstance(contents, list) else 1
                return FakeResponse(fake_vectors[:n])

        class FakeClient:
            @property
            def models(self):
                return FakeModels()

        monkeypatch.setattr("src.embedding_gemini._get_client", lambda: FakeClient())
        monkeypatch.setattr("src.embedding_gemini.os.getenv", lambda k, default=None: "fake-key" if k == "GEMINI_API_KEY" else (default or ""))
        from src.embedding_gemini import embed_texts
        result = embed_texts(["texto 1", "texto 2"], dimension=768)
        assert len(result) == 2
        assert result[0] == fake_vectors[0]
        assert result[1] == fake_vectors[1]

    def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.setattr("src.embedding_gemini.os.getenv", lambda k, default=None: None)
        from src.embedding_gemini import embed_texts
        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            embed_texts(["x"])
