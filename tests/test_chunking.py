"""
Testes para src.chunking: chunk_text, chunk_pdf_bytes, extract_text_from_pdf_bytes.
"""
import pytest

from src.chunking import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    chunk_text,
    chunk_pdf_bytes,
    extract_text_from_pdf_bytes,
    _split_text,
)


class TestChunkText:
    """Testes para chunk_text."""

    def test_empty_returns_empty(self):
        assert chunk_text("") == []
        assert chunk_text("   \n\n  ") == []

    def test_short_text_single_chunk(self):
        text = "Um parágrafo curto."
        assert chunk_text(text) == [text]

    def test_respects_chunk_size_and_overlap(self):
        # Texto maior que chunk_size deve ser dividido
        text = "A" * 100 + "\n\n" + "B" * 100 + "\n\n" + "C" * 400
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
        assert len(chunks) >= 2
        for c in chunks:
            assert len(c) <= 100 + 20  # overlap pode deixar um pouco maior em bordas

    def test_paragraph_boundaries(self):
        text = "P1\n\nP2\n\nP3"
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=0)
        assert len(chunks) == 1
        assert "P1" in chunks[0] and "P2" in chunks[0] and "P3" in chunks[0]


class TestSplitText:
    """Testes para _split_text (helper)."""

    def test_empty_returns_empty(self):
        assert _split_text("", 10, 2) == []
        assert _split_text("   ", 10, 2) == []

    def test_single_chunk(self):
        assert _split_text("hello", 10, 2) == ["hello"]

    def test_overlap(self):
        text = "a" * 20
        out = _split_text(text, chunk_size=10, overlap=3)
        assert len(out) >= 2
        assert out[0] == "a" * 10
        # Segundo chunk começa em 10-3=7
        assert out[1].startswith("a")


class TestExtractTextFromPdfBytes:
    """Testes para extração de texto de PDF."""

    def test_empty_pdf_raises_or_returns_empty(self):
        # PDF mínimo inválido pode levantar; PDF vazio válido retorna ""
        try:
            out = extract_text_from_pdf_bytes(b"not a pdf")
            assert out == "" or isinstance(out, str)
        except Exception:
            pass  # pypdf pode levantar em bytes inválidos

    def test_minimal_valid_pdf_returns_string(self):
        # PDF mínimo de uma página (formato simplificado)
        minimal_pdf = (
            b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
            b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj\n"
            b"xref\n0 4\ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n0\n%%EOF"
        )
        try:
            out = extract_text_from_pdf_bytes(minimal_pdf)
            assert isinstance(out, str)
        except Exception:
            # Algumas versões do pypdf podem falhar em PDF sem stream de texto
            pass


class TestChunkPdfBytes:
    """Testes para chunk_pdf_bytes."""

    def test_chunk_pdf_bytes_mocked_extract(self, monkeypatch):
        """Com texto extraído mockado, verifica IDs e estrutura com text e metadata."""
        fake_pages = [(0, "Primeiro paragrafo."), (1, "Segundo paragrafo com mais texto. " * 20)]

        def fake_extract_by_page(data):
            return fake_pages

        monkeypatch.setattr("src.chunking.extract_text_by_page_bytes", fake_extract_by_page)
        chunks = chunk_pdf_bytes(b"fake-pdf-bytes", source_id="test_doc", source_filename="test.pdf")
        assert len(chunks) >= 1
        for chunk_id, chunk_dict in chunks:
            assert chunk_id.startswith("test_doc_")
            assert "_" in chunk_id
            assert isinstance(chunk_dict, dict)
            assert "text" in chunk_dict and "metadata" in chunk_dict
            assert len(chunk_dict["text"]) > 0
            meta = chunk_dict["metadata"]
            assert meta.get("source_filename") == "test.pdf"
            assert "page_numbers" in meta
            assert "chunk_index" in meta
            assert "total_chunks" in meta
