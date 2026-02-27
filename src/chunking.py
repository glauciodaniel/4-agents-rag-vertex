"""
Chunking de PDFs e texto. Sem dependência do RAG Engine.
Usado na ingestão (script e frontend).
"""
import hashlib
import re
from pathlib import Path
from typing import List, Optional, Tuple

CHUNK_SIZE = 512
CHUNK_OVERLAP = 100


def extract_text_from_pdf_bytes(data: bytes) -> str:
    """Extrai texto de um PDF a partir dos bytes."""
    from pypdf import PdfReader
    import io
    reader = PdfReader(io.BytesIO(data))
    parts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def extract_text_from_pdf_path(path: str) -> str:
    """Extrai texto de um PDF a partir do caminho do arquivo."""
    with open(path, "rb") as f:
        return extract_text_from_pdf_bytes(f.read())


def _split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Divide texto em chunks com overlap (por caracteres)."""
    if not text or not text.strip():
        return []
    text = text.strip()
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if not chunk.strip():
            start = end - overlap
            continue
        chunks.append(chunk.strip())
        start = end - overlap
        if start >= len(text):
            break
    return chunks


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Divide texto em chunks com overlap.
    Tenta quebrar em fronteiras de parágrafo/sentença quando possível.
    """
    if not text or not text.strip():
        return []
    paragraphs = re.split(r"\n\s*\n", text.strip())
    chunks = []
    current = []
    current_len = 0
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if current_len + len(p) + 2 <= chunk_size:
            current.append(p)
            current_len += len(p) + 2
        else:
            if current:
                chunks.append("\n\n".join(current))
            if len(p) <= chunk_size:
                current = [p]
                current_len = len(p) + 2
            else:
                for sub in _split_text(p, chunk_size, chunk_overlap):
                    chunks.append(sub)
                current = []
                current_len = 0
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def chunk_pdf_bytes(data: bytes, source_id: str = "doc") -> List[Tuple[str, str]]:
    """
    Extrai texto do PDF, divide em chunks e retorna lista de (datapoint_id, texto).
    source_id é usado para prefixar os IDs (ex.: nome do arquivo).
    """
    text = extract_text_from_pdf_bytes(data)
    raw_chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    out = []
    for i, c in enumerate(raw_chunks):
        chunk_id = f"{source_id}_{i}_{hashlib.sha256(c.encode()).hexdigest()[:12]}"
        out.append((chunk_id, c))
    return out


def chunk_pdf_path(path: str, source_id: Optional[str] = None) -> List[Tuple[str, str]]:
    """Extrai e chunka um PDF por caminho de arquivo."""
    source_id = source_id or Path(path).stem
    with open(path, "rb") as f:
        return chunk_pdf_bytes(f.read(), source_id=source_id)
