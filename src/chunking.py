"""
Chunking de PDFs e texto. Sem dependência do RAG Engine.
Usado na ingestão (script e frontend).
Suporta metadados por chunk: source_filename, page_numbers, chunk_index, etc.
"""
import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CHUNK_SIZE = 512
CHUNK_OVERLAP = 100


def extract_text_from_pdf_bytes(data: bytes) -> str:
    """Extrai texto de um PDF a partir dos bytes (texto concatenado de todas as páginas)."""
    pages = extract_text_by_page_bytes(data)
    return "\n\n".join(t for _, t in pages)


def extract_text_by_page_bytes(data: bytes) -> List[Tuple[int, str]]:
    """Extrai texto por página. Retorna lista de (índice_página_0-based, texto)."""
    from pypdf import PdfReader
    import io
    reader = PdfReader(io.BytesIO(data))
    out = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            out.append((i, text.strip()))
        else:
            out.append((i, ""))
    return out


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


def _page_numbers_for_chunk(
    chunk_start: int,
    chunk_end: int,
    page_starts: List[int],
    page_ends: List[int],
) -> List[int]:
    """Retorna lista de números de página (1-based) que o intervalo [chunk_start, chunk_end] sobrepõe."""
    one_based = []
    for p in range(len(page_starts)):
        if chunk_end > page_starts[p] and chunk_start < page_ends[p]:
            one_based.append(p + 1)
    return one_based


def chunk_pdf_bytes(
    data: bytes,
    source_id: str = "doc",
    source_filename: Optional[str] = None,
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Extrai texto do PDF, divide em chunks e retorna lista de (chunk_id, dict).
    dict contém "text" e "metadata" (source_filename, page_numbers, chunk_index, total_chunks, char_offset).
    source_id é usado para prefixar os IDs; source_filename para exibição (ex.: relatorio.pdf).
    """
    pages = extract_text_by_page_bytes(data)
    full_parts = [t for _, t in pages]
    full_text = "\n\n".join(full_parts)
    page_starts = []
    page_ends = []
    pos = 0
    for _, t in pages:
        page_starts.append(pos)
        pos += len(t) + 2
        page_ends.append(pos - 2)

    raw_chunks = chunk_text(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
    out = []
    pos = 0
    display_name = source_filename or f"{source_id}.pdf"
    for i, c in enumerate(raw_chunks):
        chunk_start = full_text.find(c, pos)
        if chunk_start < 0:
            chunk_start = pos
        chunk_end = chunk_start + len(c)
        pos = chunk_end
        page_nums = _page_numbers_for_chunk(chunk_start, chunk_end, page_starts, page_ends)
        if not page_nums:
            page_nums = [1]
        chunk_id = f"{source_id}_{i}_{hashlib.sha256(c.encode()).hexdigest()[:12]}"
        meta = {
            "source_filename": display_name,
            "page_numbers": page_nums,
            "chunk_index": i,
            "total_chunks": len(raw_chunks),
            "char_offset": chunk_start,
        }
        out.append((chunk_id, {"text": c, "metadata": meta}))
    return out


def chunk_pdf_path(
    path: str,
    source_id: Optional[str] = None,
    source_filename: Optional[str] = None,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Extrai e chunka um PDF por caminho de arquivo. Retorna (chunk_id, dict com text e metadata)."""
    source_id = source_id or Path(path).stem
    name = source_filename or Path(path).name
    with open(path, "rb") as f:
        return chunk_pdf_bytes(f.read(), source_id=source_id, source_filename=name)
