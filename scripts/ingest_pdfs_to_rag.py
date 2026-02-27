#!/usr/bin/env python3
"""
Ingestão de PDFs para o Vector Search (sem RAG Engine Corpus).
Chunking, embedding (Gemini text-embedding-004, 768 dims), upsert no índice e chunk store no GCS.
Requer: .env com GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION, GOOGLE_CLOUD_STORAGE_BUCKET,
        VECTOR_SEARCH_INDEX_NAME, VECTOR_SEARCH_INDEX_ENDPOINT_NAME, GEMINI_API_KEY.
Entrada: PDFs em data/pdfs ou PDF_SOURCE_DIR.
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
load_dotenv(os.path.join(REPO_ROOT, ".env"))

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")
BUCKET = os.getenv("GOOGLE_CLOUD_STORAGE_BUCKET")
INDEX_NAME = os.getenv("VECTOR_SEARCH_INDEX_NAME")
INDEX_ENDPOINT_NAME = os.getenv("VECTOR_SEARCH_INDEX_ENDPOINT_NAME")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PDF_SOURCE_DIR = os.getenv("PDF_SOURCE_DIR", os.path.join(REPO_ROOT, "data", "pdfs"))


def main() -> None:
    if not all([PROJECT_ID, LOCATION, BUCKET, INDEX_NAME, INDEX_ENDPOINT_NAME]):
        print(
            "Defina no .env: GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION, GOOGLE_CLOUD_STORAGE_BUCKET, "
            "VECTOR_SEARCH_INDEX_NAME, VECTOR_SEARCH_INDEX_ENDPOINT_NAME",
            file=sys.stderr,
        )
        sys.exit(1)
    if not GEMINI_API_KEY:
        print("Defina GEMINI_API_KEY no .env (Google AI API key).", file=sys.stderr)
        sys.exit(1)

    local_path = Path(PDF_SOURCE_DIR)
    if not local_path.is_dir():
        print(f"Diretório não encontrado: {PDF_SOURCE_DIR}. Crie o diretório e coloque PDFs nele.", file=sys.stderr)
        sys.exit(1)

    pdf_paths = [str(f) for f in local_path.glob("*.pdf")]
    if not pdf_paths:
        print(f"Nenhum PDF em {PDF_SOURCE_DIR}. Coloque arquivos .pdf no diretório.", file=sys.stderr)
        sys.exit(1)

    from src.rag_ingest import ingest_pdfs_from_paths

    print(f"Processando {len(pdf_paths)} arquivo(s) PDF...")
    count = ingest_pdfs_from_paths(pdf_paths)
    print(f"Ingestão concluída. Chunks indexados: {count}")


if __name__ == "__main__":
    main()
