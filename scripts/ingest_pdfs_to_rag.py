#!/usr/bin/env python3
"""
Ingestão de PDFs para o Vector Search (sem RAG Engine Corpus).
Chunking, embedding (Gemini, 768 dims), upsert no índice e chunk store no GCS.
Arquivos já ingeridos (mesmo hash) são pulados; use --force para re-ingerir.
Requer: .env com GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION, GOOGLE_CLOUD_STORAGE_BUCKET,
        VECTOR_SEARCH_INDEX_NAME, VECTOR_SEARCH_INDEX_ENDPOINT_NAME, e GEMINI_API_KEY ou Vertex (ADC).
Entrada: PDFs em data/pdfs ou PDF_SOURCE_DIR.
"""
import argparse
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
    parser = argparse.ArgumentParser(description="Ingerir PDFs no Vector Search (RAG)")
    parser.add_argument("--force", action="store_true", help="Re-ingerir mesmo arquivos já ingeridos")
    args = parser.parse_args()

    if not all([PROJECT_ID, LOCATION, BUCKET, INDEX_NAME, INDEX_ENDPOINT_NAME]):
        print(
            "Defina no .env: GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION, GOOGLE_CLOUD_STORAGE_BUCKET, "
            "VECTOR_SEARCH_INDEX_NAME, VECTOR_SEARCH_INDEX_ENDPOINT_NAME",
            file=sys.stderr,
        )
        sys.exit(1)
    if not GEMINI_API_KEY and os.getenv("GOOGLE_GENAI_USE_VERTEXAI") != "1":
        print("Defina GEMINI_API_KEY ou GOOGLE_GENAI_USE_VERTEXAI=1 no .env.", file=sys.stderr)
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

    print(f"Processando {len(pdf_paths)} arquivo(s) PDF..." + (" (--force: re-ingerir todos)" if args.force else ""))
    count, skipped = ingest_pdfs_from_paths(pdf_paths, force=args.force)
    print(f"Ingestão concluída. Chunks indexados: {count}")
    if skipped:
        print(f"Arquivos já ingeridos (pulados): {', '.join(skipped)}")


if __name__ == "__main__":
    main()
