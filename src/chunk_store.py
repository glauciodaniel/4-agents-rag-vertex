"""
Armazenamento do texto dos chunks no GCS (mapeamento datapoint_id -> texto).
Usado na ingestão (escrita) e no retrieval (leitura).
"""
import json
import os
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_REPO_ROOT / ".env")

CHUNKS_OBJECT = "vector-search-index/chunks.json"


def _bucket_and_client():
    from google.cloud import storage
    bucket_name = os.getenv("GOOGLE_CLOUD_STORAGE_BUCKET", "").strip()
    project = os.getenv("GOOGLE_CLOUD_PROJECT", "")
    if not bucket_name:
        raise ValueError("GOOGLE_CLOUD_STORAGE_BUCKET não configurado no .env.")
    client = storage.Client(project=project or None)
    return client.bucket(bucket_name), client


def load_chunks() -> Dict[str, str]:
    """
    Carrega o mapeamento id -> texto do GCS.
    Retorna dict vazio se o arquivo não existir.
    """
    bucket, _ = _bucket_and_client()
    blob = bucket.blob(CHUNKS_OBJECT)
    try:
        if not blob.exists():
            return {}
    except Exception:
        return {}
    data = blob.download_as_string()
    return json.loads(data)


def save_chunks(mapping: Dict[str, str], merge: bool = True) -> None:
    """
    Persiste o mapeamento id -> texto no GCS.
    Se merge=True, carrega o existente, mescla e grava.
    """
    bucket, _ = _bucket_and_client()
    if merge:
        try:
            existing = load_chunks()
        except Exception:
            existing = {}
        existing.update(mapping)
        mapping = existing
    blob = bucket.blob(CHUNKS_OBJECT)
    blob.upload_from_string(
        json.dumps(mapping, ensure_ascii=False, indent=0),
        content_type="application/json",
    )


def get_chunk_store_path() -> str:
    """Retorna o caminho GCS do arquivo de chunks (gs://bucket/...)."""
    bucket_name = os.getenv("GOOGLE_CLOUD_STORAGE_BUCKET", "").strip()
    if not bucket_name:
        return ""
    return f"gs://{bucket_name}/{CHUNKS_OBJECT}"
