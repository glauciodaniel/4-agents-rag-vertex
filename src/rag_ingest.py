"""
Lógica de ingestão RAG sem Corpus: chunking, embedding Gemini, upsert no Vector Search, chunk store no GCS.
Usado por scripts/ingest_pdfs_to_rag.py e frontend Streamlit.
Chunks são (chunk_id, dict com text e metadata); ingested_at é preenchido na ingestão.
Deduplicação: manifesto no GCS (hash SHA256) permite pular PDFs já ingeridos; use force=True para re-ingerir.
"""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_REPO_ROOT / ".env")

GCS_PREFIX = "rag-pdfs"
EMBEDDING_DIMENSION = 768
UPSERT_BATCH_SIZE = 100
INGESTION_MANIFEST_OBJECT = "vector-search-index/ingestion_manifest.json"


def get_env() -> tuple[str, str, str, Optional[str], Optional[str], str]:
    project = (os.getenv("GOOGLE_CLOUD_PROJECT") or "").strip()
    location = (os.getenv("GOOGLE_CLOUD_LOCATION") or "us-east1").strip()
    bucket = (os.getenv("GOOGLE_CLOUD_STORAGE_BUCKET") or "").strip()
    index_name = (os.getenv("VECTOR_SEARCH_INDEX_NAME") or "").strip() or None
    index_endpoint = (os.getenv("VECTOR_SEARCH_INDEX_ENDPOINT_NAME") or "").strip() or None
    return project, location, bucket, index_name, index_endpoint, str(_REPO_ROOT)


def _file_hash(data: bytes) -> str:
    """Retorna o hash SHA256 hexadecimal do conteúdo do arquivo."""
    return hashlib.sha256(data).hexdigest()


def _load_manifest(bucket_name: str, project: Optional[str] = None) -> Dict[str, Any]:
    """Carrega o manifesto de ingestão do GCS. Retorna dict vazio se não existir."""
    from google.cloud import storage
    client = storage.Client(project=project)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(INGESTION_MANIFEST_OBJECT)
    try:
        if not blob.exists():
            return {}
    except Exception:
        return {}
    data = blob.download_as_string()
    return json.loads(data) if data else {}


def _save_manifest(manifest: Dict[str, Any], bucket_name: str, project: Optional[str] = None) -> None:
    """Persiste o manifesto de ingestão no GCS."""
    from google.cloud import storage
    client = storage.Client(project=project)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(INGESTION_MANIFEST_OBJECT)
    blob.upload_from_string(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        content_type="application/json",
    )


def upload_bytes_to_gcs(
    bucket: str,
    project: str,
    files: list[tuple[str, bytes]],
    prefix: str = GCS_PREFIX,
) -> list[str]:
    """Sobe arquivos (nome, conteúdo) para gs://bucket/prefix/; retorna URIs."""
    from google.cloud import storage

    client = storage.Client(project=project)
    bucket_obj = client.bucket(bucket)
    uris = []
    for name, data in files:
        safe_name = os.path.basename(name) or "document.pdf"
        if not safe_name.lower().endswith(".pdf"):
            safe_name += ".pdf"
        blob = bucket_obj.blob(f"{prefix}/{safe_name}")
        blob.upload_from_string(data, content_type="application/pdf")
        uris.append(f"gs://{bucket}/{prefix}/{safe_name}")
    return uris


def _chunk_pdf_files(files: List[Tuple[str, bytes]]) -> List[Tuple[str, Dict[str, Any]]]:
    """Gera (chunk_id, dict com text e metadata) para uma lista de (nome_arquivo, bytes)."""
    from src.chunking import chunk_pdf_bytes

    all_chunks = []
    for name, data in files:
        source_id = Path(name).stem or "doc"
        source_filename = Path(name).name
        if not source_filename.lower().endswith(".pdf"):
            source_filename += ".pdf"
        chunks = chunk_pdf_bytes(data, source_id=source_id, source_filename=source_filename)
        all_chunks.extend(chunks)
    return all_chunks


def _chunk_pdf_paths(paths: List[str]) -> List[Tuple[str, Dict[str, Any]]]:
    """Gera (chunk_id, dict com text e metadata) para uma lista de caminhos de PDF."""
    from src.chunking import chunk_pdf_path

    all_chunks = []
    for p in paths:
        chunks = chunk_pdf_path(p)
        all_chunks.extend(chunks)
    return all_chunks


def ingest_chunks_to_vector_search(
    chunks: List[Tuple[str, Dict[str, Any]]],
    *,
    index_name: Optional[str] = None,
    project: Optional[str] = None,
    location: Optional[str] = None,
) -> int:
    """
    Gera embeddings (Gemini), faz upsert no índice Vector Search e persiste no chunk store (GCS).
    chunks: lista de (chunk_id, dict com "text" e "metadata"). ingested_at é adicionado aos metadados.
    Retorna quantidade de chunks ingeridos.
    """
    from google.cloud import aiplatform
    from google.cloud.aiplatform_v1.types import IndexDatapoint

    from src.embedding_gemini import embed_texts
    from src.chunk_store import save_chunks

    proj, loc, _bucket, idx_name, _ep, _ = get_env()
    project = (project or proj or "").strip()
    location = (location or loc or "").strip()
    index_name = (index_name or idx_name or "").strip()
    if not index_name or not project or not location:
        raise ValueError(
            "Defina GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION, VECTOR_SEARCH_INDEX_NAME no .env"
        )
    # API exige o resource name completo: projects/X/locations/Y/indexes/Z
    if not index_name.startswith("projects/"):
        index_name = f"projects/{project}/locations/{location}/indexes/{index_name}"

    ingested_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    for _cid, chunk_dict in chunks:
        if isinstance(chunk_dict, dict):
            meta = chunk_dict.get("metadata", {})
            if isinstance(meta, dict):
                meta["ingested_at"] = ingested_at

    aiplatform.init(project=project, location=location)
    index = aiplatform.MatchingEngineIndex(index_name=index_name)

    ids = [c[0] for c in chunks]
    texts = []
    for _cid, c in chunks:
        if isinstance(c, dict):
            texts.append(c.get("text", ""))
        else:
            texts.append(str(c))
    embeddings = embed_texts(texts, dimension=EMBEDDING_DIMENSION)
    if len(embeddings) != len(chunks):
        raise RuntimeError(f"Embeddings count {len(embeddings)} != chunks count {len(chunks)}")

    for i in range(0, len(ids), UPSERT_BATCH_SIZE):
        batch_ids = ids[i : i + UPSERT_BATCH_SIZE]
        batch_vectors = embeddings[i : i + UPSERT_BATCH_SIZE]
        datapoints = [
            IndexDatapoint(datapoint_id=did, feature_vector=vec)
            for did, vec in zip(batch_ids, batch_vectors)
        ]
        try:
            index.upsert_datapoints(datapoints=datapoints)
        except Exception as e:
            err_msg = str(e).lower()
            if "streamupdate is not enabled" in err_msg or "stream_update" in err_msg:
                raise RuntimeError(
                    "Este índice não tem Stream Update habilitado. A ingestão por upsert exige um índice "
                    "criado com STREAM_UPDATE. Crie um novo índice com: python scripts/create_vector_search_index.py "
                    "e atualize o .env com VECTOR_SEARCH_INDEX_NAME e VECTOR_SEARCH_INDEX_ENDPOINT_NAME."
                ) from e
            raise

    chunk_map = {c[0]: c[1] for c in chunks}
    save_chunks(chunk_map, merge=True)
    return len(chunks)


def ingest_pdfs_from_bytes(
    files: List[Tuple[str, bytes]],
    force: bool = False,
) -> Tuple[int, List[str]]:
    """
    Ingestão a partir de (nome, bytes) de PDFs: chunking, embedding, upsert, chunk store.
    Se force=False, arquivos já ingeridos (mesmo hash SHA256) são pulados.
    Retorna (quantidade de chunks ingeridos, lista de nomes de arquivos pulados por deduplicação).
    """
    proj, _, bucket, _, _, _ = get_env()
    if not bucket:
        raise ValueError("GOOGLE_CLOUD_STORAGE_BUCKET não configurado no .env.")
    manifest = _load_manifest(bucket, proj or None)
    skipped: List[str] = []
    total = 0
    for name, data in files:
        safe_name = Path(name).name or "document.pdf"
        if not safe_name.lower().endswith(".pdf"):
            safe_name += ".pdf"
        h = _file_hash(data)
        if not force and safe_name in manifest and manifest[safe_name].get("sha256") == h:
            skipped.append(safe_name)
            continue
        chunks = _chunk_pdf_files([(name, data)])
        if not chunks:
            continue
        n = ingest_chunks_to_vector_search(chunks)
        ingested_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        manifest[safe_name] = {
            "sha256": h,
            "ingested_at": ingested_at,
            "chunk_count": n,
            "chunk_ids": [c[0] for c in chunks],
        }
        _save_manifest(manifest, bucket, proj or None)
        total += n
    return total, skipped


def ingest_pdfs_from_paths(
    paths: List[str],
    force: bool = False,
) -> Tuple[int, List[str]]:
    """
    Ingestão a partir de caminhos locais de PDFs.
    Retorna (chunks ingeridos, lista de arquivos pulados por deduplicação).
    """
    files: List[Tuple[str, bytes]] = []
    for p in paths:
        with open(p, "rb") as f:
            files.append((Path(p).name, f.read()))
    return ingest_pdfs_from_bytes(files, force=force)
