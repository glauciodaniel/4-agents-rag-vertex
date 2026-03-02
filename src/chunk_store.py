"""
Armazenamento do texto dos chunks no GCS.
Suporta modo legado (um único chunks.json) e sharding por documento (chunks/{source_id}.json + _index.json).
Usado na ingestão (escrita) e no retrieval (leitura). load_chunks_by_ids() carrega apenas os shards necessários.
"""
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_REPO_ROOT / ".env")

# Legado: um único arquivo (retrocompatibilidade)
CHUNKS_OBJECT = "vector-search-index/chunks.json"

# Sharding: diretório de shards + índice chunk_id -> source_id
CHUNKS_DIR = "vector-search-index/chunks"
CHUNKS_INDEX_OBJECT = "vector-search-index/chunks/_index.json"


def _bucket_and_client():
    from google.cloud import storage
    bucket_name = os.getenv("GOOGLE_CLOUD_STORAGE_BUCKET", "").strip()
    project = os.getenv("GOOGLE_CLOUD_PROJECT", "")
    if not bucket_name:
        raise ValueError("GOOGLE_CLOUD_STORAGE_BUCKET não configurado no .env.")
    client = storage.Client(project=project or None)
    return client.bucket(bucket_name), client


def _chunk_id_to_source_id(chunk_id: str) -> str:
    """
    Extrai source_id do chunk_id. Formato esperado: {source_id}_{index}_{hash12}.
    Se não seguir o padrão, retorna o chunk_id (um único shard para todos).
    """
    parts = chunk_id.rsplit("_", 2)
    if len(parts) == 3 and parts[1].isdigit() and len(parts[2]) == 12 and re.match(r"^[a-f0-9]+$", parts[2]):
        return parts[0]
    return chunk_id


def _normalize_chunk_value(val: Any) -> Dict[str, Any]:
    """Garante que o valor seja um dict com 'text' e 'metadata' (retrocompatível com str)."""
    if isinstance(val, str):
        return {"text": val, "metadata": {}}
    if isinstance(val, dict):
        return {"text": val.get("text", ""), "metadata": val.get("metadata", {})}
    return {"text": "", "metadata": {}}


def _load_legacy_chunks(bucket) -> Dict[str, Dict[str, Any]] | None:
    """Retorna o mapeamento do arquivo legado (valores normalizados) ou None se não existir."""
    blob = bucket.blob(CHUNKS_OBJECT)
    try:
        if not blob.exists():
            return None
    except Exception:
        return None
    data = json.loads(blob.download_as_string())
    return {k: _normalize_chunk_value(v) for k, v in data.items()} if data else None


def _load_index(bucket) -> Dict[str, str]:
    """Carrega o índice chunk_id -> source_id. Retorna {} se não existir."""
    blob = bucket.blob(CHUNKS_INDEX_OBJECT)
    try:
        if not blob.exists():
            return {}
    except Exception:
        return {}
    data = blob.download_as_string()
    return json.loads(data)


def _load_shard(bucket, source_id: str) -> Dict[str, Dict[str, Any]]:
    """Carrega um shard chunks/{source_id}.json. Valores normalizados para dict com text/metadata."""
    safe_source = source_id.replace(" ", "_").strip() or "default"
    path = f"{CHUNKS_DIR}/{safe_source}.json"
    blob = bucket.blob(path)
    try:
        if not blob.exists():
            return {}
    except Exception:
        return {}
    data = json.loads(blob.download_as_string())
    if not isinstance(data, dict):
        return {}
    return {k: _normalize_chunk_value(v) for k, v in data.items()}


def _sharded_index_exists(bucket) -> bool:
    """Verifica se o índice de shards existe (modo sharding em uso)."""
    blob = bucket.blob(CHUNKS_INDEX_OBJECT)
    try:
        return blob.exists()
    except Exception:
        return False


def load_chunks() -> Dict[str, Dict[str, Any]]:
    """
    Carrega o mapeamento id -> {text, metadata} do GCS.
    Preferência: se o índice de shards existir, agrega todos os shards; senão, usa arquivo legado chunks.json.
    Cada valor é um dict com "text" (str) e "metadata" (dict). Retorna dict vazio se nada existir.
    """
    bucket, _ = _bucket_and_client()
    if _sharded_index_exists(bucket):
        index = _load_index(bucket)
        if index:
            result = {}
            for source_id in set(index.values()):
                result.update(_load_shard(bucket, source_id))
            return result
    legacy = _load_legacy_chunks(bucket)
    if legacy is not None:
        return legacy
    return {}


def load_chunks_by_ids(ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Carrega apenas os chunks cujos IDs estão em ids.
    Retorna Dict[chunk_id, {text, metadata}]. Usa legado ou shards conforme disponibilidade.
    """
    if not ids:
        return {}
    bucket, _ = _bucket_and_client()
    id_set = set(ids)
    if _sharded_index_exists(bucket):
        index = _load_index(bucket)
    else:
        index = {}
    if not index:
        legacy = _load_legacy_chunks(bucket)
        if legacy is not None:
            return {k: v for k, v in legacy.items() if k in id_set}
        return {}
    source_ids = {index[cid] for cid in ids if cid in index}
    result = {}
    for source_id in source_ids:
        shard = _load_shard(bucket, source_id)
        for k, v in shard.items():
            if k in id_set:
                result[k] = v
    return result


def save_chunks(mapping: Dict[str, Any], merge: bool = True) -> None:
    """
    Persiste o mapeamento id -> (texto ou dict com text/metadata) no GCS.
    Valores podem ser str (legado) ou dict com "text" e "metadata". São normalizados ao salvar.
    """
    bucket, _ = _bucket_and_client()
    normalized = {k: _normalize_chunk_value(v) for k, v in mapping.items()}
    if merge:
        try:
            existing = load_chunks()
        except Exception:
            existing = {}
        existing.update(normalized)
        normalized = existing

    by_source: Dict[str, Dict[str, Dict[str, Any]]] = {}
    index_update: Dict[str, str] = {}
    for chunk_id, val in normalized.items():
        source_id = _chunk_id_to_source_id(chunk_id)
        safe_source = source_id.replace(" ", "_").strip() or "default"
        if safe_source not in by_source:
            by_source[safe_source] = {}
        by_source[safe_source][chunk_id] = val
        index_update[chunk_id] = safe_source

    # Atualizar shards e índice (load_chunks prefere shards quando _index existe)
    existing_index = _load_index(bucket) if _sharded_index_exists(bucket) else {}
    existing_index.update(index_update)

    for source_id, shard_map in by_source.items():
        path = f"{CHUNKS_DIR}/{source_id}.json"
        if merge and _sharded_index_exists(bucket):
            existing_shard = _load_shard(bucket, source_id)
            existing_shard.update(shard_map)
            shard_map = existing_shard
        blob = bucket.blob(path)
        blob.upload_from_string(
            json.dumps(shard_map, ensure_ascii=False, indent=0),
            content_type="application/json",
        )

    index_blob = bucket.blob(CHUNKS_INDEX_OBJECT)
    index_blob.upload_from_string(
        json.dumps(existing_index, ensure_ascii=False, indent=0),
        content_type="application/json",
    )


def get_chunk_store_path() -> str:
    """Retorna o caminho GCS do arquivo de chunks (gs://bucket/...). Legado ou diretório de shards."""
    bucket_name = os.getenv("GOOGLE_CLOUD_STORAGE_BUCKET", "").strip()
    if not bucket_name:
        return ""
    return f"gs://{bucket_name}/{CHUNKS_OBJECT}"
