"""
Tool de retrieval: embedding da query (Gemini), busca no Vector Search (find_neighbors), lookup do texto no chunk store (GCS).
Reutilizável pelo LangGraph e ADK. Usa GEMINI_API_KEY, VECTOR_SEARCH_* e GOOGLE_CLOUD_STORAGE_BUCKET.
"""
import os
from typing import Optional

from dotenv import load_dotenv
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_REPO_ROOT / ".env")

DEFAULT_DEPLOYED_INDEX_ID = "rag_pdf_deployed"


def vertex_rag_retrieval(
    query: str,
    *,
    corpus_name: Optional[str] = None,
    top_k: int = 10,
    vector_distance_threshold: float = 0.6,
) -> str:
    """
    Executa retrieval: embed da query (Gemini), find_neighbors no endpoint Vector Search,
    recupera textos do chunk store (GCS) e retorna contexto formatado para o LLM.
    corpus_name é ignorado (mantido por compatibilidade de assinatura).
    """
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-east1")
    index_endpoint_name = os.getenv("VECTOR_SEARCH_INDEX_ENDPOINT_NAME")
    deployed_index_id = os.getenv("VECTOR_SEARCH_DEPLOYED_INDEX_ID", DEFAULT_DEPLOYED_INDEX_ID)
    if not project:
        return "Erro: GOOGLE_CLOUD_PROJECT não configurado no .env."
    if not index_endpoint_name:
        return "Erro: VECTOR_SEARCH_INDEX_ENDPOINT_NAME não configurado no .env."

    from google.cloud import aiplatform
    from src.embedding_gemini import embed_texts
    from src.chunk_store import load_chunks

    query_vectors = embed_texts([query], dimension=768, task_type="retrieval_query")
    if not query_vectors:
        return "Erro ao gerar embedding da query."
    query_vector = query_vectors[0]

    aiplatform.init(project=project, location=location)
    endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=index_endpoint_name)
    response = endpoint.find_neighbors(
        deployed_index_id=deployed_index_id,
        queries=[query_vector],
        num_neighbors=top_k,
    )

    chunk_map = load_chunks()
    parts = []
    for neighbor_list in response or []:
        for neighbor in neighbor_list or []:
            datapoint_id = getattr(neighbor, "id", None)
            if not datapoint_id:
                continue
            text = chunk_map.get(datapoint_id, "")
            if text:
                parts.append(f"[{datapoint_id}]\n{text}")
    return "\n\n---\n\n".join(parts) if parts else "Nenhum contexto encontrado para a consulta."
