"""
Embedding com Gemini. Suporta autenticação via Vertex AI (ADC) ou Google AI API (GEMINI_API_KEY).
Prioridade: GOOGLE_GENAI_USE_VERTEXAI=1 com GOOGLE_CLOUD_PROJECT → Vertex; senão GEMINI_API_KEY.
Modelo: gemini-embedding-001, output_dimensionality=768 para compatibilidade com o índice Vector Search.
SDK: google.genai.
"""
import os
import time
from typing import List

from dotenv import load_dotenv
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_REPO_ROOT / ".env")

DEFAULT_DIMENSION = 768
EMBEDDING_MODEL = "gemini-embedding-001"
BATCH_SIZE = 100
RETRY_DELAY_SEC = 2
MAX_RETRIES = 3


def _get_client():
    from google import genai
    use_vertex = (os.getenv("GOOGLE_GENAI_USE_VERTEXAI") or "").strip()
    project = (os.getenv("GOOGLE_CLOUD_PROJECT") or "").strip()
    location = (os.getenv("GOOGLE_CLOUD_LOCATION") or "us-east1").strip()
    api_key = (os.getenv("GEMINI_API_KEY") or "").strip()

    if use_vertex == "1" and project:
        return genai.Client(vertexai=True, project=project, location=location)
    if api_key:
        return genai.Client(api_key=api_key, vertexai=False)
    raise ValueError(
        "Configure GOOGLE_GENAI_USE_VERTEXAI=1 com GOOGLE_CLOUD_PROJECT (e GOOGLE_CLOUD_LOCATION), "
        "ou defina GEMINI_API_KEY no .env."
    )


def embed_texts(
    texts: List[str],
    dimension: int = DEFAULT_DIMENSION,
    task_type: str = "retrieval_document",
) -> List[List[float]]:
    """
    Gera embeddings para uma lista de textos com o modelo Gemini (google.genai).
    task_type: "retrieval_document" para documentos, "retrieval_query" para consultas.

    Args:
        texts: Lista de strings para embedar.
        dimension: Dimensão do vetor (768 para índice atual).
        task_type: Tipo de tarefa (RETRIEVAL_DOCUMENT ou RETRIEVAL_QUERY).

    Returns:
        Lista de vetores (listas de float).
    """
    if not texts:
        return []

    client = _get_client()
    results = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        for attempt in range(MAX_RETRIES):
            try:
                response = client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=batch,
                    config={
                        "output_dimensionality": dimension,
                        "task_type": task_type.upper() if task_type else "RETRIEVAL_DOCUMENT",
                    },
                )
                if not response or not response.embeddings:
                    raise ValueError("Resposta sem embeddings.")
                batch_results = []
                for emb in response.embeddings:
                    vals = getattr(emb, "values", None) if emb else None
                    if vals is None and hasattr(emb, "embedding"):
                        vals = getattr(emb.embedding, "values", None)
                    if not vals:
                        raise ValueError("Embedding sem values.")
                    batch_results.append(vals)
                results.extend(batch_results)
                break
            except Exception as e:
                if "429" in str(e) or "resource_exhausted" in str(e).lower() or "quota" in str(e).lower():
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY_SEC * (attempt + 1))
                        continue
                raise
    return results
