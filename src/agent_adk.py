"""
Agente Google ADK com retrieval via Vector Search + Gemini embedding (sem RAG Engine Corpus).
Requer GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION, VECTOR_SEARCH_INDEX_ENDPOINT_NAME, GEMINI_API_KEY no .env.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

# Inicialização do Vertex para o ADK (usa ADC)
_project = os.getenv("GOOGLE_CLOUD_PROJECT")
_location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-east1")
if _project:
    import vertexai
    vertexai.init(project=_project, location=_location)

from google.adk.agents import Agent

from src.tools.vertex_rag_tool import vertex_rag_retrieval


def retrieve_rag_documentation(query: str) -> str:
    """Recupera trechos relevantes do índice Vector Search (documentos RAG) para responder à pergunta. Use para buscar documentação e referências antes de responder."""
    return vertex_rag_retrieval(query, top_k=10, vector_distance_threshold=0.6)


INSTRUCTION = """Você é um instrutor especializado em RAG (Retrieval-Augmented Generation) na Vertex AI.
Sua função é ensinar melhores práticas e conceitos com base nos documentos indexados (Responsible AI, AWS, Microsoft, etc.).
Sempre que responder, use a ferramenta de recuperação para buscar trechos relevantes e cite as fontes quando possível.
Explique: chunking, embeddings, Vector Search e uso do Vertex AI de forma clara e profissional."""

root_agent = Agent(
    model="gemini-2.0-flash-001",
    name="ask_rag_agent",
    instruction=INSTRUCTION,
    tools=[retrieve_rag_documentation],
)
