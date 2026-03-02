"""
Agente Google ADK com retrieval via Vector Search + Gemini embedding (sem RAG Engine Corpus).
Foco em respostas diretas e concisas. Inclui tool de resumo do contexto recuperado.
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


def summarize_context(context: str) -> str:
    """Resume em 2-4 frases o texto recuperado, mantendo apenas o essencial para responder à pergunta do usuário. Use após recuperar documentação longa."""
    if not context or len(context.strip()) < 100:
        return context or "(Nenhum contexto para resumir.)"
    try:
        from vertexai.generative_models import GenerativeModel
        model = GenerativeModel("gemini-2.0-flash-001")
        response = model.generate_content(
            f"Resuma em 2 a 4 frases, de forma objetiva, o seguinte texto. Mantenha apenas informações relevantes:\n\n{context[:12000]}"
        )
        if response and response.text:
            return response.text.strip()
    except Exception as e:
        return f"(Erro ao resumir: {e}. Use o contexto completo.)\n\n{context[:2000]}"
    return context[:1500]


INSTRUCTION = """Você é um instrutor especializado em RAG (Retrieval-Augmented Generation) na Vertex AI.
Dê respostas diretas e concisas. Use a ferramenta de recuperação para buscar trechos relevantes nos documentos indexados.
Quando o contexto recuperado for longo, use summarize_context para obter um resumo antes de formular sua resposta.
Cite as fontes (nome do documento e página quando disponível). Evite rodeios; vá ao ponto."""

root_agent = Agent(
    model="gemini-2.0-flash-001",
    name="ask_rag_agent",
    instruction=INSTRUCTION,
    tools=[retrieve_rag_documentation, summarize_context],
)
