"""
Orquestrador LangGraph com tool de RAG e análise multi-hop (múltiplas buscas, comparação de fontes).
Foco em análise aprofundada e raciocínio em etapas. Requer GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION,
VECTOR_SEARCH_INDEX_ENDPOINT_NAME, GEMINI_API_KEY no .env.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

from langchain_core.tools import tool
from langchain_google_vertexai import ChatVertexAI
from langgraph.prebuilt import create_react_agent

from src.tools.vertex_rag_tool import vertex_rag_retrieval


@tool
def retrieve_rag_documentation(query: str) -> str:
    """Recupera trechos relevantes do corpus RAG (Vertex AI Vector Search) para responder à pergunta. Use para buscar documentação técnica, Responsible AI ou RAG."""
    return vertex_rag_retrieval(query, top_k=10, vector_distance_threshold=0.6)


@tool
def compare_sources(query: str) -> str:
    """Faz duas buscas no corpus com ângulos diferentes (conceitos principais e detalhes) e retorna ambos os resultados para você comparar e analisar. Use quando quiser uma visão mais completa ou verificar consistência entre fontes."""
    main = vertex_rag_retrieval(query, top_k=6)
    reformulated = vertex_rag_retrieval(f"Detalhes e exemplos sobre: {query}", top_k=6)
    return (
        "--- Primeira busca (conceito direto) ---\n\n"
        f"{main}\n\n"
        "--- Segunda busca (detalhes e exemplos) ---\n\n"
        f"{reformulated}"
    )


def build_agent():
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-east1")
    if not project:
        raise ValueError("GOOGLE_CLOUD_PROJECT não definido no .env")

    llm = ChatVertexAI(
        model="gemini-2.0-flash-001",
        project=project,
        location=location,
        temperature=0.4,
    )

    tools = [retrieve_rag_documentation, compare_sources]
    system_prompt = (
        "Você é um analista especializado em RAG e Vertex AI. "
        "Raciocine em etapas: formule hipóteses, busque evidências com retrieve_rag_documentation ou compare_sources, "
        "e só então conclua. Use compare_sources quando precisar de múltiplas perspectivas ou verificar consistência. "
        "Cite as fontes (documento e página). Explique o raciocínio quando relevante."
    )

    agent = create_react_agent(llm, tools, state_modifier=system_prompt)
    return agent


# Instância do agente (grafo)
langgraph_agent = build_agent()
