"""
Orquestrador LangGraph com tool de RAG (Vector Search + Gemini embedding) e LLM Gemini.
Requer GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION, VECTOR_SEARCH_INDEX_ENDPOINT_NAME, GEMINI_API_KEY no .env.
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
    """Recupera trechos relevantes do corpus RAG (Vertex AI Vector Search) para responder à pergunta. Use sempre que precisar de informações sobre Responsible AI, RAG ou documentação técnica."""
    return vertex_rag_retrieval(query, top_k=10, vector_distance_threshold=0.6)


def build_agent():
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-east1")
    if not project:
        raise ValueError("GOOGLE_CLOUD_PROJECT não definido no .env")

    llm = ChatVertexAI(
        model="gemini-2.0-flash-001",
        project=project,
        location=location,
        temperature=0.2,
    )

    tools = [retrieve_rag_documentation]
    system_prompt = (
        "Você é um assistente especializado em RAG e Vertex AI. "
        "Use a ferramenta retrieve_rag_documentation para buscar no corpus antes de responder. "
        "Cite as fontes quando possível."
    )

    agent = create_react_agent(llm, tools, state_modifier=system_prompt)
    return agent


# Instância do agente (grafo)
langgraph_agent = build_agent()
