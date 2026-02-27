#!/usr/bin/env python3
"""
CLI para o agente LangGraph (RAG + tools). Uso: python -m src.run_langgraph_agent
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

required = ("GOOGLE_CLOUD_PROJECT", "VECTOR_SEARCH_INDEX_ENDPOINT_NAME", "GEMINI_API_KEY")
missing = [v for v in required if not os.getenv(v)]
if missing:
    print(f"Defina no .env: {', '.join(missing)}.", file=sys.stderr)
    sys.exit(1)

from src.agent_langgraph import langgraph_agent
from langchain_core.messages import HumanMessage

def main():
    print("Agente LangGraph (RAG + Vertex AI). Digite sua pergunta (ou 'sair' para encerrar).\n")
    while True:
        try:
            q = input("Você: ").strip()
            if not q or q.lower() in ("sair", "exit", "quit"):
                break
            result = langgraph_agent.invoke({"messages": [HumanMessage(content=q)]})
            messages = result.get("messages", [])
            last = messages[-1] if messages else None
            text = getattr(last, "content", None) or str(last) if last else "Sem resposta."
            print(f"Agente: {text}\n")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Erro: {e}\n", file=sys.stderr)

if __name__ == "__main__":
    main()
