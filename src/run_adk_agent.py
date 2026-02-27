#!/usr/bin/env python3
"""
CLI para o agente ADK RAG. Carrega .env, importa o agente e roda um loop de perguntas.
Uso: python -m src.run_adk_agent
Alternativa: adk run (a partir da raiz do projeto, com agente configurado).
"""
import asyncio
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
    print(f"Defina no .env: {', '.join(missing)}. Rode o script de ingestão antes de consultar.", file=sys.stderr)
    sys.exit(1)

from src.agent_adk import root_agent


def _get_response_sync(user_message: str) -> str:
    """Obtém resposta do agente (tenta generate_response ou session/stream)."""
    if hasattr(root_agent, "generate_response"):
        out = root_agent.generate_response(user_message)
        return getattr(out, "text", None) or str(out)
    # Fallback: usar AdkApp com stream e coletar último conteúdo
    try:
        from vertexai.agent_engines import AdkApp
        app = AdkApp(agent=root_agent)
        async def _run():
            last_text = ""
            async for event in app.async_stream_query(user_id="cli", message=user_message):
                if isinstance(event, dict) and event.get("content"):
                    last_text = event["content"] if isinstance(event["content"], str) else str(event["content"])
            return last_text
        return asyncio.run(_run())
    except Exception as e:
        return f"Erro ao obter resposta: {e}"


def main():
    print("Agente RAG (ADK) – Vertex AI. Digite sua pergunta (ou 'sair' para encerrar).\n")
    while True:
        try:
            q = input("Você: ").strip()
            if not q or q.lower() in ("sair", "exit", "quit"):
                break
            text = _get_response_sync(q)
            print(f"Agente: {text}\n")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Erro: {e}\n", file=sys.stderr)


if __name__ == "__main__":
    main()
