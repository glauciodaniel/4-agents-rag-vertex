# Agente RAG com Google ADK, Vertex Vector Search e LangGraph

Projeto de ensino: agente que ensina como utilizar RAG na Vertex AI, com melhores práticas e ambiente profissional. **Não utiliza RAG Engine Corpus**: ingestão e retrieval diretos com embedding Gemini e Vector Search.

## Arquitetura

- **Vertex AI Vector Search**: índice vetorial com **768 dimensões**, criado via script (`create_vector_search_index.py`). Atualização via `upsert_datapoints` (stream).
- **Embedding**: modelo **text-embedding-004** (Google AI API / Gemini), chave API em `GEMINI_API_KEY`, dimensão 768 para compatibilidade com o índice.
- **Chunk store**: chunks em GCS com sharding por documento (`vector-search-index/chunks/{source_id}.json`) e índice leve; metadados (fonte, página, data de ingestão) por chunk.
- **Google ADK**: agente com foco em respostas diretas e concisas; tools de retrieval e de resumo do contexto.
- **LangGraph**: orquestrador com foco em análise aprofundada e multi-hop; tools de retrieval e de comparação de fontes.

## Pré-requisitos

- Python 3.10+
- Conta Google Cloud com Vertex AI e Vector Search habilitados
- Chave API Gemini (Google AI): [Google AI Studio](https://aistudio.google.com/apikey)
- Autenticação GCP: `gcloud auth application-default login`
- Variáveis em `.env` (copie de `.env.example`)

## Setup rápido

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
cp .env.example .env
# Edite .env: GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION, GOOGLE_CLOUD_STORAGE_BUCKET, GEMINI_API_KEY
```

## Ordem de uso

1. **Criar índice Vector Search (obrigatório para ingestão)**  
   - A ingestão usa `upsert_datapoints`, que só funciona em índices com **Stream Update** habilitado.  
   - Crie o índice com: `python scripts/create_vector_search_index.py` (ele já usa `STREAM_UPDATE`).  
   - Atualize `.env` com `VECTOR_SEARCH_INDEX_NAME` e `VECTOR_SEARCH_INDEX_ENDPOINT_NAME` (e opcionalmente `VECTOR_SEARCH_DEPLOYED_INDEX_ID`, default `rag_pdf_deployed`).  
   - Se o índice foi criado pelo Console sem Stream Update, crie um novo com o script acima.

2. **Ingerir PDFs**  
   - Coloque PDFs em `data/pdfs/` ou use o frontend.  
   - `python scripts/ingest_pdfs_to_rag.py` (ou use o app Streamlit).  
   - Os PDFs são fragmentados, embedados (Gemini) e enviados ao índice; os textos dos chunks são gravados no GCS.

3. **Frontend (opcional)**  
   - Na raiz: `streamlit run frontend/app.py` — drag-and-drop de PDFs, processamento e indexação no Vector Search.

4. **Agente ADK**  
   - `python -m src.run_adk_agent` — CLI para perguntas ao agente RAG.

5. **LangGraph**  
   - `python -m src.run_langgraph_agent` — orquestrador com análise multi-hop e comparação de fontes.

## Quando usar cada agente

| Agente    | Melhor para                                           | Características                                                                 |
|-----------|--------------------------------------------------------|----------------------------------------------------------------------------------|
| **ADK**   | Perguntas objetivas, respostas rápidas, ensino de conceitos | Respostas curtas; tool de resumo do contexto; menos chamadas ao corpus.         |
| **LangGraph** | Análise aprofundada, múltiplas fontes, verificação de consistência | Temperature maior; raciocínio em etapas; tool que compara duas buscas (conceito vs detalhes). |

Use o **ADK** quando quiser uma resposta direta (ex.: “O que é chunking?”). Use o **LangGraph** quando quiser uma análise mais completa ou comparar informações de vários trechos (ex.: “Compare as recomendações de RAG da documentação X e Y”).

## Variáveis de ambiente

- **Obrigatórias**: `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, `GOOGLE_CLOUD_STORAGE_BUCKET`, `VECTOR_SEARCH_INDEX_NAME`, `VECTOR_SEARCH_INDEX_ENDPOINT_NAME`
- **Embedding**: use **Vertex (ADC)** definindo `GOOGLE_GENAI_USE_VERTEXAI=1` (e `gcloud auth application-default login`); ou use **Google AI API** definindo `GEMINI_API_KEY` (uma das duas é obrigatória).
- **Opcional**: `VECTOR_SEARCH_DEPLOYED_INDEX_ID` (default: `rag_pdf_deployed`), `PDF_SOURCE_DIR` (default: `data/pdfs`)

## Estrutura

- `scripts/` — criação de índice (`create_vector_search_index.py`) e ingestão (`ingest_pdfs_to_rag.py`).
- `frontend/` — app Streamlit para upload e ingestão (drag-and-drop).
- `src/` — agentes ADK e LangGraph, tool de retrieval (`tools/vertex_rag_tool.py`), ingestão (`rag_ingest.py`), embedding (`embedding_gemini.py`), chunking (`chunking.py`), chunk store (`chunk_store.py`).
- `data/pdfs/` — PDFs locais (opcional).

## Licença

Uso educacional.
