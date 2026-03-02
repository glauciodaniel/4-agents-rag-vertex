# STATE — Geração do vector-search.html (Referência Vertex AI)

## Objetivo
Transformar `vector-search.html` em um documento de referência detalhado sobre os recursos do Vertex AI:
- RAG Engine
- Vector Search
- Vector Index
- Endpoint do índice

## Progresso

### Bloco 1 — CONCLUÍDO
- Fontes consultadas:
  - https://cloud.google.com/vertex-ai/docs/vector-search/overview
  - https://cloud.google.com/vertex-ai/docs/vector-search/configuring-indexes
  - https://cloud.google.com/vertex-ai/docs/vector-search/create-manage-index
  - https://cloud.google.com/vertex-ai/docs/vector-search/setup/format-structure
  - https://cloud.google.com/vertex-ai/docs/vector-search/filtering
  - https://cloud.google.com/vertex-ai/docs/vector-search/query-index-public-endpoint
  - https://cloud.google.com/vertex-ai/docs/vector-search/deploy-index-public
  - https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/rag-api
- Conteúdo escrito em vector-search.html:
  - Painel Introdução: visão geral
  - Painel RAG Engine: definição, componentes, fluxo
  - Painel Vector Search: serviço, métricas, filtragem, consulta
  - Painel Vector Index: criação, configuração detalhada (todos os campos), formatos de dados
  - Painel Endpoint do índice: deploy, tipos, consulta REST/Python, parâmetros de performance

### Bloco 2 — CONCLUÍDO
- Conteúdo já presente ou adicionado:
  - Update/rebuild de índice (batch) e link para documentação oficial
  - Streaming updates (STREAM_UPDATE) no painel Vector Index
  - Crowding e hybrid search no painel Vector Search
  - Links para update-rebuild-index e about-hybrid-search

### Bloco 3 — CONCLUÍDO
- RAG Engine detalhado em vector-search.html:
  - Corpus: fontes de dados (GCS, Drive, inline) e gestão
  - Retrieval configs: parâmetros e reranking
  - Grounding: integração com Gemini e citações
  - Links para RAG Engine Overview e Use RAG Engine

### Bloco 4 — CONCLUÍDO
- Revisão e consolidação:
  - Navegação da sidebar mantida (intro, rag-engine, vector-search, vector-index, endpoint, custos)
  - Links de documentação revisados e complementados
  - Conteúdo consistente entre painéis

## Seções do vector-search.html
1. intro — Introdução e visão geral
2. rag-engine — RAG Engine (corpus, retrieval, grounding)
3. vector-search — Vector Search (serviço, filtros, crowding, hybrid)
4. vector-index — Vector Index (criação, update/rebuild, streaming)
5. endpoint — Endpoint do índice
6. custos — Calculadora de custos
