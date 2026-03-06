# Lab – Tutorial Explicação Professor

Este documento é um **guia para o professor** explicar o projeto em aula: em qual **ordem abrir os arquivos** e quais **trechos de código** mostram os pontos principais do fluxo de **ingestão** (chunking, embedding, store, Vector Search) e do fluxo de **consulta** (retrieval, agente, resposta).

---

## Índice

**Parte A – Ingestão**  
1. [Ordem dos arquivos na ingestão](#parte-a--ingestão)  
2. [Ponto de entrada: script de ingestão](#1-ponto-de-entrada-script-de-ingestão)  
3. [Orquestração: rag_ingest](#2-orquestração-rag_ingest)  
4. [Chunking: extração de texto e divisão em trechos](#3-chunking-extração-de-texto-e-divisão-em-trechos)  
5. [Embedding: geração de vetores com Gemini](#4-embedding-geração-de-vetores-com-gemini)  
6. [Upsert no Vector Search e persistência no chunk store](#5-upsert-no-vector-search-e-persistência-no-chunk-store)  
7. [Chunk store: sharding e índice no GCS](#6-chunk-store-sharding-e-índice-no-gcs)  

**Parte B – Consulta**  
8. [Ordem dos arquivos na consulta](#parte-b--consulta)  
9. [Ponto de entrada: CLI do agente](#9-ponto-de-entrada-cli-do-agente)  
10. [Agente e tools de retrieval](#10-agente-e-tools-de-retrieval)  
11. [Tool de RAG: embedding da query, Vector Search, chunk store](#11-tool-de-rag-embedding-da-query-vector-search-chunk-store)  
12. [Resposta final do modelo](#12-resposta-final-do-modelo)  

---

# Parte A – Ingestão

## Ordem sugerida para abrir os arquivos (ingestão)

Siga esta sequência ao explicar o fluxo de ingestão:

| # | Arquivo | Papel no fluxo |
|---|---------|-----------------|
| 1 | `scripts/ingest_pdfs_to_rag.py` | Ponto de entrada: valida .env, lista PDFs, chama a orquestração. |
| 2 | `src/rag_ingest.py` | Orquestra: deduplicação, chunking (via chunking), embedding + upsert + chunk store. |
| 3 | `src/chunking.py` | Extrai texto do PDF e divide em chunks com metadados (fonte, página). |
| 4 | `src/embedding_gemini.py` | Gera vetores (768d) com Gemini para os textos dos chunks. |
| 5 | `src/rag_ingest.py` (trecho upsert) | Monta datapoints e chama `index.upsert_datapoints`. |
| 6 | `src/chunk_store.py` | Persiste textos no GCS (shards por documento + índice). |

---

## 1. Ponto de entrada: script de ingestão

**Arquivo:** `scripts/ingest_pdfs_to_rag.py`

Este script é o que o usuário executa (`python scripts/ingest_pdfs_to_rag.py`). Ele valida o ambiente, descobre os PDFs na pasta e delega todo o trabalho ao módulo `rag_ingest`.

**Trecho 1 – Validação e lista de PDFs (linhas 35–55)**

```python
    if not all([PROJECT_ID, LOCATION, BUCKET, INDEX_NAME, INDEX_ENDPOINT_NAME]):
        print(
            "Defina no .env: GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION, GOOGLE_CLOUD_STORAGE_BUCKET, "
            "VECTOR_SEARCH_INDEX_NAME, VECTOR_SEARCH_INDEX_ENDPOINT_NAME",
            file=sys.stderr,
        )
        sys.exit(1)
    # ...
    local_path = Path(PDF_SOURCE_DIR)
    # ...
    pdf_paths = [str(f) for f in local_path.glob("*.pdf")]
```

**Explicação:** O script exige que as variáveis do Vector Search e do GCP estejam no `.env`. A pasta de PDFs vem de `PDF_SOURCE_DIR` (padrão `data/pdfs`). A lista `pdf_paths` é só os arquivos `*.pdf` nessa pasta.

**Trecho 2 – Chamada à orquestração (linhas 56–62)**

```python
    from src.rag_ingest import ingest_pdfs_from_paths

    print(f"Processando {len(pdf_paths)} arquivo(s) PDF..." + (" (--force: re-ingerir todos)" if args.force else ""))
    count, skipped = ingest_pdfs_from_paths(pdf_paths, force=args.force)
    print(f"Ingestão concluída. Chunks indexados: {count}")
    if skipped:
        print(f"Arquivos já ingeridos (pulados): {', '.join(skipped)}")
```

**Explicação:** Toda a lógica de ingestão está em `ingest_pdfs_from_paths`: ler arquivos, chunking, embedding, upsert e chunk store. O retorno é (total de chunks ingeridos, lista de arquivos pulados por deduplicação).

---

## 2. Orquestração: rag_ingest

**Arquivo:** `src/rag_ingest.py`

Aqui acontecem: leitura dos PDFs em bytes, deduplicação por hash, geração de chunks, chamada ao embedding, upsert no índice e gravação no chunk store.

**Trecho 1 – De caminhos para bytes e deduplicação (linhas 237–241, 204–218)**

```python
def ingest_pdfs_from_paths(paths: List[str], force: bool = False) -> Tuple[int, List[str]]:
    files: List[Tuple[str, bytes]] = []
    for p in paths:
        with open(p, "rb") as f:
            files.append((Path(p).name, f.read()))
    return ingest_pdfs_from_bytes(files, force=force)
```

Dentro de `ingest_pdfs_from_bytes`, para cada arquivo:

```python
        h = _file_hash(data)
        if not force and safe_name in manifest and manifest[safe_name].get("sha256") == h:
            skipped.append(safe_name)
            continue
        chunks = _chunk_pdf_files([(name, data)])   # ou _chunk_pdf_paths para paths
        if not chunks:
            continue
        n = ingest_chunks_to_vector_search(chunks)
        # ... atualiza manifesto e salva no GCS
```

**Explicação:** Os PDFs viram `(nome, bytes)`. O manifesto no GCS guarda o hash SHA256 de cada arquivo já ingerido; se o hash for igual e `force=False`, o arquivo é pulado. Caso contrário, gera-se a lista de chunks com `_chunk_pdf_files` (ou `_chunk_pdf_paths`) e envia-se tudo para `ingest_chunks_to_vector_search`.

**Trecho 2 – Geração de chunks a partir de bytes (fluxo do script) (linhas 87–99)**

No fluxo usado pelo script, `ingest_pdfs_from_paths` converte cada caminho em `(nome, bytes)` e chama `ingest_pdfs_from_bytes`, que usa **`_chunk_pdf_files`**:

```python
def _chunk_pdf_files(files: List[Tuple[str, bytes]]) -> List[Tuple[str, Dict[str, Any]]]:
    """Gera (chunk_id, dict com text e metadata) para uma lista de (nome_arquivo, bytes)."""
    from src.chunking import chunk_pdf_bytes

    all_chunks = []
    for name, data in files:
        source_id = Path(name).stem or "doc"
        source_filename = Path(name).name
        # ...
        chunks = chunk_pdf_bytes(data, source_id=source_id, source_filename=source_filename)
        all_chunks.extend(chunks)
    return all_chunks
```

**Explicação:** Cada PDF já está em memória como bytes. `chunk_pdf_bytes` (em `chunking.py`) extrai o texto, divide em chunks e retorna `(chunk_id, {"text", "metadata"})`. Há também `_chunk_pdf_paths`, que usa `chunk_pdf_path` para entrada por caminho de arquivo; o script usa o fluxo por bytes acima.

---

## 3. Chunking: extração de texto e divisão em trechos

**Arquivo:** `src/chunking.py`

Aqui o PDF vira texto por página e o texto é fatiado em chunks de tamanho fixo, com overlap e metadados (fonte, página).

**Trecho 1 – Entrada por caminho e bytes (linhas 155–166)**

```python
def chunk_pdf_path(path: str, ...) -> List[Tuple[str, Dict[str, Any]]]:
    source_id = source_id or Path(path).stem
    name = source_filename or Path(path).name
    with open(path, "rb") as f:
        return chunk_pdf_bytes(f.read(), source_id=source_id, source_filename=name)
```

**Explicação:** `chunk_pdf_path` só abre o arquivo e delega a `chunk_pdf_bytes`. O `source_id` (ex.: nome do arquivo sem extensão) e `source_filename` são usados nos metadados e no formato do `chunk_id`.

**Trecho 2 – Extração de texto por página (linhas 20–32)**

```python
def extract_text_by_page_bytes(data: bytes) -> List[Tuple[int, str]]:
    """Extrai texto por página. Retorna lista de (índice_página_0-based, texto)."""
    from pypdf import PdfReader
    import io
    reader = PdfReader(io.BytesIO(data))
    out = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            out.append((i, text.strip()))
        else:
            out.append((i, ""))
    return out
```

**Explicação:** O PDF em bytes é aberto com `PdfReader`. Cada página gera uma entrada `(índice_0-based, texto)`. Isso permite depois mapear cada chunk para as páginas que ele cobre.

**Trecho 3 – Divisão em chunks com overlap e fronteiras de parágrafo (linhas 61–94)**

```python
CHUNK_SIZE = 512
CHUNK_OVERLAP = 100

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    paragraphs = re.split(r"\n\s*\n", text.strip())
    chunks = []
    current = []
    current_len = 0
    for p in paragraphs:
        # ... acumula parágrafos até atingir chunk_size; se um parágrafo for maior que chunk_size,
        # quebra com _split_text (overlap fixo em caracteres)
```

**Explicação:** O texto é primeiro dividido em parágrafos (`\n\s*\n`). Os parágrafos são agrupados até caber em `CHUNK_SIZE` (512 caracteres); o overlap é 100 caracteres. Assim os chunks respeitam fronteiras naturais quando possível e mantêm continuidade entre trechos.

**Trecho 4 – Montagem do chunk_id e metadados (linhas 119–154 em chunk_pdf_bytes)**

```python
    pages = extract_text_by_page_bytes(data)
    full_text = "\n\n".join(full_parts)
    # ... page_starts, page_ends para mapear posição -> página
    raw_chunks = chunk_text(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
    for i, c in enumerate(raw_chunks):
        chunk_start = full_text.find(c, pos)
        chunk_end = chunk_start + len(c)
        page_nums = _page_numbers_for_chunk(chunk_start, chunk_end, page_starts, page_ends)
        chunk_id = f"{source_id}_{i}_{hashlib.sha256(c.encode()).hexdigest()[:12]}"
        meta = {
            "source_filename": display_name,
            "page_numbers": page_nums,
            "chunk_index": i,
            "total_chunks": len(raw_chunks),
            "char_offset": chunk_start,
        }
        out.append((chunk_id, {"text": c, "metadata": meta}))
```

**Explicação:** O texto completo do PDF é chunkado. Para cada chunk calcula-se em quais páginas ele está (`_page_numbers_for_chunk`). O `chunk_id` é único: `source_id` + índice + hash de 12 caracteres. Os metadados permitem ao agente citar “documento X, página Y” na resposta.

---

## 4. Embedding: geração de vetores com Gemini

**Arquivo:** `src/embedding_gemini.py`

Os textos dos chunks (e depois a query) viram vetores de 768 dimensões via API Gemini, compatíveis com o índice Vector Search.

**Trecho 1 – Cliente (Vertex ou API key) (linhas 23–36)**

```python
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
    raise ValueError(...)
```

**Explicação:** O embedding pode usar Vertex AI (ADC) ou a API do Google AI com chave. O mesmo cliente é usado na ingestão (documentos) e na consulta (query).

**Trecho 2 – Chamada à API de embedding (linhas 40–74, núcleo 61–85)**

```python
def embed_texts(texts: List[str], dimension: int = DEFAULT_DIMENSION, task_type: str = "retrieval_document") -> List[List[float]]:
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
                # ... extrai response.embeddings[].values e adiciona em batch_results
                results.extend(batch_results)
                break
```

**Explicação:** Na **ingestão**, `task_type="retrieval_document"` (default). Os textos são enviados em lotes de 100 (`BATCH_SIZE`). A API devolve um vetor de 768 floats por texto. Há retry em caso de 429/quota. Na **consulta** o mesmo módulo é chamado com `task_type="retrieval_query"` para a pergunta do usuário.

---

## 5. Upsert no Vector Search e persistência no chunk store

**Arquivo:** `src/rag_ingest.py` (função `ingest_chunks_to_vector_search`)

Aqui os vetores são enviados ao índice Vertex AI e os textos são persistidos no GCS.

**Trecho 1 – Extração de IDs e textos, geração de embeddings (linhas 154–164)**

```python
    ids = [c[0] for c in chunks]
    texts = []
    for _cid, c in chunks:
        if isinstance(c, dict):
            texts.append(c.get("text", ""))
        else:
            texts.append(str(c))
    embeddings = embed_texts(texts, dimension=EMBEDDING_DIMENSION)
    if len(embeddings) != len(chunks):
        raise RuntimeError(...)
```

**Explicação:** Cada chunk é um par `(chunk_id, dict)`. Montam-se as listas `ids` e `texts` na mesma ordem; `embed_texts` retorna um vetor por texto. A ordem é crítica: o índice associa `ids[i]` ao vetor `embeddings[i]`.

**Trecho 2 – Montagem dos datapoints e upsert em lotes (linhas 165–182)**

```python
    for i in range(0, len(ids), UPSERT_BATCH_SIZE):
        batch_ids = ids[i : i + UPSERT_BATCH_SIZE]
        batch_vectors = embeddings[i : i + UPSERT_BATCH_SIZE]
        datapoints = [
            IndexDatapoint(datapoint_id=did, feature_vector=vec)
            for did, vec in zip(batch_ids, batch_vectors)
        ]
        try:
            index.upsert_datapoints(datapoints=datapoints)
        except Exception as e:
            err_msg = str(e).lower()
            if "streamupdate is not enabled" in err_msg or "stream_update" in err_msg:
                raise RuntimeError("Este índice não tem Stream Update habilitado. ...") from e
            raise
```

**Explicação:** O índice Vertex AI Vector Search recebe `IndexDatapoint(datapoint_id=chunk_id, feature_vector=vetor)`. O upsert é feito em lotes de 100. Só funciona se o índice tiver sido criado com **Stream Update**; caso contrário a exceção é tratada com mensagem clara.

**Trecho 3 – Gravação no chunk store (linhas 184–186)**

```python
    chunk_map = {c[0]: c[1] for c in chunks}
    save_chunks(chunk_map, merge=True)
    return len(chunks)
```

**Explicação:** O dicionário `chunk_id -> {text, metadata}` é passado para `save_chunks` em `chunk_store.py`. O `merge=True` faz com que novos chunks sejam somados aos já existentes no GCS (shards e índice).

---

## 6. Chunk store: sharding e índice no GCS

**Arquivo:** `src/chunk_store.py`

O chunk store guarda o **texto** (e metadados) de cada chunk no GCS. O Vector Search guarda só os **vetores e IDs**; na consulta, os IDs retornados são usados para buscar o texto aqui.

**Trecho 1 – Normalização e agrupamento por source (linhas 152–185)**

```python
def save_chunks(mapping: Dict[str, Any], merge: bool = True) -> None:
    normalized = {k: _normalize_chunk_value(v) for k, v in mapping.items()}
    if merge:
        try:
            existing = load_chunks()
        except Exception:
            existing = {}
        existing.update(normalized)
        normalized = existing

    by_source: Dict[str, Dict[str, Dict[str, Any]]] = {}
    index_update: Dict[str, str] = {}
    for chunk_id, val in normalized.items():
        source_id = _chunk_id_to_source_id(chunk_id)   # extrai doc do formato doc_0_abc123...
        safe_source = source_id.replace(" ", "_").strip() or "default"
        if safe_source not in by_source:
            by_source[safe_source] = {}
        by_source[safe_source][chunk_id] = val
        index_update[chunk_id] = safe_source
```

**Explicação:** Cada valor é normalizado para `{"text": ..., "metadata": ...}`. Se `merge=True`, carrega o que já existe e atualiza. Depois, os chunks são agrupados por `source_id` (extraído do `chunk_id`). O `index_update` mapeia `chunk_id -> source_id` para o arquivo `_index.json`.

**Trecho 2 – Escrita dos shards e do índice (linhas 181–199)**

```python
    existing_index = _load_index(bucket) if _sharded_index_exists(bucket) else {}
    existing_index.update(index_update)

    for source_id, shard_map in by_source.items():
        path = f"{CHUNKS_DIR}/{source_id}.json"   # vector-search-index/chunks/doc.pdf.json
        if merge and _sharded_index_exists(bucket):
            existing_shard = _load_shard(bucket, source_id)
            existing_shard.update(shard_map)
            shard_map = existing_shard
        blob = bucket.blob(path)
        blob.upload_from_string(json.dumps(shard_map, ...), content_type="application/json")

    index_blob = bucket.blob(CHUNKS_INDEX_OBJECT)   # vector-search-index/chunks/_index.json
    index_blob.upload_from_string(json.dumps(existing_index, ...), ...)
```

**Explicação:** Cada documento vira um arquivo `vector-search-index/chunks/{source_id}.json` com o mapa `chunk_id -> {text, metadata}`. O arquivo `_index.json` guarda `chunk_id -> source_id` para que, na consulta, possamos carregar só os shards necessários (por ID de chunk).

---

# Parte B – Consulta

## Ordem sugerida para abrir os arquivos (consulta)

| # | Arquivo | Papel no fluxo |
|---|---------|-----------------|
| 1 | `src/run_adk_agent.py` ou `src/run_langgraph_agent.py` | CLI: lê pergunta do usuário e chama o agente. |
| 2 | `src/agent_adk.py` ou `src/agent_langgraph.py` | Define o agente (modelo, instrução, tools). |
| 3 | `src/tools/vertex_rag_tool.py` | Tool de RAG: embed da query, find_neighbors, load_chunks_by_ids, formatação. |
| 4 | `src/embedding_gemini.py` | Embedding da **query** (`task_type="retrieval_query"`). |
| 5 | `src/chunk_store.py` | `load_chunks_by_ids`: recupera textos dos chunks no GCS. |
| 6 | Agente (ADK/LangGraph) | Usa o contexto retornado pela tool + Gemini para gerar a resposta final. |

---

## 9. Ponto de entrada: CLI do agente

**Arquivos:** `src/run_adk_agent.py` e `src/run_langgraph_agent.py`

Os dois scripts fazem: carregar `.env`, validar variáveis, importar o agente e rodar um loop de perguntas.

**ADK – Trecho principal (linhas 62–74)**

```python
def main():
    print("Agente RAG (ADK) – Vertex AI. Digite sua pergunta (ou 'sair' para encerrar).\n")
    while True:
        try:
            q = input("Você: ").strip()
            if not q or q.lower() in ("sair", "exit", "quit"):
                break
            text = _get_response_sync(q)
            print(f"Agente: {text}\n")
```

**Explicação:** A pergunta do usuário é passada para `_get_response_sync`, que por sua vez chama o agente ADK (por exemplo `root_agent.generate_response(q)` ou `AdkApp.async_stream_query`). A resposta é exibida no terminal.

**LangGraph – Trecho principal (linhas 23–38)**

```python
def main():
    print("Agente LangGraph (RAG + Vertex AI). Digite sua pergunta (ou 'sair' para encerrar).\n")
    while True:
        # ...
            q = input("Você: ").strip()
            if not q or q.lower() in ("sair", "exit", "quit"):
                break
            result = langgraph_agent.invoke({"messages": [HumanMessage(content=q)]})
            messages = result.get("messages", [])
            last = messages[-1] if messages else None
            text = getattr(last, "content", None) or str(last) if last else "Sem resposta."
            print(f"Agente: {text}\n")
```

**Explicação:** O LangGraph usa o formato de mensagens (HumanMessage). O grafo é invocado com `invoke`; a última mensagem da lista é a resposta do assistente, cujo `content` é impresso.

---

## 10. Agente e tools de retrieval

**Arquivos:** `src/agent_adk.py` e `src/agent_langgraph.py`

Os agentes expõem as **tools** que o modelo pode chamar; a principal é a de recuperação RAG, que usa o Vector Search e o chunk store.

**ADK – Definição da tool e do agente (linhas 26–57)**

```python
def retrieve_rag_documentation(query: str) -> str:
    """Recupera trechos relevantes do índice Vector Search (documentos RAG) para responder à pergunta. ..."""
    return vertex_rag_retrieval(query, top_k=10, vector_distance_threshold=0.6)

# ...
root_agent = Agent(
    model="gemini-2.0-flash-001",
    name="ask_rag_agent",
    instruction=INSTRUCTION,
    tools=[retrieve_rag_documentation, summarize_context],
)
```

**Explicação:** O agente ADK tem duas tools: `retrieve_rag_documentation` (chama `vertex_rag_retrieval`) e `summarize_context` (resume o texto com Gemini). A instrução diz ao modelo para usar a recuperação antes de responder e citar fontes.

**LangGraph – Tools e construção do agente (linhas 21–61)**

```python
@tool
def retrieve_rag_documentation(query: str) -> str:
    """Recupera trechos relevantes do corpus RAG (Vertex AI Vector Search) ..."""
    return vertex_rag_retrieval(query, top_k=10, vector_distance_threshold=0.6)

@tool
def compare_sources(query: str) -> str:
    """Faz duas buscas no corpus com ângulos diferentes ..."""
    main = vertex_rag_retrieval(query, top_k=6)
    reformulated = vertex_rag_retrieval(f"Detalhes e exemplos sobre: {query}", top_k=6)
    return "--- Primeira busca ---\n\n" + main + "\n\n--- Segunda busca ---\n\n" + reformulated

    tools = [retrieve_rag_documentation, compare_sources]
    system_prompt = "Você é um analista especializado em RAG e Vertex AI. Raciocine em etapas..."
    agent = create_react_agent(llm, tools, prompt=system_prompt)
```

**Explicação:** No LangGraph as tools são decoradas com `@tool`. `compare_sources` chama `vertex_rag_retrieval` duas vezes (pergunta direta e reformulada) e concatena os resultados para o modelo comparar. O agente é um ReAct com esse LLM e essas tools.

---

## 11. Tool de RAG: embedding da query, Vector Search, chunk store

**Arquivo:** `src/tools/vertex_rag_tool.py`

Esta é a função compartilhada que implementa o retrieval: embed da pergunta, busca por vizinhos no índice, recuperação dos textos no GCS e formatação para o LLM.

**Trecho 1 – Embedding da query (linhas 42–45)**

```python
    query_vectors = embed_texts([query], dimension=768, task_type="retrieval_query")
    if not query_vectors:
        return "Erro ao gerar embedding da query."
    query_vector = query_vectors[0]
```

**Explicação:** A pergunta do usuário é embedada com **`task_type="retrieval_query"`** (diferente da ingestão, que usa `retrieval_document`). O resultado é um único vetor de 768 dimensões.

**Trecho 2 – Busca no Vector Search (linhas 46–52)**

```python
    aiplatform.init(project=project, location=location)
    endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=index_endpoint_name)
    response = endpoint.find_neighbors(
        deployed_index_id=deployed_index_id,
        queries=[query_vector],
        num_neighbors=top_k,
    )
```

**Explicação:** O endpoint do Vector Search (configurado no `.env`) recebe uma lista de vetores; aqui é um só (`[query_vector]`). A API retorna, para cada query, os `top_k` vizinhos mais próximos (IDs dos datapoints). Esses IDs são os `chunk_id` que gravamos na ingestão.

**Trecho 3 – Coleta dos IDs e carga dos textos no chunk store (linhas 54–60)**

```python
    neighbor_ids = []
    for neighbor_list in response or []:
        for neighbor in neighbor_list or []:
            datapoint_id = getattr(neighbor, "id", None)
            if datapoint_id:
                neighbor_ids.append(datapoint_id)
    chunk_map = load_chunks_by_ids(neighbor_ids)
```

**Explicação:** Os IDs retornados pelo Vector Search são os `chunk_id`. `load_chunks_by_ids` (em `chunk_store.py`) carrega do GCS apenas os shards necessários (usando `_index.json`) e devolve um dicionário `chunk_id -> {text, metadata}`.

**Trecho 4 – Formatação do contexto para o LLM (linhas 61–81)**

```python
    parts = []
    for neighbor_list in response or []:
        for neighbor in neighbor_list or []:
            datapoint_id = getattr(neighbor, "id", None)
            # ...
            val = chunk_map.get(datapoint_id)
            if isinstance(val, dict):
                text = val.get("text", "")
                meta = val.get("metadata") or {}
                source = meta.get("source_filename", "")
                pages = meta.get("page_numbers", [])
                page_str = f", p.{'-'.join(map(str, pages))}" if pages else ""
                header = f"[Fonte: {source}{page_str}]" if source else f"[{datapoint_id}]"
            # ...
            if text:
                parts.append(f"{header}\n{text}")
    return "\n\n---\n\n".join(parts) if parts else "Nenhum contexto encontrado para a consulta."
```

**Explicação:** Para cada vizinho retornado (na ordem do Vector Search), busca-se o texto em `chunk_map`. O formato final é `[Fonte: documento.pdf, p.1-2]\n<texto do chunk>`, separado por `---`, para o modelo poder citar fonte e página na resposta.

**Chunk store na consulta – load_chunks_by_ids (chunk_store.py, linhas 124–149)**

```python
def load_chunks_by_ids(ids: List[str]) -> Dict[str, Dict[str, Any]]:
    if not ids:
        return {}
    bucket, _ = _bucket_and_client()
    id_set = set(ids)
    if _sharded_index_exists(bucket):
        index = _load_index(bucket)   # chunk_id -> source_id
    else:
        index = {}
    if not index:
        legacy = _load_legacy_chunks(bucket)
        if legacy is not None:
            return {k: v for k, v in legacy.items() if k in id_set}
        return {}
    source_ids = {index[cid] for cid in ids if cid in index}
    result = {}
    for source_id in source_ids:
        shard = _load_shard(bucket, source_id)   # carrega só chunks/{source_id}.json
        for k, v in shard.items():
            if k in id_set:
                result[k] = v
    return result
```

**Explicação:** Se existir `_index.json`, o código descobre quais `source_id` são necessários a partir dos `chunk_id` pedidos e carrega só esses shards. Assim não é preciso baixar todo o chunk store, apenas os JSONs dos documentos que contêm os chunks retornados pelo Vector Search.

---

## 12. Resposta final do modelo

Depois que a tool devolve o contexto formatado:

- **ADK:** O runtime do agente passa esse contexto de volta ao modelo Gemini (como resultado da tool). O modelo usa a instrução (“cite as fontes”) e o contexto para gerar a resposta final; em seguida `_get_response_sync` extrai o texto e o exibe.
- **LangGraph:** O nó do grafo que chama o LLM recebe a mensagem com o resultado da tool; o modelo gera a próxima mensagem (resposta do assistente), que é a última em `messages` e é impressa pelo `run_langgraph_agent`.

Em ambos os casos, **não há um trecho único de código** que “monta a resposta”: a resposta é produzida pelo **modelo generativo** (Gemini) com base no contexto retornado pela tool de RAG e na instrução/system prompt do agente.

---

## Resumo para o professor

| Fase | Ordem de arquivos | Pontos principais de código |
|------|-------------------|----------------------------|
| **Ingestão** | 1) `scripts/ingest_pdfs_to_rag.py` → 2) `src/rag_ingest.py` → 3) `src/chunking.py` → 4) `src/embedding_gemini.py` → 5) `src/rag_ingest.py` (upsert) → 6) `src/chunk_store.py` | Entrada e validação; orquestração e dedup; extração de texto e chunking (tamanho, overlap, metadados); cliente e `embed_texts` (document); montagem de `IndexDatapoint` e `upsert_datapoints`; `save_chunks` (shards + _index). |
| **Consulta** | 1) `run_adk_agent` / `run_langgraph_agent` → 2) `agent_adk` / `agent_langgraph` → 3) `vertex_rag_tool` → 4) `embedding_gemini` (query) → 5) `chunk_store.load_chunks_by_ids` → 6) agente + Gemini | Loop CLI e chamada ao agente; definição das tools; `embed_texts` (retrieval_query), `find_neighbors`, `load_chunks_by_ids`, formatação; resposta gerada pelo modelo. |

Usando esta ordem e estes trechos, o professor pode percorrer o fluxo completo de ingestão e consulta diretamente no código.
