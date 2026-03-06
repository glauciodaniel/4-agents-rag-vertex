# Lab: Ingestão de Documentos no RAG (Vertex AI)

Este tutorial descreve o passo a passo para **ingerir documentos PDF** no pipeline RAG deste projeto. Os recursos na Vertex AI (índice Vector Search, endpoint, bucket de storage e deploy do índice) **já foram criados visualmente no Console**; aqui o foco é configurar o ambiente local e executar a ingestão.

---

## Índice

1. [Visão geral do fluxo de ingestão](#1-visão-geral-do-fluxo-de-ingestão)
2. [Pré-requisitos](#2-pré-requisitos)
3. [Configurar variáveis de ambiente (.env)](#3-configurar-variáveis-de-ambiente-env)
4. [Onde obter os valores no Console Vertex AI](#4-onde-obter-os-valores-no-console-vertex-ai)
5. [Ingestão por script (PDFs em pasta)](#5-ingestão-por-script-pdfs-em-pasta)
6. [Ingestão pelo frontend Streamlit (upload)](#6-ingestão-pelo-frontend-streamlit-upload)
7. [Deduplicação e re-ingestão](#7-deduplicação-e-re-ingestão)
8. [Onde os dados ficam após a ingestão](#8-onde-os-dados-ficam-após-a-ingestão)
9. [Solução de problemas](#9-solução-de-problemas)

---

## 1. Visão geral do fluxo de ingestão

Quando você executa a ingestão, o projeto:

1. **Lê os PDFs** (de uma pasta local ou do upload no frontend).
2. **Extrai o texto** de cada página e **divide em chunks** (trechos de ~512 caracteres com overlap de 100), preservando metadados (nome do arquivo, número da página).
3. **Gera embeddings** com o modelo Gemini (768 dimensões), compatível com o índice Vector Search.
4. **Envia os vetores** para o índice Vertex AI Vector Search via `upsert_datapoints` (o índice deve ter **Stream Update** habilitado).
5. **Grava os textos dos chunks** no Google Cloud Storage (chunk store), em `vector-search-index/chunks/`, para que o agente possa recuperar o conteúdo nas buscas.
6. **Atualiza o manifesto de ingestão** no GCS (`vector-search-index/ingestion_manifest.json`) para deduplicação por hash do arquivo.

Sem um índice com **Stream Update**, a etapa 4 falha; nesse caso o índice precisa ser criado com essa opção (por script ou no Console).

---

## 2. Pré-requisitos

- **Python 3.10+** instalado.
- **Conta Google Cloud** com Vertex AI e Vector Search habilitados no projeto.
- **Recursos já criados no Console** (como combinado):
  - Índice Vector Search (768 dimensões, com **Stream Update** se for usar upsert).
  - Index Endpoint.
  - Índice implantado (deployed) nesse endpoint.
  - Bucket no Cloud Storage (para chunk store e manifesto).
- **Autenticação GCP** no ambiente local:
  ```bash
  gcloud auth application-default login
  ```
- **Chave da API Gemini** (Google AI Studio), caso não use apenas Vertex com ADC:  
  [Criar chave em Google AI Studio](https://aistudio.google.com/apikey).

### Setup do projeto

Na raiz do repositório:

```bash
# Criar e ativar o ambiente virtual
python -m venv .venv

# Windows (CMD/PowerShell)
.venv\Scripts\activate

# Linux/macOS
# source .venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

---

## 3. Configurar variáveis de ambiente (.env)

O projeto lê as configurações do arquivo `.env` na raiz. Nunca commite o `.env` (ele deve estar no `.gitignore`).

### 3.1 Criar o arquivo .env

```bash
# Na raiz do projeto
cp .env.example .env
```

Edite o arquivo `.env` com um editor de texto e preencha os valores descritos abaixo.

### 3.2 Variáveis obrigatórias para ingestão

| Variável | Descrição | Exemplo |
|----------|------------|---------|
| `GOOGLE_CLOUD_PROJECT` | ID do projeto GCP | `meu-projeto-123` |
| `GOOGLE_CLOUD_LOCATION` | Região do Vertex AI (índice/endpoint) | `us-east1` |
| `GOOGLE_CLOUD_STORAGE_BUCKET` | Nome do bucket GCS onde ficam chunks e manifesto | `meu-bucket-rag` |
| `VECTOR_SEARCH_INDEX_NAME` | **Resource name completo** do índice Vector Search | `projects/.../locations/.../indexes/...` |
| `VECTOR_SEARCH_INDEX_ENDPOINT_NAME` | **Resource name completo** do Index Endpoint | `projects/.../locations/.../indexEndpoints/...` |

### 3.3 Autenticação para embedding

O embedding é feito com a API Gemini. É necessário **uma** das opções:

- **Opção A – Vertex AI (Application Default Credentials)**  
  No `.env`:
  ```env
  GOOGLE_GENAI_USE_VERTEXAI=1
  ```
  E ter executado `gcloud auth application-default login` no mesmo projeto/contas que têm acesso ao Vertex.

- **Opção B – Chave da API Gemini (Google AI Studio)**  
  No `.env`:
  ```env
  GEMINI_API_KEY=sua_chave_aqui
  ```

### 3.4 Variáveis opcionais

| Variável | Descrição | Padrão |
|----------|------------|--------|
| `VECTOR_SEARCH_DEPLOYED_INDEX_ID` | ID do índice implantado no endpoint (ex.: nome que aparece no Console) | `rag_pdf_deployed` |
| `PDF_SOURCE_DIR` | Pasta local onde o **script** de ingestão procura PDFs | `data/pdfs` |

Exemplo de `.env` mínimo para ingestão:

```env
GOOGLE_CLOUD_PROJECT=meu-projeto
GOOGLE_CLOUD_LOCATION=us-east1
GOOGLE_CLOUD_STORAGE_BUCKET=meu-bucket-rag
VECTOR_SEARCH_INDEX_NAME=projects/meu-projeto/locations/us-east1/indexes/123456789
VECTOR_SEARCH_INDEX_ENDPOINT_NAME=projects/meu-projeto/locations/us-east1/indexEndpoints/987654321
VECTOR_SEARCH_DEPLOYED_INDEX_ID=rag_pdf_deployed
GEMINI_API_KEY=sua_chave_gemini
```

---

## 4. Onde obter os valores no Console Vertex AI

Como índice, endpoint, storage e deploy já foram feitos no Console, use-o para copiar os resource names e o ID do deployed index.

### 4.1 VECTOR_SEARCH_INDEX_NAME

1. No **Google Cloud Console**, abra **Vertex AI** → **Vector Search** (ou **Matching Engine**).
2. Vá em **Indexes**.
3. Clique no índice que você usa para RAG (768 dimensões).
4. Copie o **Resource name** completo (formato `projects/PROJECT_ID/locations/REGION/indexes/INDEX_ID`).  
   Esse valor é o `VECTOR_SEARCH_INDEX_NAME`.

### 4.2 VECTOR_SEARCH_INDEX_ENDPOINT_NAME

1. Na mesma área, vá em **Index endpoints**.
2. Clique no endpoint onde o índice está implantado.
3. Copie o **Resource name** completo (formato `projects/.../locations/.../indexEndpoints/...`).  
   Esse valor é o `VECTOR_SEARCH_INDEX_ENDPOINT_NAME`.

### 4.3 VECTOR_SEARCH_DEPLOYED_INDEX_ID

1. Na página do **Index endpoint** (acima), na seção de índices implantados (Deployed indexes).
2. O **ID** do índice implantado (nome que você deu ao fazer o deploy) é o `VECTOR_SEARCH_DEPLOYED_INDEX_ID`.  
   Se não tiver definido um, o Console pode mostrar um ID padrão; use exatamente o que aparece (ex.: `rag_pdf_deployed` ou com sufixo numérico).

### 4.4 GOOGLE_CLOUD_STORAGE_BUCKET

1. **Cloud Console** → **Cloud Storage** → **Buckets**.
2. Use o **nome** do bucket onde você quer armazenar o chunk store e o manifesto (ex.: `vector-search-index/chunks/` e `ingestion_manifest.json` dentro desse bucket).

---

## 5. Ingestão por script (PDFs em pasta)

Esta opção processa todos os arquivos `.pdf` de uma pasta local.

### 5.1 Preparar a pasta de PDFs

Por padrão o script usa a pasta `data/pdfs` na raiz do projeto.

1. Crie a pasta se não existir:
   ```bash
   mkdir -p data/pdfs
   ```
2. Copie os PDFs que deseja ingerir para `data/pdfs/`.

Para usar **outra pasta**, defina no `.env`:

```env
PDF_SOURCE_DIR=C:\caminho\para\seus\pdfs
```

(No Windows use barras ou o caminho completo; no Linux/macOS use o caminho absoluto ou relativo.)

### 5.2 Executar o script

Na raiz do projeto, com o venv ativado:

```bash
python scripts/ingest_pdfs_to_rag.py
```

O script:

- Valida as variáveis do `.env`.
- Lista os PDFs em `PDF_SOURCE_DIR` (ou `data/pdfs`).
- Para cada PDF: chunking → embedding → upsert no índice → gravação no chunk store (GCS) e atualização do manifesto.
- Exibe ao final quantos chunks foram indexados e, se houver, quais arquivos foram **pulados** por já constarem no manifesto (deduplicação).

Exemplo de saída:

```
Processando 3 arquivo(s) PDF...
Ingestão concluída. Chunks indexados: 142
Arquivos já ingeridos (pulados): manual.pdf
```

### 5.3 Re-ingerir arquivos (ignorar deduplicação)

Para reprocessar **todos** os PDFs da pasta, inclusive os que já foram ingeridos:

```bash
python scripts/ingest_pdfs_to_rag.py --force
```

---

## 6. Ingestão pelo frontend Streamlit (upload)

Você pode ingerir PDFs arrastando arquivos na interface, sem precisar colocá-los em `data/pdfs`.

### 6.1 Subir o app

Na raiz do projeto, com o venv ativado:

```bash
streamlit run frontend/app.py
```

O navegador abrirá (ou acesse o URL indicado no terminal, em geral `http://localhost:8501`).

### 6.2 Usar a interface

1. **Arraste os PDFs** para a área de upload ou use "Browse files".
2. (Opcional) Marque **"Re-ingerir mesmo que o arquivo já exista (ignorar deduplicação)"** se quiser reprocessar arquivos já ingeridos.
3. Clique em **"Processar documentos"**.
4. Aguarde o processamento. A tela mostrará:
   - **"Concluído. Chunks indexados: N."**
   - Se houver arquivos pulados por deduplicação, uma mensagem informando quais.

O frontend usa as mesmas variáveis do `.env` e a mesma lógica de `src.rag_ingest` que o script; apenas a origem dos bytes (upload em memória) é diferente.

---

## 7. Deduplicação e re-ingestão

- **Deduplicação**: o projeto mantém um manifesto no GCS (`vector-search-index/ingestion_manifest.json`) com o **hash SHA256** de cada arquivo já ingerido. Se você rodar a ingestão de novo **sem** `--force` (script) ou sem marcar "Re-ingerir..." (Streamlit), arquivos com o mesmo conteúdo serão **pulados**.
- **Re-ingestão**: para processar de novo os mesmos arquivos (por exemplo após alterar o PDF), use:
  - **Script**: `python scripts/ingest_pdfs_to_rag.py --force`
  - **Streamlit**: marque a opção "Re-ingerir mesmo que o arquivo já exista" e clique em "Processar documentos".

Os novos chunks serão enviados ao índice (upsert) e o chunk store e o manifesto serão atualizados.

---

## 8. Onde os dados ficam após a ingestão

| Recurso | Local | Descrição |
|---------|--------|-----------|
| **Vetores** | Vertex AI Vector Search (índice) | IDs dos chunks e vetores de 768 dimensões; usados em `find_neighbors` nas consultas do agente. |
| **Textos dos chunks** | GCS: `gs://BUCKET/vector-search-index/chunks/` | Um arquivo JSON por documento (shard): `{source_id}.json`. Cada entrada tem `text` e `metadata` (fonte, página, etc.). |
| **Índice chunk → shard** | GCS: `gs://BUCKET/vector-search-index/chunks/_index.json` | Mapeamento `chunk_id` → `source_id` para carregar só os shards necessários no retrieval. |
| **Manifesto de ingestão** | GCS: `gs://BUCKET/vector-search-index/ingestion_manifest.json` | Mapeamento por nome de arquivo: hash SHA256, data de ingestão, quantidade de chunks e lista de chunk IDs (para deduplicação). |

O bucket é o definido em `GOOGLE_CLOUD_STORAGE_BUCKET` no `.env`.

---

## 9. Solução de problemas

### Erro: variáveis não configuradas

**Mensagem** (script): `Defina no .env: GOOGLE_CLOUD_PROJECT, ...`  
**Solução**: Preencha todas as variáveis obrigatórias listadas na seção 3 e confira que não há espaços extras. Use o resource name **completo** para índice e endpoint.

### Erro: diretório não encontrado / nenhum PDF

**Mensagem** (script): `Diretório não encontrado: ...` ou `Nenhum PDF em ...`  
**Solução**: Crie a pasta (ex.: `data/pdfs`) e coloque pelo menos um arquivo `.pdf`, ou defina `PDF_SOURCE_DIR` no `.env` apontando para uma pasta que exista e contenha PDFs.

### Erro: Stream Update não habilitado

**Mensagem**: algo como `Stream Update is not enabled` / `stream_update` no traceback.  
**Solução**: A ingestão usa `upsert_datapoints`, que exige índice com **Stream Update**. Crie um novo índice com essa opção habilitada (no Console ou via script `create_vector_search_index.py`) e atualize `VECTOR_SEARCH_INDEX_NAME` (e endpoint/deploy se necessário) no `.env`.

### Erro: GEMINI_API_KEY ou Vertex

**Mensagem**: exige `GEMINI_API_KEY` ou `GOOGLE_GENAI_USE_VERTEXAI=1`.  
**Solução**: Defina uma das duas opções de autenticação para embedding (seção 3.3). Se usar Vertex, execute `gcloud auth application-default login`.

### Erro no frontend: "Configure o .env..."

**Solução**: O frontend lê o mesmo `.env` da raiz. Garanta que `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_STORAGE_BUCKET`, `VECTOR_SEARCH_INDEX_NAME`, `VECTOR_SEARCH_INDEX_ENDPOINT_NAME` e `GEMINI_API_KEY` (ou Vertex) estão definidos e que você iniciou o Streamlit a partir da **raiz** do projeto.

### 404 Index 'rag_pdf_deployed' is not found (ao consultar o agente)

Esse erro ocorre na **consulta** (agente), não na ingestão. Significa que o endpoint não tem um deployed index com o ID configurado em `VECTOR_SEARCH_DEPLOYED_INDEX_ID`. No Console, em **Index endpoints** → seu endpoint → **Deployed indexes**, confira o **ID** exato do índice implantado e use esse valor em `VECTOR_SEARCH_DEPLOYED_INDEX_ID` no `.env`.

---

## Resumo rápido

| Passo | Ação |
|-------|------|
| 1 | Criar `.env` a partir de `.env.example` e preencher projeto, região, bucket, resource name do índice, resource name do endpoint e embedding (Gemini key ou Vertex). |
| 2 | Opcional: definir `VECTOR_SEARCH_DEPLOYED_INDEX_ID` e `PDF_SOURCE_DIR` se necessário. |
| 3a | **Script**: colocar PDFs em `data/pdfs` (ou em `PDF_SOURCE_DIR`) e rodar `python scripts/ingest_pdfs_to_rag.py` (ou com `--force` para re-ingerir). |
| 3b | **Frontend**: rodar `streamlit run frontend/app.py`, fazer upload dos PDFs e clicar em "Processar documentos". |
| 4 | Verificar no GCS o chunk store e o manifesto; em seguida usar os agentes (`run_adk_agent` / `run_langgraph_agent`) para consultar o RAG. |

Após concluir a ingestão, você pode seguir para o lab de consulta (agentes) para fazer perguntas sobre os documentos indexados.
