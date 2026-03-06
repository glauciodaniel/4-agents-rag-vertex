# Lab: Consulta ao RAG (perguntas aos agentes)

Este tutorial descreve o passo a passo para **consultar o RAG** após a ingestão de documentos: como rodar os agentes ADK e LangGraph, exemplos de perguntas (IA Responsável na AWS e princípios de IA Responsável do Google) e **onde na Vertex AI / Google Cloud é possível ver que essa consulta foi feita**.

---

## Índice

1. [Visão geral da consulta](#1-visão-geral-da-consulta)
2. [Pré-requisitos](#2-pré-requisitos)
3. [Exemplo 1: Agente ADK – IA Responsável na AWS](#3-exemplo-1-agente-adk--ia-responsável-na-aws)
4. [Exemplo 2: Agente LangGraph – Princípios de IA Responsável do Google](#4-exemplo-2-agente-langgraph--princípios-de-ia-responsável-do-google)
5. [Onde ver na Vertex AI que a consulta aconteceu](#5-onde-ver-na-vertex-ai-que-a-consulta-aconteceu)
6. [Solução de problemas](#6-solução-de-problemas)

---

## 1. Visão geral da consulta

Quando você faz uma pergunta a um dos agentes:

1. O **agente** (ADK ou LangGraph) recebe sua pergunta e decide chamar as **tools** de retrieval.
2. A tool **`vertex_rag_retrieval`** (em `src/tools/vertex_rag_tool.py`):
   - Gera o **embedding da pergunta** com Gemini (768 dimensões).
   - Chama o **Vertex AI Vector Search** (`find_neighbors`) no endpoint configurado no `.env`.
   - Usa os IDs retornados para buscar os **textos** no chunk store (GCS).
   - Devolve o contexto formatado (trechos + fonte/página) para o agente.
3. O **modelo Gemini** (Vertex AI) usa esse contexto para gerar a resposta final.

Ou seja, cada pergunta pode gerar:
- Chamadas à **API de embedding** (Gemini).
- Chamadas ao **Vector Search** (`find_neighbors`).
- Chamadas à **API generativa** (Gemini) para a resposta (e, no ADK, possivelmente para resumir contexto).

Este lab mostra como executar duas perguntas de exemplo e onde esses acessos aparecem na plataforma.

---

## 2. Pré-requisitos

- **Ingestão já realizada**: documentos indexados no Vector Search e chunk store preenchido no GCS (veja [LAB-INGESTAO-DOCUMENTOS.md](./LAB-INGESTAO-DOCUMENTOS.md)).
- **Variáveis no `.env`** (na raiz do projeto):
  - `GOOGLE_CLOUD_PROJECT`
  - `GOOGLE_CLOUD_LOCATION`
  - `VECTOR_SEARCH_INDEX_ENDPOINT_NAME` (resource name completo do endpoint)
  - `VECTOR_SEARCH_DEPLOYED_INDEX_ID` (ID do índice implantado no endpoint — ex.: `rag_pdf_deployed`)
  - `GEMINI_API_KEY` ou uso de Vertex com `GOOGLE_GENAI_USE_VERTEXAI=1` e `gcloud auth application-default login`
- **Ambiente Python**: venv ativado e `pip install -r requirements.txt` já executado.

Na raiz do projeto, confira se o `.env` está preenchido. O agente ADK exige também `VECTOR_SEARCH_DEPLOYED_INDEX_ID`; o LangGraph usa o mesmo valor na tool de RAG.

---

## 3. Exemplo 1: Agente ADK – IA Responsável na AWS

O agente **ADK** é indicado para perguntas objetivas e respostas diretas. Ele usa uma tool de recuperação (RAG) e uma de resumo de contexto.

### 3.1 Executar o agente ADK

Na raiz do projeto, com o venv ativado:

```bash
python -m src.run_adk_agent
```

Você verá algo como:

```
Agente RAG (ADK) – Vertex AI. Digite sua pergunta (ou 'sair' para encerrar).

Você:
```

### 3.2 Fazer a pergunta de exemplo

Digite exatamente (ou adapte):

```
O que é IA responsável na AWS?
```

Pressione Enter. O agente:

1. Chama a tool de recuperação com essa pergunta (embedding + Vector Search + chunk store).
2. Se o contexto for longo, pode chamar a tool de resumo.
3. Gera a resposta com base nos trechos recuperados e exibe no terminal.

A resposta deve citar fontes (documento e página) quando os PDFs ingeridos tiverem conteúdo sobre IA responsável na AWS.

### 3.3 Encerrar

Digite `sair`, `exit` ou `quit`, ou use Ctrl+C para sair do loop.

---

## 4. Exemplo 2: Agente LangGraph – Princípios de IA Responsável do Google

O agente **LangGraph** é indicado para análise mais aprofundada e múltiplas buscas (multi-hop). Ele tem tools de recuperação e de **comparação de fontes** (duas buscas com ângulos diferentes).

### 4.1 Executar o agente LangGraph

Na raiz do projeto, com o venv ativado:

```bash
python -m src.run_langgraph_agent
```

Você verá algo como:

```
Agente LangGraph (RAG + Vertex AI). Digite sua pergunta (ou 'sair' para encerrar).

Você:
```

### 4.2 Fazer a pergunta de exemplo

Digite exatamente (ou adapte):

```
Quais são os princípios de IA Responsável do Google?
```

Pressione Enter. O agente pode:

1. Usar **retrieve_rag_documentation** para buscar trechos sobre o tema.
2. Usar **compare_sources** para fazer duas buscas (conceito direto + detalhes/exemplos) e comparar.
3. Raciocinar em etapas e citar as fontes (documento e página).

A resposta deve refletir o conteúdo dos PDFs indexados sobre princípios de IA Responsável do Google.

### 4.3 Encerrar

Digite `sair`, `exit` ou `quit`, ou use Ctrl+C.

---

## 5. Onde ver na Vertex AI que a consulta aconteceu

É possível ver evidências da consulta em três níveis: **audit logs (Data Access)** no Cloud Logging, **request-response logging** do Gemini (opcional, em BigQuery) e **métricas** do Vector Search. Nenhum deles é uma “tela de consultas” única; você monta a visão a partir dessas ferramentas.

### 5.1 Cloud Logging – Audit logs (Data Access)

As chamadas à API da Vertex AI que **leem dados** (incluindo buscas no Vector Search e predição) podem ser registradas nos **Data Access audit logs** do projeto.

**Requisito:** os [Data Access audit logs](https://cloud.google.com/logging/docs/audit/configure-data-access) precisam estar **habilitados** para o projeto (ou para a organização). Por padrão podem estar desativados.

**O que aparece:**

- **`indexEndpoints.findNeighbors`**  
  Cada vez que a tool de RAG chama o Vector Search (`find_neighbors`), essa operação pode ser registrada como Data Access. Assim você vê que “houve uma consulta ao índice” naquele momento.

- **Chamadas ao modelo generativo (Gemini)**  
  Dependendo de como a Vertex AI expõe a API (ex.: `generateContent`), chamadas de predição podem aparecer como acesso a dados no mesmo serviço (`aiplatform.googleapis.com`). O formato exato depende do tipo de recurso (endpoint vs. modelo publicado).

**Como ver no Console:**

1. No **Google Cloud Console**, abra **Logging** → **Logs Explorer**.
2. Selecione o **projeto** correto.
3. Use um filtro como o abaixo para focar em **Vertex AI** e, se quiser, só em Vector Search:

   - Para ver operações de **Vector Search** (find_neighbors):
     - **Recurso**: tipo `audited_resource` ou recurso do tipo Vertex AI.
     - **Log name** (nome do log):  
       `projects/SEU_PROJECT_ID/logs/cloudaudit.googleapis.com%2Fdata_access`
     - No **Editor de consulta**, você pode usar algo como:
       ```text
       resource.type="audited_resource"
       protoPayload.serviceName="aiplatform.googleapis.com"
       protoPayload.methodName="indexEndpoints.findNeighbors"
       ```
     - Ajuste `SEU_PROJECT_ID` e, se necessário, o `methodName` conforme a [documentação de audit do Vertex AI](https://cloud.google.com/vertex-ai/docs/general/audit-logging).

4. Defina o **intervalo de tempo** (ex.: última 1 hora) e execute a consulta.  
   As entradas correspondem a acessos ao endpoint do Vector Search; cada uma pode representar uma consulta (ou um lote) feita pelo agente.

Assim você **consegue ver que houve consulta** ao índice (e, se habilitado e filtrado, atividade de predição) na Vertex AI via Cloud Logging.

### 5.2 Request-Response Logging do Gemini (BigQuery) – ver o texto da pergunta e da resposta

Para **ver o conteúdo** da pergunta e da resposta (não só que a API foi chamada), use o **request-response logging** dos modelos Gemini na Vertex AI. Os logs vão para uma **tabela no BigQuery**.

**Como habilitar (resumo):**

1. Na documentação [Log requests and responses | Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/request-response-logging), siga os passos para **base foundation models** (Gemini).
2. Crie um **dataset e uma tabela** no BigQuery para receber os logs (ou use o URI apenas com projeto/dataset e deixe a Vertex criar a tabela).
3. Configure o **PublisherModelConfig** do modelo (ex.: `gemini-2.0-flash-001`) com:
   - `loggingConfig.enabled = true`
   - `samplingRate` (ex.: 1.0 para 100% das requisições)
   - `bigqueryDestination` com o URI da tabela (ex.: `bq://PROJECT_ID.DATASET.TABLE`).

Isso pode ser feito via **REST API** (`setPublisherModelConfig`) ou **Python SDK** (`GenerativeModel` em `vertexai.preview.generative_models` e `set_request_response_logging_config`).

**O que você vê no BigQuery:**

- Campos como `full_request` e `full_response` (ou equivalentes) trazem o conteúdo da requisição e da resposta (pergunta do usuário e texto gerado pelo modelo). Assim você **vê exatamente** que a consulta “O que é IA responsável na AWS?” ou “Quais são os princípios de IA Responsável do Google?” foi feita e qual foi a resposta.

Ou seja: **sim, é possível ver que houve a consulta** e qual foi o texto, configurando o request-response logging e consultando a tabela no BigQuery.

### 5.3 Vector Search – Monitoramento (métricas, não consultas individuais)

No **Vertex AI → Vector Search**, a página de monitoramento do **Index Endpoint** mostra **métricas** de uso/capacidade, **não** o histórico de perguntas nem o conteúdo das consultas:

- **Número de shards** do índice implantado (`current_shards`).
- **Número de réplicas** ativas (`current_replicas`), que pode subir/descer com o volume de consultas.

Ou seja: você vê **que o serviço está sendo usado** (e se há escala), mas **não** “a consulta X foi feita às Y horas”. Para “ver que a consulta aconteceu” em nível de conteúdo ou de chamada de API, use os itens 5.1 e 5.2.

### 5.4 Resumo prático

| O que você quer ver | Onde |
|---------------------|------|
| Que houve chamadas ao **Vector Search** (find_neighbors) | **Cloud Logging** → Logs Explorer → Data Access audit logs, filtro por `aiplatform.googleapis.com` e `indexEndpoints.findNeighbors`. |
| O **texto da pergunta e da resposta** do Gemini | **Request-Response Logging** do modelo Gemini → tabela no **BigQuery** (`full_request` / `full_response`). |
| Uso/carga do índice (shards, réplicas) | **Vertex AI** → **Vector Search** → monitoramento do Index Endpoint (métricas). |

Assim, após rodar os exemplos do ADK e do LangGraph, você pode:
- **Confirmar que a consulta ocorreu** via audit logs (Data Access) no Cloud Logging.
- **Ver o conteúdo da consulta e da resposta** no BigQuery, se tiver habilitado o request-response logging do Gemini.

---

## 6. Solução de problemas

### Erro: "Defina no .env: ..."

Verifique se todas as variáveis listadas na seção 2 estão definidas no `.env`, em especial `VECTOR_SEARCH_DEPLOYED_INDEX_ID`. O ADK exige essa variável.

### Erro: 404 Index 'rag_pdf_deployed' is not found

O **endpoint** não tem um índice implantado com o ID configurado em `VECTOR_SEARCH_DEPLOYED_INDEX_ID`. No Console: **Vertex AI → Vector Search → Index endpoints** → clique no endpoint → em **Deployed indexes**, confira o **ID** exato e use esse valor no `.env`.

### "Nenhum contexto encontrado para a consulta"

O Vector Search retornou poucos ou nenhum vizinho (ou o chunk store não tinha textos para os IDs). Confirme que a ingestão foi feita e que os PDFs contêm conteúdo relacionado à pergunta; pode ser necessário ajustar a pergunta ou ingerir documentos mais relevantes.

### Warning: "non-text parts in the response: ['function_call']"

É um aviso do SDK ao processar a resposta do modelo (que inclui chamada de tool). O agente pode responder normalmente; se a resposta vier vazia ou com mensagem de erro, verifique o 404 ou a configuração do endpoint/deployed index.

---

## Resumo rápido

| Passo | Ação |
|-------|------|
| 1 | Garantir ingestão feita e `.env` com endpoint e `VECTOR_SEARCH_DEPLOYED_INDEX_ID`. |
| 2 | **ADK**: `python -m src.run_adk_agent` → pergunta: *"O que é IA responsável na AWS?"* |
| 3 | **LangGraph**: `python -m src.run_langgraph_agent` → pergunta: *"Quais são os princípios de IA Responsável do Google?"* |
| 4 | **Ver que a consulta aconteceu**: Cloud Logging (Data Access audit, `indexEndpoints.findNeighbors`) e/ou BigQuery (request-response logging do Gemini). |
| 5 | **Ver uso do índice**: Vertex AI → Vector Search → monitoramento do endpoint (métricas de shards/réplicas). |

Com isso você percorre o fluxo completo de consulta ao RAG e sabe onde, na Vertex AI e no Google Cloud, verificar que a consulta foi feita.
