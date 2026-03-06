# Lab Avançado: Hybrid Search (BM25 + Vetorial) e RRF

Tutorial para implementar **busca híbrida** no projeto RAG: combinar **precisão léxica (BM25)** com **busca semântica (Vector Search)** usando **Reciprocal Rank Fusion (RRF)**. Inclui teoria, uso do código já integrado e uma ideia avançada opcional (query expansion).

**Duração estimada:** 2h–3h  
**Pré-requisitos:** ter concluído ingestão e consulta básica (LAB-INGESTAO-DOCUMENTOS e LAB-CONSULTA-RAG).

---

## Índice

1. [Objetivo e resultado esperado](#1-objetivo-e-resultado-esperado)
2. [Criar o branch de trabalho](#2-criar-o-branch-de-trabalho)
3. [Teoria: por que e como funciona o Hybrid Search](#3-teoria-por-que-e-como-funciona-o-hybrid-search)
4. [Visão da implementação no projeto](#4-visão-da-implementação-no-projeto)
5. [Passo a passo: instalar, (re)ingerir e testar](#5-passo-a-passo-instalar-reingerir-e-testar)
6. [Comparar: vetorial vs híbrido](#6-comparar-vetorial-vs-híbrido)
7. [Avançado opcional: Query Expansion](#7-avançado-opcional-query-expansion)
8. [Resumo e referências](#8-resumo-e-referências)

---

## 1. Objetivo e resultado esperado

**Objetivo:** usar no mesmo RAG duas buscas complementares e fusionar os resultados:

- **Busca vetorial (semântica):** “entende” significado (ex.: “IA responsável” e “responsible AI”).
- **Busca BM25 (léxica):** destaca **palavras exatas** e termos técnicos (ex.: “BM25”, “RAG”, siglas).

Com **RRF** você une as duas listas ranqueadas em uma só, melhorando recall e relevância em perguntas que misturam conceitos e termos técnicos.

**Resultado:** um agente (ADK ou LangGraph) que chama uma tool de **retrieval híbrido** em vez da tool só vetorial, mantendo o resto do fluxo igual (chunk store, citação de fontes, etc.).

---

## 2. Criar o branch de trabalho

Sempre que for adicionar uma funcionalidade nova, use um branch separado.

Na raiz do repositório:

```bash
git status
git checkout -b feature/hybrid-search
```

Se o projeto já tiver os arquivos do hybrid search (por exemplo após um merge ou clone atualizado), o branch `feature/hybrid-search` serve para você fazer testes e experimentos sem alterar a base. Se você for implementar do zero seguindo este lab, crie o branch antes de criar os arquivos e commits.

---

## 3. Teoria: por que e como funciona o Hybrid Search

### 3.1 Busca só vetorial (atual)

Hoje a tool de RAG:

1. Gera o **embedding** da pergunta.
2. Busca no **Vector Search** os `top_k` vizinhos mais próximos (similaridade de cosseno ou produto interno).
3. Recupera os textos no chunk store e devolve ao LLM.

**Vantagem:** captura **similaridade semântica** (sinônimos, paráfrases).  
**Limitação:** termos muito específicos, siglas ou números podem “diluir” no vetor; uma pergunta com palavra-chave exata às vezes se beneficia de match léxico.

### 3.2 BM25 (busca léxica)

BM25 é um modelo de ranking baseado em **TF-IDF** (frequência do termo no documento e raridade no corpus). Ele ranqueia documentos por **ocorrência de palavras** da query (após tokenização).

- **Vantagem:** ótimo para termos técnicos, nomes, siglas, números.
- **Limitação:** não entende sinônimos nem significado (“RAG” não encontra “retrieval-augmented generation” só por palavra).

### 3.3 Por que combinar (híbrido)?

Perguntas reais misturam:

- Conceitos (“o que é IA responsável”) → forte no **vetorial**.
- Termos exatos (“BM25”, “Vertex AI”, “768 dimensões”) → forte no **BM25**.

Combinando os dois, você tende a ter melhor **recall** e **relevância** do que usando só um.

### 3.4 Reciprocal Rank Fusion (RRF)

Temos duas listas ordenadas:

- Lista A: IDs ranqueados pela busca vetorial (1º, 2º, 3º, …).
- Lista B: IDs ranqueados pelo BM25 (1º, 2º, 3º, …).

**RRF** atribui a cada documento um score de “fusão”:

- Para cada lista, documento na posição `rank` (1-based) recebe: `1 / (k + rank)`.
- `k` é uma constante (tipicamente 60); evita que o 1º lugar domine demais.
- O **score final** do documento é a **soma** desses valores em todas as listas em que ele aparecer.
- Ordenamos por esse score e pegamos o `top_k` para o contexto do LLM.

Fórmula (para cada documento `d`):

```text
score_RRF(d) = Σ 1 / (k + rank_i(d))
```

onde `rank_i(d)` é a posição de `d` na lista `i` (ou não conta se não aparecer). Depois: ordenar por `score_RRF` decrescente e tomar os primeiros `top_k`.

Assim, um chunk que aparece bem ranqueado nas **duas** listas tende a subir; um que só é bom em uma ainda pode entrar no top_k.

---

## 4. Visão da implementação no projeto

Os arquivos abaixo já estão (ou devem estar) no repositório no branch de hybrid search. Use esta seção para **abrir na ordem** e explicar em aula.

### 4.1 Novos arquivos

| Arquivo | Função |
|--------|--------|
| `src/bm25_index.py` | Tokenização, construção do índice BM25 a partir dos chunks, **persistência no GCS** (`vector-search-index/bm25_index.json`), e função `get_top_k(query, top_k)` para retornar os melhores `chunk_id` por BM25. |
| `src/tools/hybrid_rag_tool.py` | Tool `vertex_hybrid_retrieval`: chama Vector Search + BM25, aplica **RRF**, monta a lista final de IDs, busca textos no chunk store e formata o contexto para o LLM. |
| `src/agent_adk_hybrid.py` | Agente ADK que usa a tool **híbrida** em vez da tool só vetorial. |
| `src/agent_langgraph_hybrid.py` | Agente LangGraph com tools híbridas (`retrieve_hybrid_documentation`, `compare_sources_hybrid`). |
| `src/run_adk_hybrid.py` | CLI: `python -m src.run_adk_hybrid`. |
| `src/run_langgraph_hybrid.py` | CLI: `python -m src.run_langgraph_hybrid`. |

### 4.2 Arquivos alterados

| Arquivo | Alteração |
|--------|-----------|
| `requirements.txt` | Inclusão de `rank_bm25>=0.2.2` e `numpy>=1.24.0`. |
| `src/rag_ingest.py` | Após `save_chunks`, chamada a `bm25_index.build_and_save(chunk_map, merge=True)` para construir/atualizar o índice BM25 no GCS (em try/except para não quebrar a ingestão se BM25 falhar). |

### 4.3 Fluxo de dados (resumo)

**Ingestão (já existente + BM25):**

1. Chunking → embedding → upsert no Vector Search → `save_chunks` no GCS.
2. **Novo:** `build_and_save` do BM25: lê o `chunk_map` (e, se `merge=True`, o índice BM25 já salvo no GCS), adiciona os novos chunks (tokenizados), e grava de volta em `vector-search-index/bm25_index.json`.

**Consulta híbrida:**

1. Embed da query → Vector Search `find_neighbors` → lista de IDs (ordem vetorial).
2. Query tokenizada → BM25 `get_top_k` (lendo índice do GCS) → lista de IDs (ordem BM25).
3. RRF nas duas listas → lista fusionada de IDs (ordem RRF).
4. `load_chunks_by_ids(fused_ids)` → textos no chunk store → formatação `[Fonte: doc, p.X]` → retorno para o agente.

---

## 5. Passo a passo: instalar, (re)ingerir e testar

### 5.1 Instalar dependências

No ambiente virtual do projeto (na raiz):

```bash
pip install rank_bm25 numpy
# ou
pip install -r requirements.txt
```

Assim o módulo `src.bm25_index` e a tool híbrida passam a funcionar.

### 5.2 Garantir que o índice BM25 existe no GCS

O índice BM25 é construído **durante a ingestão**. Duas situações:

- **Você já ingeriu antes de ter o código do hybrid:** é preciso **re-ingerir** pelo menos um PDF com a opção de merge, para que `build_and_save` rode e crie/atualize o `bm25_index.json` no bucket.
- **Você ainda não ingeriu:** faça a ingestão normalmente (script ou Streamlit); o BM25 será gerado no mesmo fluxo.

**Re-ingerir (para criar/atualizar o BM25):**

```bash
python scripts/ingest_pdfs_to_rag.py --force
```

Ou, no frontend Streamlit, marque “Re-ingerir mesmo que o arquivo já exista” e processe os documentos.

**Verificação (opcional):** no Google Cloud Storage, confira se existe o objeto `vector-search-index/bm25_index.json` no bucket configurado em `GOOGLE_CLOUD_STORAGE_BUCKET`.

### 5.3 Rodar o agente ADK híbrido

```bash
python -m src.run_adk_hybrid
```

Faça uma pergunta que misture conceito e termo técnico, por exemplo:

- “O que é RAG e como o BM25 pode ajudar?”
- “Quais são os princípios de IA responsável na Vertex AI?”

O agente usará a tool de retrieval híbrido (Vector + BM25 + RRF) e citará as fontes como antes.

### 5.4 Rodar o agente LangGraph híbrido

```bash
python -m src.run_langgraph_hybrid
```

Teste com a mesma pergunta ou com uma que beneficie de “compare_sources_hybrid” (duas buscas híbridas com ângulos diferentes).

---

## 6. Comparar: vetorial vs híbrido

Sugestão de experimento para aula (cerca de 15–20 min):

1. **Pergunta 1 (mais conceitual):** “O que é IA responsável?”  
   - Rodar no agente **só vetorial** (`run_adk_agent` ou `run_langgraph_agent`) e anotar a resposta.  
   - Rodar no agente **híbrido** (`run_adk_hybrid` ou `run_langgraph_hybrid`) e comparar.

2. **Pergunta 2 (termo técnico + conceito):** “Como funciona o BM25 na busca de documentos?” (assumindo que há conteúdo sobre BM25 nos PDFs.)  
   - Mesmo processo: vetorial vs híbrido.

3. **Discussão:** em qual pergunta o híbrido trouxe trechos mais relevantes ou respostas mais precisas? Por que termos como “BM25” tendem a se beneficiar do braço léxico?

Isso fixa a ideia de que **semântica + léxico** cobrem tipos diferentes de pergunta.

---

## 7. Avançado opcional: Query Expansion

**Tempo estimado:** +30–45 min.

**Ideia:** antes de chamar o retrieval híbrido, usar o **próprio LLM** para gerar uma ou duas **queries alternativas** (reformulação ou sinônimos). Depois rodar o retrieval híbrido para a query original **e** para as alternativas, e **fusionar** os resultados (por exemplo com RRF de 3 listas: vetorial, BM25, e opcionalmente “vetorial da query expandida”). Assim você aumenta o recall para perguntas ambíguas ou muito curtas.

**Passos sugeridos (esqueleto):**

1. Criar uma função `expand_query(query: str) -> list[str]` que chama o Gemini com um prompt do tipo:  
   “Dado a pergunta do usuário abaixo, gere até 2 outras formas de perguntar a mesma coisa (reformulação ou sinônimos). Retorne só as perguntas, uma por linha.”
2. Em `vertex_hybrid_retrieval` (ou em uma variante `vertex_hybrid_retrieval_with_expansion`):
   - Chamar `expand_query(query)` e obter `[query, alt1, alt2]`.
   - Para cada string, rodar Vector Search + BM25 e obter listas de IDs.
   - Aplicar RRF sobre **todas** as listas (não só 2: vetorial e BM25 da query original, mas também as listas das queries alternativas, se quiser).
   - Manter o resto igual: `load_chunks_by_ids`, formatação, retorno.

Você pode limitar a 1 query alternativa para reduzir custo e tempo; o importante é ver que **multi-query + RRF** é um padrão útil para RAG avançado.

---

## 8. Resumo e referências

- **Branch:** `feature/hybrid-search` para desenvolver e testar.
- **Teoria:** BM25 = precisão léxica; Vector = semântica; RRF = fusão de rankings sem calibrar scores.
- **Implementação:** `bm25_index.py` (índice no GCS), `hybrid_rag_tool.py` (Vector + BM25 + RRF), agentes e runners `*_hybrid`.
- **Uso:** instalar `rank_bm25` (e numpy), (re)ingerir para gerar `bm25_index.json`, rodar `run_adk_hybrid` ou `run_langgraph_hybrid`.
- **Opcional:** query expansion com LLM e RRF sobre múltiplas listas.

**Referências:**

- [BM25 (Wikipedia)](https://en.wikipedia.org/wiki/Okapi_BM25)
- [Reciprocal Rank Fusion (RRF)](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [rank_bm25 (Python)](https://github.com/dorianbrown/rank_bm25)

Com isso você cobre um lab avançado de 2h–3h (teoria + prática + comparação + opcional query expansion) aproveitando o mesmo projeto RAG com Vertex AI.
