# Lab: Observabilidade com Langfuse e @observe

Neste lab você vai **configurar o Langfuse** em um diretório separado, **conectar este projeto RAG** às variáveis de ambiente e **instrumentar a execução do agente** usando a diretiva **`@observe`** para monitorar traces no painel do Langfuse.

**Duração estimada:** 2 horas  
**Pré-requisitos:** Docker e Docker Compose instalados; projeto RAG já funcional (ingestão e agente rodando).

---

## Índice

1. [Objetivo e resultado esperado](#1-objetivo-e-resultado-esperado)
2. [O que é o Langfuse](#2-o-que-é-o-langfuse)
3. [Parte 1: Instalar e subir o Langfuse (diretório separado)](#3-parte-1-instalar-e-subir-o-langfuse-diretório-separado)
4. [Parte 2: Obter as chaves e configurar variáveis neste projeto](#4-parte-2-obter-as-chaves-e-configurar-variáveis-neste-projeto)
5. [Parte 3: Instalar o SDK e preparar a instrumentação](#5-parte-3-instalar-o-sdk-e-preparar-a-instrumentação)
6. [Parte 4: Usar @observe para monitorar a execução do agente](#6-parte-4-usar-observe-para-monitorar-a-execução-do-agente)
7. [Parte 5: Rodar o agente e visualizar os traces](#7-parte-5-rodar-o-agente-e-visualizar-os-traces)
8. [Dicas e solução de problemas](#8-dicas-e-solução-de-problemas)

---

## 1. Objetivo e resultado esperado

**Objetivo:**  
Adicionar **observabilidade** ao fluxo do agente RAG (ADK ou LangGraph) usando **Langfuse**, para ver no painel:

- **Traces** por execução (uma pergunta do usuário = um trace).
- **Spans** aninhados: chamada ao agente, tools (retrieval), latência, entradas e saídas.
- Erros e tempo de resposta.

**Resultado:**  
Ao final do lab, você terá o Langfuse rodando em um diretório à parte, este projeto configurado com as variáveis de ambiente, e a execução do agente instrumentada com **`@observe`**, enviando dados para o Langfuse. Você **não** precisa alterar a lógica de negócio do RAG; apenas envolver as funções relevantes com o decorator.

---

## 2. O que é o Langfuse

**Langfuse** é uma plataforma open-source de **observabilidade para aplicações LLM**: traces, métricas, custos e debugging. Ela permite:

- Ver cada **trace** (uma “conversa” ou request).
- Ver **spans** aninhados (agente, tools, chamadas ao modelo).
- Capturar **input/output** de funções decoradas com `@observe`.
- Analisar latência e erros.

O SDK Python oferece a diretiva **`@observe()`**, que você coloca em cima de funções. O Langfuse registra automaticamente:

- Argumentos de entrada (input),
- Valor de retorno (output),
- Tempo de execução,
- Erros (exceptions).

Assim, você **monitora a execução do agente** sem reescrever a lógica interna; basta decorar o ponto de entrada (por exemplo a função que processa a pergunta do usuário) e, se quiser, as tools ou outras funções importantes.

---

## 3. Parte 1: Instalar e subir o Langfuse (diretório separado)

O Langfuse deve ficar **fora** do diretório deste projeto RAG (por exemplo em uma pasta irmã ou em outro caminho). Assim você mantém o RAG e o servidor Langfuse separados.

### 3.1 Requisitos

- **Docker** e **Docker Compose** instalados ([Docker Desktop](https://www.docker.com/products/docker-desktop/) no Windows/Mac).
- **Git.**

### 3.2 Clonar e subir com Docker Compose

Abra um terminal e vá para um diretório **diferente** do projeto RAG (ex.: `c:\projects\fiap\langfuse` ou `~/projects/langfuse`).

```bash
# Exemplo: criar pasta e clonar
mkdir -p c:\projects\fiap\langfuse
cd c:\projects\fiap\langfuse

git clone https://github.com/langfuse/langfuse.git .
```

Ou, se preferir um nome de pasta explícito:

```bash
git clone https://github.com/langfuse/langfuse.git langfuse-server
cd langfuse-server
```

### 3.3 Ajustar segredos (obrigatório)

Antes de subir, altere as senhas e segredos no `docker-compose.yml`. Procure por linhas marcadas com `# CHANGEME` e substitua por valores fortes e aleatórios (ex.: gerador de senhas). Isso evita acesso não autorizado ao Langfuse.

### 3.4 Iniciar o Langfuse

```bash
docker compose up
```

Aguarde alguns minutos. Quando o log mostrar algo como **“Ready”** no container da web, o Langfuse estará disponível em:

**http://localhost:3000**

Abra no navegador. Na primeira vez você precisará **criar uma conta** (e-mail e senha) e, em seguida, **criar um projeto**. Anote o nome do projeto; você usará o projeto para obter as chaves de API.

---

## 4. Parte 2: Obter as chaves e configurar variáveis neste projeto

As chaves são necessárias para **este projeto RAG** enviar traces ao Langfuse. Elas ficam na interface do Langfuse.

### 4.1 Obter as chaves no Langfuse

1. Com o Langfuse rodando, acesse **http://localhost:3000** e faça login.
2. Abra o **projeto** que você criou (ou crie um novo).
3. Vá em **Settings** (ou **Project Settings**) do projeto.
4. Procure por **API Keys** / **Keys** / **Credentials**.
5. Crie um par de chaves (ou use o par padrão) e copie:
   - **Public Key** (começa com `pk-lf-...`)
   - **Secret Key** (começa com `sk-lf-...`)

Guarde a **Secret Key** em local seguro; ela não deve ser commitada no repositório.

### 4.2 Adicionar variáveis de ambiente **neste projeto**

No **projeto RAG** (onde está o `.env`), edite o arquivo **`.env`** na raiz e adicione as variáveis do Langfuse. **Não** coloque espaços em volta do `=` e não use aspas nos valores no `.env`:

```env
# Langfuse (observabilidade)
LANGFUSE_PUBLIC_KEY=pk-lf-xxxxxxxx
LANGFUSE_SECRET_KEY=sk-lf-xxxxxxxx
LANGFUSE_BASE_URL=http://localhost:3000
```

Substitua `pk-lf-xxxxxxxx` e `sk-lf-xxxxxxxx` pelas chaves que você copiou. Se o Langfuse estiver em outra máquina ou porta, ajuste `LANGFUSE_BASE_URL` (ex.: `http://192.168.1.10:3000`).

**Importante:** O `.env` já está no `.gitignore`. Nunca commite o `.env` com chaves reais.

---

## 5. Parte 3: Instalar o SDK e preparar a instrumentação

Tudo a partir daqui é feito **dentro do projeto RAG**, no ambiente virtual que você já usa.

### 5.1 Instalar o SDK Python do Langfuse

Na raiz do projeto RAG, com o venv ativado:

```bash
pip install langfuse
```

### 5.2 Garantir que o .env é carregado

O SDK do Langfuse lê as credenciais das variáveis de ambiente. Como este projeto já carrega o `.env` (por exemplo com `python-dotenv` nos scripts de run do agente), desde que você rode o agente a partir do mesmo ambiente (e o `.env` seja carregado antes de qualquer chamada ao Langfuse), as variáveis `LANGFUSE_*` estarão disponíveis.

Se você criar um script próprio de entrada, carregue o `.env` no início, por exemplo:

```python
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
```

---

## 6. Parte 4: Usar @observe para monitorar a execução do agente

A ideia é **decorar** as funções que representam a “execução do agente” e, se desejar, as **tools** de retrieval, para que o Langfuse crie um **trace** com **spans** aninhados.

### 6.1 Onde aplicar @observe

Sugestão de pontos (você implementa no código):

1. **Função que processa a pergunta do usuário e chama o agente**  
   Ex.: a função que, no `run_adk_agent.py` ou `run_langgraph_agent.py`, recebe a mensagem do usuário e devolve a resposta do agente. Essa função deve ser o **trace raiz** (a mais externa).
2. **Opcional: função da tool de RAG**  
   Ex.: a função que chama `vertex_rag_retrieval` (ou a tool de retrieval no agente). Assim você vê cada chamada de retrieval como um span filho, com input (query) e output (contexto).

A função mais externa decorada com `@observe()` vira um **trace**; as funções decoradas chamadas dentro dela viram **spans** (ou **generations**, se usar `as_type="generation"`) aninhados.

### 6.2 Importar o decorator

No arquivo onde você for instrumentar (ex.: `run_adk_agent.py` ou um módulo de serviço):

```python
from langfuse import observe
```

### 6.3 Exemplo: decorar a função que obtém a resposta do agente

Suponha que no seu `run_adk_agent.py` exista uma função `_get_response_sync(user_message)`. Para monitorar cada “pergunta → resposta” como um trace:

```python
from langfuse import observe

@observe(name="agente_rag_resposta")
def _get_response_sync(user_message: str) -> str:
    # ... implementação existente (chamada ao root_agent, etc.) ...
    return resposta
```

O Langfuse vai criar um trace por chamada, com o nome `agente_rag_resposta`, e registrar automaticamente o **input** (`user_message`) e o **output** (resposta), além do tempo e de possíveis erros.

### 6.4 Exemplo: decorar a tool de retrieval (opcional)

Se você quiser ver cada busca RAG como um span dentro do trace, pode decorar a função que chama o Vector Search. Por exemplo, na definição da tool (ou na função que a implementa):

```python
from langfuse import observe

@observe(name="retrieval_rag", as_type="span")
def retrieve_rag_documentation(query: str) -> str:
    return vertex_rag_retrieval(query, top_k=10, vector_distance_threshold=0.6)
```

Assim, cada chamada à tool aparece como um span com nome `retrieval_rag`. O parâmetro `as_type="span"` indica que é um passo de processamento; para chamadas diretas a um LLM você pode usar `as_type="generation"`.

### 6.5 Encerrar e enviar eventos (scripts curtos)

Em aplicações que terminam logo após rodar (como um CLI que processa uma pergunta e sai), o SDK pode enviar os eventos em background. Para não perder traces, **faça flush** antes de encerrar. Por exemplo, no `main()` do `run_adk_agent.py`, ao sair do loop (ou no final do script):

```python
from langfuse import get_client

def main():
    # ... loop de perguntas ...
    pass

if __name__ == "__main__":
    main()
    get_client().flush()
```

Assim você garante que os últimos eventos são enviados ao Langfuse antes do processo terminar.

---

## 7. Parte 5: Rodar o agente e visualizar os traces

### 7.1 Subir o Langfuse (se ainda não estiver rodando)

No diretório onde você clonou o Langfuse:

```bash
docker compose up
```

Acesse **http://localhost:3000** e confirme que está logado no projeto correto.

### 7.2 Rodar o agente neste projeto

No projeto RAG, com o `.env` contendo as variáveis `LANGFUSE_*` e com o SDK instalado:

```bash
python -m src.run_adk_agent
```

(ou `python -m src.run_langgraph_agent`)

Faça uma ou duas perguntas (ex.: “O que é IA responsável?”) e depois saia do loop (digite `sair` ou use Ctrl+C).

### 7.3 Ver os traces no Langfuse

1. No Langfuse, abra a seção **Traces** (ou **Trace**).
2. Você deve ver um trace por pergunta, com o nome que você definiu no `@observe` (ex.: `agente_rag_resposta`).
3. Clique em um trace para ver:
   - Input (pergunta do usuário),
   - Output (resposta do agente),
   - Duração,
   - E, se tiver decorado a tool, os spans filhos (ex.: `retrieval_rag`).

Use isso para analisar latência, erros e o fluxo completo da execução do agente.

---

## 8. Dicas e solução de problemas

### Traces não aparecem

- Confirme que `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` e `LANGFUSE_BASE_URL` estão corretos no `.env` **deste projeto** e que o `.env` é carregado antes de qualquer uso do Langfuse.
- Em scripts que terminam rápido, chame `get_client().flush()` antes de sair.
- Verifique se o Langfuse está acessível: abra `LANGFUSE_BASE_URL` no navegador (ex.: http://localhost:3000).

### LANGFUSE_BASE_URL para self-hosted

- Para Langfuse rodando na sua máquina: `LANGFUSE_BASE_URL=http://localhost:3000`.
- Sem `https`, use `http`. Sem barra no final.

### Reduzir tamanho de input/output

- Se o input ou o output das funções decoradas for muito grande, você pode desativar a captura por decorator: `@observe(capture_input=False, capture_output=False)` ou configurar a variável de ambiente mencionada na documentação do Langfuse para o decorator.

### Chamadas ao LLM

- Para marcar uma função como chamada a um modelo (ex.: geração de texto), use `@observe(as_type="generation", name="...")` e, se a API permitir, preencha metadados como `model`.

---

## Resumo do lab

| Etapa | Onde | Ação |
|--------|------|------|
| 1 | Diretório separado | Clonar Langfuse, ajustar segredos no `docker-compose.yml`, rodar `docker compose up`. |
| 2 | Navegador | Acessar http://localhost:3000, criar conta e projeto, obter Public Key e Secret Key. |
| 3 | Projeto RAG | Adicionar no `.env`: `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL`. |
| 4 | Projeto RAG | `pip install langfuse`. |
| 5 | Projeto RAG | Instrumentar: `from langfuse import observe` e decorar a função que processa a pergunta do usuário (e opcionalmente a tool de retrieval) com `@observe(...)`. Chamar `get_client().flush()` ao encerrar o script. |
| 6 | Projeto RAG | Rodar o agente e fazer algumas perguntas. |
| 7 | Langfuse | Abrir a seção Traces e analisar os traces e spans da execução do agente. |

Com isso você terá o Langfuse em um diretório à parte, as variáveis de ambiente configuradas **neste projeto** e o uso da diretiva **`@observe`** para monitorar a execução do agente em um lab de cerca de 2 horas.
