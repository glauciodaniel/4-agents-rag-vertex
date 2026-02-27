#!/usr/bin/env python3
"""
Cria o índice Vertex AI Vector Search (stream), o index endpoint e faz o deploy.
Requer: GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION, GOOGLE_CLOUD_STORAGE_BUCKET no .env.
Atualiza .env com VECTOR_SEARCH_INDEX_NAME, VECTOR_SEARCH_INDEX_ENDPOINT_NAME e VECTOR_SEARCH_DEPLOYED_INDEX_ID.
Embedding: textembedding-gecko@001 (768 dimensões).
"""
import os
import time
import sys

from dotenv import load_dotenv, set_key

# Carrega .env do diretório raiz do projeto
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(REPO_ROOT, ".env"))
ENV_PATH = os.path.join(REPO_ROOT, ".env")

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")
BUCKET = os.getenv("GOOGLE_CLOUD_STORAGE_BUCKET")

# Dimensão do textembedding-gecko@001
DIMENSIONS = 768
INDEX_DISPLAY_NAME = "rag-pdf-index"
ENDPOINT_DISPLAY_NAME = "rag-pdf-endpoint"
DEPLOYED_INDEX_ID = "rag_pdf_deployed"


def main() -> None:
    if not PROJECT_ID or not LOCATION:
        print("Defina GOOGLE_CLOUD_PROJECT e GOOGLE_CLOUD_LOCATION no .env", file=sys.stderr)
        sys.exit(1)
    if not BUCKET:
        print("Defina GOOGLE_CLOUD_STORAGE_BUCKET no .env (usado como URI base do índice)", file=sys.stderr)
        sys.exit(1)

    import google.cloud.aiplatform as aiplatform
    from google.api_core import exceptions as gcp_exceptions

    aiplatform.init(project=PROJECT_ID, location=LOCATION)

    # URI para índice vazio (stream); o RAG Engine populará depois
    contents_delta_uri = f"gs://{BUCKET.strip().rstrip('/')}/vector-search-index/"
    print(f"Índice stream será criado com base em: {contents_delta_uri}")

    # Criar índice (stream) - 768 dims para textembedding-gecko@001
    # algorithmConfig é obrigatório na API: passando tree-AH config (leaf_node_embedding_count e/ou leaf_nodes_to_search_percent)
    index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
        display_name=INDEX_DISPLAY_NAME,
        contents_delta_uri=contents_delta_uri,
        dimensions=DIMENSIONS,
        approximate_neighbors_count=150,
        distance_measure_type=aiplatform.matching_engine.matching_engine_index_config.DistanceMeasureType.DOT_PRODUCT_DISTANCE,
        leaf_node_embedding_count=1000,
        leaf_nodes_to_search_percent=10,
        index_update_method="STREAM_UPDATE",
        description="Índice para RAG com textembedding-gecko@001 (768 dims)",
    )
    print(f"Índice criado: {index.resource_name}")

    # Criar endpoint
    endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
        display_name=ENDPOINT_DISPLAY_NAME,
        public_endpoint_enabled=True,
        description="Endpoint para RAG PDF",
    )
    print(f"Endpoint criado: {endpoint.resource_name}")

    # Deploy do índice no endpoint (pode levar ~20–30 min na primeira vez)
    deployed_index_id = DEPLOYED_INDEX_ID
    print("Fazendo deploy do índice no endpoint (aguarde)...")
    try:
        endpoint.deploy_index(index=index, deployed_index_id=deployed_index_id)
    except gcp_exceptions.AlreadyExists as e:
        if "DeployedIndex with same ID" in str(e):
            deployed_index_id = f"{DEPLOYED_INDEX_ID}_{int(time.time())}"
            print(f"ID '{DEPLOYED_INDEX_ID}' já em uso em outro endpoint. Usando '{deployed_index_id}'.")
            endpoint.deploy_index(index=index, deployed_index_id=deployed_index_id)
        else:
            raise
    print("Deploy concluído.")

    try:
        set_key(ENV_PATH, "VECTOR_SEARCH_INDEX_NAME", index.resource_name)
        set_key(ENV_PATH, "VECTOR_SEARCH_INDEX_ENDPOINT_NAME", endpoint.resource_name)
        set_key(ENV_PATH, "VECTOR_SEARCH_DEPLOYED_INDEX_ID", deployed_index_id)
        print(f"Atualizado .env: VECTOR_SEARCH_INDEX_NAME, VECTOR_SEARCH_INDEX_ENDPOINT_NAME, VECTOR_SEARCH_DEPLOYED_INDEX_ID")
    except Exception as e:
        print(f"Aviso: não foi possível atualizar .env: {e}")
        print(f"Defina manualmente: VECTOR_SEARCH_INDEX_NAME={index.resource_name}")
        print(f"  VECTOR_SEARCH_INDEX_ENDPOINT_NAME={endpoint.resource_name}")
        print(f"  VECTOR_SEARCH_DEPLOYED_INDEX_ID={deployed_index_id}")


if __name__ == "__main__":
    main()
