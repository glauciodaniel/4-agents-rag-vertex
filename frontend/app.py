"""
Frontend Streamlit: drag-and-drop de PDFs e ingestão no Vector Search (chunking, Gemini embedding, upsert).
Requer .env com GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION, GOOGLE_CLOUD_STORAGE_BUCKET,
VECTOR_SEARCH_INDEX_NAME, VECTOR_SEARCH_INDEX_ENDPOINT_NAME, GEMINI_API_KEY.
Execute: streamlit run frontend/app.py
"""
import os
import sys

# Raiz do projeto para importar src
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import streamlit as st

from src.rag_ingest import get_env, ingest_pdfs_from_bytes

st.set_page_config(page_title="RAG Ingest – Vertex AI", page_icon="📄", layout="centered")
st.title("Upload de PDFs para o banco vetorial (RAG)")
st.caption("Arraste os documentos para a área abaixo. Eles serão processados (chunking, embedding Gemini) e indexados no Vector Search.")

project, location, bucket, index_name, index_endpoint, _ = get_env()
gemini_key = os.getenv("GEMINI_API_KEY")
if not all([project, bucket, index_name, index_endpoint]):
    st.error(
        "Configure o .env com GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_STORAGE_BUCKET, "
        "VECTOR_SEARCH_INDEX_NAME e VECTOR_SEARCH_INDEX_ENDPOINT_NAME."
    )
    st.stop()
if not gemini_key:
    st.error("Configure GEMINI_API_KEY no .env.")
    st.stop()

uploaded = st.file_uploader(
    "Arraste seus PDFs aqui",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded:
    st.info(f"{len(uploaded)} arquivo(s) selecionado(s). Clique em **Processar documentos** para indexar no Vector Search.")

force_reingest = st.checkbox("Re-ingerir mesmo que o arquivo já exista (ignorar deduplicação)", value=False)

if st.button("Processar documentos", type="primary", disabled=not uploaded):
    if not uploaded:
        st.warning("Selecione pelo menos um PDF.")
    else:
        with st.spinner("Processando PDFs (chunking, embedding, upsert no Vector Search)…"):
            try:
                files = [(f.name, f.read()) for f in uploaded]
                count, skipped = ingest_pdfs_from_bytes(files, force=force_reingest)
                st.success(f"Concluído. Chunks indexados: {count}.")
                if skipped:
                    st.info(f"Arquivos já ingeridos (pulados): {', '.join(skipped)}")
            except Exception as e:
                st.exception(e)
