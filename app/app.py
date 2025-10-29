# app/app.py - updated robust version with dynamic embedding-dim handling and debug-friendly retrieval
import os
import time
import traceback

import streamlit as st
from dotenv import load_dotenv

# Load .env early
load_dotenv()

# -------------------------
# Initialize session_state
# -------------------------
if "vectors" not in st.session_state:
    st.session_state.vectors = None
if "embedded" not in st.session_state:
    st.session_state.embedded = False
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chain" not in st.session_state:
    st.session_state.chain = None
if "loader" not in st.session_state:
    st.session_state.loader = None
if "docs" not in st.session_state:
    st.session_state.docs = None
if "final_documents" not in st.session_state:
    st.session_state.final_documents = None
if "embedding_dim" not in st.session_state:
    st.session_state.embedding_dim = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

# ---------- Imports that require installed packages ----------
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_objectbox.vectorstores import ObjectBox
from langchain_core.prompts import ChatPromptTemplate

# local utils (same-folder module)
import utils

# ---------- Page & paths ----------
st.set_page_config(layout="wide", page_title="ObjectBox + LangChain (RAG)")
st.title("ObjectBox VectorstoreDB with LLAMA3 — (Robust & Debuggable)")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "research_papers")
OBJECTBOX_DIR = os.path.join(PROJECT_ROOT, "objectbox")

#st.write(f"**Data dir:** `{DATA_DIR}`  •  **ObjectBox dir:** `{OBJECTBOX_DIR}`")

# ---------- Prompt ----------
prompt = ChatPromptTemplate.from_template(
    """
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question

<context>
{context}
</context>

Question: {input}
"""
)

# ---------- Helper utilities ----------
def detect_embedding_dim(emb):
    """
    Return the embedding vector dimension by embedding a short sample.
    """
    try:
        sample = "test embedding vector dimension"
        vecs = emb.embed_documents([sample])
        if not vecs or not isinstance(vecs, list) or not hasattr(vecs[0], "__len__"):
            return None
        return len(vecs[0])
    except Exception:
        return None


def try_create_embedding():
    """
    Try to create the repo embedding (utils.huggingface_instruct_embedding()) and detect its dimension.
    Stores them in session_state on success.
    """
    try:
        emb = utils.huggingface_instruct_embedding()
        dim = detect_embedding_dim(emb)
        if dim is None:
            st.error("Failed to determine embedding vector dimension from the embedding model.")
            return None
        st.session_state.embeddings = emb
        st.session_state.embedding_dim = dim
        st.info(f"Embedding loaded (dim={dim})")
        return emb
    except Exception as e:
        st.error(f"Failed to create embedding model: {e}")
        st.write(traceback.format_exc())
        return None


def try_load_persisted_vectorstore(path: str = OBJECTBOX_DIR):
    """
    Attempt to open an existing ObjectBox vectorstore using the repo embedding model and detected dim.
    Returns the vectorstore instance or None.
    """
    try:
        if not os.path.exists(path) or not os.listdir(path):
            return None

        # Ensure we have an embedding and dim
        emb = st.session_state.embeddings or try_create_embedding()
        if emb is None:
            st.warning("Cannot load persisted vectorstore because embedding could not be created.")
            return None

        dim = st.session_state.embedding_dim
        if not dim:
            st.warning("Embedding dimension unknown; cannot instantiate ObjectBox.")
            return None

        # Try to instantiate the ObjectBox store with the embedding + dim
        try:
            vs = ObjectBox(embedding=emb, embedding_dimensions=dim, db_directory=path)
            st.info(f"Loaded ObjectBox with embedding_dimensions={dim}")
            return vs
        except Exception as e:
            # Show helpful message about schema mismatch
            st.error(f"Failed to open persisted ObjectBox: {e}")
            st.write(traceback.format_exc())
            st.warning("Possible causes: embedding dimension mismatch between DB and current embedding model, or DB schema mismatch.")
            return None

    except Exception as e:
        st.error(f"Unexpected error while trying to load ObjectBox: {e}")
        st.write(traceback.format_exc())
        return None


def embed_documents_and_create_vectorstore(data_dir: str = DATA_DIR, persist_dir: str = OBJECTBOX_DIR):
    """
    Ingest PDFs -> split -> create embeddings -> persist to ObjectBox.
    Uses the repo embedding and auto-detects the embedding dimension.
    """
    try:
        if not os.path.exists(data_dir) or not os.listdir(data_dir):
            st.error(f"No documents found in `{data_dir}`. Put your PDFs inside that folder.")
            return None

        st.info("Creating embeddings (this may take time)...")

        # create embedding model and detect dim
        emb = try_create_embedding()
        if emb is None:
            st.error("Embedding creation failed; cannot proceed.")
            return None

        st.session_state.embeddings = emb
        embedding_dim = st.session_state.embedding_dim

        # Load PDFs
        loader = PyPDFDirectoryLoader(data_dir)
        docs = loader.load()
        st.session_state.loader = loader
        st.session_state.docs = docs
        st.info(f"Loaded {len(docs)} documents. Splitting into chunks...")

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(docs)
        st.session_state.final_documents = final_documents
        st.info(f"Created {len(final_documents)} document-chunks for embedding.")

        # Ensure persist dir exists
        os.makedirs(persist_dir, exist_ok=True)

        st.info(f"Indexing documents into ObjectBox using embedding_dimensions={embedding_dim} (this may take a while)...")
        vectors = ObjectBox.from_documents(
            final_documents,
            emb,
            embedding_dimensions=embedding_dim,
            db_directory=persist_dir,
        )

        st.session_state.vectors = vectors
        st.session_state.embedded = True
        st.success("Embedding complete and ObjectBox vectorstore created.")

        return vectors

    except Exception as e:
        st.error(f"Embedding failed: {e}")
        st.write(traceback.format_exc())
        return None


# Try to auto-load persisted vectorstore at startup
if st.session_state.vectors is None:
    loaded_vs = try_load_persisted_vectorstore(OBJECTBOX_DIR)
    if loaded_vs is not None:
        st.session_state.vectors = loaded_vs
        st.session_state.embedded = True

# ---------- UI Layout ----------
col1, col2 = st.columns([1, 3])
with col1:
    if st.button("Embed Documents"):
        with st.spinner("Embedding documents into ObjectBox — please wait..."):
            embed_documents_and_create_vectorstore(DATA_DIR, OBJECTBOX_DIR)

    if st.button("Clear Vectors (in memory)"):
        st.session_state.vectors = None
        st.session_state.embedded = False
        st.success("Cleared in-memory vectors. To remove persisted DB delete the objectbox folder.")

    st.markdown("---")
    st.write("**Embedding status**")
    st.write(f"Embedded in session: `{st.session_state.embedded}`")
    if st.session_state.vectors is not None:
        st.write("Vectorstore: present (in-memory)")
    else:
        if os.path.exists(OBJECTBOX_DIR) and os.listdir(OBJECTBOX_DIR):
            st.write("ObjectBox persistent files found on disk. Click 'Embed Documents' to (re)create using the current embedding.")
        else:
            st.write("No persisted ObjectBox files found on disk.")

with col2:
    user_input = st.text_input("Enter your question from documents")

    if user_input:
        try:
            # build retriever with larger k to be permissive
            retriever = None
            try:
                if st.session_state.vectors is None:
                    # attempt to auto-load persisted store if available
                    maybe_loaded = try_load_persisted_vectorstore(OBJECTBOX_DIR)
                    if maybe_loaded is not None:
                        st.session_state.vectors = maybe_loaded
                        st.session_state.embedded = True

                if st.session_state.vectors is not None:
                    retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 8})
            except Exception as e:
                st.error(f"Failed to create retriever: {e}")
                st.write(traceback.format_exc())
                retriever = None

            if retriever is None:
                st.warning("No retriever available. Click 'Embed Documents' to index the PDFs first.")
            else:
                # 1) Test retriever directly: get raw relevant documents
                try:
                    raw_docs = retriever.get_relevant_documents(user_input)
                    st.write("**DEBUG — retriever returned document count:**", len(raw_docs))
                    if len(raw_docs) > 0:
                        with st.expander("DEBUG: Top retrieved raw chunks (verbatim)"):
                            for i, d in enumerate(raw_docs[:6]):
                                content = getattr(d, "page_content", None) or (d.get("page_content") if isinstance(d, dict) else str(d))
                                st.write(f"--- chunk {i+1} ({len(content)} chars) ---")
                                st.write(content[:1000])
                    else:
                        st.warning("Retriever returned 0 documents for this query. Try a verbatim sentence from a chunk to verify retrieval.")
                except Exception as e:
                    st.error(f"Retriever error: {e}")
                    st.write(traceback.format_exc())
                    raw_docs = []

                # 2) If raw documents exist, run the chain and dump full response
                if raw_docs:
                    try:
                        document_chain = create_stuff_documents_chain(utils.groq_llm(), prompt)
                        retrieval_chain = create_retrieval_chain(retriever, document_chain)

                        start = time.process_time()
                        response = retrieval_chain.invoke({"input": user_input})
                        elapsed = time.process_time() - start

                        st.write("### DEBUG — Full chain response (raw):")
                        st.write(response)  # print entire chain output so we can inspect keys

                        # Try several common response shapes
                        answer = None
                        if isinstance(response, dict):
                            answer = response.get("answer") or response.get("output_text") or response.get("result") or response.get("response")
                        else:
                            answer = str(response)

                        st.write("### Answer")
                        st.write(answer or "No 'answer' field in chain response")
                        st.write(f"response time: {elapsed:.2f} secs")

                        # Show chain-provided context if present
                        contexts = response.get("context") if isinstance(response, dict) else None
                        if contexts:
                            with st.expander("Document Similarity Search (retrieved context)"):
                                for i, doc in enumerate(contexts):
                                    st.write(doc.page_content if hasattr(doc, "page_content") else str(doc))
                        else:
                            st.info("DEBUG: No 'context' field found in chain response. Using raw retriever docs above for verification.")
                    except Exception as e:
                        st.error(f"Chain invocation failed: {e}")
                        st.write(traceback.format_exc())

        except Exception as outer_e:
            st.error(f"An unexpected error occurred while processing your query: {outer_e}")
            st.write(traceback.format_exc())

st.markdown("---")
st.write("Tips: If you have already embedded documents but the retriever fails, try re-running the 'Embed Documents' button to re-create the store in the current session.")
