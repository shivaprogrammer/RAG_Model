# recreate_objectbox.py
import os, sys, time
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)

from app import utils
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_objectbox.vectorstores import ObjectBox

PROJECT_ROOT = THIS
DATA_DIR = os.path.join(PROJECT_ROOT, "us-census-data")
PERSIST_DIR = os.path.join(PROJECT_ROOT, "objectbox")

def main():
    print("Data dir:", DATA_DIR)
    print("Persist dir:", PERSIST_DIR)
    os.makedirs(PERSIST_DIR, exist_ok=True)

    print("Creating embedding model via utils...")
    emb = utils.huggingface_instruct_embedding()
    print("Embedding object created:", type(emb))

    print("Loading PDFs...")
    loader = PyPDFDirectoryLoader(DATA_DIR)
    docs = loader.load()
    print("Loaded documents:", len(docs))

    print("Splitting documents...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print("Chunks:", len(chunks))

    embedding_dim = 384
    print("Using embedding_dimensions =", embedding_dim)

    print("Indexing into ObjectBox (this can take a while)...")
    start = time.time()
    vectors = ObjectBox.from_documents(chunks, emb, embedding_dimensions=embedding_dim, db_directory=PERSIST_DIR)
    elapsed = time.time() - start
    print("ObjectBox created in", elapsed, "seconds.")
    print("Done.")

if __name__ == "__main__":
    main()
