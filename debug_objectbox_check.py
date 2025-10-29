# debug_objectbox_check.py
import os, sys, math, traceback
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)

print("PWD:", THIS)
print("Python:", sys.executable)

# 1) Show objectbox files
obj_dir = os.path.join(THIS, "objectbox")
print("\nobjectbox dir:", obj_dir)
if os.path.exists(obj_dir):
    for f in os.listdir(obj_dir):
        print("  ", f, "-", os.path.getsize(os.path.join(obj_dir, f)))
else:
    print("  objectbox folder does not exist")

# 2) Load a few chunks (same loader used by app)
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
data_dir = os.path.join(THIS, "us-census-data")
print("\nData dir:", data_dir)
loader = PyPDFDirectoryLoader(data_dir)
docs = loader.load()
print("Loaded docs:", len(docs))
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)
print("Created chunks:", len(chunks))
first_chunk = next((c.page_content for c in chunks if (c.page_content or "").strip()), None)
print("Example chunk length:", len(first_chunk) if first_chunk else "no chunk")

# 3) Import your repo embedding and confirm vector dim
emb = None
try:
    # try import the utils used in app
    from app import utils
    print("\nImported app.utils successfully")
    emb = utils.huggingface_instruct_embedding()
    print("Embeddings object type:", type(emb))
    # make a small embedding to detect vector length and whether encode function exists
    sample = (first_chunk[:800] if first_chunk else "hello world")
    v = emb.embed_documents([sample])
    print("Produced embedding: list len", len(v), "vector dim", len(v[0]))
except Exception as e:
    print("\nFailed to import app.utils or create embedding:")
    traceback.print_exc()
    # fallback: try to import utils from app folder directly
    try:
        sys.path.insert(0, os.path.join(THIS, "app"))
        import utils as repo_utils
        emb = repo_utils.huggingface_instruct_embedding()
        print("Fallback loaded emb:", type(emb))
        v = emb.embed_documents([sample])
        print("Fallback embedding dim:", len(v[0]))
    except Exception as e2:
        print("Fallback also failed:")
        traceback.print_exc()

# 4) Try to load the persisted ObjectBox via langchain_objectbox
try:
    from langchain_objectbox.vectorstores import ObjectBox
    print("\nAttempting to load ObjectBox via ObjectBox.load()")
    try:
        store = ObjectBox.load(db_directory=obj_dir)
        print("Loaded ObjectBox via ObjectBox.load()")
    except Exception as e:
        print("ObjectBox.load() failed:", e)
        # try constructing ObjectBox with embedding object
        try:
            store = ObjectBox.from_documents([], embedding=emb, db_directory=obj_dir)
            print("Created ObjectBox instance via from_documents() (empty docs) to inspect")
        except Exception as e2:
            print("Failed to instantiate ObjectBox:", e2)
            raise
    # Print a few store attributes if present
    try:
        print("Store type:", type(store))
        # create retriever and attempt query
        retriever = store.as_retriever(search_kwargs={"k": 8})
        print("Retriever created.")
        query = "What is the uninsured rate by state in 2022?"
        print("Querying retriever for:", query)
        docs_ret = retriever.get_relevant_documents(query)
        print("Retriever returned count:", len(docs_ret))
        if len(docs_ret) > 0:
            print("--- top doc snippet ---")
            print(docs_ret[0].page_content[:800])
    except Exception as e:
        print("Error when using store/retriever:", e)
        traceback.print_exc()
except Exception as e:
    print("Error loading/inspecting ObjectBox:", e)
    traceback.print_exc()

print("\nDEBUG CHECK COMPLETE")
