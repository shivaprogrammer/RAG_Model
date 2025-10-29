# check_objectbox_query.py
import os, sys, traceback, math
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)

from app import utils
from langchain_objectbox.vectorstores import ObjectBox

OBJ_DIR = os.path.join(THIS, "objectbox")
print("ObjectBox dir:", OBJ_DIR)
try:
    # instantiate with embedding dim — matches recreate step
    emb = utils.huggingface_instruct_embedding()
    store = ObjectBox(embedding=emb, embedding_dimensions=384, db_directory=OBJ_DIR)
    print("ObjectBox instance created.")
    retriever = store.as_retriever(search_kwargs={"k": 8})
    docs = retriever.get_relevant_documents("What is the uninsured rate by state in 2022?")
    print("Retriever returned:", len(docs))
    if docs:
        print("Top doc snippet:", docs[0].page_content[:400])
except Exception as e:
    print("Error:", e)
    traceback.print_exc()
