# ingest.py
import os
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader

PERSIST_DIR = "db"

def load_documents(folder_path):
    docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join(folder_path, filename))
        elif filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, filename))
        else:
            continue
        docs.extend(loader.load())
    return docs

def ingest_documents():
    folder_path = os.path.join("data", "sample_docs")
    documents = load_documents(folder_path)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(documents, embeddings, persist_directory=PERSIST_DIR)
    vectordb.persist()
    print("✅ Documents ingested and persisted to ChromaDB.")

if __name__ == "__main__":
    ingest_documents()
