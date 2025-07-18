# ingest.py
from sentence_transformers import SentenceTransformer
import os
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader, PyPDFLoader

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
    documents = load_documents("data/sample_docs")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(documents, embeddings, persist_directory=PERSIST_DIR)
    vectordb.persist()
    print("✅ Documents ingested and persisted to ChromaDB.")

if __name__ == "__main__":
    ingest_documents()
