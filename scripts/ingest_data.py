import os
import chromadb
from chromadb.utils import embedding_functions
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Setup Chroma Persistence
CHROMA_PATH = "data/chroma_db"
DATA_PATH = "data/Med-I-C/raw"

def ingest_medical_data():
    # Persistent client for the competition (Kaggle/Local)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    # Using the embedding model you specified
    model_name = "all-MiniLM-L6-v2"
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)

    # 2. Ingest Guidelines (PDFs)
    # We create a specific collection for cleaner retrieval
    guideline_col = client.get_or_create_collection(name="antibiotic_guidelines", embedding_function=ef)
    
    loader = DirectoryLoader(f"{DATA_PATH}/guidelines", glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    
    # 1000/100 split as discussed for clinical coherence
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    # Adding to Chroma
    guideline_col.add(
        ids=[f"guideline_{i}" for i in range(len(chunks))],
        documents=[c.page_content for c in chunks],
        metadatas=[c.metadata for c in chunks]
    )
    
    print(f"Successfully ingested {len(chunks)} guideline chunks.")

if __name__ == "__main__":
    ingest_medical_data()