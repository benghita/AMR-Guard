"""ChromaDB vector store for unstructured document RAG."""

import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
from typing import Optional
import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from ..config import get_settings

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"


def get_chroma_client() -> chromadb.PersistentClient:
    """Get ChromaDB persistent client."""
    chroma_dir = get_settings().chroma_db_dir
    chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(chroma_dir))


def get_embedding_function():
    """Get the embedding function for ChromaDB."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from PDF file."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n\n"
    return text


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    """Split text into chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_text(text)


def generate_doc_id(text: str, index: int) -> str:
    """Generate a unique document ID."""
    hash_input = f"{text[:100]}_{index}"
    return hashlib.md5(hash_input.encode()).hexdigest()


def init_idsa_guidelines_collection() -> chromadb.Collection:
    """Initialize the IDSA treatment guidelines collection."""
    client = get_chroma_client()
    ef = get_embedding_function()

    # Delete existing collection if exists
    try:
        client.delete_collection("idsa_treatment_guidelines")
    except Exception:
        pass

    collection = client.create_collection(
        name="idsa_treatment_guidelines",
        embedding_function=ef,
        metadata={
            "source": "IDSA 2024 Guidance",
            "doi": "10.1093/cid/ciae403",
            "description": "Antimicrobial-Resistant Gram-Negative Infections Treatment Guidelines"
        }
    )

    return collection


def init_mic_reference_collection() -> chromadb.Collection:
    """Initialize the MIC reference documentation collection."""
    client = get_chroma_client()
    ef = get_embedding_function()

    # Delete existing collection if exists
    try:
        client.delete_collection("mic_reference_docs")
    except Exception:
        pass

    collection = client.create_collection(
        name="mic_reference_docs",
        embedding_function=ef,
        metadata={
            "source": "EUCAST Breakpoint Tables",
            "version": "16.0",
            "description": "MIC Breakpoint Reference Documentation"
        }
    )

    return collection


def classify_chunk_pathogen(text: str) -> str:
    """Classify which pathogen type a chunk relates to."""
    text_lower = text.lower()

    pathogen_keywords = {
        "ESBL-E": ["esbl", "extended-spectrum beta-lactamase", "esbl-e", "esbl-producing"],
        "CRE": ["carbapenem-resistant enterobacterales", "cre", "carbapenemase"],
        "CRAB": ["acinetobacter baumannii", "crab", "carbapenem-resistant acinetobacter"],
        "DTR-PA": ["pseudomonas aeruginosa", "dtr-p", "difficult-to-treat resistance"],
        "S.maltophilia": ["stenotrophomonas maltophilia", "s. maltophilia"],
        "AmpC-E": ["ampc", "ampc-e", "ampc-producing"],
    }

    for pathogen, keywords in pathogen_keywords.items():
        for keyword in keywords:
            if keyword in text_lower:
                return pathogen

    return "General"


def import_idsa_guidelines() -> int:
    """Import IDSA guidelines PDF into ChromaDB."""
    print("Importing IDSA guidelines into ChromaDB...")

    pdf_path = DOCS_DIR / "antibiotic_guidelines" / "ciae403.pdf"

    if not pdf_path.exists():
        print(f"  Warning: {pdf_path} not found, skipping...")
        return 0

    # Extract text from PDF
    print("  Extracting text from PDF...")
    text = extract_pdf_text(pdf_path)

    # Chunk the text
    print("  Chunking text...")
    chunks = chunk_text(text)

    # Initialize collection
    collection = init_idsa_guidelines_collection()

    # Prepare documents for insertion
    documents = []
    metadatas = []
    ids = []

    for i, chunk in enumerate(chunks):
        documents.append(chunk)
        metadatas.append({
            "source": "ciae403.pdf",
            "chunk_index": i,
            "pathogen_type": classify_chunk_pathogen(chunk),
            "page_estimate": i // 3  # Rough estimate
        })
        ids.append(generate_doc_id(chunk, i))

    # Add to collection
    print(f"  Adding {len(documents)} chunks to collection...")
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    print(f"  Imported {len(documents)} chunks from IDSA guidelines")
    return len(documents)


def import_mic_reference() -> int:
    """Import MIC breakpoint PDF into ChromaDB."""
    print("Importing MIC reference PDF into ChromaDB...")

    pdf_path = DOCS_DIR / "mic_breakpoints" / "v_16.0_Breakpoint_Tables.pdf"

    if not pdf_path.exists():
        print(f"  Warning: {pdf_path} not found, skipping...")
        return 0

    # Extract text from PDF
    print("  Extracting text from PDF...")
    text = extract_pdf_text(pdf_path)

    # Chunk the text
    print("  Chunking text...")
    chunks = chunk_text(text, chunk_size=800, chunk_overlap=150)

    # Initialize collection
    collection = init_mic_reference_collection()

    # Prepare documents for insertion
    documents = []
    metadatas = []
    ids = []

    for i, chunk in enumerate(chunks):
        documents.append(chunk)
        metadatas.append({
            "source": "v_16.0_Breakpoint_Tables.pdf",
            "chunk_index": i,
            "document_type": "mic_reference"
        })
        ids.append(generate_doc_id(chunk, i))

    # Add to collection
    print(f"  Adding {len(documents)} chunks to collection...")
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    print(f"  Imported {len(documents)} chunks from MIC reference")
    return len(documents)


def get_collection(name: str) -> Optional[chromadb.Collection]:
    """Get a collection by name."""
    client = get_chroma_client()
    ef = get_embedding_function()

    try:
        return client.get_collection(name=name, embedding_function=ef)
    except Exception:
        return None


def search_guidelines(
    query: str,
    n_results: int = 5,
    pathogen_filter: str = None
) -> list[dict]:
    """Search the IDSA guidelines collection."""
    collection = get_collection("idsa_treatment_guidelines")

    if collection is None:
        return []

    where_filter = None
    if pathogen_filter:
        where_filter = {"pathogen_type": pathogen_filter}

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )

    # Format results
    formatted = []
    for i in range(len(results['documents'][0])):
        formatted.append({
            "content": results['documents'][0][i],
            "metadata": results['metadatas'][0][i],
            "distance": results['distances'][0][i]
        })

    return formatted


def search_mic_reference(query: str, n_results: int = 3) -> list[dict]:
    """Search the MIC reference collection."""
    collection = get_collection("mic_reference_docs")

    if collection is None:
        return []

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    # Format results
    formatted = []
    for i in range(len(results['documents'][0])):
        formatted.append({
            "content": results['documents'][0][i],
            "metadata": results['metadatas'][0][i],
            "distance": results['distances'][0][i]
        })

    return formatted


def import_all_vectors() -> dict:
    """Import all PDFs into ChromaDB."""
    print(f"\n{'='*50}")
    print("ChromaDB Vector Import")
    print(f"{'='*50}\n")

    results = {
        "idsa_guidelines": import_idsa_guidelines(),
        "mic_reference": import_mic_reference(),
    }

    print(f"\n{'='*50}")
    print("Vector Import Summary:")
    for collection, count in results.items():
        print(f"  {collection}: {count} chunks")
    print(f"{'='*50}\n")

    return results


if __name__ == "__main__":
    import_all_vectors()
