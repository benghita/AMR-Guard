"""
RAG module for AMR-Guard.

Retrieves context from four ChromaDB collections:
- idsa_treatment_guidelines: IDSA 2024 AMR guidance
- mic_reference_docs: EUCAST v16.0 breakpoint tables
- drug_safety: Drug interactions and contraindications
- pathogen_resistance: ATLAS regional susceptibility data
"""

import logging
from typing import Any, Dict, List, Optional

from .config import get_settings

logger = logging.getLogger(__name__)

# Module-level singletons; initialized lazily to avoid import-time side effects
_chroma_client = None
_embedding_function = None


def get_chroma_client():
    """Return the ChromaDB persistent client, creating it on first call."""
    global _chroma_client
    if _chroma_client is None:
        import chromadb
        chroma_path = get_settings().chroma_db_dir
        chroma_path.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(chroma_path))
    return _chroma_client


def get_embedding_function():
    """Return the SentenceTransformer embedding function, creating it on first call."""
    global _embedding_function
    if _embedding_function is None:
        from chromadb.utils import embedding_functions
        # Use only the model short name (not the full HuggingFace path)
        model_short_name = get_settings().embedding_model_name.split("/")[-1]
        _embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_short_name
        )
    return _embedding_function


def get_collection(name: str):
    """Return a ChromaDB collection by name, or None if it does not exist."""
    try:
        return get_chroma_client().get_collection(name=name, embedding_function=get_embedding_function())
    except Exception:
        logger.warning(f"Collection '{name}' not found")
        return None


def search_antibiotic_guidelines(
    query: str,
    n_results: int = 5,
    pathogen_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Search the IDSA treatment guidelines collection."""
    collection = get_collection("idsa_treatment_guidelines")
    if collection is None:
        return []
    try:
        where = {"pathogen_type": pathogen_filter} if pathogen_filter else None
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return _format_results(results)
    except Exception as e:
        logger.error(f"Error querying guidelines: {e}")
        return []


def search_mic_breakpoints(
    query: str,
    n_results: int = 5,
    organism: Optional[str] = None,
    antibiotic: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Search the EUCAST MIC breakpoint reference collection."""
    collection = get_collection("mic_reference_docs")
    if collection is None:
        return []
    # Prepend organism/antibiotic to query to narrow semantic search
    enhanced_query = " ".join(filter(None, [organism, antibiotic, query]))
    try:
        results = collection.query(
            query_texts=[enhanced_query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        return _format_results(results)
    except Exception as e:
        logger.error(f"Error querying breakpoints: {e}")
        return []


def search_drug_safety(
    query: str,
    n_results: int = 5,
    drug_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Search drug interactions from SQLite (drug_interactions table)."""
    if not drug_name:
        return []
    try:
        from .db.database import execute_query

        rows = execute_query(
            """SELECT drug_1, drug_2, interaction_description, severity
               FROM drug_interactions
               WHERE LOWER(drug_1) LIKE ? OR LOWER(drug_2) LIKE ?
               LIMIT ?""",
            (f"%{drug_name.lower()}%", f"%{drug_name.lower()}%", n_results),
        )
        return [
            {
                "content": (
                    f"{r['drug_1']} + {r['drug_2']}: {r['interaction_description']}"
                ),
                "metadata": {"severity": r.get("severity", "unknown")},
                "distance": None,
                "source": "drug_interactions (SQLite)",
                "relevance_score": 1.0,
            }
            for r in rows
        ]
    except Exception as e:
        logger.error(f"Error querying drug safety: {e}")
        return []


def search_resistance_patterns(
    query: str,
    n_results: int = 5,
    organism: Optional[str] = None,
    region: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Search the ATLAS pathogen resistance collection."""
    collection = get_collection("pathogen_resistance")
    if collection is None:
        return []
    enhanced_query = " ".join(filter(None, [region, organism, query]))
    try:
        results = collection.query(
            query_texts=[enhanced_query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        return _format_results(results)
    except Exception as e:
        logger.error(f"Error querying resistance patterns: {e}")
        return []


def get_context_for_agent(
    agent_name: str,
    query: str,
    patient_context: Optional[Dict[str, Any]] = None,
    n_results: int = 3,
) -> str:
    """
    Return a formatted context string for a specific agent.

    Each agent draws from the collections most relevant to its task:
    - intake_historian: IDSA guidelines
    - vision_specialist: MIC breakpoints
    - trend_analyst: MIC breakpoints + resistance patterns
    - clinical_pharmacologist: guidelines + drug safety
    """
    ctx = patient_context or {}
    parts = []

    if agent_name == "intake_historian":
        guidelines = search_antibiotic_guidelines(query, n_results=n_results, pathogen_filter=ctx.get("pathogen_type"))
        if guidelines:
            parts.append("RELEVANT TREATMENT GUIDELINES:")
            for g in guidelines:
                parts.append(f"- {g['content'][:500]}...")
                parts.append(f"  [Source: {g.get('source', 'IDSA Guidelines')}]")

    elif agent_name == "vision_specialist":
        breakpoints = search_mic_breakpoints(query, n_results=n_results, organism=ctx.get("organism"), antibiotic=ctx.get("antibiotic"))
        if breakpoints:
            parts.append("RELEVANT BREAKPOINT INFORMATION:")
            for b in breakpoints:
                parts.append(f"- {b['content'][:400]}...")

    elif agent_name == "trend_analyst":
        breakpoints = search_mic_breakpoints(
            f"breakpoint {ctx.get('organism', '')} {ctx.get('antibiotic', '')}",
            n_results=n_results,
        )
        resistance = search_resistance_patterns(query, n_results=n_results, organism=ctx.get("organism"), region=ctx.get("region"))
        if breakpoints:
            parts.append("EUCAST BREAKPOINT DATA:")
            for b in breakpoints:
                parts.append(f"- {b['content'][:400]}...")
        if resistance:
            parts.append("\nRESISTANCE PATTERN DATA:")
            for r in resistance:
                parts.append(f"- {r['content'][:400]}...")

    elif agent_name == "clinical_pharmacologist":
        guidelines = search_antibiotic_guidelines(query, n_results=n_results)
        safety = search_drug_safety(query, n_results=n_results, drug_name=ctx.get("proposed_antibiotic"))
        if guidelines:
            parts.append("TREATMENT GUIDELINES:")
            for g in guidelines:
                parts.append(f"- {g['content'][:400]}...")
        if safety:
            parts.append("\nDRUG SAFETY INFORMATION:")
            for s in safety:
                parts.append(f"- {s['content'][:400]}...")

    else:
        guidelines = search_antibiotic_guidelines(query, n_results=n_results)
        for g in guidelines:
            parts.append(f"- {g['content'][:500]}...")

    return "\n".join(parts) if parts else "No relevant context found in knowledge base."


def _format_results(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten ChromaDB query results into a list of dicts."""
    if not results or not results.get("documents"):
        return []

    documents = results["documents"][0] if results["documents"] else []
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    return [
        {
            "content": doc,
            "metadata": metadatas[i] if i < len(metadatas) else {},
            "distance": distances[i] if i < len(distances) else None,
            "source": metadatas[i].get("source", "Unknown") if i < len(metadatas) else "Unknown",
            "relevance_score": 1 - (distances[i] if i < len(distances) else 0),
        }
        for i, doc in enumerate(documents)
    ]


def list_available_collections() -> List[str]:
    """Return names of all ChromaDB collections that exist."""
    try:
        return [c.name for c in get_chroma_client().list_collections()]
    except Exception as e:
        logger.error(f"Error listing collections: {e}")
        return []


def get_collection_info(name: str) -> Optional[Dict[str, Any]]:
    """Return count and metadata for a collection, or None if it does not exist."""
    collection = get_collection(name)
    if collection is None:
        return None
    try:
        return {"name": collection.name, "count": collection.count(), "metadata": collection.metadata}
    except Exception as e:
        logger.error(f"Error getting collection info: {e}")
        return None
