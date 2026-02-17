"""
RAG (Retrieval Augmented Generation) module for Med-I-C.

Provides unified retrieval across multiple knowledge collections:
- antibiotic_guidelines: WHO/IDSA treatment guidelines
- mic_breakpoints: EUCAST/CLSI breakpoint tables
- drug_safety: Drug interactions, warnings, contraindications
- pathogen_resistance: Regional resistance patterns
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import get_settings

logger = logging.getLogger(__name__)


# =============================================================================
# CHROMA CLIENT & EMBEDDING SETUP
# =============================================================================

_chroma_client = None
_embedding_function = None


def get_chroma_client():
    """Get or create ChromaDB persistent client."""
    global _chroma_client
    if _chroma_client is None:
        import chromadb

        settings = get_settings()
        chroma_path = settings.chroma_db_dir
        chroma_path.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(chroma_path))
    return _chroma_client


def get_embedding_function():
    """Get or create the embedding function."""
    global _embedding_function
    if _embedding_function is None:
        from chromadb.utils import embedding_functions

        settings = get_settings()
        _embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model_name.split("/")[-1]
        )
    return _embedding_function


def get_collection(name: str):
    """
    Get a ChromaDB collection by name.

    Returns None if collection doesn't exist.
    """
    client = get_chroma_client()
    ef = get_embedding_function()

    try:
        return client.get_collection(name=name, embedding_function=ef)
    except Exception:
        logger.warning(f"Collection '{name}' not found")
        return None


# =============================================================================
# COLLECTION-SPECIFIC RETRIEVERS
# =============================================================================

def search_antibiotic_guidelines(
    query: str,
    n_results: int = 5,
    pathogen_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Search antibiotic treatment guidelines.

    Args:
        query: Search query
        n_results: Number of results to return
        pathogen_filter: Optional pathogen type filter (e.g., "ESBL-E", "CRE")

    Returns:
        List of relevant guideline excerpts with metadata
    """
    collection = get_collection("idsa_treatment_guidelines")
    if collection is None:
        logger.warning("idsa_treatment_guidelines collection not available")
        return []

    where_filter = None
    if pathogen_filter:
        where_filter = {"pathogen_type": pathogen_filter}

    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        logger.error(f"Error querying guidelines: {e}")
        return []

    return _format_results(results)


def search_mic_breakpoints(
    query: str,
    n_results: int = 5,
    organism: Optional[str] = None,
    antibiotic: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Search MIC breakpoint reference documentation.

    Args:
        query: Search query
        n_results: Number of results
        organism: Optional organism name filter
        antibiotic: Optional antibiotic name filter

    Returns:
        List of relevant breakpoint information
    """
    collection = get_collection("mic_reference_docs")
    if collection is None:
        logger.warning("mic_reference_docs collection not available")
        return []

    # Build query with organism/antibiotic context if provided
    enhanced_query = query
    if organism:
        enhanced_query = f"{organism} {enhanced_query}"
    if antibiotic:
        enhanced_query = f"{antibiotic} {enhanced_query}"

    try:
        results = collection.query(
            query_texts=[enhanced_query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        logger.error(f"Error querying breakpoints: {e}")
        return []

    return _format_results(results)


def search_drug_safety(
    query: str,
    n_results: int = 5,
    drug_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Search drug safety information (interactions, warnings, contraindications).

    Args:
        query: Search query
        n_results: Number of results
        drug_name: Optional drug name to focus search

    Returns:
        List of relevant safety information
    """
    collection = get_collection("drug_safety")
    if collection is None:
        # Fallback: try existing collections
        logger.warning("drug_safety collection not available")
        return []

    enhanced_query = f"{drug_name} {query}" if drug_name else query

    try:
        results = collection.query(
            query_texts=[enhanced_query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        logger.error(f"Error querying drug safety: {e}")
        return []

    return _format_results(results)


def search_resistance_patterns(
    query: str,
    n_results: int = 5,
    organism: Optional[str] = None,
    region: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Search pathogen resistance pattern data.

    Args:
        query: Search query
        n_results: Number of results
        organism: Optional organism filter
        region: Optional geographic region filter

    Returns:
        List of relevant resistance data
    """
    collection = get_collection("pathogen_resistance")
    if collection is None:
        logger.warning("pathogen_resistance collection not available")
        return []

    enhanced_query = query
    if organism:
        enhanced_query = f"{organism} {enhanced_query}"
    if region:
        enhanced_query = f"{region} {enhanced_query}"

    try:
        results = collection.query(
            query_texts=[enhanced_query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        logger.error(f"Error querying resistance patterns: {e}")
        return []

    return _format_results(results)


# =============================================================================
# UNIFIED CONTEXT RETRIEVER
# =============================================================================

def get_context_for_agent(
    agent_name: str,
    query: str,
    patient_context: Optional[Dict[str, Any]] = None,
    n_results: int = 3,
) -> str:
    """
    Get formatted RAG context string for a specific agent.

    This is the main entry point for agents to retrieve context.

    Args:
        agent_name: Name of the requesting agent
        query: The primary search query
        patient_context: Optional dict with patient-specific info
        n_results: Number of results per collection

    Returns:
        Formatted context string for injection into prompts
    """
    context_parts = []
    patient_context = patient_context or {}

    if agent_name == "intake_historian":
        # Get empirical therapy guidelines
        guidelines = search_antibiotic_guidelines(
            query=query,
            n_results=n_results,
            pathogen_filter=patient_context.get("pathogen_type"),
        )
        if guidelines:
            context_parts.append("RELEVANT TREATMENT GUIDELINES:")
            for g in guidelines:
                context_parts.append(f"- {g['content'][:500]}...")
                context_parts.append(f"  [Source: {g.get('source', 'IDSA Guidelines')}]")

    elif agent_name == "vision_specialist":
        # Get MIC reference info for lab interpretation
        breakpoints = search_mic_breakpoints(
            query=query,
            n_results=n_results,
            organism=patient_context.get("organism"),
            antibiotic=patient_context.get("antibiotic"),
        )
        if breakpoints:
            context_parts.append("RELEVANT BREAKPOINT INFORMATION:")
            for b in breakpoints:
                context_parts.append(f"- {b['content'][:400]}...")

    elif agent_name == "trend_analyst":
        # Get breakpoints and resistance trends
        breakpoints = search_mic_breakpoints(
            query=f"breakpoint {patient_context.get('organism', '')} {patient_context.get('antibiotic', '')}",
            n_results=n_results,
        )
        resistance = search_resistance_patterns(
            query=query,
            n_results=n_results,
            organism=patient_context.get("organism"),
            region=patient_context.get("region"),
        )

        if breakpoints:
            context_parts.append("EUCAST BREAKPOINT DATA:")
            for b in breakpoints:
                context_parts.append(f"- {b['content'][:400]}...")

        if resistance:
            context_parts.append("\nRESISTANCE PATTERN DATA:")
            for r in resistance:
                context_parts.append(f"- {r['content'][:400]}...")

    elif agent_name == "clinical_pharmacologist":
        # Get comprehensive context for final recommendation
        guidelines = search_antibiotic_guidelines(
            query=query,
            n_results=n_results,
        )
        safety = search_drug_safety(
            query=query,
            n_results=n_results,
            drug_name=patient_context.get("proposed_antibiotic"),
        )

        if guidelines:
            context_parts.append("TREATMENT GUIDELINES:")
            for g in guidelines:
                context_parts.append(f"- {g['content'][:400]}...")

        if safety:
            context_parts.append("\nDRUG SAFETY INFORMATION:")
            for s in safety:
                context_parts.append(f"- {s['content'][:400]}...")

    else:
        # Generic retrieval
        guidelines = search_antibiotic_guidelines(query, n_results=n_results)
        if guidelines:
            for g in guidelines:
                context_parts.append(f"- {g['content'][:500]}...")

    if not context_parts:
        return "No relevant context found in knowledge base."

    return "\n".join(context_parts)


def get_context_string(
    query: str,
    collections: Optional[List[str]] = None,
    n_results_per_collection: int = 3,
    **filters,
) -> str:
    """
    Get a combined context string from multiple collections.

    This is a simpler interface for general-purpose RAG retrieval.

    Args:
        query: Search query
        collections: List of collection names to search (defaults to all)
        n_results_per_collection: Results per collection
        **filters: Additional filters (organism, antibiotic, region, etc.)

    Returns:
        Combined context string
    """
    default_collections = [
        "idsa_treatment_guidelines",
        "mic_reference_docs",
    ]
    collections = collections or default_collections

    context_parts = []

    for collection_name in collections:
        if collection_name == "idsa_treatment_guidelines":
            results = search_antibiotic_guidelines(
                query,
                n_results=n_results_per_collection,
                pathogen_filter=filters.get("pathogen_type"),
            )
        elif collection_name == "mic_reference_docs":
            results = search_mic_breakpoints(
                query,
                n_results=n_results_per_collection,
                organism=filters.get("organism"),
                antibiotic=filters.get("antibiotic"),
            )
        elif collection_name == "drug_safety":
            results = search_drug_safety(
                query,
                n_results=n_results_per_collection,
                drug_name=filters.get("drug_name"),
            )
        elif collection_name == "pathogen_resistance":
            results = search_resistance_patterns(
                query,
                n_results=n_results_per_collection,
                organism=filters.get("organism"),
                region=filters.get("region"),
            )
        else:
            continue

        if results:
            context_parts.append(f"=== {collection_name.upper()} ===")
            for r in results:
                context_parts.append(r["content"])
                context_parts.append(f"[Relevance: {1 - r.get('distance', 0):.2f}]")
                context_parts.append("")

    return "\n".join(context_parts) if context_parts else "No relevant context found."


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _format_results(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Format ChromaDB query results into a standard format."""
    if not results or not results.get("documents"):
        return []

    formatted = []
    documents = results["documents"][0] if results["documents"] else []
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for i, doc in enumerate(documents):
        formatted.append({
            "content": doc,
            "metadata": metadatas[i] if i < len(metadatas) else {},
            "distance": distances[i] if i < len(distances) else None,
            "source": metadatas[i].get("source", "Unknown") if i < len(metadatas) else "Unknown",
            "relevance_score": 1 - (distances[i] if i < len(distances) else 0),
        })

    return formatted


def list_available_collections() -> List[str]:
    """List all available ChromaDB collections."""
    client = get_chroma_client()
    try:
        collections = client.list_collections()
        return [c.name for c in collections]
    except Exception as e:
        logger.error(f"Error listing collections: {e}")
        return []


def get_collection_info(name: str) -> Optional[Dict[str, Any]]:
    """Get information about a specific collection."""
    collection = get_collection(name)
    if collection is None:
        return None

    try:
        return {
            "name": collection.name,
            "count": collection.count(),
            "metadata": collection.metadata,
        }
    except Exception as e:
        logger.error(f"Error getting collection info: {e}")
        return None


__all__ = [
    "get_chroma_client",
    "get_embedding_function",
    "get_collection",
    "search_antibiotic_guidelines",
    "search_mic_breakpoints",
    "search_drug_safety",
    "search_resistance_patterns",
    "get_context_for_agent",
    "get_context_string",
    "list_available_collections",
    "get_collection_info",
]
