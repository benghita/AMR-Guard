"""Database modules for AMR-Guard."""

from .database import (
    init_database,
    get_connection,
    execute_query,
    execute_insert,
    execute_many,
    DB_PATH,
    DATA_DIR,
    DOCS_DIR,
)

from .vector_store import (
    get_chroma_client,
    search_guidelines,
    search_mic_reference,
    import_all_vectors,
)

__all__ = [
    "init_database",
    "get_connection",
    "execute_query",
    "execute_insert",
    "execute_many",
    "DB_PATH",
    "DATA_DIR",
    "DOCS_DIR",
    "get_chroma_client",
    "search_guidelines",
    "search_mic_reference",
    "import_all_vectors",
]
