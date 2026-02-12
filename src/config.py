
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field


# Load variables from a local .env if present (handy for local dev)
load_dotenv()


class Settings(BaseModel):
    """
    Central configuration object for Med-I-C.

    Values are read from environment variables where possible so that
    the same code can run locally, on Kaggle, and in production.
    """

    # ------------------------------------------------------------------
    # General environment
    # ------------------------------------------------------------------
    environment: Literal["local", "kaggle", "production"] = Field(
        default_factory=lambda: os.getenv("MEDIC_ENV", "local")
    )

    project_root: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[1]
    )

    data_dir: Path = Field(
        default_factory=lambda: Path(
            os.getenv("MEDIC_DATA_DIR", "data")
        )
    )

    chroma_db_dir: Path = Field(
        default_factory=lambda: Path(
            os.getenv("MEDIC_CHROMA_DB_DIR", "data/chroma_db")
        )
    )

    # ------------------------------------------------------------------
    # Model + deployment preferences
    # ------------------------------------------------------------------
    default_backend: Literal["vertex", "local"] = Field(
        default_factory=lambda: os.getenv("MEDIC_DEFAULT_BACKEND", "vertex")  # type: ignore[arg-type]
    )

    # Quantization mode for local models
    quantization: Literal["none", "4bit"] = Field(
        default_factory=lambda: os.getenv("MEDIC_QUANTIZATION", "4bit")  # type: ignore[arg-type]
    )

    # Embedding model used for ChromaDB / RAG
    embedding_model_name: str = Field(
        default_factory=lambda: os.getenv(
            "MEDIC_EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
    )

    # ------------------------------------------------------------------
    # Vertex AI configuration (MedGemma / TxGemma hosted on Vertex)
    # ------------------------------------------------------------------
    use_vertex: bool = Field(
        default_factory=lambda: os.getenv("MEDIC_USE_VERTEX", "true").lower()
        in {"1", "true", "yes"}
    )

    vertex_project_id: Optional[str] = Field(
        default_factory=lambda: os.getenv("MEDIC_VERTEX_PROJECT_ID")
    )
    vertex_location: str = Field(
        default_factory=lambda: os.getenv("MEDIC_VERTEX_LOCATION", "us-central1")
    )

    # Model IDs as expected by Vertex / langchain-google-vertexai
    vertex_medgemma_4b_model: str = Field(
        default_factory=lambda: os.getenv(
            "MEDIC_VERTEX_MEDGEMMA_4B_MODEL",
            "med-gemma-4b-it",
        )
    )
    vertex_medgemma_27b_model: str = Field(
        default_factory=lambda: os.getenv(
            "MEDIC_VERTEX_MEDGEMMA_27B_MODEL",
            "med-gemma-27b-text-it",
        )
    )
    vertex_txgemma_9b_model: str = Field(
        default_factory=lambda: os.getenv(
            "MEDIC_VERTEX_TXGEMMA_9B_MODEL",
            "tx-gemma-9b",
        )
    )
    vertex_txgemma_2b_model: str = Field(
        default_factory=lambda: os.getenv(
            "MEDIC_VERTEX_TXGEMMA_2B_MODEL",
            "tx-gemma-2b",
        )
    )

    # Standard GOOGLE_APPLICATION_CREDENTIALS path, if needed
    google_application_credentials: Optional[Path] = Field(
        default_factory=lambda: (
            Path(os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
            if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ
            else None
        )
    )

    # ------------------------------------------------------------------
    # Local model paths (for offline / Kaggle GPU usage)
    # ------------------------------------------------------------------
    local_medgemma_4b_model: Optional[str] = Field(
        default_factory=lambda: os.getenv("MEDIC_LOCAL_MEDGEMMA_4B_MODEL")
    )
    local_medgemma_27b_model: Optional[str] = Field(
        default_factory=lambda: os.getenv("MEDIC_LOCAL_MEDGEMMA_27B_MODEL")
    )
    local_txgemma_9b_model: Optional[str] = Field(
        default_factory=lambda: os.getenv("MEDIC_LOCAL_TXGEMMA_9B_MODEL")
    )
    local_txgemma_2b_model: Optional[str] = Field(
        default_factory=lambda: os.getenv("MEDIC_LOCAL_TXGEMMA_2B_MODEL")
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a cached Settings instance.

    Use this helper everywhere instead of instantiating Settings directly:

        from src.config import get_settings
        settings = get_settings()
    """

    return Settings()


__all__ = ["Settings", "get_settings"]

