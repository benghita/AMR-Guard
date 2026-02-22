
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Explicitly load .env from the project root so it works regardless of CWD
# (e.g. when imported from a Kaggle notebook whose CWD is /kaggle/working/)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_PROJECT_ROOT / ".env")


class Settings(BaseModel):
    """
    All configuration for AMR-Guard, read from environment variables.

    Supports three deployment targets via MEDIC_ENV: local, kaggle, production.
    """

    environment: Literal["local", "kaggle", "production"] = Field(
        default_factory=lambda: os.getenv("MEDIC_ENV", "local")
    )
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[1]
    )
    data_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("MEDIC_DATA_DIR", "data"))
    )
    chroma_db_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("MEDIC_CHROMA_DB_DIR", "data/chroma_db"))
    )

    # 4-bit quantization via bitsandbytes
    quantization: Literal["none", "4bit"] = Field(
        default_factory=lambda: os.getenv("MEDIC_QUANTIZATION", "4bit")  # type: ignore[arg-type]
    )
    embedding_model_name: str = Field(
        default_factory=lambda: os.getenv("MEDIC_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )

    # Local HuggingFace model paths
    medgemma_4b_model: Optional[str] = Field(
        default_factory=lambda: os.getenv("MEDIC_LOCAL_MEDGEMMA_4B_MODEL")
    )
    medgemma_27b_model: Optional[str] = Field(
        default_factory=lambda: os.getenv("MEDIC_LOCAL_MEDGEMMA_27B_MODEL")
    )
    txgemma_9b_model: Optional[str] = Field(
        default_factory=lambda: os.getenv("MEDIC_LOCAL_TXGEMMA_9B_MODEL")
    )
    txgemma_2b_model: Optional[str] = Field(
        default_factory=lambda: os.getenv("MEDIC_LOCAL_TXGEMMA_2B_MODEL")
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached Settings singleton. Import this instead of instantiating Settings directly."""
    return Settings()
