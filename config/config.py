# config/config.py
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root if present
BASE_DIR = Path(__file__).parent.parent
ENV_PATH = BASE_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    load_dotenv()  # fallback: load from wherever


def get_openai_api_key() -> str | None:
    """Return the OpenAI API key from environment, or None if missing."""
    return os.getenv("OPENAI_API_KEY")


def get_openai_model() -> str:
    """
    Return the default OpenAI model to use with CrewAI.

    You can override this in your .env as OPENAI_MODEL.
    """
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")
