# agents/knowledge_agents.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from config.config import get_openai_api_key, get_openai_model

# ---- Optional FAISS integration ----
try:
    import faiss  # type: ignore
    # Plugin path for recent llama-index versions (0.14.x with faiss plugin)
    from llama_index.vector_stores.faiss import FaissVectorStore  # type: ignore

    HAS_FAISS = True
except Exception:
    # If import fails, we will gracefully fall back to default vector store
    FaissVectorStore = None  # type: ignore
    HAS_FAISS = False


BASE_DIR = Path(__file__).parent.parent
DOCS_DIR = BASE_DIR / "data" / "documents"
FAISS_DIR = BASE_DIR / "data" / "faiss_store"
FAISS_INDEX_PATH = FAISS_DIR / "faiss_index.bin"

_INDEX: VectorStoreIndex | None = None


def _init_llama_settings() -> None:
    """Configure LlamaIndex to use your OpenAI key + model + embeddings."""
    api_key = get_openai_api_key()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Please add it to your .env file."
        )

    Settings.llm = LlamaOpenAI(
        model=get_openai_model(),
        api_key=api_key,
    )
    Settings.embed_model = OpenAIEmbedding(api_key=api_key)


def _load_documents():
    """Load knowledge base documents from data/documents."""
    if not DOCS_DIR.exists():
        raise RuntimeError(
            f"Knowledge base folder not found: {DOCS_DIR}. "
            "Create it and add docs like service_plans.md, network_guide.md, "
            "billing_faq.md, technical_support.md."
        )

    reader = SimpleDirectoryReader(
        input_dir=str(DOCS_DIR),
        required_exts=[".md", ".txt", ".pdf"],
        recursive=True,
    )
    documents = reader.load_data()
    if not documents:
        raise RuntimeError(
            f"No documents found in knowledge base folder: {DOCS_DIR}"
        )
    return documents


def _build_index_default() -> VectorStoreIndex:
    """Build a simple in-memory vector index (fallback when FAISS not available)."""
    documents = _load_documents()
    return VectorStoreIndex.from_documents(documents)


def _build_index_faiss() -> VectorStoreIndex:
    """
    Build a FAISS-backed index using LlamaIndex's FaissVectorStore.

    If anything fails, the caller should fall back to _build_index_default().
    """
    if not HAS_FAISS or FaissVectorStore is None:
        raise RuntimeError("FAISS or FaissVectorStore is not available")

    documents = _load_documents()

    # Ensure directory exists
    FAISS_DIR.mkdir(parents=True, exist_ok=True)

    # If an index file exists, load it
    if FAISS_INDEX_PATH.exists():
        faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    else:
        # Create a new FAISS index with typical OpenAI embedding dimension
        d = 1536  # works for text-embedding-3-small / ada-002 etc.
        faiss_index = faiss.IndexFlatL2(d)

        # Attach FAISS index to the vector store
        vector_store = FaissVectorStore(faiss_index=faiss_index)

        # Build index from documents (this will populate faiss_index via the vector store)
        index = VectorStoreIndex.from_documents(documents, vector_store=vector_store)

        # 🚨 Explicitly save FAISS index to disk
        faiss.write_index(faiss_index, str(FAISS_INDEX_PATH))

    return index


def _get_index() -> VectorStoreIndex:
    """
    Get or build the global index.

    Preference:
      1. Use FAISS vector store if available.
      2. Otherwise, use default in-memory vector store.
    """
    global _INDEX
    if _INDEX is None:
        _init_llama_settings()
        if HAS_FAISS:
            try:
                _INDEX = _build_index_faiss()
            except Exception:
                # If FAISS fails for any reason, safely fall back
                _INDEX = _build_index_default()
        else:
            _INDEX = _build_index_default()
    return _INDEX


def answer_knowledge_query(
    query: str,
    customer_email: Optional[str] = None,
) -> str:
    """
    Answer 'how-to' and technical support questions using LlamaIndex.

    Examples:
      - How do I enable VoLTE?
      - What are the APN settings?
      - How to activate international roaming?
      - How to troubleshoot network issues on my phone?
    """
    try:
        index = _get_index()
    except Exception as e:
        return (
            "Knowledge assistant is not configured correctly.\n\n"
            f"Technical details: {type(e).__name__}: {e}"
        )

    query_engine = index.as_query_engine(similarity_top_k=3)

    full_question = (
        "You are a Telecom Technical Support Assistant. Use ONLY the provided "
        "documentation and retrieved context to answer precisely.\n\n"
        f"Customer: {customer_email or 'unknown'}\n"
        f"Question: {query}\n\n"
        "If the exact steps are not clearly in the docs, say that and provide a "
        "generic best-practice answer, noting that details may vary by device/operator."
    )

    try:
        response = query_engine.query(full_question)
        return str(response)
    except Exception as e:
        return (
            "Sorry, I ran into a problem while searching the knowledge base.\n\n"
            f"Technical details: {type(e).__name__}: {e}"
        )
