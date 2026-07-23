# semantic_search/embedder.py

import logging
import os
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)

# TÜBİTAK self-hosted, OpenAI-compatible embeddings endpoint (vLLM), reachable
# only over the TÜBİTAK VPN. Mirrors the chat endpoint's URL pattern:
#   https://ai-api.tubitak.gov.tr/vllm/<model>/v1/<route>
# so the embeddings route becomes .../vllm/multilingual-e5-large/v1/embeddings.
DEFAULT_BASE_URL = "https://ai-api.tubitak.gov.tr/vllm/multilingual-e5-large/v1"
DEFAULT_MODEL = "multilingual-e5-large"
DEFAULT_DIMENSION = 1024  # multilingual-e5-large output dimension


def is_embedder_available() -> bool:
    """
    Check if the embedder can be used, i.e. whether the `openai` client
    library (used to talk to the TÜBİTAK OpenAI-compatible endpoint) is
    installed. No API key is required for this backend, so there's no
    credential to check for anymore - reachability (VPN) is only verified
    at actual call time, same as any other network dependency in this codebase.
    """
    try:
        import openai  # noqa: F401
        return True
    except ImportError:
        return False


# Backward-compatible alias. Older call sites (e.g. mevzuat_mcp_server.py)
# import `is_openrouter_available` - keep that name working so nothing
# breaks while callers are migrated over.
is_openrouter_available = is_embedder_available


class TubitakEmbedder:
    """
    Embedder using TÜBİTAK's self-hosted, OpenAI-compatible vLLM endpoint,
    serving `multilingual-e5-large`.

    Reachable only over the TÜBİTAK VPN. No API key is required - the
    OpenAI SDK still needs *some* non-empty string for `api_key`, so a
    placeholder ("EMPTY") is sent and ignored by the server.

    Configurable via env vars (all optional, defaults match the current
    TÜBİTAK deployment):
    - TUBITAK_EMBEDDINGS_BASE_URL  (default: https://ai-api.tubitak.gov.tr/vllm/multilingual-e5-large/v1)
    - TUBITAK_EMBEDDINGS_MODEL     (default: multilingual-e5-large)
    - TUBITAK_API_KEY              (default: "EMPTY", not checked by the server)
    """

    def __init__(self):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package is required. Install with: pip install openai")

        base_url = os.getenv("TUBITAK_EMBEDDINGS_BASE_URL", DEFAULT_BASE_URL)
        api_key = os.getenv("TUBITAK_API_KEY", "EMPTY")

        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.model = os.getenv("TUBITAK_EMBEDDINGS_MODEL", DEFAULT_MODEL)
        self.dimension = DEFAULT_DIMENSION
        self._is_e5 = "e5" in self.model.lower()

        logger.info(f"Tubitak Embedder initialized: model={self.model}, base_url={base_url}")

    def _format_query(self, query: str) -> str:
        """Format query text based on model requirements."""
        if self._is_e5:
            return f"query: {query}"
        return query

    def _format_document(self, text: str, title: str) -> str:
        """Format document text based on model requirements."""
        if self._is_e5:
            return f"passage: {title} {text}" if title and title != "none" else f"passage: {text}"
        return f"{title}\n{text}" if title and title != "none" else text

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a search query into an embedding vector."""
        text = self._format_query(query)

        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
                encoding_format="float",
            )

            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            if embedding.shape[0] != self.dimension:
                # Self-correct if the deployed model's real output size
                # differs from our assumed default - don't silently
                # mismatch the VectorStore dimension elsewhere.
                self.dimension = embedding.shape[0]

            # L2 normalize for cosine similarity
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            logger.debug(f"Encoded query: {query[:50]}... -> shape: {embedding.shape}")
            return embedding

        except Exception as e:
            logger.error(f"Failed to encode query: {e}")
            raise

    def encode_documents(self, documents: List[str], titles: Optional[List[str]] = None,
                         batch_size: int = 50) -> np.ndarray:
        """Encode multiple documents, batching to avoid request-size limits."""
        if not documents:
            return np.array([])

        texts = []
        for i, doc in enumerate(documents):
            title = titles[i] if titles and i < len(titles) else "none"
            texts.append(self._format_document(doc, title))

        try:
            all_embeddings = []

            for start in range(0, len(texts), batch_size):
                batch = texts[start:start + batch_size]
                logger.info(f"Encoding batch {start // batch_size + 1}/{(len(texts) - 1) // batch_size + 1} ({len(batch)} docs)")

                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                    encoding_format="float",
                )

                batch_embeddings = np.array(
                    [d.embedding for d in sorted(response.data, key=lambda x: x.index)],
                    dtype=np.float32
                )
                all_embeddings.append(batch_embeddings)

            embeddings = np.vstack(all_embeddings) if len(all_embeddings) > 1 else all_embeddings[0]
            if embeddings.shape[1] != self.dimension:
                self.dimension = embeddings.shape[1]

            # L2 normalize each embedding for cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)

            logger.info(f"Encoded {len(documents)} documents -> shape: {embeddings.shape}")
            return embeddings

        except Exception as e:
            logger.error(f"Failed to encode documents: {e}")
            raise


# Backward-compatible alias. Older call sites (e.g. mevzuat_mcp_server.py)
# do `from semantic_search import OpenRouterEmbedder` - keep that name
# resolvable so existing code keeps working while callers are migrated.
OpenRouterEmbedder = TubitakEmbedder