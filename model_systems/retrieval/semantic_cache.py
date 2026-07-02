from typing import Any, Dict, Optional


class SemanticCache:
    """
    Future embedding-based cache interface.

    This is intentionally dependency-free for now. When sentence-transformers,
    Redis vector search, pgvector, or another vector database is available, the
    implementation can be added behind this interface without changing API
    routes.
    """

    def find_similar_request(
        self,
        text: str,
        task: str,
        mode: str,
    ) -> Optional[Dict[str, Any]]:
        return None

    def save_embedding_cache(
        self,
        *,
        text: str,
        task: str,
        mode: str,
        response: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        return False
