"""LanceDB-backed caching for Q&A pairs tied to documents."""

import hashlib
import logging
from typing import Any, Dict, List, Optional, Set

from lancedb_cache import get_lancedb_store

logger = logging.getLogger(__name__)


class QACacheManager:
    """Manages Q&A caching in LanceDB per document set."""

    def __init__(self):
        self.store = get_lancedb_store()

    @staticmethod
    def _normalize_doc_ids(doc_ids: Optional[Set[str]]) -> Set[str]:
        return set(doc_ids) if doc_ids else set()

    def _make_cache_key(self, question: str, doc_ids: Set[str]) -> str:
        sorted_docs = "|".join(sorted(doc_ids))
        combined = f"{question.strip().lower()}||{sorted_docs}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def cache_answer(
        self,
        *,
        question: str,
        response: str,
        thinking: Optional[str],
        doc_ids: Set[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        normalized = self._normalize_doc_ids(doc_ids)
        if not question or not normalized:
            return False
        return self.store.cache_qa(
            question=question,
            response=response,
            thinking=thinking,
            doc_ids=normalized,
            metadata=metadata,
        )

    def get_cached_answer(self, *, question: str, doc_ids: Set[str]) -> Optional[Dict[str, Any]]:
        normalized = self._normalize_doc_ids(doc_ids)
        if not question or not normalized:
            return None
        return self.store.get_cached_qa(question=question, doc_ids=normalized)

    def get_doc_history(self, doc_id: str) -> List[Dict[str, Any]]:
        return self.store.get_doc_qa_history(doc_id)

    def clear_doc_cache(self, doc_id: str) -> bool:
        return self.store.clear_doc_cache(doc_id)

    def clear_all_cache(self) -> bool:
        return self.store.clear_all_cache()

    def get_stats(self) -> Dict[str, Any]:
        try:
            df = self.store.qa_df()
            return {
                "total_qa_pairs": len(df),
                "total_documents_indexed": len({doc for ids in df.get("doc_ids", []) for doc in (ids or [])}),
                "storage": "lancedb",
            }
        except Exception as exc:  # pragma: no cover
            logger.warning(f"Failed to compute QA cache stats: {exc}")
            return {}


# Singleton instance
_qa_cache: Optional[QACacheManager] = None


def get_qa_cache() -> QACacheManager:
    """Get or create QA cache manager singleton."""
    global _qa_cache
    if _qa_cache is None:
        _qa_cache = QACacheManager()
    return _qa_cache
