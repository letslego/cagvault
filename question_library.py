"""
Question Library Management (LanceDB)

Stores and manages a library of common questions for documents.
Provides autocomplete suggestions and question recommendation without Redis.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Set

from lancedb_cache import get_lancedb_store

logger = logging.getLogger(__name__)


class QuestionLibraryManager:
    """Manages question library in LanceDB with autocomplete support."""

    CATEGORIES = {
        "definitions": "Term/Concept Definitions",
        "parties": "Parties & Obligations",
        "financial": "Financial Metrics & Covenants",
        "conditions": "Conditions & Events",
        "rights": "Rights & Remedies",
        "restrictions": "Restrictions & Limitations",
        "procedures": "Procedures & Mechanics",
        "timeline": "Timeline & Dates",
        "calculations": "Calculations & Formulas",
        "other": "Other",
    }

    def __init__(self):
        self.store = get_lancedb_store()
        self._initialize_default_library()

    def _initialize_default_library(self) -> None:
        """Initialize with default common questions if library is empty."""
        df = self.store.question_df()
        if not df.empty:
            return
        default_questions = {
            "What are the key parties to this agreement?": "parties",
            "Who is the borrower?": "parties",
            "Who is the administrative agent?": "parties",
            "What are the borrower's obligations?": "parties",
            "What are the guarantor's obligations?": "parties",
            "Who can be a lender under this agreement?": "parties",
            "What are the financial covenants?": "financial",
            "What is the leverage ratio requirement?": "financial",
            "What is the interest coverage ratio?": "financial",
            "What is the debt service coverage ratio?": "financial",
            "What is the minimum liquidity requirement?": "financial",
            "What is the commitment amount?": "financial",
            "What are the fees and charges?": "financial",
            "What triggers an event of default?": "conditions",
            "What are the conditions precedent to borrowing?": "conditions",
            "What happens if the borrower defaults?": "conditions",
            "What is the interest rate and how is it calculated?": "calculations",
            "What are the borrower's restrictions on debt?": "restrictions",
            "What covenants must the borrower maintain?": "restrictions",
            "How are prepayments handled?": "procedures",
            "When is the loan due?": "timeline",
            "What are the lender's remedies upon default?": "rights",
            "What is the governing law?": "other",
            "Are there any guarantees?": "other",
        }
        for q, category in default_questions.items():
            self.add_question(q, category=category, is_default=True)

    def add_question(
        self,
        question: str,
        doc_ids: Optional[Set[str]] = None,
        category: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        is_default: bool = False,
    ) -> bool:
        return self.store.add_question(
            question=question,
            doc_ids=doc_ids,
            category=category,
            metadata=metadata,
            is_default=is_default,
        )

    def increment_usage(self, question: str) -> bool:
        try:
            self.store.increment_usage(question)
            return True
        except Exception as exc:
            logger.warning(f"Failed to increment usage: {exc}")
            return False

    def get_autocomplete_suggestions(
        self,
        query: str,
        doc_id: Optional[str] = None,
        max_results: int = 10,
        include_global: bool = True,
    ) -> List[Dict[str, Any]]:
        if not query:
            return []
        df = self.store.question_df().copy()
        if df.empty:
            return []
        query_lower = query.lower().strip()
        if doc_id:
            df = df[df["doc_ids"].apply(lambda ids: doc_id in (ids or []))]
        elif not include_global:
            df = df[df["doc_ids"].apply(lambda ids: bool(ids))]
        df["usage"] = df.apply(lambda r: json.loads(r["metadata_json"] or "{}").get("usage_count", r.get("usage_count", 0)), axis=1)
        matched = df[df["question"].str.contains(query_lower, case=False, na=False)]
        matched = matched.sort_values("usage", ascending=False).head(max_results)
        results: List[Dict[str, Any]] = []
        for _, row in matched.iterrows():
            meta = json.loads(row["metadata_json"] or "{}")
            cat = row["category"] or "other"
            results.append({
                "question": row["question"],
                "category": cat,
                "category_label": self.CATEGORIES.get(cat, "Other"),
                "metadata": meta,
            })
        return results

    def get_popular_questions(
        self,
        doc_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        df = self.store.question_df().copy()
        if df.empty:
            return []
        if doc_id:
            df = df[df["doc_ids"].apply(lambda ids: doc_id in ids if isinstance(ids, (list, tuple)) else False)]
        
        # Extract usage count from metadata_json
        def get_usage(row):
            try:
                meta = json.loads(row["metadata_json"] or "{}")
                return meta.get("usage_count", row.get("usage_count", 0))
            except:
                return 0
        
        df["usage"] = df.apply(get_usage, axis=1)
        top = df.sort_values("usage", ascending=False).head(limit)
        enriched: List[Dict[str, Any]] = []
        for _, row in top.iterrows():
            meta = json.loads(row["metadata_json"] or "{}")
            cat = row["category"] or "other"
            enriched.append({
                "question": row["question"],
                "category": cat,
                "category_label": self.CATEGORIES.get(cat, "Other"),
                "metadata": meta,
            })
        return enriched

    def get_questions_by_category(
        self,
        category: str,
        doc_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        if category not in self.CATEGORIES:
            return []
        df = self.store.question_df().copy()
        if df.empty:
            return []
        df = df[df["category"] == category]
        if doc_id:
            def filter_by_doc(ids):
                try:
                    # Handle None, lists, and numpy arrays
                    if ids is None:
                        return False
                    if isinstance(ids, list):
                        return doc_id in ids
                    if hasattr(ids, '__iter__') and not isinstance(ids, str):
                        return doc_id in list(ids)
                    return False
                except (TypeError, ValueError):
                    return False
            df = df[df["doc_ids"].apply(filter_by_doc)]
        df = df.head(limit)
        results: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            meta = json.loads(row["metadata_json"] or "{}")
            results.append({
                "question": row["question"],
                "category": category,
                "category_label": self.CATEGORIES.get(category, "Other"),
                "metadata": meta,
            })
        return results

    def get_all_questions(self, doc_id: Optional[str] = None) -> List[str]:
        df = self.store.question_df()
        if df.empty:
            return []
        if doc_id:
            # Handle both lists and numpy arrays
            df = df[df["doc_ids"].apply(lambda ids: doc_id in list(ids) if ids is not None and len(ids) > 0 else False)]
        return df["question"].tolist()

    def remove_question(self, question: str, doc_id: Optional[str] = None) -> bool:
        if not question:
            return False
        q_lower = question.strip()
        df = self.store.question_df()
        if df.empty:
            return False
        if doc_id:
            mask = (df["question"] == q_lower)
            if mask.any():
                row = df[mask].iloc[0]
                doc_ids = set(row["doc_ids"] or [])
                if doc_id in doc_ids:
                    doc_ids.remove(doc_id)
                    meta = json.loads(row["metadata_json"] or "{}")
                    self.store.add_question(
                        question=q_lower,
                        doc_ids=doc_ids,
                        category=row["category"],
                        metadata=meta,
                        is_default=bool(row["is_default"]),
                    )
            return True
        try:
            self.store.question_table.delete(where=f"question = '{q_lower}'")
            self.store.invalidate_question_cache()
            return True
        except Exception as exc:
            logger.warning(f"Failed to remove question: {exc}")
            return False

    def clear_library(self) -> bool:
        try:
            self.store.question_table.delete(where="question IS NOT NULL")
            self.store.invalidate_question_cache()
            return True
        except Exception as exc:
            logger.warning(f"Failed to clear library: {exc}")
            return False


_question_library = None


def get_question_library() -> QuestionLibraryManager:
    """Get or create question library manager singleton."""
    global _question_library
    if _question_library is None:
        _question_library = QuestionLibraryManager()
    return _question_library
