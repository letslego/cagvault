import json
import hashlib
from dataclasses import asdict, is_dataclass
from datetime import datetime, UTC
from threading import Lock
from time import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import lancedb
import pyarrow as pa


class LanceDBStore:
    """LanceDB-backed cache for documents, sections, QA history, and question library."""

    def __init__(self, db_path: str = "./lancedb"):
        self.db = lancedb.connect(db_path)
        self._df_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl_seconds = 3.0
        self._cache_lock = Lock()
        self.doc_table = self._ensure_doc_table()
        self.qa_table = self._ensure_qa_table()
        self.question_table = self._ensure_question_table()
        self.warm_cache()

    def _invalidate_cache(self, *names: str) -> None:
        with self._cache_lock:
            for name in names:
                self._df_cache.pop(name, None)

    def _get_df(self, name: str, table) -> Any:
        now = time()
        with self._cache_lock:
            cached = self._df_cache.get(name)
            if cached and now - cached["ts"] <= self._cache_ttl_seconds:
                return cached["df"]
        df = table.to_pandas()
        with self._cache_lock:
            self._df_cache[name] = {"df": df, "ts": now}
        return df

    def warm_cache(self) -> None:
        """Prime frequently-used tables into the in-process cache."""
        try:
            self._get_df("doc_sections", self.doc_table)
            self._get_df("qa_cache", self.qa_table)
            self._get_df("question_library", self.question_table)
        except Exception:
            # Warm-up is best-effort; ignore failures.
            pass

    def doc_df(self):
        return self._get_df("doc_sections", self.doc_table)

    def qa_df(self):
        return self._get_df("qa_cache", self.qa_table)

    def question_df(self):
        return self._get_df("question_library", self.question_table)

    def invalidate_doc_cache(self):
        self._invalidate_cache("doc_sections")

    def invalidate_qa_cache(self):
        self._invalidate_cache("qa_cache")

    def invalidate_question_cache(self):
        self._invalidate_cache("question_library")

    def _ensure_table(self, name: str, schema: pa.Schema):
        try:
            tables = self.db.list_tables().tables
        except Exception:
            tables = []
        if name in tables:
            return self.db.open_table(name)
        return self.db.create_table(name, schema=schema)

    def _ensure_doc_table(self):
        schema = pa.schema([
            pa.field("document_id", pa.string()),
            pa.field("document_name", pa.string()),
            pa.field("section_id", pa.string()),
            pa.field("parent_id", pa.string()),
            pa.field("level", pa.int32()),
            pa.field("order_idx", pa.int32()),
            pa.field("title", pa.string()),
            pa.field("content", pa.string()),
            pa.field("metadata_json", pa.string()),
            pa.field("keywords", pa.list_(pa.string())),
            pa.field("entities_json", pa.string()),
            pa.field("total_pages", pa.int32()),
            pa.field("extraction_method", pa.string()),
            pa.field("stored_at", pa.string()),
            pa.field("source", pa.string()),
            pa.field("file_type", pa.string()),
            pa.field("file_size", pa.int64()),
            pa.field("upload_date", pa.string()),
        ])
        tbl = self._ensure_table("doc_sections", schema)
        # Best-effort indexes; ignore errors if already exist
        try:
            tbl.create_fts_index("content", replace=False)
            tbl.create_fts_index("title", replace=False)
            tbl.create_fts_index("document_name", replace=False)
        except Exception:
            pass
        return tbl

    def _ensure_qa_table(self):
        schema = pa.schema([
            pa.field("cache_key", pa.string()),
            pa.field("question", pa.string()),
            pa.field("response", pa.string()),
            pa.field("thinking", pa.string()),
            pa.field("doc_ids", pa.list_(pa.string())),
            pa.field("timestamp", pa.string()),
            pa.field("metadata_json", pa.string()),
        ])
        return self._ensure_table("qa_cache", schema)

    def _ensure_question_table(self):
        schema = pa.schema([
            pa.field("question", pa.string()),
            pa.field("doc_ids", pa.list_(pa.string())),
            pa.field("category", pa.string()),
            pa.field("metadata_json", pa.string()),
            pa.field("usage_count", pa.int64()),
            pa.field("is_default", pa.bool_()),
            pa.field("created_at", pa.string()),
        ])
        tbl = self._ensure_table("question_library", schema)
        try:
            tbl.create_fts_index("question", replace=False)
        except Exception:
            pass
        return tbl

    # -------- Sections --------
    def upsert_sections(
        self,
        *,
        document_id: str,
        document_name: str,
        sections: Sequence[Any],
        total_pages: int,
        extraction_method: str,
        source: str,
        file_type: Optional[str],
        file_size: Optional[int],
        upload_date: Optional[str],
        keywords_map: Optional[Dict[str, List[str]]] = None,
        entities_map: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> None:
        """Replace all sections for a document."""
        keywords_map = keywords_map or {}
        entities_map = entities_map or {}
        # Delete existing rows for this document
        try:
            self.doc_table.delete(where=f"document_id = '{document_id}'")
        except Exception:
            pass

        records: List[Dict[str, Any]] = []
        order_idx = 0

        def _flatten(sec_list: Sequence[Any]):
            nonlocal order_idx
            for sec in sec_list:
                # Handle both Section objects and dicts
                if hasattr(sec, "metadata"):
                    # It's a Section dataclass
                    if is_dataclass(sec.metadata):
                        meta_dict = asdict(sec.metadata)
                    else:
                        meta_dict = sec.metadata.__dict__ if hasattr(sec.metadata, "__dict__") else {}
                    content = sec.content
                    subs = sec.subsections or []
                else:
                    # It's a dict
                    meta_dict = sec.get("metadata", {})
                    content = sec.get("content", "")
                    subs = meta_dict.get("subsections", [])
                
                meta_json = json.dumps(meta_dict)
                sid = meta_dict.get("id")
                keywords = keywords_map.get(sid, [])
                entities = entities_map.get(sid, [])
                records.append({
                    "document_id": document_id,
                    "document_name": document_name,
                    "section_id": sid,
                    "parent_id": meta_dict.get("parent_id"),
                    "level": int(meta_dict.get("level", 1)),
                    "order_idx": order_idx,
                    "title": meta_dict.get("title", ""),
                    "content": content,
                    "metadata_json": meta_json,
                    "keywords": keywords,
                    "entities_json": json.dumps(entities),
                    "total_pages": int(total_pages or 0),
                    "extraction_method": extraction_method,
                    "stored_at": datetime.now(UTC).isoformat(),
                    "source": source,
                    "file_type": file_type,
                    "file_size": int(file_size or 0),
                    "upload_date": upload_date,
                })
                order_idx += 1
                if subs:
                    _flatten(subs)

        _flatten(sections)
        if records:
            self.doc_table.add(records)
        self._invalidate_cache("doc_sections")

    def load_sections(self, document_id: str):
        """Return section rows for a document."""
        df = self._get_df("doc_sections", self.doc_table)
        doc_df = df[df["document_id"] == document_id]
        return doc_df.to_dict(orient="records")

    def list_documents(self) -> List[Dict[str, Any]]:
        df = self._get_df("doc_sections", self.doc_table)
        if df.empty:
            return []
        latest = (
            df.sort_values(["document_id", "stored_at"], ascending=[True, False])
            .drop_duplicates(subset=["document_id"])
        )
        cols = ["document_id", "document_name", "total_pages", "file_type", "file_size", "upload_date", "source", "stored_at"]
        return latest[cols].to_dict(orient="records")

    # -------- QA Cache --------
    @staticmethod
    def _qa_cache_key(question: str, doc_ids: Set[str]) -> str:
        sorted_docs = "|".join(sorted(doc_ids))
        combined = f"{question.strip().lower()}||{sorted_docs}"
        digest = hashlib.sha256(combined.encode()).hexdigest()[:16]
        return f"qa_cache:{digest}"

    def cache_qa(
        self,
        *,
        question: str,
        response: str,
        thinking: Optional[str],
        doc_ids: Set[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if not question or not doc_ids:
            return False
        cache_key = self._qa_cache_key(question, doc_ids)
        try:
            self.qa_table.delete(where=f"cache_key = '{cache_key}'")
        except Exception:
            pass
        record = {
            "cache_key": cache_key,
            "question": question,
            "response": response,
            "thinking": thinking or "",
            "doc_ids": list(doc_ids),
            "timestamp": datetime.now(UTC).isoformat(),
            "metadata_json": json.dumps(metadata or {}),
        }
        self.qa_table.add([record])
        self._invalidate_cache("qa_cache")
        return True

    def get_cached_qa(self, *, question: str, doc_ids: Set[str]) -> Optional[Dict[str, Any]]:
        if not question or not doc_ids:
            return None
        cache_key = self._qa_cache_key(question, doc_ids)
        df = self._get_df("qa_cache", self.qa_table)
        hit = df[df["cache_key"] == cache_key]
        if hit.empty:
            return None
        row = hit.iloc[0]
        return {
            "question": row["question"],
            "response": row["response"],
            "thinking": row["thinking"] or None,
            "doc_ids": list(row["doc_ids"] or []),
            "timestamp": row["timestamp"],
            "metadata": json.loads(row["metadata_json"] or "{}"),
        }

    def get_doc_qa_history(self, doc_id: str) -> List[Dict[str, Any]]:
        df = self._get_df("qa_cache", self.qa_table)
        if df.empty:
            return []
        mask = df["doc_ids"].apply(lambda ids: doc_id in (ids or []))
        rows = df[mask].sort_values("timestamp", ascending=False)
        out = []
        for _, row in rows.iterrows():
            out.append({
                "question": row["question"],
                "response": row["response"],
                "thinking": row["thinking"] or None,
                "doc_ids": list(row["doc_ids"] or []),
                "timestamp": row["timestamp"],
                "metadata": json.loads(row["metadata_json"] or "{}"),
            })
        return out

    def clear_doc_cache(self, doc_id: str) -> bool:
        df = self.qa_table.to_pandas()
        if df.empty:
            return False
        mask = df["doc_ids"].apply(lambda ids: doc_id in (ids or []))
        keys = df[mask]["cache_key"].tolist()
        for key in keys:
            try:
                self.qa_table.delete(where=f"cache_key = '{key}'")
            except Exception:
                pass
        self._invalidate_cache("qa_cache")
        return True

    def clear_all_cache(self) -> bool:
        try:
            self.qa_table.delete(where="cache_key IS NOT NULL")
            self._invalidate_cache("qa_cache")
            return True
        except Exception:
            return False

    # -------- Question Library --------
    def add_question(
        self,
        *,
        question: str,
        doc_ids: Optional[Set[str]] = None,
        category: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        is_default: bool = False,
    ) -> bool:
        if not question:
            return False
        q_lower = question.strip()
        doc_list = list(doc_ids) if doc_ids else []
        meta = metadata or {}
        meta.setdefault("created_at", datetime.now(UTC).isoformat())
        meta.setdefault("is_default", is_default)
        meta.setdefault("usage_count", 0)
        meta_json = json.dumps(meta)
        try:
            self.question_table.delete(where=f"question = '{q_lower}'")
        except Exception:
            pass
        record = {
            "question": q_lower,
            "doc_ids": doc_list,
            "category": category,
            "metadata_json": meta_json,
            "usage_count": int(meta.get("usage_count", 0)),
            "is_default": is_default,
            "created_at": meta.get("created_at", datetime.now(UTC).isoformat()),
        }
        self.question_table.add([record])
        self._invalidate_cache("question_library")
        return True

    def increment_usage(self, question: str, amount: int = 1) -> None:
        if not question:
            return
        q_lower = question.strip()
        df = self._get_df("question_library", self.question_table)
        row = df[df["question"] == q_lower]
        if row.empty:
            return
        meta = json.loads(row.iloc[0]["metadata_json"] or "{}")
        meta["usage_count"] = int(meta.get("usage_count", 0)) + amount
        self.add_question(
            question=q_lower,
            doc_ids=set(row.iloc[0]["doc_ids"] or []),
            category=row.iloc[0]["category"],
            metadata=meta,
            is_default=bool(row.iloc[0]["is_default"]),
        )

    def get_popular(self, limit: int = 20) -> List[str]:
        df = self._get_df("question_library", self.question_table)
        if df.empty:
            return []
        df["usage"] = df.apply(lambda r: json.loads(r["metadata_json"] or "{}").get("usage_count", r.get("usage_count", 0)), axis=1)
        top = df.sort_values("usage", ascending=False).head(limit)
        return top["question"].tolist()

    def get_questions_for_doc(self, doc_id: str) -> List[str]:
        df = self._get_df("question_library", self.question_table)
        if df.empty:
            return []
        mask = df["doc_ids"].apply(lambda ids: doc_id in (ids or []))
        return df[mask]["question"].tolist()

    # -------- Migration from Redis --------
    def migrate_from_redis(self, redis_client) -> None:
        """One-time import from Redis if available."""
        if redis_client is None:
            return
        try:
            redis_client.ping()
        except Exception:
            return

        # Migrate documents
        for meta_key in redis_client.scan_iter(match="doc:*:meta"):
            meta_raw = redis_client.get(meta_key)
            if not meta_raw:
                continue
            meta = json.loads(meta_raw)
            doc_id = meta.get("document_id")
            doc_name = meta.get("document_name", doc_id)
            total_pages = meta.get("pages", 0)
            sections_full_key = meta_key.replace(":meta", ":sections_full")
            sections_raw = redis_client.get(sections_full_key)
            if not sections_raw:
                continue
            all_payloads = json.loads(sections_raw)
            # Rebuild sections into a simple tree (parent_id links already present)
            by_id = {p["id"]: p for p in all_payloads}
            children: Dict[str, List[Dict[str, Any]]] = {}
            roots: List[Dict[str, Any]] = []
            for p in all_payloads:
                pid = p.get("metadata", {}).get("parent_id")
                children.setdefault(pid, []).append(p)
            for p in all_payloads:
                pid = p.get("metadata", {}).get("parent_id")
                if not pid:
                    roots.append(p)
            def _build(node):
                node["subsections"] = children.get(node.get("metadata", {}).get("id")) or []
                for sub in node["subsections"]:
                    _build(sub)
            for r in roots:
                _build(r)
            self.upsert_sections(
                document_id=doc_id,
                document_name=doc_name,
                sections=roots,
                total_pages=total_pages,
                extraction_method=meta.get("extraction_method", "redis_import"),
                source=meta.get("source", "redis"),
                file_type=meta.get("file_type"),
                file_size=meta.get("file_size"),
                upload_date=meta.get("stored_at"),
                keywords_map={},
                entities_map={},
            )

        # Migrate QA cache
        for cache_key in redis_client.keys("qa_cache:*"):
            data = redis_client.hgetall(cache_key)
            if not data:
                continue
            doc_ids = json.loads(data.get("doc_ids", "[]"))
            self.cache_qa(
                question=data.get("question", ""),
                response=data.get("response", ""),
                thinking=data.get("thinking", ""),
                doc_ids=set(doc_ids),
                metadata=json.loads(data.get("metadata", "{}")),
            )

        # Migrate question library
        global_questions = redis_client.zrange("global_question_library", 0, -1) or []
        for q in global_questions:
            meta_key = f"question_meta:{q}"
            meta_raw = redis_client.hget(meta_key, "metadata") or "{}"
            meta = json.loads(meta_raw)
            doc_ids_raw = redis_client.hget(meta_key, "doc_ids") or "[]"
            doc_ids = set(json.loads(doc_ids_raw))
            category = redis_client.hget("question_categories", q)
            usage = meta.get("usage_count", 0)
            meta.setdefault("usage_count", usage)
            self.add_question(
                question=q,
                doc_ids=doc_ids,
                category=category,
                metadata=meta,
                is_default=bool(meta.get("is_default", False)),
            )


def get_lancedb_store(db_path: str = "./lancedb") -> LanceDBStore:
    """Singleton-style accessor."""
    global _LANCEDB_STORE_SINGLETON
    try:
        store = _LANCEDB_STORE_SINGLETON
    except NameError:
        store = LanceDBStore(db_path=db_path)
        _LANCEDB_STORE_SINGLETON = store
    return store
