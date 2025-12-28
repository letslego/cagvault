"""
Data Lineage Tracker for RAG System

Implements OpenLineage standard for tracking data flow through:
- Document ingestion → PDF parsing → Section extraction
- Embedding generation → LanceDB storage
- Query retrieval → Cache hits/misses
- LLM response generation

Uses SQLite for persistence and provides dashboard visualization.
"""

import json
import sqlite3
import logging
from datetime import datetime
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import uuid

logger = logging.getLogger(__name__)

# Database location
LINEAGE_DB = Path(".cache/lineage.db")


@dataclass
class DataAsset:
    """Represents a data asset in the pipeline."""
    name: str
    type: str  # document, section, embedding, cache_entry, query, answer
    format: Optional[str] = None
    size_bytes: Optional[int] = None
    record_count: Optional[int] = None
    metadata: Optional[Dict] = None


@dataclass
class LineageEvent:
    """Represents a lineage event following OpenLineage standard."""
    event_id: str
    timestamp: str
    producer: str  # Component that produced the event
    input_assets: List[DataAsset]
    output_assets: List[DataAsset]
    operation: str  # parse, embed, retrieve, cache_hit, llm_response
    status: str  # SUCCESS, FAILED
    duration_ms: float
    metadata: Dict[str, Any]


class LineageDB:
    """SQLite database for storing lineage events."""
    
    def __init__(self, db_path: Path = LINEAGE_DB):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS lineage_events (
                    event_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    producer TEXT,
                    operation TEXT,
                    status TEXT,
                    duration_ms REAL,
                    input_assets TEXT,
                    output_assets TEXT,
                    metadata TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS data_assets (
                    asset_id TEXT PRIMARY KEY,
                    name TEXT,
                    type TEXT,
                    format TEXT,
                    size_bytes INTEGER,
                    record_count INTEGER,
                    first_seen TEXT,
                    last_seen TEXT,
                    metadata TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON lineage_events(timestamp DESC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_operation ON lineage_events(operation)
            """)
            conn.commit()
    
    def record_event(self, event: LineageEvent):
        """Record a lineage event."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO lineage_events 
                (event_id, timestamp, producer, operation, status, duration_ms, 
                 input_assets, output_assets, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.event_id,
                event.timestamp,
                event.producer,
                event.operation,
                event.status,
                event.duration_ms,
                json.dumps([asdict(a) for a in event.input_assets]),
                json.dumps([asdict(a) for a in event.output_assets]),
                json.dumps(event.metadata)
            ))
            conn.commit()
    
    def record_asset(self, asset_id: str, asset: DataAsset):
        """Record a data asset."""
        now = datetime.utcnow().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            # Check if asset exists
            cursor = conn.execute(
                "SELECT asset_id FROM data_assets WHERE asset_id = ?", 
                (asset_id,)
            )
            if cursor.fetchone():
                # Update last_seen
                conn.execute(
                    "UPDATE data_assets SET last_seen = ? WHERE asset_id = ?",
                    (now, asset_id)
                )
            else:
                # Insert new asset
                conn.execute("""
                    INSERT INTO data_assets
                    (asset_id, name, type, format, size_bytes, record_count, 
                     first_seen, last_seen, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    asset_id,
                    asset.name,
                    asset.type,
                    asset.format,
                    asset.size_bytes,
                    asset.record_count,
                    now,
                    now,
                    json.dumps(asset.metadata or {})
                ))
            conn.commit()
    
    def get_events(self, limit: int = 100, operation: Optional[str] = None) -> List[Dict]:
        """Get recent lineage events."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if operation:
                cursor = conn.execute(
                    "SELECT * FROM lineage_events WHERE operation = ? ORDER BY timestamp DESC LIMIT ?",
                    (operation, limit)
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM lineage_events ORDER BY timestamp DESC LIMIT ?",
                    (limit,)
                )
            return [dict(row) for row in cursor.fetchall()]
    
    def get_asset_lineage(self, asset_name: str) -> Dict[str, Any]:
        """Get full lineage for a specific asset."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Get asset
            cursor = conn.execute(
                "SELECT * FROM data_assets WHERE name = ?", 
                (asset_name,)
            )
            asset = dict(cursor.fetchone() or {})
            
            if not asset:
                return {"error": f"Asset '{asset_name}' not found"}
            
            # Get events involving this asset
            cursor = conn.execute("""
                SELECT * FROM lineage_events 
                WHERE input_assets LIKE ? OR output_assets LIKE ?
                ORDER BY timestamp
            """, (f'%"{asset_name}"%', f'%"{asset_name}"%'))
            
            events = [dict(row) for row in cursor.fetchall()]
            
            return {
                "asset": asset,
                "event_count": len(events),
                "events": events
            }
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Total events
            total = conn.execute("SELECT COUNT(*) as count FROM lineage_events").fetchone()
            
            # Events by operation
            ops = conn.execute("""
                SELECT operation, COUNT(*) as count, AVG(duration_ms) as avg_duration
                FROM lineage_events
                GROUP BY operation
            """).fetchall()
            
            # Success/failure ratio
            status = conn.execute("""
                SELECT status, COUNT(*) as count
                FROM lineage_events
                GROUP BY status
            """).fetchall()
            
            # Unique assets
            assets = conn.execute(
                "SELECT COUNT(*) as count FROM data_assets"
            ).fetchone()
            
            return {
                "total_events": total["count"],
                "operations": [dict(op) for op in ops],
                "status_breakdown": [dict(s) for s in status],
                "total_assets": assets["count"]
            }


class LineageTracker:
    """Main lineage tracking system."""
    
    def __init__(self):
        self.db = LineageDB()
    
    def track_pdf_ingestion(self, pdf_name: str, size_bytes: int) -> str:
        """Track PDF document ingestion."""
        event_id = str(uuid.uuid4())
        asset = DataAsset(
            name=pdf_name,
            type="document",
            format="pdf",
            size_bytes=size_bytes
        )
        asset_id = f"doc_{pdf_name}"
        self.db.record_asset(asset_id, asset)
        
        event = LineageEvent(
            event_id=event_id,
            timestamp=datetime.utcnow().isoformat(),
            producer="PDFParser",
            input_assets=[],
            output_assets=[asset],
            operation="ingest",
            status="SUCCESS",
            duration_ms=0,
            metadata={"pdf_name": pdf_name, "size_bytes": size_bytes}
        )
        self.db.record_event(event)
        logger.info(f"Tracked PDF ingestion: {pdf_name}")
        return asset_id
    
    def track_section_extraction(self, pdf_id: str, section_name: str, 
                                  section_text: str, duration_ms: float):
        """Track document section extraction."""
        event_id = str(uuid.uuid4())
        output_asset = DataAsset(
            name=section_name,
            type="section",
            format="text",
            size_bytes=len(section_text.encode('utf-8')),
            metadata={"source_pdf": pdf_id}
        )
        asset_id = f"section_{section_name[:50]}"
        self.db.record_asset(asset_id, output_asset)
        
        input_asset = DataAsset(name=pdf_id, type="document")
        
        event = LineageEvent(
            event_id=event_id,
            timestamp=datetime.utcnow().isoformat(),
            producer="SectionExtractor",
            input_assets=[input_asset],
            output_assets=[output_asset],
            operation="extract_section",
            status="SUCCESS",
            duration_ms=duration_ms,
            metadata={"section_name": section_name}
        )
        self.db.record_event(event)
        logger.info(f"Tracked section extraction: {section_name}")
        return asset_id
    
    def track_embedding(self, section_id: str, embedding_dim: int, 
                       duration_ms: float):
        """Track embedding generation."""
        event_id = str(uuid.uuid4())
        output_asset = DataAsset(
            name=f"{section_id}_embedding",
            type="embedding",
            format="vector",
            record_count=embedding_dim,
            metadata={"dimensions": embedding_dim, "model": "sentence-transformers"}
        )
        asset_id = f"embedding_{section_id}"
        self.db.record_asset(asset_id, output_asset)
        
        input_asset = DataAsset(name=section_id, type="section")
        
        event = LineageEvent(
            event_id=event_id,
            timestamp=datetime.utcnow().isoformat(),
            producer="EmbeddingGenerator",
            input_assets=[input_asset],
            output_assets=[output_asset],
            operation="embed",
            status="SUCCESS",
            duration_ms=duration_ms,
            metadata={"model": "sentence-transformers", "dimensions": embedding_dim}
        )
        self.db.record_event(event)
        logger.info(f"Tracked embedding: {section_id}")
        return asset_id
    
    def track_lancedb_storage(self, embedding_ids: List[str], 
                             table_name: str, duration_ms: float):
        """Track storage in LanceDB."""
        event_id = str(uuid.uuid4())
        output_asset = DataAsset(
            name=table_name,
            type="vector_store",
            format="lancedb",
            record_count=len(embedding_ids),
            metadata={"table": table_name}
        )
        asset_id = f"lancedb_{table_name}"
        self.db.record_asset(asset_id, output_asset)
        
        input_assets = [DataAsset(name=eid, type="embedding") for eid in embedding_ids]
        
        event = LineageEvent(
            event_id=event_id,
            timestamp=datetime.utcnow().isoformat(),
            producer="LanceDBWriter",
            input_assets=input_assets,
            output_assets=[output_asset],
            operation="store_lancedb",
            status="SUCCESS",
            duration_ms=duration_ms,
            metadata={"table": table_name, "record_count": len(embedding_ids)}
        )
        self.db.record_event(event)
        logger.info(f"Tracked LanceDB storage: {table_name} ({len(embedding_ids)} embeddings)")
        return asset_id
    
    def track_retrieval(self, query: str, retrieved_sections: List[str], 
                       duration_ms: float, cache_hit: bool = False):
        """Track retrieval from LanceDB."""
        event_id = str(uuid.uuid4())
        query_asset = DataAsset(
            name=f"query_{query[:30]}",
            type="query",
            format="text",
            metadata={"query_text": query}
        )
        query_id = f"query_{event_id}"
        self.db.record_asset(query_id, query_asset)
        
        input_asset = DataAsset(name="doc_sections", type="vector_store")
        output_assets = [DataAsset(name=s, type="section") for s in retrieved_sections]
        
        operation = "retrieve_cache" if cache_hit else "retrieve"
        
        event = LineageEvent(
            event_id=event_id,
            timestamp=datetime.utcnow().isoformat(),
            producer="LanceDBRetriever",
            input_assets=[input_asset],
            output_assets=output_assets,
            operation=operation,
            status="SUCCESS",
            duration_ms=duration_ms,
            metadata={
                "query": query,
                "result_count": len(retrieved_sections),
                "cache_hit": cache_hit
            }
        )
        self.db.record_event(event)
        logger.info(f"Tracked retrieval: {len(retrieved_sections)} sections, cache_hit={cache_hit}")
        return query_id
    
    def track_llm_response(self, query_id: str, response: str, 
                          duration_ms: float, model: str = "Qwen3-14B"):
        """Track LLM response generation."""
        event_id = str(uuid.uuid4())
        output_asset = DataAsset(
            name=f"answer_{query_id}",
            type="answer",
            format="text",
            size_bytes=len(response.encode('utf-8')),
            metadata={"model": model}
        )
        answer_id = f"answer_{event_id}"
        self.db.record_asset(answer_id, output_asset)
        
        input_asset = DataAsset(name=query_id, type="query")
        
        event = LineageEvent(
            event_id=event_id,
            timestamp=datetime.utcnow().isoformat(),
            producer="LLMGenerator",
            input_assets=[input_asset],
            output_assets=[output_asset],
            operation="llm_response",
            status="SUCCESS",
            duration_ms=duration_ms,
            metadata={"model": model, "response_length": len(response)}
        )
        self.db.record_event(event)
        logger.info(f"Tracked LLM response: {model}")
        return answer_id
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return self.db.get_pipeline_stats()
    
    def get_events(self, limit: int = 50, operation: Optional[str] = None) -> List[Dict]:
        """Get recent events."""
        return self.db.get_events(limit, operation)
    
    def get_asset_lineage(self, asset_name: str) -> Dict[str, Any]:
        """Get lineage for a specific asset."""
        return self.db.get_asset_lineage(asset_name)


# Global instance
_lineage_tracker: Optional[LineageTracker] = None


def get_lineage_tracker() -> LineageTracker:
    """Get or create global lineage tracker."""
    global _lineage_tracker
    if _lineage_tracker is None:
        _lineage_tracker = LineageTracker()
    return _lineage_tracker
