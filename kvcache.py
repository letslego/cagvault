"""
KV Cache implementation for Cache-Augmented Generation (CAG).

This module provides efficient caching of the LLM's key-value states for preloaded contexts,
enabling fast multi-turn conversations without reprocessing documents.

Architecture:
1. Preload Phase: Documents are processed and KV-cache is generated
2. Cache Storage: KV-state is stored in memory or disk
3. Inference Phase: Queries use cached state for instant answers
4. Cache Reset: Efficient truncation for new queries without reprocessing
"""

import json
import hashlib
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheMetadata:
    """Metadata about a cached KV-state."""
    cache_id: str
    context_hash: str
    context_size: int  # Token count estimate
    created_at: float
    last_accessed_at: float
    hit_count: int = 0
    source_ids: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert metadata to dictionary."""
        return {
            "cache_id": self.cache_id,
            "context_hash": self.context_hash,
            "context_size": self.context_size,
            "created_at": self.created_at,
            "last_accessed_at": self.last_accessed_at,
            "hit_count": self.hit_count,
            "source_ids": self.source_ids,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "CacheMetadata":
        """Create metadata from dictionary."""
        return cls(**data)


@dataclass
class KVCacheEntry:
    """Represents a single KV-cache entry."""
    cache_id: str
    context: str  # The preloaded context (concatenated documents)
    metadata: CacheMetadata
    kv_state: Optional[Any] = None  # Placeholder for actual KV-cache from model
    
    def is_valid(self) -> bool:
        """Check if cache entry is still valid."""
        return self.kv_state is not None


class KVCache:
    """
    Manager for KV-cache storage and retrieval.
    
    Implements efficient caching of LLM inference states (KV-states) for preloaded contexts.
    Supports both in-memory and disk-based storage.
    
    Example:
        >>> cache = KVCache(cache_dir=Path(".cache"))
        >>> cache_id = cache.create("my_context_text")
        >>> entry = cache.get(cache_id)
        >>> cache.update_kv_state(cache_id, kv_state_from_model)
    """
    
    def __init__(self, cache_dir: Optional[Path] = None, memory_only: bool = False):
        """
        Initialize KV-Cache manager.
        
        Args:
            cache_dir: Directory for storing cache files. If None, uses .cache/
            memory_only: If True, only keep caches in memory (faster but not persistent)
        """
        self.memory_only = memory_only
        self.cache_dir = Path(cache_dir) if cache_dir else Path(".cache/kvcache")
        
        # In-memory storage
        self._memory_cache: Dict[str, KVCacheEntry] = {}
        
        # Create cache directory if not memory-only
        if not memory_only:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._metadata_file = self.cache_dir / "cache_metadata.json"
        
        logger.info(f"KVCache initialized (memory_only={memory_only}, cache_dir={self.cache_dir})")
    
    def _hash_context(self, context: str) -> str:
        """Generate hash of context for deduplication."""
        return hashlib.sha256(context.encode()).hexdigest()
    
    def _estimate_token_count(self, text: str) -> int:
        """
        Estimate token count using simple heuristic.
        (In production, use proper tokenizer from the LLM)
        """
        # Rough approximation: ~4 chars per token
        return len(text) // 4
    
    def create(self, context: str, source_ids: Optional[List[str]] = None) -> str:
        """
        Create a new KV-cache entry for the given context.
        
        Args:
            context: The preloaded context (concatenated documents)
            source_ids: Optional list of source IDs that make up this context
            
        Returns:
            cache_id: Unique identifier for this cache entry
        """
        context_hash = self._hash_context(context)
        
        # Check for duplicate context
        existing = self.find_by_hash(context_hash)
        if existing:
            logger.info(f"Cache hit: context already cached as {existing.cache_id}")
            return existing.cache_id
        
        # Generate unique cache ID
        cache_id = f"cache_{int(time.time() * 1000)}"
        
        # Create metadata
        metadata = CacheMetadata(
            cache_id=cache_id,
            context_hash=context_hash,
            context_size=self._estimate_token_count(context),
            created_at=time.time(),
            last_accessed_at=time.time(),
            source_ids=source_ids or [],
        )
        
        # Create cache entry
        entry = KVCacheEntry(
            cache_id=cache_id,
            context=context,
            metadata=metadata,
            kv_state=None,  # Will be set by update_kv_state
        )
        
        # Store in memory
        self._memory_cache[cache_id] = entry
        
        # Save to disk if not memory-only
        if not self.memory_only:
            self._save_entry(entry)
            self._update_metadata_index()
        
        logger.info(f"Created cache {cache_id} (size: {metadata.context_size} tokens)")
        return cache_id
    
    def get(self, cache_id: str) -> Optional[KVCacheEntry]:
        """
        Retrieve a KV-cache entry.
        
        Args:
            cache_id: The cache ID to retrieve
            
        Returns:
            KVCacheEntry if found, None otherwise
        """
        # Check memory first
        if cache_id in self._memory_cache:
            entry = self._memory_cache[cache_id]
            entry.metadata.last_accessed_at = time.time()
            entry.metadata.hit_count += 1
            return entry
        
        # Try to load from disk
        if not self.memory_only:
            entry = self._load_entry(cache_id)
            if entry:
                self._memory_cache[cache_id] = entry
                entry.metadata.last_accessed_at = time.time()
                entry.metadata.hit_count += 1
                return entry
        
        return None
    
    def find_by_hash(self, context_hash: str) -> Optional[KVCacheEntry]:
        """Find cache entry by content hash."""
        for entry in self._memory_cache.values():
            if entry.metadata.context_hash == context_hash:
                return entry
        
        # Search disk cache metadata if available
        if not self.memory_only and self._metadata_file.exists():
            metadata = self._load_metadata_index()
            for cache_id, meta in metadata.items():
                if meta["context_hash"] == context_hash:
                    return self.get(cache_id)
        
        return None
    
    def update_kv_state(self, cache_id: str, kv_state: Any) -> bool:
        """
        Update the KV-state for a cache entry.
        
        Args:
            cache_id: The cache ID to update
            kv_state: The KV-state object from the model
            
        Returns:
            True if successful, False if cache not found
        """
        entry = self.get(cache_id)
        if not entry:
            logger.warning(f"Cache {cache_id} not found")
            return False
        
        entry.kv_state = kv_state
        
        # Save to disk if not memory-only
        if not self.memory_only:
            self._save_entry(entry)
        
        logger.info(f"Updated KV-state for cache {cache_id}")
        return True
    
    def delete(self, cache_id: str) -> bool:
        """Delete a cache entry."""
        if cache_id in self._memory_cache:
            del self._memory_cache[cache_id]
        
        if not self.memory_only:
            cache_file = self.cache_dir / f"{cache_id}.pkl"
            if cache_file.exists():
                cache_file.unlink()
        
        logger.info(f"Deleted cache {cache_id}")
        return True
    
    def clear_all(self) -> int:
        """Clear all caches. Returns number of cleared entries."""
        count = len(self._memory_cache)
        self._memory_cache.clear()
        
        if not self.memory_only:
            # Remove all pickle files
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            
            # Clear metadata
            if self._metadata_file.exists():
                self._metadata_file.unlink()
        
        logger.info(f"Cleared {count} cache entries")
        return count
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        total_size = sum(e.metadata.context_size for e in self._memory_cache.values())
        total_hits = sum(e.metadata.hit_count for e in self._memory_cache.values())
        
        return {
            "total_entries": len(self._memory_cache),
            "total_tokens": total_size,
            "total_hits": total_hits,
            "memory_only": self.memory_only,
            "entries": [
                {
                    "cache_id": e.cache_id,
                    "size": e.metadata.context_size,
                    "hits": e.metadata.hit_count,
                    "sources": len(e.metadata.source_ids),
                }
                for e in self._memory_cache.values()
            ]
        }
    
    def _save_entry(self, entry: KVCacheEntry) -> None:
        """Save cache entry to disk."""
        if self.memory_only:
            return
        
        cache_file = self.cache_dir / f"{entry.cache_id}.pkl"
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(entry, f)
        except Exception as e:
            logger.error(f"Failed to save cache {entry.cache_id}: {e}")
    
    def _load_entry(self, cache_id: str) -> Optional[KVCacheEntry]:
        """Load cache entry from disk."""
        if self.memory_only:
            return None
        
        cache_file = self.cache_dir / f"{cache_id}.pkl"
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load cache {cache_id}: {e}")
            return None
    
    def _update_metadata_index(self) -> None:
        """Update the metadata index file."""
        if self.memory_only:
            return
        
        metadata = {
            cache_id: entry.metadata.to_dict()
            for cache_id, entry in self._memory_cache.items()
        }
        
        try:
            with open(self._metadata_file, "w") as f:
                json.dump(metadata, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to update metadata index: {e}")
    
    def _load_metadata_index(self) -> Dict[str, dict]:
        """Load metadata index from disk."""
        if self.memory_only or not self._metadata_file.exists():
            return {}
        
        try:
            with open(self._metadata_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load metadata index: {e}")
            return {}


# Global cache instance
_kv_cache_instance: Optional[KVCache] = None


def get_kv_cache(cache_dir: Optional[Path] = None, memory_only: bool = False) -> KVCache:
    """
    Get or create the global KV-cache instance.
    
    Args:
        cache_dir: Directory for cache storage
        memory_only: If True, use memory-only mode
        
    Returns:
        KVCache instance
    """
    global _kv_cache_instance
    
    if _kv_cache_instance is None:
        _kv_cache_instance = KVCache(cache_dir=cache_dir, memory_only=memory_only)
    
    return _kv_cache_instance


def reset_kv_cache() -> None:
    """Reset the global KV-cache instance."""
    global _kv_cache_instance
    _kv_cache_instance = None
