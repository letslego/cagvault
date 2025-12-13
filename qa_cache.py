"""
Redis-based caching for Q&A pairs tied to specific documents.

Stores:
- Question text
- LLM thinking/reasoning
- Full response
- Associated document IDs
- Timestamp and metadata
"""

import json
import hashlib
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Set
import redis

logger = logging.getLogger(__name__)


class QACacheManager:
    """Manages Q&A caching in Redis per document."""
    
    # Redis key patterns
    QA_CACHE_KEY_PREFIX = "qa_cache"
    QA_INDEX_KEY_PREFIX = "qa_index"  # For tracking Q&A pairs per document
    DOC_QUESTIONS_KEY_PREFIX = "doc_questions"  # For retrieving all questions for a doc
    
    def __init__(self, redis_client: redis.Redis = None):
        """Initialize Q&A cache manager.
        
        Args:
            redis_client: Redis client instance (uses default if None)
        """
        if redis_client is None:
            self.redis = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
        else:
            self.redis = redis_client
        
        # Test connection
        try:
            self.redis.ping()
            logger.info("Connected to Redis for Q&A cache")
        except Exception as e:
            logger.warning(f"Could not connect to Redis: {e}")
            self.redis = None
    
    def _generate_cache_key(self, question: str, doc_ids: Set[str]) -> str:
        """Generate a unique cache key based on question and associated docs.
        
        Args:
            question: The question text
            doc_ids: Set of document IDs this question is about
            
        Returns:
            Unique cache key
        """
        # Sort doc_ids for consistent hashing
        sorted_docs = "|".join(sorted(doc_ids))
        combined = f"{question.strip().lower()}||{sorted_docs}"
        hash_digest = hashlib.sha256(combined.encode()).hexdigest()[:16]
        return f"{self.QA_CACHE_KEY_PREFIX}:{hash_digest}"
    
    def _generate_question_index_key(self, doc_id: str) -> str:
        """Generate Redis key for storing question hashes for a document."""
        return f"{self.DOC_QUESTIONS_KEY_PREFIX}:{doc_id}"
    
    def cache_qa(
        self,
        question: str,
        response: str,
        thinking: Optional[str] = None,
        doc_ids: Optional[Set[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Cache a Q&A pair for specific documents.
        
        Args:
            question: User question
            response: LLM response
            thinking: LLM thinking/reasoning (optional)
            doc_ids: Set of document IDs this question relates to
            metadata: Additional metadata (model name, temperature, etc.)
            
        Returns:
            True if cached successfully, False otherwise
        """
        if not self.redis or not doc_ids:
            return False
        
        try:
            cache_key = self._generate_cache_key(question, doc_ids)
            
            # Prepare cache data
            cache_data = {
                "question": question,
                "response": response,
                "thinking": thinking or "",
                "doc_ids": json.dumps(sorted(list(doc_ids))),
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": json.dumps(metadata or {})
            }
            
            # Store main Q&A cache
            self.redis.hset(cache_key, mapping=cache_data)
            
            # Set TTL (24 hours)
            self.redis.expire(cache_key, 86400)
            
            # Index this question per document for quick lookup
            for doc_id in doc_ids:
                index_key = self._generate_question_index_key(doc_id)
                self.redis.sadd(index_key, cache_key)
                self.redis.expire(index_key, 86400)
            
            logger.debug(f"Cached Q&A: {cache_key}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to cache Q&A: {e}")
            return False
    
    def get_cached_qa(
        self,
        question: str,
        doc_ids: Set[str]
    ) -> Optional[Dict[str, Any]]:
        """Retrieve cached Q&A for a question and document set.
        
        Args:
            question: Question to look up
            doc_ids: Set of document IDs
            
        Returns:
            Cached Q&A data or None if not found
        """
        if not self.redis or not doc_ids:
            return None
        
        try:
            cache_key = self._generate_cache_key(question, doc_ids)
            data = self.redis.hgetall(cache_key)
            
            if not data:
                return None
            
            # Parse JSON fields
            return {
                "question": data.get("question"),
                "response": data.get("response"),
                "thinking": data.get("thinking") or None,
                "doc_ids": json.loads(data.get("doc_ids", "[]")),
                "timestamp": data.get("timestamp"),
                "metadata": json.loads(data.get("metadata", "{}"))
            }
        except Exception as e:
            logger.warning(f"Failed to retrieve cached Q&A: {e}")
            return None
    
    def get_doc_qa_history(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get all cached Q&A pairs for a specific document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            List of Q&A pairs for this document
        """
        if not self.redis or not doc_id:
            return []
        
        try:
            index_key = self._generate_question_index_key(doc_id)
            cache_keys = self.redis.smembers(index_key)
            
            qa_pairs = []
            for cache_key in cache_keys:
                data = self.redis.hgetall(cache_key)
                if data:
                    qa_pairs.append({
                        "question": data.get("question"),
                        "response": data.get("response"),
                        "thinking": data.get("thinking") or None,
                        "doc_ids": json.loads(data.get("doc_ids", "[]")),
                        "timestamp": data.get("timestamp"),
                        "metadata": json.loads(data.get("metadata", "{}"))
                    })
            
            # Sort by timestamp (newest first)
            qa_pairs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            return qa_pairs
        
        except Exception as e:
            logger.warning(f"Failed to retrieve Q&A history for {doc_id}: {e}")
            return []
    
    def clear_doc_cache(self, doc_id: str) -> bool:
        """Clear all cached Q&A for a specific document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if cleared successfully
        """
        if not self.redis or not doc_id:
            return False
        
        try:
            index_key = self._generate_question_index_key(doc_id)
            cache_keys = self.redis.smembers(index_key)
            
            # Delete all Q&A pairs for this document
            for cache_key in cache_keys:
                self.redis.delete(cache_key)
            
            # Delete the index
            self.redis.delete(index_key)
            
            logger.info(f"Cleared Q&A cache for document {doc_id}")
            return True
        
        except Exception as e:
            logger.warning(f"Failed to clear Q&A cache for {doc_id}: {e}")
            return False
    
    def clear_all_cache(self) -> bool:
        """Clear all Q&A caches (use with caution)."""
        if not self.redis:
            return False
        
        try:
            # Find all Q&A cache keys
            qa_keys = self.redis.keys(f"{self.QA_CACHE_KEY_PREFIX}:*")
            doc_index_keys = self.redis.keys(f"{self.DOC_QUESTIONS_KEY_PREFIX}:*")
            
            all_keys = qa_keys + doc_index_keys
            
            if all_keys:
                self.redis.delete(*all_keys)
            
            logger.info(f"Cleared {len(all_keys)} Q&A cache entries")
            return True
        
        except Exception as e:
            logger.warning(f"Failed to clear all Q&A caches: {e}")
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about cached Q&A pairs.
        
        Returns:
            Cache statistics
        """
        if not self.redis:
            return {}
        
        try:
            qa_keys = self.redis.keys(f"{self.QA_CACHE_KEY_PREFIX}:*")
            doc_index_keys = self.redis.keys(f"{self.DOC_QUESTIONS_KEY_PREFIX}:*")
            
            stats = {
                "total_qa_pairs": len(qa_keys),
                "total_documents_indexed": len(doc_index_keys),
                "redis_memory_usage": self.redis.info("memory").get("used_memory_human", "N/A")
            }
            
            return stats
        except Exception as e:
            logger.warning(f"Failed to get cache stats: {e}")
            return {}


# Singleton instance
_qa_cache = None


def get_qa_cache() -> QACacheManager:
    """Get or create QA cache manager singleton."""
    global _qa_cache
    if _qa_cache is None:
        _qa_cache = QACacheManager()
    return _qa_cache
