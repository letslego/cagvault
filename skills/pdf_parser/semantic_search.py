"""
Semantic Search for Sections using TF-IDF and Cosine Similarity

Complements keyword search with semantic similarity matching to find
sections that are conceptually similar even if they don't share exact keywords.
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


class SemanticSearchEngine:
    """Semantic search using TF-IDF vectors and cosine similarity."""
    
    def __init__(self):
        """Initialize semantic search engine."""
        self.section_contents: Dict[str, str] = {}  # section_id -> content
        self.section_ids_ordered: List[str] = []  # Preserve order for matrix
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix: Optional[np.ndarray] = None
        self.sklearn_available = SKLEARN_AVAILABLE
    
    def add_section(self, section_id: str, content: str) -> None:
        """Add section content to semantic index.
        
        Args:
            section_id: Unique section identifier
            content: Full section content text
        """
        self.section_contents[section_id] = content
        if section_id not in self.section_ids_ordered:
            self.section_ids_ordered.append(section_id)
        # Reset vectorizer to rebuild with new content
        self.rebuild_index()
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for semantically similar sections.
        
        Uses TF-IDF vectorization and cosine similarity to find sections
        conceptually related to the query, even without exact keyword matches.
        
        Args:
            query: Search query text
            top_k: Number of top results to return
            
        Returns:
            List of section results ranked by semantic similarity score
        """
        if not self.sklearn_available or not self.section_contents:
            return []
        
        try:
            # Build TF-IDF matrix if not already done
            if self.vectorizer is None and len(self.section_contents) > 0:
                contents = [self.section_contents[sid] for sid in self.section_ids_ordered]
                self.vectorizer = TfidfVectorizer(
                    max_features=1000,
                    stop_words='english',
                    min_df=1,
                    max_df=0.95
                )
                self.tfidf_matrix = self.vectorizer.fit_transform(contents)
            
            if self.vectorizer is None or self.tfidf_matrix is None:
                return []
            
            # Vectorize query
            query_vec = self.vectorizer.transform([query])
            
            # Compute cosine similarity with all sections
            similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
            
            # Get top-k results with non-zero similarity
            top_indices = np.argsort(similarities)[::-1][:top_k]
            results = []
            for idx in top_indices:
                score = float(similarities[idx])
                if score > 0:  # Only include results with some similarity
                    section_id = self.section_ids_ordered[idx]
                    results.append({
                        'section_id': section_id,
                        'relevance_score': score,
                        'content_preview': self.section_contents[section_id][:300]
                    })
            
            return results
        except Exception as e:
            logger.warning(f"Semantic search failed: {e}")
            return []
    
    def rebuild_index(self) -> None:
        """Rebuild TF-IDF matrix when sections change."""
        self.vectorizer = None
        self.tfidf_matrix = None
    
    def clear(self) -> None:
        """Clear all indexed sections."""
        self.section_contents.clear()
        self.section_ids_ordered.clear()
        self.rebuild_index()
