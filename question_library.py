"""
Question Library Management

Stores and manages a library of common questions for documents.
Provides autocomplete suggestions and question recommendation.
"""

import json
import logging
from typing import Dict, Any, Optional, List, Set
from datetime import datetime
import redis

logger = logging.getLogger(__name__)


class QuestionLibraryManager:
    """Manages question library in Redis with autocomplete support."""
    
    # Redis key patterns
    QUESTION_LIBRARY_KEY = "question_library"  # Sorted set: question -> usage_count
    GLOBAL_LIBRARY_KEY = "global_question_library"  # Global questions across all docs
    DOC_LIBRARY_KEY_PREFIX = "doc_library"  # Per-document library: doc_id -> questions
    QUESTION_CATEGORIES_KEY = "question_categories"  # Map: question -> category
    CATEGORY_INDEX_KEY = "category_index"  # Index: category -> [questions]
    QUESTION_METADATA_KEY_PREFIX = "question_meta"  # Per-question metadata
    
    # Predefined question categories
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
        "other": "Other"
    }
    
    def __init__(self, redis_client: redis.Redis = None):
        """Initialize question library manager.
        
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
            logger.info("Connected to Redis for question library")
            self._initialize_default_library()
        except Exception as e:
            logger.warning(f"Could not connect to Redis: {e}")
            self.redis = None
    
    def _initialize_default_library(self) -> None:
        """Initialize with default common questions if library is empty."""
        if not self.redis:
            return
        
        if self.redis.exists(self.GLOBAL_LIBRARY_KEY):
            return  # Already initialized
        
        default_questions = {
            # Parties & Obligations
            "What are the key parties to this agreement?": "parties",
            "Who is the borrower?": "parties",
            "Who is the administrative agent?": "parties",
            "What are the borrower's obligations?": "parties",
            "What are the guarantor's obligations?": "parties",
            "Who can be a lender under this agreement?": "parties",
            
            # Financial Covenants & Metrics
            "What are the financial covenants?": "financial",
            "What is the leverage ratio requirement?": "financial",
            "What is the interest coverage ratio?": "financial",
            "What is the debt service coverage ratio?": "financial",
            "What is the minimum liquidity requirement?": "financial",
            "What is the commitment amount?": "financial",
            "What are the fees and charges?": "financial",
            "What is the total net leverage ratio?": "financial",
            "What is the secured net leverage ratio?": "financial",
            "What is the fixed charge coverage ratio?": "financial",
            
            # Definitions
            "What is EBITDA and how is it defined?": "definitions",
            "What is Consolidated EBITDA?": "definitions",
            "What is Material Adverse Effect?": "definitions",
            "What qualifies as a Permitted Acquisition?": "definitions",
            "What is Excess Cash Flow?": "definitions",
            "What are Permitted Liens?": "definitions",
            "What is a Change of Control?": "definitions",
            "What constitutes Available Amount?": "definitions",
            
            # Events of Default & Conditions
            "What triggers an event of default?": "conditions",
            "What are the conditions precedent to borrowing?": "conditions",
            "What happens if the borrower defaults?": "conditions",
            "What is a Material Adverse Effect event?": "conditions",
            "What are the conditions precedent to closing?": "conditions",
            "When does a cross-default occur?": "conditions",
            "What triggers mandatory prepayment?": "conditions",
            
            # Interest & Calculations
            "What is the interest rate and how is it calculated?": "calculations",
            "What is the applicable margin?": "calculations",
            "How is SOFR calculated?": "calculations",
            "What is the interest rate floor?": "calculations",
            "How are pro forma adjustments calculated?": "calculations",
            "How is Excess Cash Flow calculated?": "calculations",
            
            # Debt Restrictions & Baskets
            "What are the borrower's restrictions on debt?": "restrictions",
            "Can the borrower incur additional debt?": "restrictions",
            "What debt is permitted under the agreement?": "restrictions",
            "Can the borrower incur secured debt?": "restrictions",
            "What is the general debt basket?": "restrictions",
            "What is the ratio debt basket?": "restrictions",
            "Can the borrower make acquisitions?": "restrictions",
            "What are the restrictions on investments?": "restrictions",
            "What are the restrictions on dividends?": "restrictions",
            "Can the borrower make restricted payments?": "restrictions",
            
            # Covenants & Compliance
            "What covenants must the borrower maintain?": "restrictions",
            "What are the affirmative covenants?": "restrictions",
            "What are the negative covenants?": "restrictions",
            "What financial reports must be delivered?": "restrictions",
            "What are the insurance requirements?": "restrictions",
            "What are the representations and warranties?": "restrictions",
            
            # Liens & Security
            "What liens are permitted?": "restrictions",
            "Can the borrower grant security interests?": "restrictions",
            "What collateral secures the loan?": "restrictions",
            "What is the lien basket?": "restrictions",
            
            # Procedures & Mechanics
            "How are prepayments handled?": "procedures",
            "How does the borrower request a loan?": "procedures",
            "What is the borrowing procedure?": "procedures",
            "How are payments made?": "procedures",
            "How can the agreement be amended?": "procedures",
            "What is the voting requirement for amendments?": "procedures",
            "How are notices delivered?": "procedures",
            "Can the borrower assign its rights?": "procedures",
            
            # Timeline & Maturity
            "When is the loan due?": "timeline",
            "What is the maturity date?": "timeline",
            "When must financial statements be delivered?": "timeline",
            "What are the key dates in the agreement?": "timeline",
            "When do the commitments terminate?": "timeline",
            
            # Rights & Remedies
            "What are the lender's remedies upon default?": "rights",
            "Can the lender accelerate the loan?": "rights",
            "What are the borrower's rights to prepay?": "rights",
            "Can the borrower cure a covenant breach?": "rights",
            "What are the indemnification provisions?": "rights",
            "What are the agent's rights and powers?": "rights",
            
            # Other Common Questions
            "What is the governing law?": "other",
            "What is the choice of forum?": "other",
            "Are there any guarantees?": "other",
            "What are the successor and assignment provisions?": "other",
            "Can a lender assign to affiliates without consent?": "rights",
            "Is there a judgment event of default?": "conditions",
            "Can the borrower create a superpriority lender group?": "restrictions",
            "Can obligors cure default after acceleration?": "rights",
            "Can the revolving loan be extended?": "timeline",
            "Can the swing line loan be prepaid?": "procedures",
            "What are the notice requirements for borrowing?": "procedures",
            "Can borrowing continue if there is a default?": "conditions",
            "Can lenders transfer interests to securities facilities?": "rights",
            "Can investments be made to unrestricted subsidiaries?": "restrictions",
            "Can debt fund acquisitions using restricted payments?": "restrictions",
            "What is the cap on new indebtedness?": "restrictions",
            "What are the prepayment rights?": "procedures",
            "What are the agent indemnities?": "rights",
            "What are the continuing deliverables?": "restrictions",
            "Can an agent be replaced?": "rights",
            "What steps must a lender take to assign rights?": "procedures",
            "What are restrictions during default vs event of default?": "conditions",
            "What are the acquisition indebtedness thresholds?": "restrictions",
            "What is the impact of cross-defaults?": "conditions",
            "What is the RCF springer covenant threshold?": "financial",
            "What is the threshold for acquisition funding?": "restrictions",
            "What are the limitations in assignment and transfer?": "restrictions",
            "Which defaults have materiality qualifiers?": "conditions",
            "What rights does the agent have without lender consent?": "rights",
            "What steps must a lender take to enter into a participation?": "procedures",
            "Is the borrower in default for missed interest payments?": "conditions",
            "What is the cure period for interest payment defaults?": "conditions",
            "What happens if the interest rate benchmark changes?": "calculations",
            "What is the fallback rate mechanism?": "calculations",
            "Can lenders enter into participations?": "procedures",
            "What is the participation process?": "procedures",
            "Can the borrower block a participation?": "restrictions",
            "Can the lender accelerate on change of control?": "conditions",
            "How is change of control defined?": "definitions",
            "Is a prepayment penalty applicable?": "procedures",
            "When does the penalty period expire?": "timeline",
            "Will the lender consent to asset sales?": "restrictions",
            "Who approves material asset sales?": "restrictions",
            "Where must sale proceeds be deposited?": "procedures",
            "Can the borrower incur subordinated liens?": "restrictions",
            "What are the conditions for subordinated liens?": "restrictions",
            "Who must agree to intercreditor terms?": "restrictions",
            "Are indemnity provisions included?": "rights",
            "What events are covered by indemnities?": "rights",
            "How does the borrower notify of indemnity claims?": "procedures",
            "Is material adverse effect defined?": "definitions",
            "What factors are considered in MAE definition?": "definitions",
            "Why is the MAE definition critical?": "definitions",
            "Can the borrower issue letters of credit?": "restrictions",
            "When must letters of credit be cash-collateralized?": "procedures",
            "How is the letter of credit fee determined?": "calculations",
            "Is the loan recourse or non-recourse?": "definitions",
            "Who are the guarantors?": "parties",
            "What are the limitations on guarantees?": "rights",
            "What are the agent bank obligations to obligors?": "rights",
            "What are the agent bank obligations to finance parties?": "rights",
            "Is the agent indemnified for all obligations?": "rights",
            "What are the gaps in agent indemnification?": "rights",
            "What are the drivers in the leverage ratio?": "financial",
            "What are the leverage ratio numerator components?": "financial",
            "What are the leverage ratio denominator components?": "financial",
            "Are there unusual omissions from the leverage ratio?": "financial",
            "Is there anything unusual in the EBITDA definition?": "definitions",
        }
        
        for question, category in default_questions.items():
            self.add_question(question, category=category, is_default=True)
    
    def add_question(
        self,
        question: str,
        doc_ids: Optional[Set[str]] = None,
        category: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        is_default: bool = False
    ) -> bool:
        """Add a question to the library.
        
        Args:
            question: Question text
            doc_ids: Optional set of document IDs this question relates to
            category: Question category (from CATEGORIES keys)
            metadata: Additional metadata
            is_default: Whether this is a default question
            
        Returns:
            True if added successfully
        """
        if not self.redis or not question:
            return False
        
        try:
            question_lower = question.strip()
            
            # Add to global library (sorted set by usage count, default 0)
            self.redis.zadd(self.GLOBAL_LIBRARY_KEY, {question_lower: 0})
            
            # Add to per-document library if doc_ids provided
            if doc_ids:
                for doc_id in doc_ids:
                    doc_lib_key = f"{self.DOC_LIBRARY_KEY_PREFIX}:{doc_id}"
                    self.redis.sadd(doc_lib_key, question_lower)
                    self.redis.expire(doc_lib_key, 86400 * 7)  # 7 day TTL
            
            # Store category
            if category and category in self.CATEGORIES:
                self.redis.hset(self.QUESTION_CATEGORIES_KEY, question_lower, category)
                
                # Index by category
                cat_index_key = f"{self.CATEGORY_INDEX_KEY}:{category}"
                self.redis.sadd(cat_index_key, question_lower)
                self.redis.expire(cat_index_key, 86400 * 7)
            
            # Store metadata
            meta = metadata or {}
            meta["created_at"] = datetime.utcnow().isoformat()
            meta["is_default"] = is_default
            meta["usage_count"] = 0
            
            meta_key = f"{self.QUESTION_METADATA_KEY_PREFIX}:{question_lower}"
            self.redis.hset(meta_key, mapping={
                "metadata": json.dumps(meta),
                "doc_ids": json.dumps(list(doc_ids)) if doc_ids else "[]"
            })
            self.redis.expire(meta_key, 86400 * 7)
            
            logger.debug(f"Added question to library: {question_lower}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to add question: {e}")
            return False
    
    def increment_usage(self, question: str) -> bool:
        """Increment usage count for a question (tracks popularity).
        
        Args:
            question: Question text
            
        Returns:
            True if incremented successfully
        """
        if not self.redis or not question:
            return False
        
        try:
            question_lower = question.strip()
            
            # Increment in global library
            self.redis.zincrby(self.GLOBAL_LIBRARY_KEY, 1, question_lower)
            
            # Update metadata
            meta_key = f"{self.QUESTION_METADATA_KEY_PREFIX}:{question_lower}"
            data = self.redis.hgetall(meta_key)
            if data:
                meta = json.loads(data.get("metadata", "{}"))
                meta["usage_count"] = meta.get("usage_count", 0) + 1
                meta["last_used"] = datetime.utcnow().isoformat()
                self.redis.hset(meta_key, "metadata", json.dumps(meta))
            
            return True
        except Exception as e:
            logger.warning(f"Failed to increment usage: {e}")
            return False
    
    def get_autocomplete_suggestions(
        self,
        query: str,
        doc_id: Optional[str] = None,
        max_results: int = 10,
        include_global: bool = True
    ) -> List[Dict[str, Any]]:
        """Get autocomplete suggestions for a query.
        
        Args:
            query: Partial question text
            doc_id: Filter to document-specific questions
            max_results: Maximum suggestions to return
            include_global: Whether to include global questions
            
        Returns:
            List of matching questions with metadata
        """
        if not self.redis or not query:
            return []
        
        try:
            query_lower = query.lower().strip()
            suggestions = []
            seen = set()
            
            # Get document-specific questions if doc_id provided
            if doc_id:
                doc_lib_key = f"{self.DOC_LIBRARY_KEY_PREFIX}:{doc_id}"
                doc_questions = self.redis.smembers(doc_lib_key)
                
                for q in doc_questions:
                    if query_lower in q.lower() and q not in seen:
                        suggestions.append(q)
                        seen.add(q)
                        if len(suggestions) >= max_results:
                            break
            
            # Get global questions
            if include_global and len(suggestions) < max_results:
                all_questions = self.redis.zrange(
                    self.GLOBAL_LIBRARY_KEY, 0, -1, withscores=True
                )
                
                # Sort by score (usage count) descending
                all_questions = sorted(all_questions, key=lambda x: x[1], reverse=True)
                
                for q, score in all_questions:
                    if query_lower in q.lower() and q not in seen:
                        suggestions.append(q)
                        seen.add(q)
                        if len(suggestions) >= max_results:
                            break
            
            # Enrich with metadata
            enriched = []
            for q in suggestions:
                meta_key = f"{self.QUESTION_METADATA_KEY_PREFIX}:{q}"
                data = self.redis.hgetall(meta_key)
                category = self.redis.hget(self.QUESTION_CATEGORIES_KEY, q) or "other"
                
                enriched.append({
                    "question": q,
                    "category": category,
                    "category_label": self.CATEGORIES.get(category, "Other"),
                    "metadata": json.loads(data.get("metadata", "{}")) if data else {}
                })
            
            return enriched
        
        except Exception as e:
            logger.warning(f"Failed to get autocomplete suggestions: {e}")
            return []
    
    def get_popular_questions(
        self,
        doc_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get most popular questions by usage.
        
        Args:
            doc_id: Filter to document-specific questions
            limit: Number of questions to return
            
        Returns:
            List of popular questions
        """
        if not self.redis:
            return []
        
        try:
            if doc_id:
                doc_lib_key = f"{self.DOC_LIBRARY_KEY_PREFIX}:{doc_id}"
                questions = self.redis.smembers(doc_lib_key)
                
                # Get usage scores for each
                scores = []
                for q in questions:
                    score = self.redis.zscore(self.GLOBAL_LIBRARY_KEY, q) or 0
                    scores.append((q, score))
                
                scores.sort(key=lambda x: x[1], reverse=True)
                popular = [q for q, _ in scores[:limit]]
            else:
                popular = self.redis.zrange(
                    self.GLOBAL_LIBRARY_KEY, -limit, -1, withscores=False
                )
                popular.reverse()  # Most popular first
            
            # Enrich with metadata
            enriched = []
            for q in popular:
                meta_key = f"{self.QUESTION_METADATA_KEY_PREFIX}:{q}"
                data = self.redis.hgetall(meta_key)
                category = self.redis.hget(self.QUESTION_CATEGORIES_KEY, q) or "other"
                
                enriched.append({
                    "question": q,
                    "category": category,
                    "category_label": self.CATEGORIES.get(category, "Other"),
                    "metadata": json.loads(data.get("metadata", "{}")) if data else {}
                })
            
            return enriched
        
        except Exception as e:
            logger.warning(f"Failed to get popular questions: {e}")
            return []
    
    def get_questions_by_category(
        self,
        category: str,
        doc_id: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get questions in a specific category.
        
        Args:
            category: Category key
            doc_id: Filter to document-specific questions
            limit: Maximum results
            
        Returns:
            List of questions in category
        """
        if not self.redis or category not in self.CATEGORIES:
            return []
        
        try:
            cat_index_key = f"{self.CATEGORY_INDEX_KEY}:{category}"
            questions = self.redis.smembers(cat_index_key)
            
            # Filter by doc if provided
            if doc_id:
                doc_lib_key = f"{self.DOC_LIBRARY_KEY_PREFIX}:{doc_id}"
                doc_questions = self.redis.smembers(doc_lib_key)
                questions = questions.intersection(doc_questions)
            
            questions = list(questions)[:limit]
            
            # Enrich with metadata
            enriched = []
            for q in questions:
                meta_key = f"{self.QUESTION_METADATA_KEY_PREFIX}:{q}"
                data = self.redis.hgetall(meta_key)
                
                enriched.append({
                    "question": q,
                    "category": category,
                    "category_label": self.CATEGORIES.get(category, "Other"),
                    "metadata": json.loads(data.get("metadata", "{}")) if data else {}
                })
            
            return enriched
        
        except Exception as e:
            logger.warning(f"Failed to get questions by category: {e}")
            return []
    
    def get_all_questions(self, doc_id: Optional[str] = None) -> List[str]:
        """Get all questions in library.
        
        Args:
            doc_id: Filter to document-specific questions
            
        Returns:
            List of all questions
        """
        if not self.redis:
            return []
        
        try:
            if doc_id:
                doc_lib_key = f"{self.DOC_LIBRARY_KEY_PREFIX}:{doc_id}"
                return list(self.redis.smembers(doc_lib_key))
            else:
                return self.redis.zrange(self.GLOBAL_LIBRARY_KEY, 0, -1)
        except Exception as e:
            logger.warning(f"Failed to get all questions: {e}")
            return []
    
    def remove_question(self, question: str, doc_id: Optional[str] = None) -> bool:
        """Remove a question from library.
        
        Args:
            question: Question text
            doc_id: Only remove from specific document (None = remove globally)
            
        Returns:
            True if removed successfully
        """
        if not self.redis or not question:
            return False
        
        try:
            question_lower = question.strip()
            
            if doc_id:
                # Remove from document library only
                doc_lib_key = f"{self.DOC_LIBRARY_KEY_PREFIX}:{doc_id}"
                self.redis.srem(doc_lib_key, question_lower)
            else:
                # Remove globally
                self.redis.zrem(self.GLOBAL_LIBRARY_KEY, question_lower)
                self.redis.hdel(self.QUESTION_CATEGORIES_KEY, question_lower)
                
                # Remove from category indexes
                category = self.redis.hget(self.QUESTION_CATEGORIES_KEY, question_lower)
                if category:
                    cat_index_key = f"{self.CATEGORY_INDEX_KEY}:{category}"
                    self.redis.srem(cat_index_key, question_lower)
                
                # Remove metadata
                meta_key = f"{self.QUESTION_METADATA_KEY_PREFIX}:{question_lower}"
                self.redis.delete(meta_key)
            
            logger.debug(f"Removed question: {question_lower}")
            return True
        
        except Exception as e:
            logger.warning(f"Failed to remove question: {e}")
            return False
    
    def clear_library(self) -> bool:
        """Clear entire question library (use with caution)."""
        if not self.redis:
            return False
        
        try:
            keys_to_delete = self.redis.keys(f"{self.QUESTION_METADATA_KEY_PREFIX}:*")
            keys_to_delete.extend(self.redis.keys(f"{self.CATEGORY_INDEX_KEY}:*"))
            keys_to_delete.extend(self.redis.keys(f"{self.DOC_LIBRARY_KEY_PREFIX}:*"))
            keys_to_delete.extend([
                self.GLOBAL_LIBRARY_KEY,
                self.QUESTION_CATEGORIES_KEY
            ])
            
            if keys_to_delete:
                self.redis.delete(*keys_to_delete)
            
            logger.info("Cleared entire question library")
            return True
        except Exception as e:
            logger.warning(f"Failed to clear library: {e}")
            return False


# Singleton instance
_question_library = None


def get_question_library() -> QuestionLibraryManager:
    """Get or create question library manager singleton."""
    global _question_library
    if _question_library is None:
        _question_library = QuestionLibraryManager()
    return _question_library
