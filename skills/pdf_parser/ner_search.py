"""
Enhanced PDF Parser with NER and Full-Text Search

Adds:
- Named Entity Recognition (NER) for entities in sections
- Full-text search across sections and subsections
- Search results with ranking and relevance scores
- Entity indexing by type and document
"""

import re
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Extracted named entity."""
    text: str
    type: str  # PERSON, ORG, LOCATION, MONEY, DATE, AGREEMENT, PARTY, etc.
    section_id: str
    document_id: str
    position: int
    confidence: float = 1.0
    
    def __hash__(self):
        return hash((self.text.lower(), self.type, self.section_id))


@dataclass
class SearchResult:
    """Full-text search result."""
    section_id: str
    document_id: str
    section_title: str
    relevance_score: float
    match_count: int
    matched_text: str
    context_before: str
    context_after: str
    content: str


class NamedEntityRecognizer:
    """Simple NER for financial documents."""
    
    # Patterns for entities common in credit agreements
    PATTERNS = {
        'MONEY': [
            r'\$\s*[\d,]+(?:\.\d{2})?',
            r'[\d,]+\s*(?:million|billion|thousand|dollars?)',
        ],
        'DATE': [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}',
        ],
        'PARTY': [
            r'(?:Borrower|Lender|Agent|Guarantor|Obligor|Creditor)',
        ],
        'AGREEMENT': [
            r'(?:Credit Agreement|Loan Agreement|Security Agreement|Guarantee|Covenant)',
        ],
        'PERCENTAGE': [
            r'\d+(?:\.\d+)?\s*%',
        ],
        'RATIO': [
            r'[A-Z]+\s+to\s+[A-Z]+\s+(?:ratio|test)',
        ]
    }
    
    def extract_entities(self, text: str, section_id: str, document_id: str) -> List[Entity]:
        """Extract entities from text."""
        entities = []
        
        for entity_type, patterns in self.PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    entity = Entity(
                        text=match.group(),
                        type=entity_type,
                        section_id=section_id,
                        document_id=document_id,
                        position=match.start()
                    )
                    entities.append(entity)
        
        return entities


class FullTextSearchEngine:
    """Full-text search for sections."""
    
    def __init__(self):
        """Initialize search engine."""
        self.index: Dict[str, List[str]] = {}  # term -> [section_ids]
        self.entities_by_type: Dict[str, List[Entity]] = {}
        self.entities_by_document: Dict[str, List[Entity]] = {}
    
    def add_section(self, section_id: str, content: str) -> None:
        """Index section content for full-text search."""
        # Tokenize content
        tokens = self._tokenize(content)
        
        # Index each token
        for token in tokens:
            if token not in self.index:
                self.index[token] = []
            if section_id not in self.index[token]:
                self.index[token].append(section_id)
    
    def add_entities(self, entities: List[Entity]) -> None:
        """Index entities."""
        for entity in entities:
            # Index by type
            if entity.type not in self.entities_by_type:
                self.entities_by_type[entity.type] = []
            self.entities_by_type[entity.type].append(entity)
            
            # Index by document
            if entity.document_id not in self.entities_by_document:
                self.entities_by_document[entity.document_id] = []
            self.entities_by_document[entity.document_id].append(entity)
    
    def search(self, query: str, section_contents: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Search for query in indexed sections.
        
        Args:
            query: Search query
            section_contents: Dict of section_id -> content
            
        Returns:
            List of search results ranked by relevance
        """
        tokens = self._tokenize(query)
        matching_sections = {}
        
        # Find sections containing query terms
        for token in tokens:
            if token in self.index:
                for section_id in self.index[token]:
                    if section_id not in matching_sections:
                        matching_sections[section_id] = 0
                    matching_sections[section_id] += 1
        
        # Rank by relevance
        results = []
        for section_id, match_count in sorted(matching_sections.items(), 
                                               key=lambda x: x[1], reverse=True):
            if section_id in section_contents:
                content = section_contents[section_id]
                relevance = match_count / len(tokens) if tokens else 0
                results.append({
                    'section_id': section_id,
                    'match_count': match_count,
                    'relevance_score': min(relevance, 1.0),
                    'content_preview': content[:500]
                })
        
        return results
    
    def search_entities(self, document_id: str, entity_type: Optional[str] = None) -> List[Entity]:
        """
        Search for entities in document.
        
        Args:
            document_id: Document to search
            entity_type: Optional entity type filter
            
        Returns:
            List of entities matching criteria
        """
        entities = self.entities_by_document.get(document_id, [])
        
        if entity_type:
            entities = [e for e in entities if e.type == entity_type]
        
        return entities
    
    def get_entities_by_type(self, entity_type: str) -> List[Entity]:
        """Get all entities of a specific type."""
        return self.entities_by_type.get(entity_type, [])
    
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple tokenization."""
        # Convert to lowercase, remove punctuation, split on whitespace
        text = text.lower()
        # Remove special characters but keep some for financial terms
        text = re.sub(r'[^\w\s%-]', ' ', text)
        tokens = text.split()
        # Filter out very short tokens and common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        return [t for t in tokens if len(t) > 2 and t not in stop_words]


class EnhancedSearchableParser:
    """Enhanced parser with search and NER capabilities."""
    
    def __init__(self):
        """Initialize enhanced parser."""
        self.ner = NamedEntityRecognizer()
        self.search_engine = FullTextSearchEngine()
        self.section_entities: Dict[str, List[Entity]] = {}
        self.section_contents: Dict[str, str] = {}
    
    def index_section(self, section_id: str, content: str, document_id: str) -> Dict[str, Any]:
        """
        Index a section with search and NER.
        
        Args:
            section_id: Section ID
            content: Section content
            document_id: Document ID
            
        Returns:
            Indexing summary with entities found
        """
        # Add to search index
        self.search_engine.add_section(section_id, content)
        self.section_contents[section_id] = content
        
        # Extract entities
        entities = self.ner.extract_entities(content, section_id, document_id)
        self.section_entities[section_id] = entities
        self.search_engine.add_entities(entities)
        
        return {
            'section_id': section_id,
            'indexed_tokens': len(self.search_engine._tokenize(content)),
            'entities_found': len(entities),
            'entity_types': list(set(e.type for e in entities)),
            'top_entities': [
                {'text': e.text, 'type': e.type}
                for e in entities[:5]
            ]
        }
    
    def full_text_search(self, query: str) -> List[Dict[str, Any]]:
        """
        Full-text search across all sections.
        
        Args:
            query: Search query
            
        Returns:
            Ranked search results
        """
        return self.search_engine.search(query, self.section_contents)
    
    def search_by_entity_type(self, document_id: str, entity_type: str) -> List[Dict[str, Any]]:
        """
        Search for entities of a specific type in document.
        
        Args:
            document_id: Document to search
            entity_type: Entity type (MONEY, DATE, PARTY, etc.)
            
        Returns:
            List of found entities with context
        """
        entities = self.search_engine.search_entities(document_id, entity_type)
        
        return [
            {
                'text': e.text,
                'type': e.type,
                'section_id': e.section_id,
                'position': e.position,
                'in_section': self.section_contents.get(e.section_id, '')[:100]
            }
            for e in entities
        ]
    
    def get_section_entities(self, section_id: str) -> List[Dict[str, Any]]:
        """Get entities in a specific section."""
        entities = self.section_entities.get(section_id, [])
        return [
            {
                'text': e.text,
                'type': e.type,
                'position': e.position
            }
            for e in entities
        ]


# Global instance
_searchable_parser: Optional[EnhancedSearchableParser] = None


def get_searchable_parser() -> EnhancedSearchableParser:
    """Get or create searchable parser instance."""
    global _searchable_parser
    if _searchable_parser is None:
        _searchable_parser = EnhancedSearchableParser()
    return _searchable_parser


# Claude Skill functions
def full_text_search_document(document_id: str, query: str) -> Dict[str, Any]:
    """
    Claude Skill: Full-text search across document sections.
    
    Args:
        document_id: Document ID to search
        query: Search query (can be multiple terms)
        
    Returns:
        Ranked search results with relevance scores
    """
    parser = get_searchable_parser()
    results = parser.full_text_search(query)
    
    return {
        'query': query,
        'document_id': document_id,
        'results_found': len(results),
        'results': results[:10]  # Top 10 results
    }


def extract_entities_from_document(document_id: str, entity_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Claude Skill: Extract entities from document.
    
    Args:
        document_id: Document ID
        entity_type: Optional entity type filter (MONEY, DATE, PARTY, AGREEMENT, etc.)
        
    Returns:
        List of entities found
    """
    parser = get_searchable_parser()
    
    if entity_type:
        entities = parser.search_engine.search_entities(document_id, entity_type)
        return {
            'document_id': document_id,
            'entity_type': entity_type,
            'count': len(entities),
            'entities': [
                {'text': e.text, 'section_id': e.section_id, 'position': e.position}
                for e in entities[:20]
            ]
        }
    else:
        entities = parser.search_engine.entities_by_document.get(document_id, [])
        entity_types = {}
        for entity in entities:
            if entity.type not in entity_types:
                entity_types[entity.type] = []
            entity_types[entity.type].append({
                'text': entity.text,
                'section_id': entity.section_id
            })
        
        return {
            'document_id': document_id,
            'total_entities': len(entities),
            'entity_types': entity_types
        }


def get_section_entities(section_id: str) -> Dict[str, Any]:
    """
    Claude Skill: Get entities in a section.
    
    Args:
        section_id: Section ID
        
    Returns:
        Entities found in section
    """
    parser = get_searchable_parser()
    entities = parser.get_section_entities(section_id)
    
    # Group by type
    by_type = {}
    for entity in entities:
        ent_type = entity['type']
        if ent_type not in by_type:
            by_type[ent_type] = []
        by_type[ent_type].append(entity)
    
    return {
        'section_id': section_id,
        'total_entities': len(entities),
        'by_type': by_type,
        'entities': entities
    }
