"""
Enhanced PDF Parser with Section Memory and Hierarchical Metadata

Stores sections and subsections in memory with full document and section references.
Includes metadata extraction, full-text search, and named entity recognition.
"""

import hashlib
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import logging
import re

from .ner_search import NamedEntityRecognizer, FullTextSearchEngine, EnhancedSearchableParser
from .credit_analyst_prompt import get_credit_analyst

logger = logging.getLogger(__name__)


@dataclass
class SectionMetadata:
    """Metadata for a section or subsection."""
    
    title: str
    level: int  # 1 = H1, 2 = H2, etc.
    start_line: int
    end_line: int
    content_length: int
    word_count: int
    has_code: bool
    has_tables: bool
    subsection_count: int
    parent_id: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    page_estimate: int = 1  # Estimated starting page
    page_range: Optional[str] = None  # e.g., "3-7" for multi-page sections
    start_page: int = 1  # Actual start page
    end_page: int = 1  # Actual end page
    section_type: Optional[str] = None  # Section classification (DEFINITIONS, COVENANT, DEFAULT, etc.)
    importance_score: float = 0.5  # Importance score 0-1 from credit analyst
    typical_dependencies: List[str] = field(default_factory=list)  # Related section types
    
    def __post_init__(self):
        """Generate unique ID based on hierarchy."""
        self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique ID for this section."""
        combined = f"{self.title}_{self.level}_{self.start_line}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]


@dataclass
class Section:
    """A section or subsection with content and metadata."""
    metadata: SectionMetadata
    content: str
    document_id: str
    subsections: List['Section'] = field(default_factory=list)
    
    @property
    def full_id(self) -> str:
        """Get full hierarchical ID path."""
        if self.metadata.parent_id:
            return f"{self.document_id}#{self.metadata.parent_id}#{self.metadata.id}"
        return f"{self.document_id}#{self.metadata.id}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert section to dictionary."""
        return {
            "id": self.metadata.id,
            "full_id": self.full_id,
            "title": self.metadata.title,
            "level": self.metadata.level,
            "content_length": self.metadata.content_length,
            "word_count": self.metadata.word_count,
            "has_code": self.metadata.has_code,
            "has_tables": self.metadata.has_tables,
            "subsection_count": self.metadata.subsection_count,
            "subsections": [s.to_dict() for s in self.subsections],
            "metadata": asdict(self.metadata)
        }


class SectionMemoryStore:
    """In-memory store for document sections with hierarchical organization."""
    
    def __init__(self):
        """Initialize the section memory store."""
        # Structure: {document_id: {section_id: Section}}
        self.documents: Dict[str, Dict[str, Section]] = {}
        # Structure: {document_id: [Section]} - flat list for quick access
        self.section_index: Dict[str, List[Section]] = {}
        # Metadata index: {full_section_id: SectionMetadata}
        self.metadata_index: Dict[str, SectionMetadata] = {}
        logger.info("SectionMemoryStore initialized")
    
    def add_section(
        self,
        document_id: str,
        section: Section,
        parent_id: Optional[str] = None
    ) -> str:
        """
        Add a section to the store.
        
        Args:
            document_id: ID of parent document
            section: Section object to store
            parent_id: ID of parent section (for subsections)
            
        Returns:
            Full section ID
        """
        if document_id not in self.documents:
            self.documents[document_id] = {}
            self.section_index[document_id] = []
        
        # Set parent ID if provided
        if parent_id:
            section.metadata.parent_id = parent_id
        
        # Store section
        section_id = section.metadata.id
        self.documents[document_id][section_id] = section
        self.section_index[document_id].append(section)
        
        # Index metadata
        full_id = section.full_id
        self.metadata_index[full_id] = section.metadata
        
        logger.info(f"Added section '{section.metadata.title}' to {document_id}")
        return full_id
    
    def get_section(self, full_section_id: str) -> Optional[Section]:
        """Get a section by its full ID."""
        # Parse full_id format: document_id#section_id or document_id#parent_id#section_id
        parts = full_section_id.split("#")
        if len(parts) < 2:
            return None
        
        document_id = parts[0]
        if document_id not in self.documents:
            return None
        
        section_id = parts[-1]
        return self.documents[document_id].get(section_id)
    
    def get_document_sections(self, document_id: str) -> List[Section]:
        """Get all sections for a document."""
        return self.section_index.get(document_id, [])
    
    def get_section_metadata(self, full_section_id: str) -> Optional[SectionMetadata]:
        """Get metadata for a section."""
        return self.metadata_index.get(full_section_id)
    
    def get_document_hierarchy(self, document_id: str) -> List[Dict[str, Any]]:
        """Get hierarchical view of document sections."""
        sections = self.section_index.get(document_id, [])
        return [self._build_hierarchy(s) for s in sections]
    
    @staticmethod
    def _build_hierarchy(section: Section) -> Dict[str, Any]:
        """Build hierarchical representation of section."""
        return {
            "id": section.metadata.id,
            "title": section.metadata.title,
            "level": section.metadata.level,
            "word_count": section.metadata.word_count,
            "subsections": [
                SectionMemoryStore._build_hierarchy(s)
                for s in section.subsections
            ]
        }
    
    def search_sections(
        self,
        document_id: str,
        query: str,
        search_titles: bool = True,
        search_content: bool = True
    ) -> List[Tuple[Section, List[int]]]:
        """
        Search sections by title or content.
        
        Args:
            document_id: Document to search in
            query: Search term
            search_titles: Search section titles
            search_content: Search section content
            
        Returns:
            List of (Section, match_positions) tuples
        """
        sections = self.section_index.get(document_id, [])
        results = []
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        
        for section in sections:
            matches = []
            
            if search_titles and pattern.search(section.metadata.title):
                matches.append(-1)  # Special marker for title match
            
            if search_content:
                for i, line in enumerate(section.content.split("\n")):
                    if pattern.search(line):
                        matches.append(i)
            
            if matches:
                results.append((section, matches))
        
        return results
    
    def get_statistics(self, document_id: str) -> Dict[str, Any]:
        """Get statistics about document sections."""
        sections = self.section_index.get(document_id, [])
        
        if not sections:
            return {}
        
        total_words = sum(s.metadata.word_count for s in sections)
        total_content = sum(s.metadata.content_length for s in sections)
        sections_with_code = sum(1 for s in sections if s.metadata.has_code)
        sections_with_tables = sum(1 for s in sections if s.metadata.has_tables)
        
        return {
            "total_sections": len(sections),
            "total_subsections": sum(len(s.subsections) for s in sections),
            "total_words": total_words,
            "total_content": total_content,
            "average_section_length": total_words // len(sections) if sections else 0,
            "sections_with_code": sections_with_code,
            "sections_with_tables": sections_with_tables,
        }


class EnhancedPDFParserSkill:
    """Enhanced PDF Parser with section memory, search, and NER."""
    
    def __init__(self):
        """Initialize enhanced parser with section memory and search."""
        from skills.pdf_parser.pdf_parser import PDFParserSkill
        self.parser = PDFParserSkill()
        self.memory = SectionMemoryStore()
        self.search_parser = EnhancedSearchableParser()
        logger.info("EnhancedPDFParserSkill initialized with section memory, search, and NER")
    
    def parse_and_extract_sections(self, file_path: str) -> Dict[str, Any]:
        """
        Parse PDF and extract sections into memory with coverage tracking.
        """
        # Parse document
        doc = self.parser.parse_pdf(file_path)

        logger.info(f"Extracting sections from {doc.name}")

        # Use coverage from base parser if available (avoids re-conversion)
        coverage = getattr(doc, "coverage", None)
        coverage_message = getattr(doc, "coverage_message", "Coverage verification: not available")

        # Extract sections
        section_count = 0
        for section_data in doc.sections:
            section_obj = self._create_section_from_data(
                document_id=doc.id,
                section_data=section_data
            )
            self.memory.add_section(doc.id, section_obj)
            section_count += 1

        # Get statistics
        stats = self.memory.get_statistics(doc.id)

        return {
            "document_id": doc.id,
            "document_name": doc.name,
            "pages": doc.metadata.pages,
            "sections_extracted": section_count,
            "statistics": stats,
            "coverage": coverage,
            "coverage_message": coverage_message,
            "message": f"Extracted {section_count} sections from {doc.name}"
        }
    
    def _create_section_from_data(
        self,
        document_id: str,
        section_data: Dict[str, Any]
    ) -> Section:
        """Create Section object from parsed section data with page tracking."""
        title = section_data.get("title", "Untitled")
        level = section_data.get("level", 1)
        content = section_data.get("content", "")
        start_line = section_data.get("start_line", 0)
        end_line = section_data.get("end_line", 0)
        page_estimate = section_data.get("page_estimate", 1)
        page_end_estimate = section_data.get("page_end_estimate", page_estimate)
        start_page = section_data.get("start_page", page_estimate)
        end_page = section_data.get("end_page", page_end_estimate)
        
        # Extract metadata
        metadata = self._extract_section_metadata(
            title=title,
            level=level,
            content=content,
            start_line=start_line,
            end_line=end_line,
            page_estimate=page_estimate,
            page_end_estimate=page_end_estimate,
            start_page=start_page,
            end_page=end_page
        )
        
        # Create section
        section = Section(
            metadata=metadata,
            content=content,
            document_id=document_id
        )
        
        # Process subsections
        for subsection_data in section_data.get("subsections", []):
            subsection = self._create_section_from_data(
                document_id=document_id,
                section_data=subsection_data
            )
            section.subsections.append(subsection)
        
        return section
    
    @staticmethod
    def _extract_section_metadata(
        title: str,
        level: int,
        content: str,
        start_line: int = 0,
        end_line: int = 0,
        page_estimate: int = 1,
        page_end_estimate: int = 1,
        start_page: int = 1,
        end_page: int = 1
    ) -> SectionMetadata:
        """Extract metadata for a section including page information and credit analyst classification."""
        # Count words precisely
        words = content.split()
        word_count = len(words)
        
        # Check for code blocks
        has_code = bool(re.search(r'```|def |class |import ', content, re.IGNORECASE))
        
        # Check for tables (simple heuristic)
        has_tables = bool(re.search(r'\|.*\|', content))
        
        # Use actual start/end pages if provided, otherwise use estimates
        actual_start = start_page if start_page > 0 else page_estimate
        actual_end = end_page if end_page > 0 else page_end_estimate
        
        # Ensure end >= start
        actual_end = max(actual_start, actual_end)
        
        # Create page range display
        page_range = f"{actual_start}-{actual_end}" if actual_end > actual_start else str(actual_start)
        
        # Analyze section with credit analyst
        analyst = get_credit_analyst()
        analysis = analyst.analyze_section_importance(title, content, level)
        
        return SectionMetadata(
            title=title,
            level=level,
            start_line=start_line,
            end_line=end_line,
            content_length=len(content),
            word_count=word_count,
            has_code=has_code,
            has_tables=has_tables,
            subsection_count=0,
            page_estimate=page_estimate,
            page_range=page_range,
            start_page=actual_start,
            end_page=actual_end,
            section_type=analysis.get("classification"),
            importance_score=analysis.get("importance_score", 0.5),
            typical_dependencies=analysis.get("typical_dependencies", [])
        )
    
    def get_section(self, full_section_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific section with full details."""
        section = self.memory.get_section(full_section_id)
        if not section:
            return None
        
        return {
            "section": section.to_dict(),
            "content": section.content,
            "metadata": asdict(section.metadata)
        }
    
    def get_document_index(self, document_id: str) -> Dict[str, Any]:
        """Get complete index of document sections."""
        hierarchy = self.memory.get_document_hierarchy(document_id)
        stats = self.memory.get_statistics(document_id)
        
        return {
            "document_id": document_id,
            "hierarchy": hierarchy,
            "statistics": stats
        }
    
    def search_document(
        self,
        document_id: str,
        query: str
    ) -> Dict[str, Any]:
        """Search across all sections in document."""
        results = self.memory.search_sections(
            document_id=document_id,
            query=query,
            search_titles=True,
            search_content=True
        )
        
        return {
            "query": query,
            "document_id": document_id,
            "results": [
                {
                    "section_id": section.metadata.id,
                    "title": section.metadata.title,
                    "level": section.metadata.level,
                    "match_count": len(matches),
                    "matches": matches
                }
                for section, matches in results
            ],
            "total_matches": sum(len(m) for _, m in results)
        }
    
    def full_text_search(self, query: str) -> Dict[str, Any]:
        """Full-text search across all indexed sections."""
        results = self.search_parser.full_text_search(query)
        return {
            "query": query,
            "results_count": len(results),
            "results": results
        }
    
    def extract_named_entities(
        self,
        document_id: str,
        entity_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract named entities from document sections.
        
        Args:
            document_id: Document ID
            entity_type: Optional entity type filter
            
        Returns:
            Extracted entities grouped by type
        """
        return self.search_parser.search_engine.search_entities(document_id, entity_type) if not entity_type else \
            self.search_parser.search_engine.search_entities(document_id, entity_type)
    
    def get_entities_in_section(self, section_id: str) -> Dict[str, Any]:
        """Get entities in a specific section."""
        return self.search_parser.get_section_entities(section_id)
    
    def get_all_documents(self) -> Dict[str, Any]:
        """Get summary of all documents in memory."""
        return {
            "document_count": len(self.memory.documents),
            "documents": [
                {
                    "document_id": doc_id,
                    "section_count": len(sections),
                    "statistics": self.memory.get_statistics(doc_id)
                }
                for doc_id, sections in self.memory.section_index.items()
            ]
        }
    
    def verify_document_coverage(self, document_id: str) -> Dict[str, Any]:
        """
        Verify that document parsing captured all pages and sections.
        
        Analyzes page ranges and line counts to ensure no content was missed.
        
        Args:
            document_id: Document ID to verify
            
        Returns:
            Coverage report with pages, sections, and content analysis
        """
        sections = self.memory.get_document_sections(document_id)
        
        if not sections:
            return {
                "document_id": document_id,
                "status": "no_sections",
                "message": "No sections found for document"
            }
        
        # Collect page information
        min_page = float('inf')
        max_page = 0
        total_word_count = 0
        total_content_length = 0
        section_count = 0
        pages_with_sections = set()
        
        def collect_section_info(secs: List[Section]) -> None:
            nonlocal min_page, max_page, total_word_count, total_content_length, section_count
            for section in secs:
                section_count += 1
                total_word_count += section.metadata.word_count
                total_content_length += section.metadata.content_length
                
                # Track page coverage
                if section.metadata.page_estimate < min_page:
                    min_page = section.metadata.page_estimate
                
                page_end = section.metadata.page_estimate
                if section.metadata.page_range and '-' in section.metadata.page_range:
                    try:
                        _, page_end = section.metadata.page_range.split('-')
                        page_end = int(page_end)
                    except (ValueError, IndexError):
                        pass
                
                if page_end > max_page:
                    max_page = page_end
                
                # Track which pages have content
                for page in range(section.metadata.page_estimate, page_end + 1):
                    pages_with_sections.add(page)
                
                # Recurse into subsections
                if section.subsections:
                    collect_section_info(section.subsections)
        
        collect_section_info(sections)
        
        # Reset infinity value if no pages found
        if min_page == float('inf'):
            min_page = 1
        
        return {
            "document_id": document_id,
            "status": "verified",
            "coverage_analysis": {
                "total_sections": section_count,
                "estimated_page_range": f"{min_page}-{max_page}",
                "pages_with_content": len(pages_with_sections),
                "total_word_count": total_word_count,
                "total_content_length": total_content_length,
                "average_section_words": total_word_count // section_count if section_count > 0 else 0
            },
            "quality_checks": {
                "has_content": total_content_length > 0,
                "has_sections": section_count > 0,
                "multiple_pages": max_page > min_page if max_page != 0 else False,
                "adequate_coverage": len(pages_with_sections) > 0
            }
        }


# Global instance
_enhanced_parser: Optional[EnhancedPDFParserSkill] = None


def get_enhanced_parser() -> EnhancedPDFParserSkill:
    """Get or create enhanced parser instance."""
    global _enhanced_parser
    if _enhanced_parser is None:
        _enhanced_parser = EnhancedPDFParserSkill()
    return _enhanced_parser


# Claude Skill functions
def extract_document_sections(file_path: str) -> Dict[str, Any]:
    """
    Claude Skill: Parse PDF and extract all sections into memory.
    
    Returns complete section hierarchy with metadata.
    
    Args:
        file_path: Path to PDF file
        
    Returns:
        Extraction summary with section count and statistics
    """
    parser = get_enhanced_parser()
    return parser.parse_and_extract_sections(file_path)


def get_section_details(full_section_id: str) -> Dict[str, Any]:
    """
    Claude Skill: Get details of a specific section.
    
    Args:
        full_section_id: Full hierarchical section ID
        
    Returns:
        Complete section details with content and metadata
    """
    parser = get_enhanced_parser()
    result = parser.get_section(full_section_id)
    return result or {"error": f"Section not found: {full_section_id}"}


def get_document_hierarchy(document_id: str) -> Dict[str, Any]:
    """
    Claude Skill: Get hierarchical index of document sections.
    
    Args:
        document_id: Document ID
        
    Returns:
        Hierarchical view of all sections with metadata
    """
    parser = get_enhanced_parser()
    return parser.get_document_index(document_id)


def search_sections(document_id: str, query: str) -> Dict[str, Any]:
    """
    Claude Skill: Search sections within a document.
    
    Args:
        document_id: Document to search
        query: Search term
        
    Returns:
        List of matching sections with context
    """
    parser = get_enhanced_parser()
    return parser.search_document(document_id, query)


def get_section_metadata(full_section_id: str) -> Dict[str, Any]:
    """
    Claude Skill: Get metadata for a section.
    
    Args:
        full_section_id: Full hierarchical section ID
        
    Returns:
        Section metadata (word count, has_code, has_tables, etc.)
    """
    parser = get_enhanced_parser()
    section = parser.memory.get_section(full_section_id)
    
    if not section:
        return {"error": f"Section not found: {full_section_id}"}
    
    return {
        "section_id": section.metadata.id,
        "title": section.metadata.title,
        "level": section.metadata.level,
        "word_count": section.metadata.word_count,
        "content_length": section.metadata.content_length,
        "has_code": section.metadata.has_code,
        "has_tables": section.metadata.has_tables,
        "subsection_count": len(section.subsections),
        "created_at": section.metadata.created_at
    }


def list_all_documents() -> Dict[str, Any]:
    """
    Claude Skill: List all documents currently in memory.
    
    Returns:
        Summary of all loaded documents and their sections
    """
    parser = get_enhanced_parser()
    return parser.get_all_documents()


def verify_document_coverage(document_id: str) -> Dict[str, Any]:
    """
    Claude Skill: Verify that a document was completely parsed.
    
    Analyzes page coverage and section extraction to ensure no content was missed,
    especially important for documents with multiple tables of contents.
    
    Args:
        document_id: Document ID to verify
        
    Returns:
        Coverage report with quality checks and statistics
    """
    parser = get_enhanced_parser()
    return parser.verify_document_coverage(document_id)


def create_search_strategy(document_id: str, question: str) -> Dict[str, Any]:
    """
    Claude Skill: Create intelligent search strategy using credit analyst expertise.
    
    Uses expert knowledge of credit agreement structures to prioritize sections
    for question answering, following complete logical chains rather than stopping
    at the first relevant section.
    
    Args:
        document_id: Document ID to search
        question: The analytical question to answer
        
    Returns:
        Prioritized list of sections with reasoning and search strategy
    """
    parser = get_enhanced_parser()
    sections = parser.memory.get_document_sections(document_id)
    
    # Convert sections to dict format for analyst
    section_dicts = []
    for section in sections:
        section_dicts.append({
            "id": section.metadata.id,
            "title": section.metadata.title,
            "level": section.metadata.level,
            "content": section.content[:1000],  # First 1000 chars for classification
            "section_type": section.metadata.section_type,
            "importance_score": section.metadata.importance_score,
            "page_range": section.metadata.page_range
        })
    
    # Create search strategy
    analyst = get_credit_analyst()
    strategy = analyst.create_search_strategy(question, section_dicts)
    
    return {
        "document_id": document_id,
        "question": question,
        "search_strategy": strategy,
        "system_prompt_used": "Expert Syndicated Credit Analyst Framework"
    }
