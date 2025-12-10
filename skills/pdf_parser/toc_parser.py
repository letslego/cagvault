"""
TOC-Based PDF Parser Integration

Integrates the advanced doc_parse_utils TOC-based parsing with the existing
enhanced_parser framework. Provides intelligent section detection using LLM
and Table of Contents analysis.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

# Add parent directory to path to import doc_parse_utils
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from doc_parse_utils import parse_pdf_by_toc
    TOC_PARSER_AVAILABLE = True
except ImportError as e:
    logging.warning(f"doc_parse_utils not available: {e}")
    TOC_PARSER_AVAILABLE = False

from .enhanced_parser import (
    SectionMetadata,
    Section,
    SectionMemoryStore,
    get_enhanced_parser
)

logger = logging.getLogger(__name__)


@dataclass
class TOCParseResult:
    """Result from TOC-based parsing."""
    document_id: str
    document_name: str
    sections_extracted: int
    total_pages: int
    sections: List[Section]
    metadata: Dict[str, Any]
    toc_found: bool
    classification: Dict[str, Any]


def parse_pdf_with_toc(
    pdf_path: str,
    document_id: Optional[str] = None,
    use_llm_matching: bool = True,
    debug_toc: bool = False,
    debug_matching: bool = False
) -> TOCParseResult:
    """
    Parse PDF using TOC-based detection and integrate with enhanced parser.
    
    This function uses the advanced doc_parse_utils.parse_pdf_by_toc() function
    which provides:
    - Intelligent TOC detection and parsing with LLM
    - Document classification (standard vs amended & restated)
    - Accurate section matching with context awareness
    - Page range tracking for each section
    
    Args:
        pdf_path: Path to PDF file
        document_id: Optional document ID (generated if not provided)
        use_llm_matching: Use LLM for header matching (more accurate)
        debug_toc: Enable debug output for TOC parsing
        debug_matching: Enable debug output for header matching
        
    Returns:
        TOCParseResult with parsed sections and metadata
        
    Raises:
        ImportError: If doc_parse_utils is not available
        ValueError: If PDF path is invalid
    """
    if not TOC_PARSER_AVAILABLE:
        raise ImportError(
            "doc_parse_utils module not available. "
            "Ensure doc_parse_utils.py is in the skills directory."
        )
    
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise ValueError(f"PDF file not found: {pdf_path}")
    
    # Generate document ID if not provided
    if not document_id:
        import hashlib
        document_id = f"pdf_{hashlib.sha256(str(pdf_path).encode()).hexdigest()[:16]}"
    
    logger.info(f"Parsing PDF with TOC-based detection: {pdf_path}")
    
    # Call the advanced TOC parser
    result = parse_pdf_by_toc(
        pdf_path=str(pdf_path),
        output_dir=f".cache/toc_sections/{document_id}",
        generate_summaries=False,  # We'll generate summaries separately if needed
        use_llm_matching=use_llm_matching,
        debug_toc=debug_toc,
        debug_matching=debug_matching,
    )
    
    # Extract sections metadata from the result
    sections_metadata = result.get("sections_metadata", {})
    toc_data = result.get("toc_data", {})
    doc_classification = result.get("doc_classification", {})
    total_pages = result.get("total_pages", 0)
    
    # Get the enhanced parser's memory store
    parser = get_enhanced_parser()
    memory = parser.memory
    
    # Convert sections to our Section format
    sections = []
    for idx, (section_id, section_meta) in enumerate(sections_metadata.items()):
        # Parse page range
        page_range = section_meta.get("page_range", "1-1")
        try:
            start_page, end_page = map(int, page_range.split("-"))
        except (ValueError, AttributeError):
            start_page, end_page = 1, 1
        
        # Read section content
        content_path = Path(section_meta.get("file_path", ""))
        content = ""
        if content_path.exists():
            try:
                with open(content_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                logger.warning(f"Failed to read section content: {e}")
        
        # Count words
        word_count = len(content.split()) if content else 0
        
        # Create section metadata
        metadata = SectionMetadata(
            title=section_meta.get("title", f"Section {idx+1}"),
            level=1,  # TOC sections are top-level
            start_line=0,
            end_line=len(content.split('\n')) if content else 0,
            content_length=len(content),
            word_count=word_count,
            has_code=False,
            has_tables="table" in content.lower() or "|" in content,
            subsection_count=0,
            page_estimate=start_page,
            page_range=page_range,
            start_page=start_page,
            end_page=end_page,
        )
        
        # Create section
        section = Section(
            metadata=metadata,
            content=content,
            document_id=document_id,
            subsections=[]
        )
        
        sections.append(section)
        
        # Add to memory store
        memory.add_section(
            document_id=document_id,
            section_id=metadata.id,
            section=section
        )
        
        logger.debug(f"Added section: {metadata.title} (pages {start_page}-{end_page})")
    
    logger.info(
        f"TOC parsing complete: {len(sections)} sections extracted, "
        f"TOC found: {toc_data.get('is_toc_found', False)}"
    )
    
    return TOCParseResult(
        document_id=document_id,
        document_name=pdf_path.name,
        sections_extracted=len(sections),
        total_pages=total_pages,
        sections=sections,
        metadata=sections_metadata,
        toc_found=toc_data.get("is_toc_found", False),
        classification=doc_classification
    )


def get_toc_sections(document_id: str) -> List[Section]:
    """
    Retrieve sections parsed with TOC detection for a document.
    
    Args:
        document_id: Document ID
        
    Returns:
        List of sections
    """
    parser = get_enhanced_parser()
    return parser.memory.get_document_sections(document_id)


def list_toc_documents() -> List[Dict[str, Any]]:
    """
    List all documents parsed with TOC detection.
    
    Returns:
        List of document summaries
    """
    parser = get_enhanced_parser()
    return parser.memory.list_documents()


# Claude Skills interface
__claude_skills__ = [
    {
        "name": "parse_pdf_with_toc",
        "description": (
            "Parse PDF using advanced TOC-based detection with LLM intelligence. "
            "Automatically detects table of contents, classifies document type "
            "(standard vs amended & restated), and accurately matches section headers "
            "using context-aware LLM matching. Provides accurate page ranges for "
            "each section."
        ),
        "parameters": {
            "pdf_path": {
                "type": "string",
                "description": "Path to the PDF file to parse",
                "required": True
            },
            "document_id": {
                "type": "string",
                "description": "Optional document ID (auto-generated if not provided)",
                "required": False
            },
            "use_llm_matching": {
                "type": "boolean",
                "description": "Use LLM for header matching (default: True, more accurate)",
                "required": False,
                "default": True
            }
        }
    },
    {
        "name": "get_toc_sections",
        "description": "Retrieve all sections for a document parsed with TOC detection",
        "parameters": {
            "document_id": {
                "type": "string",
                "description": "Document ID",
                "required": True
            }
        }
    },
    {
        "name": "list_toc_documents",
        "description": "List all documents parsed with TOC-based detection",
        "parameters": {}
    }
]
