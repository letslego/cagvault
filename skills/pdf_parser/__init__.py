"""
PDF Parser Skill Package

Claude Skill for parsing PDF documents and extracting structured content.
Integrates with CAG (Cache-Augmented Generation) application.
"""

from .pdf_parser import (
    PDFParserSkill,
    ParsedDocument,
    DocumentMetadata,
    get_pdf_parser_skill,
    parse_document,
    parse_web_content,
    search_document,
    get_document_metadata,
)

from .integration import (
    CAGPDFIntegration,
    claude_parse_pdf,
    claude_ingest_pdf,
    claude_search_pdf,
    claude_get_structure,
)

from .enhanced_parser import (
    EnhancedPDFParserSkill,
    SectionMemoryStore,
    Section,
    SectionMetadata,
    get_enhanced_parser,
    extract_document_sections,
    get_section_details,
    get_document_hierarchy,
    search_sections,
    get_section_metadata,
    list_all_documents,
)

__version__ = "1.1.0"
__author__ = "CAG Skills Team"
__all__ = [
    # Basic parser
    "PDFParserSkill",
    "ParsedDocument",
    "DocumentMetadata",
    "get_pdf_parser_skill",
    "parse_document",
    "parse_web_content",
    "search_document",
    "get_document_metadata",
    # Integration
    "CAGPDFIntegration",
    "claude_parse_pdf",
    "claude_ingest_pdf",
    "claude_search_pdf",
    "claude_get_structure",
    # Enhanced parser with section memory
    "EnhancedPDFParserSkill",
    "SectionMemoryStore",
    "Section",
    "SectionMetadata",
    "get_enhanced_parser",
    "extract_document_sections",
    "get_section_details",
    "get_document_hierarchy",
    "search_sections",
    "get_section_metadata",
    "list_all_documents",
]
