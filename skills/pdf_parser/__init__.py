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

__version__ = "1.0.0"
__author__ = "CAG Skills Team"
__all__ = [
    "PDFParserSkill",
    "ParsedDocument",
    "DocumentMetadata",
    "get_pdf_parser_skill",
    "parse_document",
    "parse_web_content",
    "search_document",
    "get_document_metadata",
    "CAGPDFIntegration",
    "claude_parse_pdf",
    "claude_ingest_pdf",
    "claude_search_pdf",
    "claude_get_structure",
]
