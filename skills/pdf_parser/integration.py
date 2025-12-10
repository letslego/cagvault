"""
Integration module to connect PDF Parser Skill with CAG application.

This module bridges the pdf_parser skill with the existing knowledge.py
and provides Claude with direct access to CAG's document management.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from knowledge import KnowledgeSource, KnowledgeType, load_from_file, load_from_url
from .pdf_parser import (
    PDFParserSkill,
    ParsedDocument,
    get_pdf_parser_skill
)


class CAGPDFIntegration:
    """Integrates PDF Parser Skill with CAG knowledge base."""
    
    def __init__(self):
        """Initialize the integration."""
        self.skill = get_pdf_parser_skill()
    
    def parse_and_store(self, file_path: str) -> KnowledgeSource:
        """
        Parse a PDF and store it in the CAG knowledge base.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            KnowledgeSource object ready for CAG
        """
        # Parse with skill
        parsed_doc = self.skill.parse_pdf(file_path)
        
        # Create KnowledgeSource for CAG
        knowledge_source = KnowledgeSource(
            id=parsed_doc.id,
            name=parsed_doc.name,
            type=KnowledgeType.DOCUMENT,
            content=parsed_doc.content
        )
        
        return knowledge_source
    
    def parse_url_and_store(self, url: str) -> KnowledgeSource:
        """
        Parse web content and store in CAG knowledge base.
        
        Args:
            url: URL to parse
            
        Returns:
            KnowledgeSource object ready for CAG
        """
        parsed_doc = self.skill.parse_url(url)
        
        knowledge_source = KnowledgeSource(
            id=parsed_doc.id,
            name=parsed_doc.name,
            type=KnowledgeType.URL,
            content=parsed_doc.content
        )
        
        return knowledge_source
    
    def get_document_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get comprehensive information about a parsed document.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Dictionary with metadata, sections, tables, and summary
        """
        parsed_doc = self.skill.parse_pdf(file_path)
        
        return {
            "summary": self.skill.get_summary(parsed_doc),
            "metadata": self.skill.get_metadata(parsed_doc),
            "sections": parsed_doc.sections,
            "tables": parsed_doc.tables,
            "id": parsed_doc.id,
            "name": parsed_doc.name,
            "content_length": len(parsed_doc.content),
            "cache_path": parsed_doc.cache_path
        }
    
    def search_and_highlight(self, file_path: str, query: str) -> Dict[str, Any]:
        """
        Search document and return results with context.
        
        Args:
            file_path: Path to PDF file
            query: Search term
            
        Returns:
            Search results with line numbers and content
        """
        parsed_doc = self.skill.parse_pdf(file_path)
        results = self.skill.search_content(parsed_doc, query)
        
        return {
            "query": query,
            "document": parsed_doc.name,
            "matches_found": len(results),
            "results": results
        }


# Claude Skill functions for direct use with Claude Code
def claude_parse_pdf(file_path: str) -> Dict[str, Any]:
    """
    Claude Skill: Parse PDF and integrate with CAG.
    
    Returns document ready for credit analyst Q&A.
    
    Args:
        file_path: Path to PDF file
        
    Returns:
        Document info with metadata and structure
    """
    integration = CAGPDFIntegration()
    return integration.get_document_info(file_path)


def claude_ingest_pdf(file_path: str) -> Dict[str, Any]:
    """
    Claude Skill: Parse PDF and add to CAG knowledge base.
    
    Args:
        file_path: Path to PDF file
        
    Returns:
        Confirmation with document ID and details
    """
    integration = CAGPDFIntegration()
    knowledge_source = integration.parse_and_store(file_path)
    
    return {
        "status": "ingested",
        "document_id": knowledge_source.id,
        "name": knowledge_source.name,
        "type": knowledge_source.type.value,
        "content_length": len(knowledge_source.content),
        "message": f"Document '{knowledge_source.name}' successfully parsed and stored in CAG knowledge base"
    }


def claude_search_pdf(file_path: str, query: str) -> Dict[str, Any]:
    """
    Claude Skill: Search within a PDF document.
    
    Args:
        file_path: Path to PDF file
        query: What to search for
        
    Returns:
        Search results with context
    """
    integration = CAGPDFIntegration()
    return integration.search_and_highlight(file_path, query)


def claude_get_structure(file_path: str) -> Dict[str, Any]:
    """
    Claude Skill: Extract table of contents and structure.
    
    Args:
        file_path: Path to PDF file
        
    Returns:
        Hierarchical document structure
    """
    integration = CAGPDFIntegration()
    skill = get_pdf_parser_skill()
    parsed_doc = skill.parse_pdf(file_path)
    
    return {
        "document": parsed_doc.name,
        "pages": parsed_doc.metadata.pages,
        "sections": parsed_doc.sections,
        "tables": len(parsed_doc.tables),
        "toc": [
            {
                "level": s.get("level", 1),
                "title": s.get("title", "Untitled"),
                "subsections": len(s.get("subsections", []))
            }
            for s in parsed_doc.sections
        ]
    }


if __name__ == "__main__":
    # Example usage
    integration = CAGPDFIntegration()
    
    # Example: parse_and_store
    # knowledge_source = integration.parse_and_store("/path/to/document.pdf")
    # print(f"Stored: {knowledge_source.id}")
    
    # Example: search
    # results = integration.search_and_highlight("/path/to/document.pdf", "covenant")
    # print(f"Found {results['matches_found']} matches")
