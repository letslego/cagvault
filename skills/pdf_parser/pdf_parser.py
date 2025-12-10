"""
PDF Document Parser Skill Implementation

Provides Claude with the ability to parse PDF documents and extract
structured content for the CAG application.
"""

import os
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

from docling.document_converter import (
    DocumentConverter,
    DocumentStream,
    InputFormat,
    PdfFormatOption,
    StandardPdfPipeline,
)
from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DocumentMetadata:
    """Metadata extracted from document."""
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    creation_date: Optional[str] = None
    modification_date: Optional[str] = None
    pages: int = 0
    file_size: int = 0
    file_path: Optional[str] = None
    parse_date: str = ""
    format: str = "pdf"


@dataclass
class ParsedDocument:
    """Structured output from document parsing."""
    id: str
    name: str
    content: str
    metadata: DocumentMetadata
    sections: List[Dict[str, Any]]
    tables: List[Dict[str, Any]]
    source_url: Optional[str] = None
    cache_path: Optional[str] = None


class PDFParserSkill:
    """Claude Skill for parsing PDF documents."""
    
    def __init__(self, cache_dir: str = ".cache/documents"):
        """Initialize PDF parser with optional caching."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.converter = self._create_converter()
        logger.info(f"PDFParserSkill initialized with cache at {cache_dir}")
    
    @staticmethod
    def _create_converter() -> DocumentConverter:
        """Create and cache the DocumentConverter."""
        return DocumentConverter(
            allowed_formats=[
                InputFormat.PDF,
                InputFormat.HTML,
            ],
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=StandardPdfPipeline,
                    backend=DoclingParseV4DocumentBackend,
                ),
            },
        )
    
    def _generate_cache_key(self, file_path: str) -> str:
        """Generate cache key from file path and modification time."""
        try:
            mtime = os.path.getmtime(file_path)
            cache_input = f"{file_path}_{mtime}"
        except OSError:
            cache_input = file_path
        
        return hashlib.sha256(cache_input.encode()).hexdigest()
    
    def _get_cached_document(self, cache_key: str) -> Optional[ParsedDocument]:
        """Retrieve document from cache if available."""
        import pickle
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    doc = pickle.load(f)
                logger.info(f"Retrieved cached document: {cache_key}")
                return doc
            except Exception as e:
                logger.warning(f"Failed to load cached document: {e}")
        
        return None
    
    def _cache_document(self, doc: ParsedDocument, cache_key: str) -> None:
        """Save parsed document to cache."""
        import pickle
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(doc, f)
            doc.cache_path = str(cache_file)
            logger.info(f"Cached document: {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to cache document: {e}")
    
    def _extract_sections(self, content: str) -> List[Dict[str, Any]]:
        """Extract hierarchical sections from markdown content."""
        sections = []
        current_section = None
        
        for line in content.split("\n"):
            # Detect headers
            if line.startswith("#"):
                level = len(line.split()[0])
                text = line.lstrip("#").strip()
                
                section = {
                    "level": level,
                    "title": text,
                    "content": "",
                    "subsections": []
                }
                
                if current_section is None or level == 1:
                    sections.append(section)
                    current_section = section
                else:
                    # Add as subsection
                    if current_section:
                        current_section["subsections"].append(section)
            elif current_section and line.strip():
                current_section["content"] += line + "\n"
        
        return sections
    
    def _extract_tables(self, content: str) -> List[Dict[str, Any]]:
        """Extract table information from markdown content."""
        tables = []
        lines = content.split("\n")
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Detect markdown tables
            if "|" in line and i + 1 < len(lines) and "|" in lines[i + 1]:
                table = {
                    "start_line": i,
                    "headers": [],
                    "rows": []
                }
                
                # Parse header
                header_cells = [cell.strip() for cell in line.split("|")[1:-1]]
                table["headers"] = header_cells
                
                # Skip separator line
                i += 2
                
                # Parse rows
                while i < len(lines) and "|" in lines[i]:
                    row_cells = [cell.strip() for cell in lines[i].split("|")[1:-1]]
                    table["rows"].append(row_cells)
                    i += 1
                
                tables.append(table)
                continue
            
            i += 1
        
        return tables
    
    def parse_pdf(self, file_path: str) -> ParsedDocument:
        """
        Parse a local PDF file and extract structured content.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            ParsedDocument with extracted content and metadata
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format not supported
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not file_path.suffix.lower() in [".pdf"]:
            raise ValueError(f"Unsupported format: {file_path.suffix}")
        
        # Check cache
        cache_key = self._generate_cache_key(str(file_path))
        cached_doc = self._get_cached_document(cache_key)
        if cached_doc:
            return cached_doc
        
        logger.info(f"Parsing PDF: {file_path}")
        
        try:
            # Convert document
            result = self.converter.convert(str(file_path))
            
            # Extract content
            content = result.document.export_to_markdown()
            
            # Build metadata
            metadata = DocumentMetadata(
                pages=len(result.document.pages),
                file_size=file_path.stat().st_size,
                file_path=str(file_path),
                parse_date=datetime.now().isoformat(),
                format="pdf"
            )
            
            # Extract structure
            sections = self._extract_sections(content)
            tables = self._extract_tables(content)
            
            # Create parsed document
            doc = ParsedDocument(
                id=f"pdf_{cache_key}",
                name=file_path.name,
                content=content,
                metadata=metadata,
                sections=sections,
                tables=tables
            )
            
            # Cache the result
            self._cache_document(doc, cache_key)
            
            logger.info(f"Successfully parsed: {file_path.name} ({metadata.pages} pages)")
            return doc
            
        except Exception as e:
            logger.error(f"Failed to parse PDF {file_path}: {e}")
            raise
    
    def parse_url(self, url: str) -> ParsedDocument:
        """
        Parse content from a URL (HTML to Markdown).
        
        Args:
            url: URL to parse
            
        Returns:
            ParsedDocument with extracted content
        """
        # Check cache
        cache_key = hashlib.sha256(url.encode()).hexdigest()
        cached_doc = self._get_cached_document(cache_key)
        if cached_doc:
            return cached_doc
        
        logger.info(f"Parsing URL: {url}")
        
        try:
            result = self.converter.convert(url)
            content = result.document.export_to_markdown()
            
            metadata = DocumentMetadata(
                pages=len(result.document.pages),
                parse_date=datetime.now().isoformat(),
                format="html"
            )
            
            sections = self._extract_sections(content)
            tables = self._extract_tables(content)
            
            doc = ParsedDocument(
                id=f"url_{cache_key}",
                name=url,
                content=content,
                metadata=metadata,
                sections=sections,
                tables=tables,
                source_url=url
            )
            
            self._cache_document(doc, cache_key)
            
            logger.info(f"Successfully parsed URL: {url}")
            return doc
            
        except Exception as e:
            logger.error(f"Failed to parse URL {url}: {e}")
            raise
    
    def get_metadata(self, doc: ParsedDocument) -> Dict[str, Any]:
        """Extract metadata as dictionary."""
        return asdict(doc.metadata)
    
    def get_summary(self, doc: ParsedDocument) -> str:
        """Generate a summary of the document."""
        summary = f"""
Document: {doc.name}
Pages: {doc.metadata.pages}
Size: {doc.metadata.file_size / 1024:.1f} KB
Sections: {len(doc.sections)}
Tables: {len(doc.tables)}
Parsed: {doc.metadata.parse_date}
"""
        return summary.strip()
    
    def search_content(self, doc: ParsedDocument, query: str) -> List[Dict[str, Any]]:
        """Search for query terms in document content."""
        import re
        
        results = []
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        
        for i, line in enumerate(doc.content.split("\n")):
            if pattern.search(line):
                results.append({
                    "line_number": i,
                    "content": line.strip(),
                    "match": pattern.search(line).group()
                })
        
        return results
    
    def export_json(self, doc: ParsedDocument) -> Dict[str, Any]:
        """Export document as structured JSON."""
        return {
            "id": doc.id,
            "name": doc.name,
            "metadata": asdict(doc.metadata),
            "sections": doc.sections,
            "tables": doc.tables,
            "content_preview": doc.content[:500] + "..." if len(doc.content) > 500 else doc.content
        }


# Global skill instance
_skill_instance: Optional[PDFParserSkill] = None


def get_pdf_parser_skill() -> PDFParserSkill:
    """Get or create the PDF parser skill instance."""
    global _skill_instance
    if _skill_instance is None:
        _skill_instance = PDFParserSkill()
    return _skill_instance


# Claude Skill functions
def parse_document(file_path: str) -> Dict[str, Any]:
    """
    Claude Skill: Parse a PDF document and extract structured content.
    
    Args:
        file_path: Path to the PDF file to parse
        
    Returns:
        Structured document with metadata, sections, and tables
    """
    skill = get_pdf_parser_skill()
    doc = skill.parse_pdf(file_path)
    return skill.export_json(doc)


def parse_web_content(url: str) -> Dict[str, Any]:
    """
    Claude Skill: Parse web content from a URL and convert to structured format.
    
    Args:
        url: URL to parse
        
    Returns:
        Structured document with metadata and sections
    """
    skill = get_pdf_parser_skill()
    doc = skill.parse_url(url)
    return skill.export_json(doc)


def search_document(file_path: str, query: str) -> List[Dict[str, Any]]:
    """
    Claude Skill: Search for content in a parsed document.
    
    Args:
        file_path: Path to the PDF file
        query: Search term
        
    Returns:
        List of matching lines with context
    """
    skill = get_pdf_parser_skill()
    doc = skill.parse_pdf(file_path)
    return skill.search_content(doc, query)


def get_document_metadata(file_path: str) -> Dict[str, Any]:
    """
    Claude Skill: Extract metadata from a document.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Document metadata dictionary
    """
    skill = get_pdf_parser_skill()
    doc = skill.parse_pdf(file_path)
    return skill.get_metadata(doc)


if __name__ == "__main__":
    # Example usage
    skill = get_pdf_parser_skill()
    
    # Parse example
    # doc = skill.parse_pdf("/path/to/document.pdf")
    # print(skill.get_summary(doc))
    # print(f"Sections: {len(doc.sections)}")
    # print(f"Tables: {len(doc.tables)}")
