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
import re

from .page_tracker import PageTracker

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
    coverage: Optional[Dict[str, Any]] = None
    coverage_message: Optional[str] = None
    coverage_complete: Optional[bool] = None


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
        """Create and cache the DocumentConverter with speed optimizations."""
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        
        # Fast pipeline options - skip OCR for speed
        pipeline_options = PdfPipelineOptions(
            do_ocr=False,  # Skip OCR for speed (assumes text-based PDFs)
            do_table_structure=True,  # Keep table detection with default settings
        )
        
        return DocumentConverter(
            allowed_formats=[
                InputFormat.PDF,
                InputFormat.HTML,
            ],
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=StandardPdfPipeline,
                    backend=DoclingParseV4DocumentBackend,
                    pipeline_options=pipeline_options,
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
        """
        Extract hierarchical sections from markdown content.
        
        Ensures all content is captured, handles multiple tables of contents,
        and maintains proper parent-child relationships for nested sections.
        """
        lines = content.split("\n")
        sections = []
        section_stack = []  # Stack to track current section hierarchy
        
        line_num = 0
        while line_num < len(lines):
            line = lines[line_num]
            
            # Detect headers
            if line.startswith("#"):
                # Count level by # symbols
                match = re.match(r'^(#+)', line)
                if match:
                    level = len(match.group(1))
                    text = line[level:].strip()
                    
                    if text:  # Only process non-empty headers
                        section = {
                            "level": level,
                            "title": text,
                            "content": "",
                            "subsections": [],
                            "start_line": line_num
                        }
                        
                        # Pop stack until we find the correct parent level
                        while section_stack and section_stack[-1]["level"] >= level:
                            section_stack.pop()
                        
                        # Add to parent or root
                        if section_stack:
                            section_stack[-1]["subsections"].append(section)
                        else:
                            sections.append(section)
                        
                        section_stack.append(section)
                    
                    line_num += 1
                else:
                    # Fallback: treat as content if header parsing fails
                    if section_stack:
                        section_stack[-1]["content"] += line + "\n"
                    line_num += 1
            else:
                # Add content to current section
                if section_stack:
                    # Strip excessive whitespace but preserve structure
                    if line.strip():
                        section_stack[-1]["content"] += line + "\n"
                    elif section_stack[-1]["content"].strip():
                        # Preserve paragraph breaks (single empty line)
                        if not section_stack[-1]["content"].endswith("\n\n"):
                            section_stack[-1]["content"] += "\n"
                elif line.strip():
                    # Content before first header - create a preamble section
                    if not sections or sections[0]["level"] != 0:
                        preamble = {
                            "level": 0,
                            "title": "Preamble",
                            "content": "",
                            "subsections": [],
                            "start_line": 0
                        }
                        sections.insert(0, preamble)
                        section_stack.append(preamble)
                    
                    if section_stack:
                        section_stack[-1]["content"] += line + "\n"
                
                line_num += 1
        
        # Clean up sections - remove empty ones but keep structure
        def clean_sections(secs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            """Remove empty content but preserve section structure."""
            cleaned = []
            for sec in secs:
                # Clean content
                sec["content"] = sec["content"].strip()
                
                # Recursively clean subsections
                if sec["subsections"]:
                    sec["subsections"] = clean_sections(sec["subsections"])
                
                # Skip preamble if empty, but keep others to preserve document structure
                if sec["content"] or sec["subsections"] or sec.get("level", 1) != 0:
                    cleaned.append(sec)
            
            return cleaned
        
        return clean_sections(sections)
    
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
        
        # Check cache; re-parse if cached doc lacks coverage info
        cache_key = self._generate_cache_key(str(file_path))
        cached_doc = self._get_cached_document(cache_key)
        if cached_doc:
            cov = getattr(cached_doc, "coverage", None)
            if cov and cov.get("pages_with_content", 0) > 0:
                return cached_doc
        
        logger.info(f"Parsing PDF: {file_path}")
        
        try:
            # Convert document
            result = self.converter.convert(str(file_path))
            
            # Extract content
            content = result.document.export_to_markdown()
            
            # Build page tracker for accurate page estimates
            try:
                tracker = PageTracker(result.document, content)
            except Exception as e:
                logger.warning(f"Page tracker initialization failed: {e}")
                tracker = None
            
            # Build metadata
            metadata = DocumentMetadata(
                pages=len(result.document.pages),
                file_size=file_path.stat().st_size,
                file_path=str(file_path),
                parse_date=datetime.now().isoformat(),
                format="pdf"
            )
            
            # Extract structure with page information
            sections = self._extract_sections_with_pages(content, tracker)
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
            # Attach coverage info if available
            if tracker:
                try:
                    complete, cov_msg = tracker.verify_complete_coverage()
                    doc.coverage = tracker.get_coverage_report()
                    doc.coverage_message = cov_msg
                    doc.coverage_complete = complete
                except Exception as e:
                    logger.debug(f"Could not attach coverage info: {e}")
            else:
                doc.coverage_message = "Coverage verification: not available"
            
            # Cache the result
            self._cache_document(doc, cache_key)
            
            logger.info(f"Successfully parsed: {file_path.name} ({metadata.pages} pages, {len(sections)} sections)")
            return doc
            
        except Exception as e:
            logger.error(f"Failed to parse PDF {file_path}: {e}")
            raise
    
    def _extract_page_markers(self, document: Any) -> Dict[int, str]:
        """
        Extract page boundary information from Docling document.
        
            Maps content blocks to their page numbers for accurate page tracking.
        
            Returns:
                Mapping of block indices to page numbers
            """
        page_markers = {}

        try:
            # Iterate through pages and collect block page information
            for page_idx, page in enumerate(document.pages, 1):
                if hasattr(page, 'blocks'):
                    for block in page.blocks:
                        # Store page number for each block
                        if hasattr(block, 'block_id'):
                            page_markers[block.block_id] = page_idx
        except Exception as e:
            logger.debug(f"Could not extract detailed page markers: {e}")

        return page_markers
    
    def _extract_sections_with_pages(
        self,
        content: str,
        tracker: Optional[PageTracker]
    ) -> List[Dict[str, Any]]:
        """
        Extract hierarchical sections from markdown content with page tracking.
        
        Ensures all content is captured and tracks which page range contains each section.
        Maximizes granularity by creating dedicated sections for preamble content blocks.
        """
        lines = content.split("\n")
        sections = []
        section_stack = []  # Stack to track current section hierarchy
        preamble_lines = []  # Collect preamble content before first header
        
        line_num = 0
        while line_num < len(lines):
            line = lines[line_num]
            
            # Detect headers
            if line.startswith("#"):
                # First, flush any accumulated preamble content as separate sections
                if preamble_lines and not section_stack:
                    # Clear section stack when we hit first header
                    self._create_preamble_sections(preamble_lines, sections, tracker, len(lines))
                    preamble_lines = []
                
                # Count level by # symbols
                match = re.match(r'^(#+)', line)
                if match:
                    level = len(match.group(1))
                    text = line[level:].strip()
                    
                    if text:  # Only process non-empty headers
                        page_est = tracker.get_page_for_line(line_num) if tracker else self._estimate_page_from_line(line_num, len(lines))
                        section = {
                            "level": level,
                            "title": text,
                            "content": "",
                            "subsections": [],
                            "start_line": line_num,
                            "page_estimate": page_est,
                            "start_page": page_est,
                            "end_page": page_est
                        }
                        
                        # Pop stack until we find the correct parent level
                        while section_stack and section_stack[-1]["level"] >= level:
                            popped = section_stack.pop()
                            popped["end_line"] = line_num
                            end_page = tracker.get_page_for_line(line_num) if tracker else self._estimate_page_from_line(line_num, len(lines))
                            popped["page_end_estimate"] = end_page
                            popped["end_page"] = end_page
                        
                        # Add to parent or root
                        if section_stack:
                            section_stack[-1]["subsections"].append(section)
                        else:
                            sections.append(section)
                        
                        section_stack.append(section)
                    
                    line_num += 1
                else:
                    # Fallback: treat as content if header parsing fails
                    if section_stack:
                        section_stack[-1]["content"] += line + "\n"
                    else:
                        preamble_lines.append(line)
                    line_num += 1
            else:
                # Add content to current section or preamble
                if section_stack:
                    # Strip excessive whitespace but preserve structure
                    if line.strip():
                        section_stack[-1]["content"] += line + "\n"
                    elif section_stack[-1]["content"].strip():
                        # Preserve paragraph breaks (single empty line)
                        if not section_stack[-1]["content"].endswith("\n\n"):
                            section_stack[-1]["content"] += "\n"
                else:
                    # Content before first header - accumulate for preamble
                    preamble_lines.append(line)
                
                line_num += 1
        
        # Flush any remaining preamble
        if preamble_lines and not section_stack:
            self._create_preamble_sections(preamble_lines, sections, tracker, len(lines))
        
        # Mark end lines for final sections in stack
        for section in section_stack:
            section["end_line"] = len(lines)
            end_page = tracker.get_page_for_line(len(lines)) if tracker else self._estimate_page_from_line(len(lines), len(lines))
            section["page_end_estimate"] = end_page
            section["end_page"] = end_page
        
        # Clean up sections - remove truly empty ones but keep structure
        def clean_sections(secs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            """Remove empty content but preserve section structure."""
            cleaned = []
            for sec in secs:
                # Clean content
                sec["content"] = sec["content"].strip()
                
                # Recursively clean subsections
                if sec["subsections"]:
                    sec["subsections"] = clean_sections(sec["subsections"])
                
                # Keep sections with content, subsections, or headers (skip only if truly empty)
                if sec["content"] or sec["subsections"]:
                    cleaned.append(sec)
            
            return cleaned
        
        return clean_sections(sections)
    
    def _create_preamble_sections(
        self,
        preamble_lines: List[str],
        sections: List[Dict[str, Any]],
        tracker: Optional[PageTracker],
        total_lines: int
    ) -> None:
        """
        Create preamble section(s) from accumulated lines.
        
        If preamble is large, split it into multiple sections for granularity.
        """
        if not preamble_lines:
            return
        
        # Join and filter
        preamble_text = "\n".join(preamble_lines).strip()
        if not preamble_text:
            return
        
        # Create single preamble section
        start_page = 1
        end_page = tracker.get_page_for_line(len(preamble_lines)) if tracker else self._estimate_page_from_line(len(preamble_lines), total_lines)
        
        preamble = {
            "level": 0,
            "title": "Preamble",
            "content": preamble_text,
            "subsections": [],
            "start_line": 0,
            "page_estimate": start_page,
            "start_page": start_page,
            "end_page": end_page,
            "page_end_estimate": end_page,
            "end_line": len(preamble_lines)
        }
        sections.insert(0, preamble)
    
    def _estimate_page_from_line(self, line_num: int, total_lines: int) -> int:
        """
        Estimate page number based on line position and typical page length.
        
        Assumes roughly 50 lines per page (approximate for markdown).
        """
        lines_per_page = 50
        estimated_page = max(1, (line_num // lines_per_page) + 1)
        return estimated_page
    
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
