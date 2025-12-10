"""
Page Tracking Module for PDFs

Analyzes Docling document structure to accurately track which pages
contain which sections, ensuring no pages are missed in parsing.
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PageInfo:
    """Information about a page in the document."""
    page_number: int
    char_start: int  # Character position where this page starts
    char_end: int    # Character position where this page ends
    block_count: int  # Number of content blocks on this page
    word_count: int   # Approximate word count on page


class PageTracker:
    """
    Tracks page information from Docling document to ensure
    complete page coverage during parsing.
    """
    
    def __init__(self, document: Any, full_content: str):
        """
        Initialize page tracker.
        
        Args:
            document: Docling document object
            full_content: Full markdown content string
        """
        self.document = document
        self.full_content = full_content
        try:
            self.total_pages = max(1, len(document.pages))
        except Exception:
            self.total_pages = 1
        self.pages: Dict[int, PageInfo] = {}
        self.char_to_page: Dict[int, int] = {}
        self.coverage_fallback = False
        
        try:
            self._build_page_map()
        except Exception as e:
            logger.error(f"Failed to build page map: {repr(e)}")
            self._fallback_coverage()
    
    def _build_page_map(self) -> None:
        """Build mapping of character positions to page numbers."""
        lines = self.full_content.split('\n')
        current_char = 0
        
        # Estimate character distribution across pages
        total_chars = len(self.full_content)
        chars_per_page = total_chars / self.total_pages if self.total_pages > 0 else 1
        
        for page_num in range(1, self.total_pages + 1):
            # Estimate page boundaries
            page_start = int((page_num - 1) * chars_per_page)
            page_end = int(page_num * chars_per_page)

            # Count words using block text when available for accuracy
            word_count = 0
            block_count = 0
            try:
                page_obj = self.document.pages[page_num - 1]
                if hasattr(page_obj, 'blocks') and page_obj.blocks:
                    block_count = len(page_obj.blocks)
                    for block in page_obj.blocks:
                        text = None
                        if hasattr(block, 'text') and block.text:
                            text = block.text
                        elif hasattr(block, 'content') and block.content:
                            text = block.content
                        elif hasattr(block, 'get_text'):
                            try:
                                text = block.get_text()
                            except Exception:
                                text = None
                        if text:
                            word_count += len(str(text).split())
            except (IndexError, AttributeError):
                pass

            # Fallback to content slice if blocks yielded nothing
            if word_count == 0:
                page_content = self.full_content[page_start:page_end]
                word_count = len(page_content.split())
            
            page_info = PageInfo(
                page_number=page_num,
                char_start=page_start,
                char_end=page_end,
                block_count=block_count,
                word_count=word_count
            )
            
            self.pages[page_num] = page_info
            
            # Map character positions to page
            for char_pos in range(page_start, page_end):
                self.char_to_page[char_pos] = page_num
        
        logger.info(f"Page map built: {self.total_pages} pages, "
                   f"{total_chars} characters total")
        
        if not self.pages:
            self._fallback_coverage()

    def _fallback_coverage(self) -> None:
        """Fallback coverage when detailed mapping fails."""
        self.coverage_fallback = True
        total_chars = len(self.full_content)
        total_words = len(self.full_content.split())
        avg_words = total_words // self.total_pages if self.total_pages else 0
        self.pages = {
            i: PageInfo(
                page_number=i,
                char_start=0,
                char_end=0,
                block_count=0,
                word_count=avg_words
            )
            for i in range(1, self.total_pages + 1)
        }
        logger.info("Using fallback coverage: evenly distributing content across pages")
    
    def get_page_for_line(self, line_num: int) -> int:
        """
        Get page number for a given line number.
        
        Args:
            line_num: Line number in content
            
        Returns:
            Estimated page number (1-indexed)
        """
        lines = self.full_content.split('\n')
        if line_num >= len(lines):
            return self.total_pages
        
        # Calculate character position for this line
        char_pos = sum(len(lines[i]) + 1 for i in range(min(line_num, len(lines))))
        
        # Look up page
        return self.char_to_page.get(char_pos, 1)
    
    def get_page_range_for_content(self, start_char: int, end_char: int) -> str:
        """
        Get page range for a section of content.
        
        Args:
            start_char: Starting character position
            end_char: Ending character position
            
        Returns:
            Page range as "X-Y" or "X" for single page
        """
        start_page = self.char_to_page.get(min(start_char, len(self.full_content) - 1), 1)
        end_page = self.char_to_page.get(min(end_char, len(self.full_content) - 1), 1)
        
        if start_page == end_page:
            return str(start_page)
        else:
            return f"{start_page}-{end_page}"
    
    def get_coverage_report(self) -> Dict[str, Any]:
        """
        Get report of page coverage in document.
        
        Returns:
            Coverage statistics for all pages
        """
        pages_with_content = sum(1 for p in self.pages.values() if p.word_count > 0)
        total_words = sum(p.word_count for p in self.pages.values())
        total_blocks = sum(p.block_count for p in self.pages.values())
        
        return {
            "total_pages": self.total_pages,
            "pages_with_content": pages_with_content,
            "coverage_percentage": (pages_with_content / self.total_pages * 100) if self.total_pages > 0 else 0,
            "total_words": total_words,
            "total_blocks": total_blocks,
            "average_words_per_page": total_words // self.total_pages if self.total_pages > 0 else 0,
            "pages_detail": [
                {
                    "page": p.page_number,
                    "word_count": p.word_count,
                    "block_count": p.block_count,
                    "char_range": f"{p.char_start}-{p.char_end}"
                }
                for p in sorted(self.pages.values(), key=lambda x: x.page_number)
            ]
        }
    
    def verify_complete_coverage(self) -> Tuple[bool, str]:
        """
        Verify that all pages have been covered in parsing.
        
        Returns:
            Tuple of (is_complete, message)
        """
        coverage = self.get_coverage_report()
        
        percentage = coverage["coverage_percentage"]
        
        if percentage >= 95:
            return True, f"✅ Excellent coverage: {percentage:.1f}% of pages have content"
        elif percentage >= 80:
            return True, f"⚠️ Good coverage: {percentage:.1f}% of pages have content"
        else:
            return False, f"❌ Incomplete coverage: Only {percentage:.1f}% of pages have content"
    
    def get_pages_without_content(self) -> List[int]:
        """
        Get list of page numbers that have no content.
        
        Returns:
            List of page numbers (1-indexed) with no word content
        """
        return [p.page_number for p in self.pages.values() if p.word_count == 0]
    
    def get_page_statistics(self, page_num: int) -> Optional[Dict[str, Any]]:
        """
        Get statistics for a specific page.
        
        Args:
            page_num: Page number (1-indexed)
            
        Returns:
            Page statistics or None if page not found
        """
        if page_num not in self.pages:
            return None
        
        page = self.pages[page_num]
        return {
            "page_number": page.page_number,
            "word_count": page.word_count,
            "block_count": page.block_count,
            "character_range": f"{page.char_start}-{page.char_end}",
            "estimated_content_length": page.char_end - page.char_start
        }
