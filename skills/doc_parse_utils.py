"""
Enhanced PDF parsing utilities with intelligent section detection.

This module provides comprehensive PDF document parsing using Docling V4 backend
with enhanced article/section detection using regex patterns from docvals.py and doc.py.

PRIMARY FUNCTION: parse_pdf_by_toc()
- Uses intelligent TOC-based detection with LLM assistance
- Supports both standard Credit Agreements and Amended & Restated documents
- Falls back to article-based detection if no TOC found
- More accurate and structured than legacy parse_pdf_by_articles()

ALIASES: parse_pdf, enhanced_parse_pdf (both point to parse_pdf_by_toc)
"""

import logging
from pathlib import Path
import re
import json
import sys
import subprocess
import uuid
from rapidfuzz import fuzz, process
from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import SectionHeaderItem, TableItem, TextItem, ListItem
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from config import Config

# Regex patterns for section detection (inline definitions to avoid external dependencies)
re_sec_roman = re.compile(r'^\s*\([ivxlcdm]+\)', re.IGNORECASE)
re_sec_roman_big = re.compile(r'^\s*[IVXLCDM]+\.')
re_sec_roman_little = re.compile(r'^\s*\([ivxlcdm]+\)')
re_sec_alpha = re.compile(r'^\s*\([a-z]\)')
re_sec_dotno = re.compile(r'^\s*\d+\.\d+')
re_sec_intno = re.compile(r'^\s*\d+\.')
re_sec_lstno = re.compile(r'^\s*\(\d+\)')
re_sec_lstno2 = re.compile(r'^\s*\d+\)')
re_sec_parno = re.compile(r'^\s*¶\s*\d+')
re_sec_suffx = re.compile(r'\s*\.\s*$')
re_art_wording = re.compile(r'^\s*article\s+', re.IGNORECASE)
re_preface_art = re.compile(r'^\s*article\s+[ivxlcdm0-9]+', re.IGNORECASE)
re_art_not_following = re.compile(r'article\s+(?!of\b|and\b|or\b|the\b)', re.IGNORECASE)
re_wdptag = re.compile(r'<[^>]+>')
re_pgbreak_tag = re.compile(r'<!-- pagebreak -->')
re_table_beg_marker = re.compile(r'<!-- table_begin -->')
re_table_end_marker = re.compile(r'<!-- table_end -->')

# Simple implementations to avoid dependency issues
def num_leading_spaces(text):
    """Count leading spaces in text."""
    if not text:
        return 0
    return len(text) - len(text.lstrip(' '))

def is_block_indented(text):
    """Check if text block is indented."""
    if not text:
        return False
    return num_leading_spaces(text) > 4

def has_hanging_indent(text):
    """Check if text has hanging indent."""
    if not text:
        return False
    lines = text.split('\n')
    if len(lines) < 2:
        return False
    first_spaces = num_leading_spaces(lines[0])
    second_spaces = num_leading_spaces(lines[1]) if len(lines) > 1 else 0
    return second_spaces > first_spaces


logger = logging.getLogger()


# Initialize local Ollama LLM
try:
    # Use Qwen3-14B for both tasks (high quality local model)
    llm_sonnet = ChatOllama(
        model="hf.co/unsloth/Qwen3-14B-GGUF:Q4_K_XL",
        base_url=Config.OLLAMA_BASE_URL,
        temperature=0.0,
        num_ctx=8192,
        keep_alive=-1,
        num_parallel=Config.OLLAMA_NUM_PARALLEL,
        timeout=Config.REQUEST_TIMEOUT,
    )
    LLM_SONNET_AVAILABLE = True
    logger.info("✓ Local LLM (Qwen3-14B) initialized for detailed parsing")
except Exception as e:
    logger.warning("Failed to initialize local LLM for sonnet tasks: %s", e)
    llm_sonnet = None
    LLM_SONNET_AVAILABLE = False

try:
    # Use same model for faster tasks (can be swapped to a lighter model if needed)
    llm_haiku = ChatOllama(
        model="hf.co/unsloth/Qwen3-14B-GGUF:Q4_K_XL",
        base_url=Config.OLLAMA_BASE_URL,
        temperature=0.0,
        num_ctx=8192,
        keep_alive=-1,
        num_parallel=Config.OLLAMA_NUM_PARALLEL,
        timeout=Config.REQUEST_TIMEOUT,
    )
    LLM_HAIKU_AVAILABLE = True
    logger.info("✓ Local LLM (Qwen3-14B) initialized for quick tasks")
except Exception as e:
    logger.warning("Failed to initialize local LLM for haiku tasks: %s", e)
    llm_haiku = None
    LLM_HAIKU_AVAILABLE = False

def create_section_id(section_title: str, index: int) -> str:
    """Create a section ID from the title.

    Uses index-based ID to ensure uniqueness, especially for A&R documents
    with duplicate section titles.
    """
    return f"section_{index}"

# ==============================================================================
# ⭐ PRIMARY PARSING FUNCTION - USE THIS FOR ALL NEW CODE ⭐
# ==============================================================================

def parse_pdf_by_toc(
    pdf_path = None,
    output_dir: str = "sections",
    generate_summaries: bool = True,
    use_sonnet_for_summaries: bool = True,
    toc_max_pages: int = 20,
    fuzzy_threshold: int = 75,
    debug_toc: bool = False,
    use_llm_matching: bool = True,
    debug_matching: bool = False,
    pre_filter_article: bool = True,
    debug_classification: bool = False
) -> dict:
    """
    Parse PDF and save each high-level section as a separate text file.

    Uses intelligent TOC-based detection with LLM parsing and matching.
    Falls back to ARTICLE-based detection if no TOC is found.

    Args:
        pdf_path: Path to the PDF file (optional if doc_id is provided)
        output_dir: Directory to save output files
        generate_summaries: Whether to generate AI summaries (costs money)
        use_sonnet_for_summaries: Use Sonnet (better) vs Haiku (cheaper) for summaries
        toc_max_pages: How many pages to search for TOC tables
        fuzzy_threshold: Minimum similarity score for header matching (0-100) - used if use_llm_matching=False
        debug_toc: If True, print LLM input/output and exit early (for debugging TOC parsing)
        use_llm_matching: If True, use Haiku to match headers to TOC (more accurate, costs more)
                          If False, use fuzzy string matching (faster, free)
        debug_matching: If True, print each header matching prompt and response
        pre_filter_article: If True, only call LLM for headers containing "Article" (saves cost)
                            If False, call LLM for all section headers
        debug_classification: If True, print document classification prompt and response

    Returns:
        Dictionary of section metadata
    """
    # Validate input
    if not pdf_path:
        raise ValueError("pdf_path must be provided")

    # Convert string path to Path object if needed
    if isinstance(pdf_path, str):
        pdf_path = Path(pdf_path)

    # Validate file exists
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    output_path = Path(output_dir)

    # Clear output directory if it exists (remove old files from previous runs)
    if output_path.exists():
        import shutil
        shutil.rmtree(output_path)
        print(f"   🗑️  Cleared existing output directory: {output_dir}/")
    output_path.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"PDF Ingestion and Deconstruction")
    print(f"{'='*60}")
    print(f"Input: {pdf_path}")
    print(f"Output: {output_dir}/")
    print(f"{'='*60}\n")

    # -------------------------------------------------------------------------
    # Step 1: Convert PDF with Docling
    # -------------------------------------------------------------------------
    print("STEP 1: Converting PDF with Docling...")
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    doc = result.document
    print("   ✓ PDF converted successfully\n")

    # -------------------------------------------------------------------------
    # Step 1.5: Classify Document Type (Credit Agreement vs Amended & Restated)
    # -------------------------------------------------------------------------
    print("STEP 1.5: Classifying document type with Claude Haiku...")
    doc_classification = classify_document_type(doc, debug=debug_classification)
    is_amended_restated = doc_classification["document_type"] == "amended_and_restated"

    print(f"   → Document type: {doc_classification['document_type'].upper().replace('_', ' ')}")
    print(f"   → Confidence: {doc_classification['confidence']}")
    print(f"   → Reasoning: {doc_classification['reasoning']}")
    if is_amended_restated:
        print(f"   ⚠️  Amended & Restated document detected - duplicate articles will be captured")
    print()

    # -------------------------------------------------------------------------
    # Step 2: Extract tables from first N pages
    # -------------------------------------------------------------------------
    print(f"STEP 2: Extracting tables from first {toc_max_pages} pages...")
    tables, toc_pages, toc_locations = extract_tables_from_first_pages(doc, max_pages=toc_max_pages)
    print(f"   ✓ Found {len(tables)} tables on pages: {sorted(toc_pages) if toc_pages else 'none'}\n")

    # -------------------------------------------------------------------------
    # Step 3: Parse TOC with LLM
    # -------------------------------------------------------------------------
    print("STEP 3: Parsing Table of Contents with Claude Sonnet...")
    toc_data = parse_toc_with_llm(tables, toc_locations, debug=debug_toc)

    use_toc_matching = toc_data["is_toc_found"] and len(toc_data["high_level_sections"]) > 0

    if use_toc_matching:
        toc_sections = toc_data["high_level_sections"]
        toc_by_source = toc_data.get("toc_sections", {})

        print(f"   ✓ Found {len(toc_sections)} high-level sections across {len(toc_by_source)} TOCs:")

        # Show sections grouped by TOC source
        if toc_by_source:
            for source_name, sections in toc_by_source.items():
                if sections:
                    print(f"      {source_name}: {len(sections)} sections")
                    for i, sec in enumerate(sections[:3]):  # Show first 3 per TOC
                        print(f"        {i+1}. {sec['title']} (page {sec.get('page', '?')})")
                    if len(sections) > 3:
                        print(f"        ... and {len(sections) - 3} more")
        else:
            # Fallback to flat display
            for i, sec in enumerate(toc_sections[:5]):  # Show first 5
                source = sec.get('source_toc', 'Unknown TOC')
                print(f"      {i+1}. {sec['title']} (page {sec.get('page', '?')}) [{source}]")
            if len(toc_sections) > 5:
                print(f"      ... and {len(toc_sections) - 5} more")
    else:
        print("   ⚠ No TOC found - falling back to ARTICLE-based detection")
        toc_sections = []
        toc_by_source = {}
    print()

    # -------------------------------------------------------------------------
    # Step 4: Iterate through document and split by sections
    # -------------------------------------------------------------------------
    print("STEP 4: Processing document sections...")
    if use_toc_matching:
        print(f"   Matching method: {'LLM (Haiku)' if use_llm_matching else 'Fuzzy string matching'}")

    # Initialize tracking variables
    current_section_title = "Cover Page / Preamble"
    current_content = []
    section_counter = 0
    current_page_start = None
    current_page_end = None
    sections_metadata = {}
    matched_sections = set()  # Track which TOC sections we've matched (for standard docs)
    section_occurrence_count = {}  # Track occurrences of each section (for A&R docs)

    # Context buffer for LLM matching (stores recent text items)
    context_buffer = []  # List of recent text strings
    CONTEXT_BUFFER_SIZE = 10  # Number of recent items to keep for context

    def save_current_section():
        """Helper to save the current section and generate metadata."""
        nonlocal section_counter, current_page_start, current_page_end

        filename = f"{section_counter:02d}_{sanitize_filename(current_section_title)}.txt"
        filepath = output_path / filename
        section_text = '\n'.join(current_content) if current_content else ""

        # Save the text file (even if empty)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(section_text)
        content_lines = len(current_content) if current_content else 0
        print(f"   ✓ Saved: {filename} ({content_lines} lines)")

        # Generate metadata
        section_id = create_section_id(current_section_title, section_counter)
        page_range = f"{current_page_start or 1}-{current_page_end or 1}"

        # Generate AI summary if requested
        if generate_summaries:
            summary = generate_section_summary(
                current_section_title,
                section_text,
                llm_sonnet
            )
        else:
            summary = "Summary not generated (set generate_summaries=True to enable)"

        # Store metadata
        sections_metadata[section_id] = {
            "section_id": section_id,
            "title": current_section_title,
            "description": summary,
            "page_range": page_range,
            "file_path": str(filepath),
            "content": section_text  # Add content to metadata
        }

        section_counter += 1

    # First pass: collect all items with their indices for context lookup
    all_items = list(doc.iterate_items())

    # Process each item in the document
    for item_idx, (item, level) in enumerate(all_items):

        # Get page number for this item
        page_no = None
        if hasattr(item, 'prov') and item.prov and len(item.prov) > 0:
            page_no = item.prov[0].page_no

        # Track page numbers for current section
        if page_no is not None:
            if current_page_start is None:
                current_page_start = page_no
            current_page_end = page_no

        # Check if this is a section header that matches our TOC
        is_new_section = False
        new_section_title = None

        if isinstance(item, SectionHeaderItem):
            # Get the page number for this header
            header_page = page_no

            # DEBUG: Always print when we encounter a section header
            header_preview = item.text[:60] + "..." if len(item.text) > 60 else item.text
            print(f"\n   📄 Header found (page {header_page}): '{header_preview}'")

            # Skip headers on TOC pages - they're from the table of contents, not actual content
            if header_page and header_page in toc_pages:
                print(f"      ⏭️  SKIPPED - Header is on TOC page {header_page}")
                current_content.append(f"\n{item.text}")
                continue

            if use_toc_matching:
                match = None

                if use_llm_matching:
                    # Pre-filter: Only call LLM if header contains "Article" or "Section" (when enabled)
                    header_lower = item.text.lower()
                    if pre_filter_article and "article" not in header_lower and "section" not in header_lower:
                        print(f"      ⏭️  Skipping LLM - no 'Article' or 'Section' in header")
                        match = None
                    else:
                        # Gather context before (from buffer)
                        context_before = "\n".join(context_buffer[-5:]) if context_buffer else ""

                        # Gather context after (look ahead a few items)
                        context_after_items = []
                        for future_idx in range(item_idx + 1, min(item_idx + 6, len(all_items))):
                            future_item, _ = all_items[future_idx]
                            if isinstance(future_item, TextItem) and future_item.text.strip():
                                context_after_items.append(future_item.text.strip())
                            elif isinstance(future_item, SectionHeaderItem):
                                break  # Stop at next header
                        context_after = "\n".join(context_after_items)

                        # Use LLM to match
                        print(f"      🤖 Asking Haiku to match...")
                        match = match_header_with_llm(
                            item.text,
                            context_before,
                            context_after,
                            toc_sections,
                            debug=debug_matching
                        )

                    if match:
                        print(f"      ✅ LLM MATCHED → '{match['section']['title']}' (confidence: {match.get('confidence', 'unknown')})")
                else:
                    # Fuzzy string matching (legacy)
                    match = find_matching_toc_section_fuzzy(
                        item.text,
                        toc_sections,
                        threshold=fuzzy_threshold
                    )
                    if match:
                        print(f"      ✅ FUZZY MATCHED → '{match['section']['title']}' (score: {match['score']})")

                if match:
                    section_title = match["section"]["title"]

                    if is_amended_restated:
                        # For A&R documents: Allow duplicate sections, track occurrences
                        occurrence = section_occurrence_count.get(section_title, 0) + 1
                        section_occurrence_count[section_title] = occurrence

                        is_new_section = True
                        if occurrence > 1:
                            # Append occurrence number for duplicates
                            new_section_title = f"{section_title} (Occurrence {occurrence})"
                            print(f"      📋 Duplicate section #{occurrence} captured (A&R document)")
                        else:
                            new_section_title = section_title
                        matched_sections.add(section_title)
                    else:
                        # For standard documents: Skip duplicates
                        if section_title not in matched_sections:
                            is_new_section = True
                            new_section_title = section_title
                            matched_sections.add(new_section_title)
                        else:
                            print(f"      ⚠️  Already matched: '{section_title}' - SKIPPING")
                else:
                    print(f"      ❌ No match found")
            else:
                # Fallback: ARTICLE-based detection
                if is_article_header(item.text):
                    is_new_section = True
                    new_section_title = item.text
                    print(f"      ✅ ARTICLE detected: {item.text}")
                else:
                    print(f"      ❌ Not an ARTICLE header")

        # Update context buffer with text content
        if isinstance(item, TextItem) and item.text.strip():
            context_buffer.append(item.text.strip())
            if len(context_buffer) > CONTEXT_BUFFER_SIZE:
                context_buffer.pop(0)

        # Handle section boundary
        if is_new_section and new_section_title:
            # Save previous section with its page range
            save_current_section()

            # Start new section
            current_section_title = new_section_title
            current_content = [f"{new_section_title}\n{'='*len(new_section_title)}\n"]
            # IMPORTANT: Set start page to current page for this new section header
            current_page_start = page_no  # The page where this header appears
            current_page_end = page_no

        # Add content to current section
        else:
            if isinstance(item, SectionHeaderItem):
                current_content.append(f"\n{item.text}")
            elif isinstance(item, TextItem) and item.text.strip():
                current_content.append(item.text)
            elif isinstance(item, TableItem):
                try:
                    table_df = item.export_to_dataframe()
                    table_markdown = table_df.to_markdown(index=False)
                    current_content.append(f"\n{table_markdown}\n")
                except Exception:
                    current_content.append("\n[Table content could not be extracted]\n")

    # Save final section
    save_current_section()

    # -------------------------------------------------------------------------
    # Step 5: Save metadata JSON
    # -------------------------------------------------------------------------
    print(f"\nSTEP 5: Saving metadata...")

    # Create final output with document classification info
    final_metadata = {
        "_document_info": {
            "source_file": str(pdf_path),
            "document_type": doc_classification["document_type"],
            "classification_confidence": doc_classification["confidence"],
            "classification_reasoning": doc_classification["reasoning"],
            "is_amended_restated": is_amended_restated,
            "total_sections": section_counter,
            "duplicate_sections_found": sum(1 for v in section_occurrence_count.values() if v > 1) if is_amended_restated else 0
        },
        "sections": sections_metadata
    }

    metadata_file = output_path / "section_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(final_metadata, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Saved: {metadata_file}")

    # -------------------------------------------------------------------------
    # Step 6: In-Memory Storage (sections already saved to files in output_dir)
    # -------------------------------------------------------------------------
    print(f"\n✓ Sections saved to files in: {output_dir}/")
    print(f"   {len(sections_metadata)} section files created")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Document type: {doc_classification['document_type'].upper().replace('_', ' ')}")
    print(f"Total sections: {section_counter}")
    if is_amended_restated:
        duplicates = sum(1 for v in section_occurrence_count.values() if v > 1)
        print(f"Duplicate sections captured: {duplicates}")
    print(f"Output directory: {output_dir}/")
    print(f"Metadata file: {metadata_file}")
    if use_toc_matching:
        print(f"Detection method: TOC-based ({len(matched_sections)}/{len(toc_sections)} sections matched)")
    else:
        print(f"Detection method: ARTICLE-based (fallback)")
    print(f"{'='*60}\n")

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------
    # Clean up temporary file if we created one
    if temp_file_cleanup and temp_file_cleanup.exists():
        try:
            temp_file_cleanup.unlink()
            logger.debug("Cleaned up temporary file: %s", temp_file_cleanup)
        except Exception as e:
            logger.warning("Could not clean up temporary file %s: %s", temp_file_cleanup, e)

    return final_metadata


# ==============================================================================
# Document Classification Functions
# ==============================================================================

def extract_first_pages_text(doc, num_pages: int = 2) -> str:
    """
    Extract text from the first N pages of the document.

    Args:
        doc: Docling document object
        num_pages: Number of pages to extract text from

    Returns:
        Concatenated text from the first N pages
    """
    text_parts = []

    for item, level in doc.iterate_items():
        if hasattr(item, 'prov') and item.prov and len(item.prov) > 0:
            page_no = item.prov[0].page_no
            if page_no > num_pages:
                break  # Stop after N pages

            if isinstance(item, (SectionHeaderItem, TextItem)):
                if item.text.strip():
                    text_parts.append(item.text.strip())

    return "\n".join(text_parts)


def classify_document_type(doc, debug: bool = False) -> dict:
    """
    Use Claude Haiku to classify the document type based on the first 2 pages.

    Determines if the document is:
    - A standard Credit Agreement (single set of articles)
    - An Amended and Restated Agreement (may have duplicate articles)

    Args:
        doc: Docling document object
        debug: If True, print the classification prompt and response

    Returns:
        {
            "document_type": "credit_agreement" | "amended_and_restated",
            "confidence": "high" | "medium" | "low",
            "reasoning": "brief explanation"
        }
    """
    first_pages_text = extract_first_pages_text(doc, num_pages=2)

    if not first_pages_text:
        return {
            "document_type": "credit_agreement",
            "confidence": "low",
            "reasoning": "No text found in first 2 pages - defaulting to standard credit agreement"
        }

    prompt = f"""Analyze the beginning of this legal document to classify its type.

    DOCUMENT TEXT (First 2 pages):
    {first_pages_text[:4000]}

    TASK: Classify this document as one of the following types:

    1. "credit_agreement" - A standard Credit Agreement with a single set of articles/sections
    2. "amended_and_restated" - An Amended and Restated Agreement that contains:
    - An amendment section with its own articles (often shorter, referencing changes)
    - The original/restated credit agreement with its own complete set of articles

    IMPORTANT INDICATORS for "amended_and_restated":
    - Title contains "AMENDED AND RESTATED" or "AMENDMENT AND RESTATEMENT"
    - References to an "Existing Credit Agreement" being amended
    - Mentions of "Third Amendment", "Second Amendment", etc.
    - Structure that restates/incorporates an existing agreement
    - Phrases like "hereby amend and restate" or "amended and restated in its entirety"

    IMPORTANT INDICATORS for standard "credit_agreement":
    - Title is just "CREDIT AGREEMENT" or "LOAN AGREEMENT"
    - No references to amending a prior agreement
    - Appears to be an original/new agreement

    RESPOND WITH ONLY a JSON object (no markdown, no explanation):
    {{"document_type": "credit_agreement"|"amended_and_restated", "confidence": "high"|"medium"|"low", "reasoning": "brief explanation"}}"""

    if debug:
        print("\n" + "="*80)
        print("DEBUG: DOCUMENT CLASSIFICATION PROMPT")
        print("="*80)
        print(prompt)
        print("="*80)

    try:
        response = llm_haiku.invoke(prompt)
        content = response.content.strip()

        if debug:
            print("\nDEBUG: CLASSIFICATION RESPONSE")
            print("-"*40)
            print(content)
            print("-"*40 + "\n")

        # Handle potential markdown code blocks
        if content.startswith("```"):
            content = re.sub(r'^```(?:json)?\n?', '', content)
            content = re.sub(r'\n?```$', '', content)

        result = json.loads(content)
        return result

    except (json.JSONDecodeError, Exception) as e:
        print(f"   ⚠️  Classification error: {e}")
        return {
            "document_type": "credit_agreement",
            "confidence": "low",
            "reasoning": f"Classification failed: {str(e)} - defaulting to standard"
        }


# ==============================================================================
# Main Parsing Function (Primary API)
# ==============================================================================

# Main parsing function - use this for all new code
parse_pdf = parse_pdf_by_toc  # Primary API
enhanced_parse_pdf = parse_pdf_by_toc  # Alternative alias

# ==============================================================================
# TOC Extraction Functions
# ==============================================================================

def find_toc_start_page(doc, search_pages: int = 50) -> int | None:
    """
    Find the page where Table of Contents begins by looking for "Table of Contents" text.

    Args:
        doc: Docling document object
        search_pages: Maximum number of pages to search through

    Returns:
        Page number where TOC starts, or None if not found
    """
    for item, level in doc.iterate_items():
        if hasattr(item, 'prov') and item.prov and len(item.prov) > 0:
            page_no = item.prov[0].page_no
            if page_no > search_pages:
                break  # Stop searching after N pages

            # Check both SectionHeaderItem and TextItem
            if isinstance(item, (SectionHeaderItem, TextItem)):
                text_lower = item.text.lower().strip()
                # Look for "table of contents" - return first match
                if "table of contents" in text_lower:
                    return page_no

    return None


def find_all_tocs(doc, search_pages: int = 200) -> list[dict]:
    """
    Find all Table of Contents occurrences in the document.

    This handles documents with multiple TOCs, such as:
    - Main document TOC
    - Annex/Exhibit TOCs
    - Schedule TOCs
    - Appendix TOCs

    Args:
        doc: Docling document object
        search_pages: Maximum number of pages to search through

    Returns:
        List of dicts with TOC information: [{"page": int, "text": str, "section_type": str}, ...]
    """
    toc_locations = []

    for item, level in doc.iterate_items():
        if hasattr(item, 'prov') and item.prov and len(item.prov) > 0:
            page_no = item.prov[0].page_no
            if page_no > search_pages:
                break  # Stop searching after N pages

            # Check both SectionHeaderItem and TextItem
            if isinstance(item, (SectionHeaderItem, TextItem)):
                text_lower = item.text.lower().strip()
                original_text = item.text.strip()

                # Look for various TOC patterns
                toc_patterns = [
                    ("table of contents", "Main TOC"),
                    ("contents", "Contents"),
                    ("annex", "Annex TOC"),
                    ("exhibit", "Exhibit TOC"),
                    ("schedule", "Schedule TOC"),
                    ("appendix", "Appendix TOC"),
                ]

                for pattern, section_type in toc_patterns:
                    if pattern in text_lower:
                        # Avoid duplicates on same page
                        if not any(loc['page'] == page_no for loc in toc_locations):
                            toc_info = {
                                "page": page_no,
                                "text": original_text,
                                "section_type": section_type
                            }
                            toc_locations.append(toc_info)
                            break  # Only match first pattern per item

    return toc_locations


def extract_tables_from_first_pages(
    doc,
    max_pages: int = 20,
    toc_search_pages: int = 50,
    toc_range: int = 10
) -> tuple[list[str], set[int], list[dict]]:
    """
    Extract tables from TOC pages using smart detection with multiple TOC support.

    Strategy:
    1. Find all "Table of Contents" occurrences in the document
    2. For each TOC, extract tables from toc_page to toc_page + toc_range
    3. If no TOC marker found, fall back to first max_pages

    Args:
        doc: Docling document object
        max_pages: Fallback - how many pages to scan if no TOC marker found
        toc_search_pages: How many pages to search for "Table of Contents" text
        toc_range: How many pages after TOC start to scan for tables

    Returns:
        Tuple of (tables_markdown_list, set_of_page_numbers_with_tables, list_of_toc_locations)
    """
    # Find all TOC occurrences
    toc_locations = find_all_tocs(doc, search_pages=toc_search_pages)

    tables = []
    toc_pages = set()  # Track pages that contain tables (likely TOC)

    if toc_locations:
        print(f"   → Found {len(toc_locations)} Table of Contents:")
        for i, toc_info in enumerate(toc_locations):
            page = toc_info['page']
            section_type = toc_info['section_type']
            start_page = max(1, page - 2)  # Include pages before in case header is separate
            end_page = page + toc_range
            print(f"     {i+1}. {section_type} on page {page}, scanning pages {start_page}-{end_page}")

            # Extract tables for this TOC range
            for item, level in doc.iterate_items():
                if isinstance(item, TableItem):
                    if hasattr(item, 'prov') and item.prov and len(item.prov) > 0:
                        page_no = item.prov[0].page_no
                        if start_page <= page_no <= end_page:
                            try:
                                df = item.export_to_dataframe()
                                table_md = df.to_markdown(index=False)
                                # Tag table with TOC section info
                                table_entry = f"[Table from page {page_no} - {section_type}]\n{table_md}"
                                tables.append(table_entry)
                                toc_pages.add(page_no)
                            except Exception as e:
                                print(f"     ⚠ Error extracting table from page {page_no}: {e}")
    else:
        start_page = 1
        end_page = max_pages
        print(f"   → No 'Table of Contents' markers found, falling back to pages 1-{max_pages}")

        for item, level in doc.iterate_items():
            if isinstance(item, TableItem):
                if hasattr(item, 'prov') and item.prov and len(item.prov) > 0:
                    page_no = item.prov[0].page_no
                    if start_page <= page_no <= end_page:
                        try:
                            df = item.export_to_dataframe()
                            table_md = df.to_markdown(index=False)
                            tables.append(f"[Table from page {page_no}]\n{table_md}")
                            toc_pages.add(page_no)
                        except Exception as e:
                            print(f"   ⚠ Error extracting table from page {page_no}: {e}")

    return tables, toc_pages, toc_locations


def parse_toc_with_llm(tables_markdown: list[str], toc_locations: list[dict] = None, debug: bool = False) -> dict:
    """
    Use Claude Sonnet to identify TOCs and extract high-level sections from multiple TOCs.

    Args:
        tables_markdown: List of markdown table strings
        toc_locations: List of TOC location info from find_all_tocs
        debug: If True, print input/output and exit after LLM call

    Returns:
        {
            "is_toc_found": bool,
            "high_level_sections": [
                {"title": "DEFINITIONS AND CONSTRUCTION", "page": 1, "source_toc": "Main TOC"},
                ...
            ],
            "toc_sections": {
                "Main TOC": [...],
                "Annex TOC": [...],
                ...
            }
        }
    """
    if not tables_markdown:
        return {"is_toc_found": False, "high_level_sections": [], "toc_sections": {}}

    tables_text = "\n\n---\n\n".join(tables_markdown)

    # Include TOC location context if available
    toc_context = ""
    if toc_locations:
        toc_context = "\n\nTOC LOCATIONS FOUND:\n"
        for i, toc in enumerate(toc_locations):
            toc_context += f"{i+1}. {toc['section_type']} on page {toc['page']}: '{toc['text']}'\n"

    prompt = f"""Analyze these tables extracted from a legal/financial document that may contain multiple Table of Contents sections.

    TABLES:
    {tables_text}{toc_context}

    TASK:
    1. Identify ALL tables that represent Table of Contents (TOC) sections
    2. For EACH TOC found, extract ONLY the HIGH-LEVEL sections (main chapters/articles, NOT sub-sections)
    - High-level sections are main divisions like: "ARTICLE 1", "ARTICLE II", "SECTION 1", "1. DEFINITIONS", etc.
    - Do NOT include sub-sections like "1.1 Definitions", "2.3 Payments", "Section 4.5"
    - If a section has a number like "1." or "2." without decimals, it's likely high-level
    - ALL CAPS entries without decimal numbers are often high-level
    3. Group sections by their source TOC (Main TOC, Annex TOC, etc.)
    4. Clean up any OCR artifacts (e.g., "DEFINITIONSAND" should become "DEFINITIONS AND")
    5. Return as valid JSON

    OUTPUT FORMAT (strict JSON only, no markdown code blocks):
    {{
        "is_toc_found": true,
        "high_level_sections": [
            {{"title": "SECTION TITLE HERE", "page": 1, "source_toc": "Main TOC"}},
            {{"title": "ANNEX SECTION", "page": 50, "source_toc": "Annex TOC"}}
        ],
        "toc_sections": {{
            "Main TOC": [{{"title": "ARTICLE I", "page": 1}}, ...],
            "Annex TOC": [{{"title": "SECTION 1", "page": 50}}, ...]
        }}
    }}

    If no TOC is found, return:
    {{"is_toc_found": false, "high_level_sections": [], "toc_sections": {{}}}}

    IMPORTANT: Return ONLY the JSON object, nothing else."""

    # DEBUG: Print what goes into the LLM
    if debug:
        print("\n" + "="*80)
        print("DEBUG: LLM INPUT (PROMPT)")
        print("="*80)
        print(prompt)
        print("="*80 + "\n")

    try:
        print("   → Asking Sonnet to parse TOC...")
        response = llm_sonnet.invoke(prompt)
        content = response.content.strip()

        # DEBUG: Print what comes out of the LLM
        if debug:
            print("\n" + "="*80)
            print("DEBUG: LLM OUTPUT (RAW RESPONSE)")
            print("="*80)
            print(content)
            print("="*80 + "\n")

        # Handle potential markdown code blocks in response
        if content.startswith("```"):
            content = re.sub(r'^```(?:json)?\n?', '', content)
            content = re.sub(r'\n?```$', '', content)

        result = json.loads(content)

        # DEBUG: Print parsed result and exit
        if debug:
            print("\n" + "="*80)
            print("DEBUG: PARSED JSON RESULT")
            print("="*80)
            print(json.dumps(result, indent=2))
            print("="*80)
            print("\n🛑 DEBUG MODE: Exiting early after LLM call")
            import sys
            sys.exit(0)

        print(f"   ✓ TOC parsing complete")
        return result
    except json.JSONDecodeError as e:
        print(f"   ✗ Error parsing LLM response as JSON: {e}")
        print(f"   Raw response: {response.content[:500]}...")
        if debug:
            print("\n🛑 DEBUG MODE: Exiting after JSON parse error")
            import sys
            sys.exit(1)
        return {"is_toc_found": False, "high_level_sections": []}
    except Exception as e:
        print(f"   ✗ Error calling LLM: {e}")
        if debug:
            print("\n🛑 DEBUG MODE: Exiting after LLM error")
            import sys
            sys.exit(1)
        return {"is_toc_found": False, "high_level_sections": []}


# ==============================================================================
# Fuzzy Matching Functions (Legacy - kept for fallback)
# ==============================================================================

def normalize_text(text: str) -> str:
    """Normalize text for fuzzy matching comparison.

    Only normalizes whitespace and case. We keep ARTICLE/SECTION prefixes
    and numbers because they help with matching accuracy.
    """
    # Just normalize whitespace and uppercase - let fuzzy matching handle the rest
    text = ' '.join(text.split())
    return text.upper()


def find_matching_toc_section_fuzzy(header_text: str, toc_sections: list[dict], threshold: int = 75) -> dict | None:
    """
    Find the best matching TOC section for a given header using fuzzy matching.
    (Legacy function - kept for fallback/comparison)

    Args:
        header_text: The header text from the document
        toc_sections: List of {"title": ..., "page": ...} from TOC
        threshold: Minimum similarity score (0-100) to consider a match

    Returns:
        Matching TOC section dict or None if no match
    """
    if not toc_sections:
        return None

    normalized_header = normalize_text(header_text)

    # Build list of normalized TOC titles for matching
    normalized_toc = [(normalize_text(s["title"]), i) for i, s in enumerate(toc_sections)]
    toc_titles = [t[0] for t in normalized_toc]

    # Find best match using rapidfuzz if available
    match = process.extractOne(
        normalized_header,
        toc_titles,
        scorer=fuzz.ratio,
        score_cutoff=threshold
    )

    if match:
        matched_title, score, index = match
        return {
            "section": toc_sections[index],
            "score": score,
            "matched_title": matched_title,
            "match_method": "fuzzy"
        }

    return None


# ==============================================================================
# LLM-Based Section Matching
# ==============================================================================

def match_header_with_llm(
    header_text: str,
    context_before: str,
    context_after: str,
    toc_sections: list[dict],
    debug: bool = False
) -> dict | None:
    """
    Use Claude Haiku to determine if a header matches a TOC section.

    Args:
        header_text: The header text found in the document
        context_before: Text appearing before the header (for context)
        context_after: Text appearing after the header (for context)
        toc_sections: List of {"title": ..., "page": ...} from TOC
        debug: If True, print the prompt and response

    Returns:
        Matching TOC section dict or None if no match
    """
    if not toc_sections:
        return None

    # Build numbered list of TOC sections for the prompt
    toc_list = "\n".join([
        f"{i+1}. {sec['title']}"
        for i, sec in enumerate(toc_sections)
    ])

    prompt = f"""You are analyzing a legal document to identify section boundaries based on the Table of Contents.

    I found this header in the document:
    HEADER: "{header_text}"

    Context before header:
    {context_before[:500] if context_before else "(start of document)"}

    Context after header:
    {context_after[:500] if context_after else "(end of document)"}

    Here are the high-level sections extraacted from the Table of Contents:
    {toc_list}

    TASK: Does this header text LITERALLY match one of the TOC section titles based on the WORDS present?

    MATCHING RULES (STRICT - NO SEMANTIC INTERPRETATION):
    - Match ONLY if the header contains the same key words as a TOC entry and context shows that is is a valid Header
    - Reject headers that appear to be extracted from within the actual TOC table (look at context to figure this out). If it is directly pulled from within TOC, it is not valid.
    - Allow for minor variations: OCR artifacts, spacing differences, roman vs arabic numerals (I vs 1)
    - Handle both "ARTICLE X" and "SECTION X" style headers - match with corresponding TOC entries regardless of format
    - Sub-sections like "Section 1.05", "Section 2.3", "1.1 Something" are NOT high-level matches - return no match
    - "Section 1.05 Additional Currencies" does NOT match "ARTICLE II The Credits" - different words entirely
    - Use context ONLY to verify this is a real header (not a reference), NOT to determine which TOC entry it belongs to.
    - If the context shows the header being used in-line as part of a sentence (like "X" is defined by Article ...), it is not a valid header. Valid headers must appear to be a standalone section header.
    - DO NOT interpret meaning. "Currencies" and "Credits" are different words, not a match.

    RESPOND WITH ONLY a JSON object (no markdown, no explanation):
    - If it matches a TOC section: {{"match": true, "toc_index": <1-based index>, "confidence": "high"|"medium"|"low"}}
    - If no match: {{"match": false}}"""

    # DEBUG: Print the prompt being sent
    if debug:
        print("\n" + "="*80)
        print("DEBUG: HEADER MATCHING PROMPT")
        print("="*80)
        print(prompt)
        print("="*80)

    try:
        response = llm_haiku.invoke(prompt)
        content = response.content.strip()

        # DEBUG: Print the response
        if debug:
            print("\nDEBUG: HAIKU RESPONSE")
            print("-"*40)
            print(content)
            print("-"*40 + "\n")

        # Handle potential markdown code blocks
        if content.startswith("```"):
            content = re.sub(r'^```(?:json)?\n?', '', content)
            content = re.sub(r'\n?```$', '', content)

        result = json.loads(content)

        if result.get("match") and result.get("toc_index"):
            index = result["toc_index"] - 1  # Convert to 0-based
            if 0 <= index < len(toc_sections):
                return {
                    "section": toc_sections[index],
                    "confidence": result.get("confidence", "medium"),
                    "match_method": "llm"
                }

        return None

    except (json.JSONDecodeError, Exception) as e:
        print(f"      ⚠️  LLM matching error: {e}")
        return None


def clean_text_content(text):
    """Clean and normalize text content using doc.py cleaning logic.

    Adapted from clean_raw_text and filter_paragraph_text functions.
    """
    try:
        if not text:
            return ""

        # Remove page break tags
        text = re.sub(re_pgbreak_tag, "", text)
        # Remove document end tag, if present
        text = re.sub(r"[#]{2}DOCUMENT_END[#]{2}", "", text)
        # Remove table markers
        text = re.sub(re_table_beg_marker, "", text).strip()
        text = re.sub(re_table_end_marker, "", text).strip()
        # Remove any wdp tags
        text = re.sub(re_wdptag, "", text)
        # Handle special case where line(s) ends with a dash
        text = re.sub(r"-\n", r"-", text)
        # Normalize multiple newlines to single spaces
        text = re.sub(r"\n{1,2}", " ", text)
        # Normalize spacing
        text = " ".join(text.split())

        return text.strip()

    except Exception as e:
        logger.warning("Error cleaning text content: %s", e)
        return text if text else ""


def get_text_statistics(text):
    """Get text statistics for section validation (adapted from f_section logic).

    Returns dict with counts of punctuation patterns that help identify sentences vs headers.
    """
    try:
        clean_text = clean_text_content(text)

        # Patterns to identify sentences vs section headers (from f_section)
        re_puncts = r"[,;:]\s+\S"
        re_periods = r"\.\s{1,2}\S"
        re_lcwords = r"\b[a-z]{2,15}\b"
        re_sentend = rf'(\.\"?|[:;])({re_wdptag})?$'

        stats = {
            'length': len(clean_text),
            'n_puncts': len(list(re.finditer(re_puncts, clean_text))),
            'n_periods': len(list(re.finditer(re_periods, clean_text))),
            'n_lcwords': len(list(re.finditer(re_lcwords, clean_text))),
            'ends_like_sentence': bool(re.search(re_sentend, clean_text))
        }

        return stats

    except Exception as e:
        logger.warning("Error getting text statistics: %s", e)
        return {'length': 0, 'n_puncts': 0, 'n_periods': 0, 'n_lcwords': 0, 'ends_like_sentence': False}


def is_sentence_not_section(text):
    """Determine if text is a sentence rather than a section header.

    Uses logic from doc.py f_section to detect sentences that start with
    section references but are not actual section headers.
    """
    try:
        stats = get_text_statistics(text)

        # Threshold from f_section logic
        TEXT_LONG_THRESHOLD = 50
        is_long_enough = stats['length'] >= TEXT_LONG_THRESHOLD

        if not is_long_enough:
            return False

        clean_text = clean_text_content(text)

        # Pattern to identify sentences starting with section references
        re_secref = r"(Section|Clause|Article)\s+[^\s]{1,15}\s+[a-z]"
        starts_with_ref = re.match(re_secref, clean_text)

        # Strong indicators this is a sentence, not a section header
        has_sentence_structure = (
            stats['n_puncts'] > 1 or
            stats['n_periods'] >= 1 or
            stats['n_lcwords'] > 5
        )

        is_sentence = (
            starts_with_ref and
            has_sentence_structure and
            stats['ends_like_sentence']
        )

        if is_sentence:
            logger.debug("Identified as sentence (not section): %s", clean_text[:50])

        return is_sentence

    except Exception as e:
        logger.warning("Error checking if text is sentence: %s", e)
        return False


def validate_section_text(text):
    """Validate if text is a legitimate section using logic from doc.py f_section.

    This function implements the sentence vs section detection logic from the
    original f_section function to avoid false positives.
    """
    try:
        if not text or len(text.strip()) == 0:
            return False

        # Skip if text is unreasonably long (adapted from f_section)
        TEXT_CRAZY_LONG_THRESHOLD = 100000
        if len(text) > TEXT_CRAZY_LONG_THRESHOLD:
            logger.debug("Text is too long (%d chars) to be a section header", len(text))
            return False

        # Check if this is a sentence rather than a section header
        if is_sentence_not_section(text):
            return False

        return True

    except Exception as e:
        logger.warning("Error validating section text '%s': %s", text[:50] if text else "None", e)
        return False


def is_list_item(text):
    """Check if text represents a list item rather than a section (from doc.py)."""
    try:
        if not text:
            return False

        clean_text = clean_text_content(text)
        # Patterns for list items (adapted from is_list logic)
        # Be more specific to avoid matching section headers

        # Simple bullet patterns
        re_bullet = r"^\s*[•·-]\s+"

        # Alphabetic lists in parentheses: (a), (b), (i), (ii), etc.
        re_alpha_paren = r"^\s*\([a-z]+\)\s+"

        # Roman numeral lists in parentheses: (i), (ii), (iii), etc.
        re_roman_paren = r"^\s*\([ivxlcdm]+\)\s+"

        # Number lists in parentheses: (1), (2), etc.
        re_num_paren = r"^\s*\(\d+\)\s+"

        # Single lowercase letters followed by period: a., b., c. (but not Roman numerals)
        # Exclude Roman numerals (i, ii, iii, iv, v, x, etc.) and uppercase letters
        re_single_letter = r"^\s*[a-h,j-z]\.\s+(?!\d)"  # Exclude i,v,x which are Roman numerals

        list_patterns = [re_bullet, re_alpha_paren, re_roman_paren, re_num_paren, re_single_letter]

        for pattern in list_patterns:
            if re.match(pattern, clean_text, re.IGNORECASE):
                logger.debug("Identified as list item: %s", clean_text[:30])
                return True
        return False

    except Exception as e:
        logger.warning("Error checking if text is list item: %s", e)
        return False


def has_proper_section_formatting(text):
    """Check if text has proper section header formatting."""
    try:
        if not text:
            return False

        # Check indentation and formatting using docutils functions
        leading_spaces = num_leading_spaces(text)
        is_indented = is_block_indented(text)
        has_hanging = has_hanging_indent(text)

        # Section headers typically are not heavily indented or have hanging indents
        # unless they are subsections
        if leading_spaces > 20 or has_hanging:
            logger.debug("Rejected due to formatting: spaces=%d, hanging=%s", leading_spaces, has_hanging)
            return False

        return True

    except Exception as e:
        logger.warning("Error checking section formatting: %s", e)
        return True  # Default to allowing if we can't check


def check_text_quality(text):
    """Check if text meets quality standards for section headers."""
    try:
        if not text:
            return False

        clean_text = clean_text_content(text)

        # Basic quality checks
        # Too short to be meaningful
        if len(clean_text.strip()) < 3:
            return False

        # Too long to be a header (adapted from docparse.py errorCheck logic)
        if len(clean_text) > 500:  # Reasonable limit for section headers
            logger.debug("Text too long to be section header: %d chars", len(clean_text))
            return False

        # Check for excessive special characters (corrupted text)
        special_char_ratio = len(re.findall(r'[^\w\s\-\.\(\),:]', clean_text)) / len(clean_text)
        if special_char_ratio > 0.3:  # More than 30% special chars suggests corruption
            logger.debug("Too many special characters, likely corrupted: %.2f ratio", special_char_ratio)
            return False

        # Check for reasonable word structure
        words = clean_text.split()
        if len(words) == 0:
            return False

        # Sections should have reasonable word lengths
        avg_word_length = sum(len(word.strip('.,()')) for word in words) / len(words)
        if avg_word_length < 1 or avg_word_length > 20:  # Unusually short/long words
            logger.debug("Unusual average word length: %.2f", avg_word_length)
            return False

        return True

    except Exception as e:
        logger.warning("Error checking text quality: %s", e)
        return True  # Default to allowing if we can't check
def extract_section_number_and_type(text):
    """Extract section number and determine section type using docvals patterns.

    Returns tuple of (section_number, section_type, prefix) or None if not found.
    """
    try:
        clean_text = text.strip()

        # Try different section number patterns
        patterns = [
            # ARTICLE with Roman numerals
            (rf"^({re_art_wording})\s+({re_sec_roman}){re_sec_suffx}?", "article", "roman"),
            # ARTICLE with integers
            (rf"^({re_art_wording})\s+({re_sec_intno})", "article", "integer"),
            # SECTION with integer
            (rf"^(SECTION|Section)\s+({re_sec_intno})", "section", "integer"),
            # SECTION with dotted number
            (rf"^(SECTION|Section)\s+({re_sec_dotno})", "section", "dotted"),
            # SECTION with Roman numeral
            (rf"^(SECTION|Section)\s+({re_sec_roman})", "section", "roman"),
            # Direct dotted numbers (e.g., "1.1 Defined Terms", "2.05 Payment Terms")
            (rf"^({re_sec_dotno})\s+", "section", "dotted"),
            # Simple decimal numbers (e.g., "1.1", "2.01")
            (r"^(\d+\.\d+)\s+", "section", "decimal"),
            # Simple integers with period (e.g., "1.", "2.")
            (r"^(\d+\.)\s+", "section", "numbered"),
            # Simple list numbers
            (rf"^({re_sec_lstno})\s+", "section", "list"),
            # Roman numerals standalone
            (rf"^({re_sec_roman}){re_sec_suffx}?\s+", "section", "roman"),
            # Alphabetic sections
            (rf"^({re_sec_alpha}){re_sec_suffx}\s+", "section", "alpha"),
        ]

        for pattern, sec_type, num_type in patterns:
            match = re.match(pattern, clean_text, re.IGNORECASE | re.VERBOSE)
            if match:
                if sec_type == "article":
                    prefix = match.group(1)
                    number = match.group(2)
                else:
                    if "SECTION" in pattern:
                        prefix = match.group(1) if match.lastindex >= 1 else ""
                        number = match.group(2) if match.lastindex >= 2 else match.group(1)
                    else:
                        prefix = ""
                        number = match.group(1)

                logger.debug(f"Extracted {sec_type} {num_type}: '{number}' from '{clean_text[:50]}'")
                return number.strip(), sec_type, prefix.strip()

        return None

    except Exception as e:
        logger.warning("Error extracting section info from '%s': %s", text[:50], e)
        return None


def is_article_header(text):
    """Enhanced article header detection using regex patterns from docvals.py and doc.py.

    This method uses established parsing logic to detect article/section headers
    with regex-first approach and minimal LLM fallback for edge cases.
    """
    try:

        # First validate that this could be a section using doc.py logic
        if not validate_section_text(text):
            return False

        # Check if this is a list item rather than a section header
        if is_list_item(text):
            return False

        # Check proper section formatting
        if not has_proper_section_formatting(text):
            return False

        # Check text quality
        if not check_text_quality(text):
            return False

        # Extract section information using docvals patterns
        section_info = extract_section_number_and_type(text)
        if section_info:
            number, sec_type, prefix = section_info

            # Accept any valid section pattern, even without ARTICLE/SECTION prefix
            # This handles formats like "1.1 Defined Terms", "I. GENERAL PROVISIONS"
            if sec_type in ["article", "section"]:
                return True
        # Additional manual patterns for common legal document formats
        clean_text = text.strip()

        # Check for simple capitalized section titles (common in legal docs)
        capitalized_section_patterns = [
            r"^[A-Z][A-Z\s&,-]+$",  # All caps like "DEFINITIONS AND INTERPRETATION"
            r"^\d+\.\s*[A-Z]",      # Number + period + space + capital letter
            r"^[IVX]+\.\s*[A-Z]",   # Roman + period + space + capital letter
        ]

        for pattern in capitalized_section_patterns:
            if re.match(pattern, clean_text):
                # Additional checks to avoid false positives
                if len(clean_text) < 100 and not re.search(r'\blower\b|\bcase\b', clean_text):
                    return True

        return False

        # LLM fallback disabled - relying on regex patterns only
        logger.debug("No article header patterns matched for text: %s", clean_text)
        return False

    except (AttributeError, TypeError) as e:
        logger.warning("Error checking article header for text '%s': %s", text, e)
        return False


def sanitize_filename(text):
    """Sanitize filename with defensive validation."""
    if not text or not isinstance(text, str):
        logger.debug("Filename sanitization failed: invalid input")
        return "unknown_section"

    try:
        filename = re.sub(r"[^\w\s-]", "", text)
        filename = re.sub(r"[-\s]+", "_", filename)
        result = filename[:100]
        return result if result else "unnamed_section"
    except (AttributeError, TypeError) as e:
        logger.warning("Error sanitizing filename for text '%s': %s", text, e)
        return "error_section"


def safe_get_page_number(item, default: int = 1) -> int:
    """Safely extract page number from item with fallbacks."""
    try:
        if hasattr(item, "prov") and item.prov and len(item.prov) > 0:
            page_no = getattr(item.prov[0], "page_no", default)
            return page_no if isinstance(page_no, int) and page_no > 0 else default
    except (AttributeError, IndexError, TypeError):
        logger.debug("Could not extract page number from item, using default: %d", default)

    return default


def generate_section_summary(section_title: str, section_content: str, llm_sonnet) -> str:
    """Generate a comprehensive summary of a section using Claude Sonnet 4.5 with defensive error handling."""
    # Input validation
    if not section_title or not isinstance(section_title, str):
        section_title = "Unknown Section"
        logger.warning("Invalid section title provided, using default")

    if not section_content or not isinstance(section_content, str):
        logger.warning("Invalid or empty section content provided")
        return "No content available for summary generation."

    # Truncate content if too long to prevent API errors
    max_content_length = 50000  # Adjust based on model limits
    if len(section_content) > max_content_length:
        section_content = section_content[:max_content_length] + "... [Content truncated]"
        logger.warning("Section content truncated for summary generation")

    # Check if LLM SONNET is available
    if not LLM_SONNET_AVAILABLE or llm_sonnet is None:
        logger.warning("LLM SONNET not available, returning basic summary")
        return f"Summary unavailable - LLM SONNET service not accessible. Section contains {len(section_content)} characters of content."

    # Create prompt for summary generation
    system_prompt = """You are analyzing a section from a credit agreement. Generate a comprehensive 3-4 sentence summary that serves as a quick lookup guide for credit analysts. The summary should highlight ALL key concepts, provisions, terms, and topics covered in this section so analysts can quickly determine if this section contains information relevant to their question."""

    user_prompt = f"""Section Title: {section_title}

    Section Content:
    {section_content}

    Provide only the summary, no preamble."""

    template = ChatPromptTemplate([("system", system_prompt), ("user", user_prompt)])

    try:
        logger.debug("   → Generating AI summary for: %s", section_title)
        response = llm_sonnet.invoke(template.invoke({}))

        if hasattr(response, "content") and response.content:
            summary = response.content.strip()
            if summary:
                logger.debug("   ✓ Summary generated successfully")
                return summary

        # Fallback if response is empty
        logger.warning("Empty response from LLM for section: %s", section_title)
        return f"Summary generation returned empty response for section: {section_title}"

    except Exception as e:
        logger.exception("   ✗ Error generating summary for '%s': %s", section_title, e)
        return f"Summary generation failed due to error: {str(e)[:200]}..."


def extract_article_number(section_title: str) -> str:
    """Extract article number from section title to create section_id with defensive validation."""
    try:
        if not section_title or not isinstance(section_title, str):
            logger.debug("Invalid section title for article extraction")
            return "unknown_article"

        # Enhanced regex-based extraction using patterns from doc.py and docparse.py
        logger.debug("Extracting identifier using enhanced regex patterns from: %s", section_title)

        # Clean the title using existing text cleaning methods
        clean_title = clean_text_content(section_title)

        # Extended Roman numeral mapping for comprehensive coverage
        roman_map = {
            "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8,
            "IX": 9, "X": 10, "XI": 11, "XII": 12, "XIII": 13, "XIV": 14, "XV": 15,
            "XVI": 16, "XVII": 17, "XVIII": 18, "XIX": 19, "XX": 20
        }

        # Use direct regex patterns to avoid syntax issues with imported patterns
        # Pattern 1: ARTICLE with Roman numerals
        match = re.search(r"(ARTICLE|Article)\s+([IVX]+)", clean_title, re.IGNORECASE)
        if match:
            roman = match.group(2).upper()
            article_num = roman_map.get(roman, 0)
            logger.debug("Extracted Roman article number: %s -> %d", roman, article_num)
            return f"article_{article_num}"

        # Pattern 2: ARTICLE with Arabic numerals
        match = re.search(r"(ARTICLE|Article)\s+(\d+)", clean_title, re.IGNORECASE)
        if match:
            article_num = match.group(2)
            logger.debug("Extracted Arabic article number: %s", article_num)
            return f"article_{article_num}"

        # Pattern 3: SECTION with dotted numbers
        match = re.search(r"(SECTION|Section)\s+(\d+\.\d+)", clean_title, re.IGNORECASE)
        if match:
            section_num = match.group(2).split('.')[0]  # Take first part before dot
            logger.debug("Extracted dotted section number: %s", section_num)
            return f"section_{section_num}"

        # Pattern 4: SECTION with integer numbers
        match = re.search(r"(SECTION|Section)\s+(\d+)", clean_title, re.IGNORECASE)
        if match:
            section_num = match.group(2)
            logger.debug("Extracted integer section number: %s", section_num)
            return f"section_{section_num}"

        # Pattern 5: Direct numbered sections (1., 2., 12.)
        match = re.search(r"^(\d+\.)\s+", clean_title)
        if match:
            section_num = match.group(1).rstrip('.')
            logger.debug("Extracted direct section number: %s", section_num)
            return f"section_{section_num}"

        # Pattern 6: Direct decimal sections (1.1, 2.05, 3.4)
        match = re.search(r"^(\d+\.\d+)\s+", clean_title)
        if match:
            section_num = match.group(1).split('.')[0]  # Take integer part
            logger.debug("Extracted decimal section number: %s", section_num)
            return f"section_{section_num}"

        # Pattern 7: Roman numeral sections standalone
        match = re.search(r"^([IVX]+)\.\s+", clean_title)
        if match:
            roman = match.group(1).upper()
            section_num = roman_map.get(roman, 0)
            logger.debug("Extracted standalone Roman section: %s -> %d", roman, section_num)
            return f"section_{section_num}"        # Pattern 7: PART sections
        part_pattern = r"(PART|Part)\s+([IVX]+|\d+)"
        match = re.search(part_pattern, clean_title, re.IGNORECASE)
        if match:
            part_identifier = match.group(2).upper()
            part_num = roman_map.get(part_identifier, part_identifier)
            logger.debug("Extracted part number: %s", part_num)
            return f"part_{part_num}"

        # Pattern 8: CHAPTER sections
        chapter_pattern = r"(CHAPTER|Chapter)\s+([IVX]+|\d+)"
        match = re.search(chapter_pattern, clean_title, re.IGNORECASE)
        if match:
            chapter_identifier = match.group(2).upper()
            chapter_num = roman_map.get(chapter_identifier, chapter_identifier)
            logger.debug("Extracted chapter number: %s", chapter_num)
            return f"chapter_{chapter_num}"

        # Special cases and fallback patterns
        upper_title = clean_title.upper()

        # Handle special document sections
        if any(word in upper_title for word in ["COVER PAGE", "TITLE PAGE"]):
            return "article_0"
        if "TABLE OF CONTENTS" in upper_title:
            return "section_toc"
        if "PRELIMINARY STATEMENTS" in upper_title:
            return "section_preliminary"
        if "SCHEDULES" in upper_title and len(clean_title) < 20:
            return "section_schedules"
        if "EXHIBITS" in upper_title and len(clean_title) < 20:
            return "section_exhibits"
        if "DEFINITIONS" in upper_title and "ACCOUNTING" in upper_title:
            return "section_definitions"

        # Common legal document sections
        section_mappings = {
            "DEFINITIONS": "definitions",
            "REPRESENTATIONS": "representations",
            "WARRANTIES": "warranties",
            "COVENANTS": "covenants",
            "CONDITIONS PRECEDENT": "conditions_precedent",
            "CONDITIONS SUBSEQUENT": "conditions_subsequent",
            "EVENTS OF DEFAULT": "events_default",
            "REMEDIES": "remedies",
            "INDEMNIFICATION": "indemnification",
            "MISCELLANEOUS": "miscellaneous",
            "GENERAL PROVISIONS": "general_provisions",
            "GOVERNING LAW": "governing_law",
            "DISPUTE RESOLUTION": "dispute_resolution",
            "SIGNATURES": "signatures"
        }

        for key_phrase, identifier in section_mappings.items():
            if key_phrase in upper_title:
                logger.debug("Matched common section pattern: %s -> %s", key_phrase, identifier)
                return f"section_{identifier}"

        # Generate descriptive identifier from title words
        # Use first 2-3 meaningful words, cleaned and joined
        words = [word.lower() for word in clean_title.split() if len(word) > 2 and word.isalpha()]
        if words:
            # Take first 2-3 words, limit length
            desc_words = words[:3]
            desc_id = "_".join(desc_words)[:30]  # Limit to 30 chars
            # Remove any remaining special characters
            desc_id = re.sub(r'[^a-z0-9_]', '', desc_id)
            if desc_id:
                logger.debug("Generated descriptive identifier: %s", desc_id)
                return f"section_{desc_id}"

        # Final fallback using sanitized filename
        safe_id = sanitize_filename(section_title).lower()
        return f"section_{safe_id}" if safe_id != "unnamed_section" else "section_unknown"

    except (AttributeError, TypeError) as e:
        logger.warning("Error extracting article number from '%s': %s", section_title, e)
        return "error_article"


def safe_table_to_markdown(table_item: TableItem) -> str:
    """Safely convert table to markdown with error handling."""
    try:
        if not hasattr(table_item, "export_to_dataframe"):
            logger.debug("Table item does not support export_to_dataframe")
            return "[Table content - export method not available]"

        table_df = table_item.export_to_dataframe()
        if table_df is None or table_df.empty:
            logger.debug("Table is empty or None")
            return "[Empty table]"

        # Limit table size to prevent memory issues
        max_rows = 100
        max_cols = 20
        if len(table_df) > max_rows:
            table_df = table_df.head(max_rows)
            logger.warning("Table truncated to %d rows", max_rows)

        if len(table_df.columns) > max_cols:
            table_df = table_df.iloc[:, :max_cols]
            logger.warning("Table truncated to %d columns", max_cols)

        table_markdown = table_df.to_markdown(index=False)
        return table_markdown if table_markdown else "[Table conversion failed]"

    except Exception as e:
        logger.warning("Error converting table to markdown: %s", e)
        return f"[Table content - conversion error: {str(e)[:100]}]"


def safe_list_to_markdown(list_item: ListItem) -> str:
    """Safely convert list item to markdown with error handling."""
    try:
        # Check if list item has text content
        if not hasattr(list_item, "text") or not list_item.text:
            logger.debug("List item has no text content")
            return "[Empty list item]"

        text_content = list_item.text.strip()
        if not text_content:
            return "[Empty list item]"

        # Get list properties
        is_enumerated = getattr(list_item, "enumerated", False)
        marker = getattr(list_item, "marker", None)

        # Format based on list type
        if is_enumerated:
            # For numbered lists, use the marker if available, otherwise default numbering
            if marker and marker.strip():
                return f"{marker} {text_content}"
            else:
                return f"1. {text_content}"
        else:
            # For bulleted lists, use the marker if available, otherwise default bullet
            if marker and marker.strip():
                return f"{marker} {text_content}"
            else:
                return f"• {text_content}"

    except Exception as e:
        logger.warning("Error converting list item to markdown: %s", e)
        return f"[List item - conversion error: {str(e)[:100]}]"


def parse_pdf_with_fallback(pdf_path: Path) -> dict:
    """Parse PDF with simple text extraction as fallback when docling fails."""
    logger.debug("Using fallback PDF parser for %s...", pdf_path)
    sections = {}
    current_section = None
    current_content = []
    page_count = 0

    try:
        # Try to use pdftotext command line utility if available
        import subprocess

        logger.debug("Attempting text extraction using pdftotext...")
        result = subprocess.run(
            ['pdftotext', str(pdf_path), '-'],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            raise Exception(f"pdftotext failed: {result.stderr}")

        all_text = result.stdout
        logger.debug("Successfully extracted text using pdftotext")

    except Exception as e:
        logger.error("pdftotext extraction failed: %s", e)

        # Fallback to pypdf if pdftotext is not available
        try:
            from pypdf import PdfReader
            logger.debug("Attempting text extraction using pypdf...")

            reader = PdfReader(str(pdf_path))
            all_text = ""
            page_count = len(reader.pages)

            for page_num, page in enumerate(reader.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    all_text += f"\n\n--- Page {page_num} ---\n\n" + page_text

        except Exception as pypdf_error:
            logger.error("pypdf extraction also failed: %s", pypdf_error)
            # Create minimal fallback based on filename
            doc_name = pdf_path.stem
            sections = {
                'section_1': {
                    'title': f'Document Content - {doc_name}',
                    'content': f'Unable to extract text from {pdf_path.name}. Document processing failed with both docling and fallback parsers.',
                    'page_range': '1-?',
                    'description': f'Text extraction failed for {doc_name}. Manual review required.'
                }
            }
            logger.warning("Created minimal fallback section due to extraction failures")
            return {'sections': sections}

    # Parse extracted text for article headers
    if not all_text.strip():
        logger.warning("No text extracted from PDF")
        sections = {
            'section_1': {
                'title': 'Empty Document',
                'description': 'No readable text content found in the document.',
                'page_range': '1-1',
                'content': 'Document appears to be empty or contains only non-text content.'
            }
        }
        return {'sections': sections}

    # Split text into lines and process
    lines = all_text.split('\n')
    current_section = 'Cover Page'
    current_content = []
    section_start_page = 1
    current_page = 1

    for line in lines:
        line = line.strip()

        # Track page numbers from page markers
        if line.startswith('--- Page ') and line.endswith(' ---'):
            try:
                current_page = int(line.split()[2])
            except (IndexError, ValueError):
                pass
            continue

        if not line:
            continue

        # Check if this line is an article header
        if is_article_header(line):
            # Save previous section if it has content
            if current_content:
                section_text = '\n'.join(current_content)
                section_id = extract_article_number(current_section)
                page_range = f"{section_start_page}-{current_page - 1}" if current_page > section_start_page else str(section_start_page)

                # Generate AI summary if LLM is available
                if llm_sonnet and section_text.strip():
                    summary = generate_section_summary(current_section, section_text, llm_sonnet)
                else:
                    summary = f"Section contains {len(section_text)} characters. LLM summary not available."

                sections[section_id] = {
                    'section_id': section_id,
                    'title': current_section,
                    'description': summary,
                    'page_range': page_range,
                    'content': section_text,
                }

            # Start new section
            current_section = line
            current_content = [line]
            section_start_page = current_page
            logger.debug("Found section: %s", line)
        else:
            # Add line to current section content
            current_content.append(line)

    # Don't forget the last section
    if current_content:
        section_text = '\n'.join(current_content)
        section_id = extract_article_number(current_section)
        page_range = f"{section_start_page}-{current_page}" if current_page > section_start_page else str(section_start_page)

        if llm_sonnet and section_text.strip():
            summary = generate_section_summary(current_section, section_text, llm_sonnet)
        else:
            summary = f"Section contains {len(section_text)} characters. LLM summary not available."

        sections[section_id] = {
            "section_id": section_id,
            "title": current_section,
            "description": summary,
            "page_range": page_range,
            "content": section_text,
        }

    # If no sections were found, create a single section with all content
    if not sections:
        logger.warning("No article headers detected, creating single section with all content")
        content_preview = all_text[:5000] + "..." if len(all_text) > 5000 else all_text

        if llm_sonnet:
            summary = generate_section_summary('Document Content', content_preview, llm_sonnet)
        else:
            summary = f"Complete document content ({len(all_text)} characters). LLM summary not available."

        sections["section_1"] = {
            "section_id": "section_1",
            "title": "Document Content",
            "description": summary,
            "page_range": f"1-{page_count or '?'}",
            "content": all_text,
        }

    logger.debug("Fallback parser found %d sections", len(sections))
    return {"sections": sections}


def extract_table_of_contents(doc) -> list:
    """
    Extract table of contents from the document.

    DISABLED: Returns empty list to skip TOC extraction.

    Args:
        doc: The parsed document object

    Returns:
        Empty list (TOC extraction disabled)
    """
    logger.debug("Table of contents extraction disabled")
    return []


def parse_pdf_by_articles(pdf_path: Path, llm_instance=None) -> dict:
    """
    Parse PDF by articles with comprehensive defensive programming.

    ⚠️  DEPRECATED: This function is deprecated. Use parse_pdf_by_toc() instead
    for better TOC-based parsing with LLM assistance.

    Args:
        pdf_path: Path to the PDF file to parse for validation
        llm_instance: LLM instance for generating summaries (optional, uses global if None)
    """
    import warnings
    warnings.warn(
        "parse_pdf_by_articles is deprecated. Use parse_pdf_by_toc for better TOC-based parsing.",
        DeprecationWarning,
        stacklevel=2
    )
    # Use provided LLM instance or fall back to global llm_sonnet
    llm = llm_instance if llm_instance is not None else llm_sonnet
    # Input validation
    if not pdf_path:
        logger.error("PDF path is required")
        raise ValueError("PDF path is required")

    if isinstance(pdf_path, str):
        pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        logger.error("PDF file not found: %s", pdf_path)
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if not pdf_path.is_file():
        logger.error("Path is not a file: %s", pdf_path)
        raise ValueError(f"Path is not a file: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        logger.error("File is not a PDF: %s", pdf_path)
        raise ValueError(f"File is not a PDF: {pdf_path}")

    try:
        logger.debug("Converting %s...", pdf_path)
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    backend=DoclingParseV4DocumentBackend,
                ),
            }
        )
        result = converter.convert(pdf_path)

        if not result or not hasattr(result, "document"):
            logger.error("Document conversion failed - no result returned")
            raise ValueError("Document conversion failed - no result returned")

        doc = result.document
        if not doc:
            logger.error("Document conversion failed - empty document")
            raise ValueError("Document conversion failed - empty document")

        logger.debug("✓ Document converted successfully")

    except Exception as e:
        logger.error("Failed to convert PDF '%s'", pdf_path)
        logger.exception("Docling conversion failed, trying fallback parser")
        # Try fallback method
        logger.debug("Attempting fallback PDF parsing...")
        return parse_pdf_with_fallback(pdf_path)

    # Initialize variables with defensive defaults
    current_section = "ARTICLE 0 - Cover Page"
    current_content = []
    section_counter = 0

    # Track page numbers and metadata
    current_page_start = None
    current_page_end = None
    sections_data = {}
    items_processed = 0
    errors_encountered = 0

    logger.debug("=== Processing Document ===")
    logger.debug("→ Collecting: %s", current_section)

    try:
        for item, _ in doc.iterate_items():
            items_processed += 1

            try:
                # Track page numbers from items with provenance
                page_no = safe_get_page_number(item)
                if current_page_start is None:
                    current_page_start = page_no
                current_page_end = page_no

                if isinstance(item, TextItem):
                    if hasattr(item, "text") and item.text and item.text.strip():
                        # Check if this text is an article header BEFORE adding to content
                        if is_article_header(item.text):
                            # Save previous section (including ARTICLE 0)
                            if current_content:
                                section_text = "\n".join(current_content)

                                # Generate metadata for this section
                                section_id = extract_article_number(current_section)
                                page_range = f"{current_page_start or 1}-{current_page_end or 1}"

                                # Generate AI summary with error handling
                                summary = generate_section_summary(current_section, section_text, llm_sonnet)

                                # Store metadata
                                sections_data[section_id] = {
                                    "section_id": section_id,
                                    "title": current_section,
                                    "description": summary,
                                    "page_range": page_range,
                                    "content": section_text,
                                }

                                section_counter += 1
                                logger.debug("   ✓ Processed section: %s", current_section)

                            # Start new ARTICLE section
                            current_section = item.text if item.text else "Unnamed Article"
                            current_content = [f"{current_section}\n{'=' * len(current_section)}\n"]
                            current_page_start = None  # Reset for new section
                            current_page_end = None
                            logger.debug("→ Found: %s", current_section)
                        else:
                            # Regular text, add to current section
                            current_content.append(item.text)
                elif isinstance(item, TableItem):
                    # Extract actual table content with error handling
                    table_markdown = safe_table_to_markdown(item)
                    current_content.append(f"\n{table_markdown}\n")
                elif isinstance(item, ListItem):
                    # Extract list item content with error handling
                    list_markdown = safe_list_to_markdown(item)
                    current_content.append(f"\n{list_markdown}")
                # Also check SectionHeaderItem objects
                elif isinstance(item, SectionHeaderItem):
                    if hasattr(item, "text") and item.text and item.text.strip():
                        # Check if this is an article header
                        if is_article_header(item.text):
                            # Save previous section (including ARTICLE 0)
                            if current_content:
                                section_text = "\n".join(current_content)

                                # Generate metadata for this section
                                section_id = extract_article_number(current_section)
                                page_range = f"{current_page_start or 1}-{current_page_end or 1}"

                                # Generate AI summary with error handling
                                summary = generate_section_summary(current_section, section_text, llm_sonnet)

                                # Store metadata
                                sections_data[section_id] = {
                                    "section_id": section_id,
                                    "title": current_section,
                                    "description": summary,
                                    "page_range": page_range,
                                    "content": section_text,
                                }

                                section_counter += 1
                                logger.debug("   ✓ Processed section: %s", current_section)

                            # Start new ARTICLE section
                            current_section = item.text if item.text else "Unnamed Article"
                            current_content = [f"{current_section}\n{'=' * len(current_section)}\n"]
                            current_page_start = None  # Reset for new section
                            current_page_end = None
                            logger.debug("→ Found: %s", current_section)
                        else:
                            # Regular section header, add to current content
                            current_content.append(item.text)

                # Add ALL content to current section with defensive checks


            except Exception as e:
                errors_encountered += 1
                logger.warning("Error processing item %d: %s", items_processed, e)
                if errors_encountered > 10:  # Prevent too many errors
                    logger.error("Too many errors encountered (%d), continuing with caution", errors_encountered)

    except Exception as e:
        logger.exception("Error during document iteration")
        # Continue processing with what we have

    # Save final section with defensive checks
    if current_content:
        try:
            section_text = "\n".join(current_content)

            # Generate metadata for final section
            section_id = extract_article_number(current_section)
            page_range = f"{current_page_start or 1}-{current_page_end or 1}"

            # Generate AI summary
            summary = generate_section_summary(current_section, section_text, llm_sonnet)

            # Store metadata
            sections_data[section_id] = {
                "section_id": section_id,
                "title": current_section,
                "description": summary,
                "page_range": page_range,
                "content": section_text,
            }
            logger.debug("   ✓ Processed final section: %s", current_section)
        except Exception as e:
            logger.exception("Error processing final section")

    logger.debug("=== Done! ===")
    logger.debug("Found %d sections", section_counter + 1)
    logger.debug("Processed %d items with %d errors", items_processed, errors_encountered)

    # Ensure we return something even if processing partially failed
    if not sections_data:
        logger.warning("No sections were successfully processed")
        sections_data = {
            "error_section": {
                "section_id": "error_section",
                "title": "Processing Error",
                "description": "Document processing encountered errors and no sections were extracted successfully.",
                "page_range": "1-1",
                "content": ""
            }
        }

        return {"sections": sections_data}
