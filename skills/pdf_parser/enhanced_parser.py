"""
Enhanced PDF Parser with Section Memory and Hierarchical Metadata

Stores sections and subsections in memory with full document and section references.
Includes metadata extraction, full-text search, and named entity recognition.
"""

import hashlib
import json
import os
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import logging
import re
import sys
from pathlib import Path
import redis 

from .ner_search import NamedEntityRecognizer, FullTextSearchEngine, EnhancedSearchableParser
from .credit_analyst_prompt import get_credit_analyst
from .llm_section_evaluator import get_llm_evaluator

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

    def clear_document(self, document_id: str) -> None:
        """Remove all stored sections for a document to prevent duplication."""
        self.documents.pop(document_id, None)
        self.section_index.pop(document_id, None)
        self.metadata_index = {
            k: v for k, v in self.metadata_index.items()
            if not k.startswith(f"{document_id}#")
        }
    
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
        self.search_parser = EnhancedSearchableParser(llm=None)  # Agentic uses Claude SDK directly
        self.redis_client = self._init_redis()
        logger.info("EnhancedPDFParserSkill initialized with section memory, search (keyword/semantic/agentic), and NER")

    def _init_redis(self):
        """Create Redis client if Redis is available."""
        try:
            # Try to use REDIS_URL if set, otherwise use default localhost
            url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            if redis is None:
                return None
            client = redis.Redis.from_url(url)
            client.ping()
            logger.info("Redis client initialized")
            return client
        except Exception as exc:  # pragma: no cover
            logger.warning(f"Redis unavailable: {exc}")
            return None
    
    def parse_and_extract_sections(self, file_path: str) -> Dict[str, Any]:
        """
        Parse PDF and extract sections into memory with coverage tracking.
        
        Uses LLM to intelligently detect sections and track accurate page ranges.
        """
        # Short-circuit if already cached in Redis using the same document id as the parser cache key
        document_id: Optional[str] = None
        if self.redis_client:
            cache_key = self.parser._generate_cache_key(str(file_path))
            document_id = f"pdf_{cache_key}"
            cached = self._load_document_from_redis(document_id)
            if cached:
                return cached
        
        # Parse document using standard parser
        doc = self.parser.parse_pdf(file_path)
        document_id = document_id or doc.id

        # Clear any existing in-memory copy before adding fresh sections
        self.memory.clear_document(document_id)

        logger.info(f"Extracting sections from {doc.name}")

        # Use LLM-based extraction with parallel processing for speed
        return self._extract_sections_with_llm(doc, file_path)
    
    def _extract_sections_with_llm(self, doc: "ParsedDocument", file_path: str) -> Dict[str, Any]:
        """
        Extract sections using LLM-enhanced markdown parsing with parallel processing.
        
        Uses the markdown-based section extraction to get granular sections,
        then uses LLM to assign accurate page ranges using concurrent requests.
        """
        from langchain_ollama import ChatOllama
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from config import Config
        import json
        
        # Initialize LLM with correct model name - single instance handles concurrent requests
        llm = ChatOllama(
            model="hf.co/unsloth/Qwen3-14B-GGUF:Q4_K_XL",
            base_url=Config.OLLAMA_BASE_URL,
            temperature=0.0,
            top_k=2,
            num_ctx=4096,  # Context window
            num_parallel=Config.OLLAMA_NUM_PARALLEL,  # Enable concurrent requests
            timeout=Config.REQUEST_TIMEOUT,
        )
        
        markdown_content = doc.content
        total_pages = doc.metadata.pages
        sections_from_markdown = doc.sections
        
        # Skip slow page range LLM call - use fast word-based estimation
        # This is accurate enough and avoids a blocking LLM call
        logger.info(f"Using fast word-based page estimation for {len(sections_from_markdown)} sections")
        page_ranges = {}
        
        # Create sections with improved page tracking
        section_count = 0
        current_page = 1
        
        for section_data in sections_from_markdown:
            title = section_data.get("title", "Untitled")
            content = section_data.get("content", "")
            
            # Get page range from LLM if available
            if title in page_ranges:
                start_page, end_page = page_ranges[title]
                # Update current_page to continue from where this section ends
                current_page = end_page + 1
            else:
                # Estimate pages: ~250 words per page
                word_count = len(content.split()) if content else 0
                pages_needed = max(1, word_count // 250)
                start_page = current_page
                end_page = min(total_pages, current_page + pages_needed - 1)
                current_page = end_page + 1
            
            # Ensure valid page range
            start_page = max(1, min(start_page, total_pages))
            end_page = max(start_page, min(end_page, total_pages))
            # Always update current_page to ensure no gaps
            current_page = end_page + 1
            
            # Create section metadata
            word_count = len(content.split()) if content else 0
            metadata = SectionMetadata(
                title=title,
                level=section_data.get("level", 1),
                start_line=section_data.get("start_line", 0),
                end_line=section_data.get("end_line", 0),
                content_length=len(content),
                word_count=word_count,
                has_code=False,
                has_tables="table" in content.lower() or "|" in content,
                subsection_count=len(section_data.get("subsections", [])),
                page_estimate=start_page,
                page_range=f"{start_page}-{end_page}",
                start_page=start_page,
                end_page=end_page,
                section_type="OTHER",
                importance_score=0.5,
                typical_dependencies=[]
            )
            
            # Skip detailed analysis during initial upload for speed
            # Sections will be analyzed on-demand when user asks questions
            # This avoids blocking Streamlit with N LLM calls
            
            # Create and add section
            section = Section(
                metadata=metadata,
                content=content,
                document_id=doc.id,
                subsections=[]
            )
            self.memory.add_section(doc.id, section)
            section_count += 1
        
        # Batch analyze section importance in parallel using ThreadPoolExecutor
        logger.info(f"Starting parallel importance analysis for {section_count} sections...")
        all_sections = self.memory.get_document_sections(doc.id)
        self._batch_analyze_sections_parallel(all_sections, max_workers=4)
        
        # Index for search and NER before persistence
        self._index_sections(doc.id, self.memory.get_document_sections(doc.id))

        # Persist to Redis if available
        self._store_document_in_redis(
            document_id=doc.id,
            document_name=doc.name,
            total_pages=total_pages,
            extraction_method="llm_enhanced_markdown",
            sections=self.memory.get_document_sections(doc.id)
        )
        
        return {
            "document_id": doc.id,
            "document_name": doc.name,
            "pages": total_pages,
            "sections_extracted": section_count,
            "extraction_method": "llm_enhanced_markdown",
            "message": f"Extracted {section_count} sections with parallel LLM analysis"
        }
    
    def _batch_analyze_sections_parallel(self, sections: List[Section], max_workers: int = 4) -> None:
        """
        Analyze section importance in parallel using ThreadPoolExecutor.
        
        Args:
            sections: List of sections to analyze
            max_workers: Number of parallel worker threads (default 4)
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def analyze_single_section(section: Section) -> tuple:
            """Analyze a single section and return (section, analysis)."""
            try:
                analyst = get_credit_analyst()
                analysis = analyst.analyze_section_importance(
                    section.metadata.title,
                    section.content,
                    section.metadata.level
                )
                return (section, analysis)
            except Exception as e:
                logger.warning(f"Failed to analyze section {section.metadata.title}: {e}")
                return (section, {
                    "classification": "OTHER",
                    "importance_score": 0.5,
                    "typical_dependencies": []
                })
        
        # Flatten sections to include subsections
        all_sections = self._flatten_sections(sections)
        
        # Process sections in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_section = {
                executor.submit(analyze_single_section, section): section 
                for section in all_sections
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_section):
                section, analysis = future.result()
                # Update section metadata with analysis results
                section.metadata.section_type = analysis.get("classification", "OTHER")
                section.metadata.importance_score = analysis.get("importance_score", 0.5)
                section.metadata.typical_dependencies = analysis.get("typical_dependencies", [])
                
                completed += 1
                if completed % 10 == 0:
                    logger.info(f"Analyzed {completed}/{len(all_sections)} sections...")
        
        logger.info(f"✓ Completed parallel analysis of {len(all_sections)} sections")

    def _store_document_in_redis(
        self,
        document_id: str,
        document_name: str,
        total_pages: int,
        extraction_method: str,
        sections: List[Section]
    ) -> None:
        """Store document metadata, sections, and subsections in Redis (idempotent)."""
        if not self.redis_client:
            return
        try:
            meta_key = f"doc:{document_id}:meta"
            sections_key = f"doc:{document_id}:sections"
            section_ids_key = f"doc:{document_id}:section_ids"
            sections_full_key = f"doc:{document_id}:sections_full"
            doc_keywords_key = f"doc:{document_id}:keywords"
            doc_entities_key = f"doc:{document_id}:entities"
            all_sections = self._flatten_sections(sections)
            # Metadata
            meta_payload = {
                "document_id": document_id,
                "document_name": document_name,
                "pages": total_pages,
                "extraction_method": extraction_method,
                "stored_at": datetime.now().isoformat()
            }
            self.redis_client.set(meta_key, json.dumps(meta_payload))
            # Sections and subsections
            pipe = self.redis_client.pipeline()
            pipe.delete(section_ids_key)
            pipe.delete(doc_keywords_key)
            pipe.delete(doc_entities_key)
            pipe.delete(sections_full_key)

            doc_keywords: set[str] = set()
            doc_entities: List[Dict[str, Any]] = []
            all_payloads: List[Dict[str, Any]] = []
            for section in all_sections:
                sid = section.metadata.id
                sec_key = f"doc:{document_id}:section:{sid}"
                # Derive keywords and entities from search/NER index
                keywords = self.search_parser.search_engine._tokenize(section.content) if self.search_parser else []
                entities = self.search_parser.get_section_entities(sid)
                doc_keywords.update(keywords)
                for ent in entities:
                    ent_with_section = dict(ent)
                    ent_with_section["section_id"] = sid
                    doc_entities.append(ent_with_section)
                payload = {
                    "id": sid,
                    "title": section.metadata.title,
                    "level": section.metadata.level,
                    "parent_id": section.metadata.parent_id,
                    "content": section.content,
                    "metadata": asdict(section.metadata),
                    "document_id": document_id,
                    "keywords": keywords,
                    "entities": entities,
                }
                pipe.set(sec_key, json.dumps(payload))
                pipe.rpush(section_ids_key, sid)
                all_payloads.append(payload)
            # Keep ordered list of sections + subsections
            pipe.set(sections_key, json.dumps([s.metadata.id for s in all_sections]))
            pipe.set(sections_full_key, json.dumps(all_payloads))
            pipe.set(doc_keywords_key, json.dumps(sorted(doc_keywords)))
            pipe.set(doc_entities_key, json.dumps(doc_entities))
            pipe.execute()
            logger.info(f"Stored document {document_id} with {len(all_sections)} sections in Redis")
        except Exception as exc:  # pragma: no cover
            logger.warning(f"Failed to store in Redis: {exc}")

    def _load_document_from_redis(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Load document metadata and sections from Redis into memory.

        Returns cached stats dict matching the parse response shape or None on miss.
        """
        if not self.redis_client:
            return None

        meta_key = f"doc:{document_id}:meta"
        sections_key = f"doc:{document_id}:sections"
        section_ids_key = f"doc:{document_id}:section_ids"

        try:
            meta_raw = self.redis_client.get(meta_key)
            if not meta_raw:
                return None

            meta = json.loads(meta_raw)

            # Skip rehydration if sections are already in memory to avoid duplicate adds/logs
            existing_sections = self.memory.get_document_sections(document_id)
            if existing_sections:
                logger.info(
                    f"Document {document_id} already hydrated in memory; skipping Redis reload"
                )
                return {
                    "document_id": document_id,
                    "document_name": meta.get("document_name", ""),
                    "pages": meta.get("pages", 0),
                    "sections_extracted": len(existing_sections),
                    "extraction_method": meta.get("extraction_method", "redis_cache"),
                    "message": "Document already in memory; skipped Redis reload",
                }

            # Clear any existing in-memory copy for this document to avoid duplication
            self.memory.clear_document(document_id)

            # Retrieve ordered section ids; fall back to stored list value if list is empty
            section_ids = [
                sid.decode("utf-8") if isinstance(sid, (bytes, bytearray)) else sid
                for sid in self.redis_client.lrange(section_ids_key, 0, -1)
            ]
            if not section_ids:
                stored_list = self.redis_client.get(sections_key)
                if stored_list:
                    section_ids = json.loads(stored_list)

            section_map: Dict[str, Section] = {}
            for sid in section_ids:
                sec_raw = self.redis_client.get(f"doc:{document_id}:section:{sid}")
                if not sec_raw:
                    continue
                payload = json.loads(sec_raw)
                metadata_dict = payload.get("metadata", {}) or {}
                # Ensure optional list fields are not None
                metadata_dict["typical_dependencies"] = metadata_dict.get("typical_dependencies") or []
                meta_obj = SectionMetadata(**metadata_dict)
                meta_obj.id = payload.get("id", sid)
                section = Section(
                    metadata=meta_obj,
                    content=payload.get("content", ""),
                    document_id=document_id,
                    subsections=[]
                )
                section_map[meta_obj.id] = section

            # Rebuild hierarchy and repopulate memory store in the original order
            root_sections: List[Section] = []
            for sid in section_ids:
                section = section_map.get(sid)
                if not section:
                    continue
                parent_id = section.metadata.parent_id
                if parent_id and parent_id in section_map:
                    section_map[parent_id].subsections.append(section)
                else:
                    root_sections.append(section)
                self.memory.add_section(document_id, section, parent_id=parent_id)

            # Re-index for search and NER
            self._index_sections(document_id, root_sections)

            logger.info(f"Loaded document {document_id} from Redis cache")
            return {
                "document_id": document_id,
                "document_name": meta.get("document_name", ""),
                "pages": meta.get("pages", 0),
                "sections_extracted": len(section_ids),
                "extraction_method": meta.get("extraction_method", "redis_cache"),
                "message": "Loaded document from Redis cache (skipped parsing)",
            }
        except Exception as exc:  # pragma: no cover
            logger.warning(f"Failed to load from Redis: {exc}")
            return None

    def load_document_from_redis(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Public wrapper to hydrate a document from Redis cache."""
        return self._load_document_from_redis(document_id)
    
    def _annotate_with_pages(self, markdown_content: str, total_pages: int) -> str:
        """
        Annotate markdown content with approximate page boundaries.
        
        Helps LLM understand where pages break in the document.
        """
        lines = markdown_content.split('\n')
        lines_per_page = len(lines) / total_pages if total_pages > 0 else 1
        
        annotated = []
        for i, line in enumerate(lines):
            current_page = int(i / lines_per_page) + 1
            if i % int(lines_per_page) == 0 and i > 0:
                annotated.append(f"\n--- PAGE {current_page} ---\n")
            annotated.append(line)
        
        # Truncate if too long for LLM context
        result = '\n'.join(annotated)
        max_chars = 8000  # Leave room for prompt
        if len(result) > max_chars:
            result = result[:max_chars] + f"\n... [document continues for {total_pages} pages total] ..."
        
        return result
    
    def _extract_sections_from_markdown(self, doc: "ParsedDocument") -> Dict[str, Any]:
        """Extract sections using markdown-based approach (original method)."""
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

        # Index for search and NER before persistence
        self._index_sections(doc.id, self.memory.get_document_sections(doc.id))

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
    
    def _extract_sections_with_toc(self, file_path: str) -> Dict[str, Any]:
        """
        Extract sections using TOC-based parsing for better granularity.
        
        This method uses doc_parse_utils which intelligently detects table of contents
        and matches section headers using LLM guidance, providing:
        - More sections (typically 30+ for credit agreements)
        - Accurate page ranges
        - Document type classification
        """
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from doc_parse_utils import parse_pdf_by_toc
        
        import tempfile
        import hashlib
        
        # Generate document ID
        document_id = f"pdf_{hashlib.sha256(str(file_path).encode()).hexdigest()[:16]}"
        
        # Parse with TOC detection
        result = parse_pdf_by_toc(
            pdf_path=file_path,
            output_dir=f".cache/toc_sections/{document_id}",
            generate_summaries=False,
            use_llm_matching=False,  # Use fuzzy for speed
            debug_toc=False
        )
        
        # Extract sections metadata
        sections_metadata = result.get("sections_metadata", {})
        doc_classification = result.get("doc_classification", {})
        total_pages = result.get("total_pages", 0)
        
        # Convert to Section objects
        section_count = 0
        for section_id, section_meta in sections_metadata.items():
            # Parse page range
            page_range_str = section_meta.get("page_range", "1-1")
            try:
                start_page, end_page = map(int, page_range_str.split("-"))
            except (ValueError, AttributeError):
                start_page, end_page = 1, 1
            
            # Get content from file or metadata
            content = section_meta.get("content", "")
            if not content:
                content_path = Path(section_meta.get("file_path", ""))
                if content_path.exists():
                    try:
                        with open(content_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                    except Exception as e:
                        logger.warning(f"Failed to read section content: {e}")
            
            # Create section metadata
            word_count = len(content.split()) if content else 0
            metadata = SectionMetadata(
                title=section_meta.get("title", f"Section {section_count+1}"),
                level=1,
                start_line=0,
                end_line=len(content.split('\n')) if content else 0,
                content_length=len(content),
                word_count=word_count,
                has_code=False,
                has_tables="table" in content.lower() or "|" in content,
                subsection_count=0,
                page_estimate=start_page,
                page_range=page_range_str,
                start_page=start_page,
                end_page=end_page,
            )
            
            # Store basic metadata now, will batch analyze later
            metadata.section_type = "OTHER"
            metadata.importance_score = 0.5
            metadata.typical_dependencies = []
            
            # Create and add section
            section = Section(
                metadata=metadata,
                content=content,
                document_id=document_id,
                subsections=[]
            )
            self.memory.add_section(document_id, section)
            section_count += 1
        
        logger.info(f"TOC extraction: {section_count} sections extracted")
        
        # Batch analyze sections in parallel
        all_sections = self.memory.get_document_sections(document_id)
        self._batch_analyze_sections_parallel(all_sections, max_workers=4)
        
        return {
            "document_id": document_id,
            "document_name": Path(file_path).name,
            "pages": total_pages,
            "sections_extracted": section_count,
            "document_type": doc_classification.get("document_type", "unknown"),
            "toc_found": result.get("toc_data", {}).get("is_toc_found", False),
            "extraction_method": "toc_based",
            "message": f"Extracted {section_count} sections from {Path(file_path).name} using TOC detection"
        }
    
    def _flatten_sections(self, sections: List[Section]) -> List[Section]:
        """Flatten a list of sections and subsections preserving order."""
        flattened: List[Section] = []

        def walk(section: Section) -> None:
            flattened.append(section)
            for child in section.subsections:
                if child.metadata.parent_id is None:
                    child.metadata.parent_id = section.metadata.id
                walk(child)

        for sec in sections:
            walk(sec)
        return flattened

    def _index_sections(self, document_id: str, sections: List[Section]) -> None:
        """Index sections (and subsections) for search and NER."""
        if not sections:
            return
        flattened = self._flatten_sections(sections)
        for section in flattened:
            self.search_parser.index_section(
                section_id=section.metadata.id,
                content=section.content,
                document_id=document_id,
                title=section.metadata.title
            )

    def _create_section_from_data(
        self,
        document_id: str,
        section_data: Dict[str, Any],
        parent_id: Optional[str] = None
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
        if parent_id:
            section.metadata.parent_id = parent_id
        
        # Process subsections
        for subsection_data in section_data.get("subsections", []):
            subsection = self._create_section_from_data(
                document_id=document_id,
                section_data=subsection_data,
                parent_id=metadata.id
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

    def list_redis_documents(self) -> List[Dict[str, Any]]:
        """List documents stored in Redis (metadata only)."""
        if not self.redis_client:
            return []
        docs = []
        try:
            for meta_key in self.redis_client.scan_iter(match="doc:*:meta"):
                raw = self.redis_client.get(meta_key)
                if not raw:
                    continue
                meta = json.loads(raw)
                docs.append(meta)
        except Exception as exc:  # pragma: no cover
            logger.warning(f"Failed to list Redis documents: {exc}")
        return docs
    
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


def evaluate_section_for_question(
    document_id: str,
    section_id: str,
    question: str,
    previous_notes: Optional[str] = None,
    already_checked_sections: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Claude Skill: Evaluate whether a section provides a complete answer to a question.
    
    Uses expert credit analyst evaluation framework to determine if this section
    fully answers the question or if additional sections need to be retrieved.
    
    Args:
        document_id: Document ID
        section_id: Section ID to evaluate
        question: The user's analytical question
        previous_notes: Accumulated notes from previously analyzed sections
        already_checked_sections: List of section titles already analyzed
        
    Returns:
        Evaluation with decision (ANSWER/PASS), reasoning, and next sections to retrieve
    """
    parser = get_enhanced_parser()
    evaluator = get_llm_evaluator()
    
    # Get the section
    section = parser.memory.get_section(f"{document_id}#{section_id}")
    if not section:
        return {
            "error": f"Section not found: {section_id}",
            "section_id": section_id
        }
    
    # Create evaluation prompt with system prompt
    evaluation_prompt = evaluator.create_evaluation_prompt(
        question=question,
        section_title=section.metadata.title,
        section_content=section.content,
        previous_notes=previous_notes,
        already_checked_sections=already_checked_sections
    )
    
    return {
        "document_id": document_id,
        "section_id": section_id,
        "section_title": section.metadata.title,
        "system_prompt": evaluator.get_system_prompt(),
        "evaluation_prompt": evaluation_prompt,
        "section_type": section.metadata.section_type,
        "importance_score": section.metadata.importance_score,
        "cross_references": evaluator.extract_section_references(section.content),
        "instruction": "Send system_prompt + evaluation_prompt to Claude for complete evaluation"
    }


def get_section_evaluation_summary(
    document_id: str,
    question: str
) -> Dict[str, Any]:
    """
    Claude Skill: Get summary of section evaluation requirements for a question.
    
    Returns which sections should be evaluated and in what order.
    
    Args:
        document_id: Document ID
        question: The user's question
        
    Returns:
        Evaluation plan with section order and mandatory checks
    """
    parser = get_enhanced_parser()
    evaluator = get_llm_evaluator()
    
    # Get all sections
    sections = parser.memory.get_document_sections(document_id)
    
    # Create evaluation summary
    summary = {
        "document_id": document_id,
        "question": question,
        "total_sections": len(sections),
        "evaluation_plan": [],
        "mandatory_checks": {}
    }
    
    # For each section type that exists, add evaluation guidance
    section_types_found = set(s.metadata.section_type for s in sections if s.metadata.section_type)
    
    for section_type in section_types_found:
        mandatory = evaluator.get_mandatory_cross_references(question, section_type)
        summary["mandatory_checks"][section_type] = mandatory
        
        sections_of_type = [s for s in sections if s.metadata.section_type == section_type]
        for section in sections_of_type:
            summary["evaluation_plan"].append({
                "section_id": section.metadata.id,
                "section_title": section.metadata.title,
                "section_type": section_type,
                "importance": section.metadata.importance_score,
                "page_range": section.metadata.page_range
            })
    
    return summary
