from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from io import BytesIO
from docling.document_converter import (
    DocumentConverter,
    DocumentStream,
    InputFormat,
    PdfFormatOption,
    StandardPdfPipeline,
)
from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend

class KnowledgeType(str, Enum):
    DOCUMENT = "document"
    URL = "url"
 
@dataclass
class KnowledgeSource:
    id: str          # Unique identifier (e.g., "file_report.pdf", "url_hash(...)")
    name: str        # Original filename or URL
    type: KnowledgeType # Type of source (DOCUMENT or URL)
    content: str     # The extracted text content (usually Markdown)
    

@lru_cache(maxsize=1)
def create_document_converter() -> DocumentConverter:
    """Initializes and caches the DocumentConverter."""
    return DocumentConverter(
        allowed_formats=[
            InputFormat.PDF,
            InputFormat.HTML, # For handling URLs
        ],
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=StandardPdfPipeline,
                backend=DoclingParseV4DocumentBackend,
            ),
        },
    )
    
def load_from_url(url: str) -> KnowledgeSource:
    converter = create_document_converter()
    result = converter.convert(url)
    source_id = f"url_{hash(url)}" # Create a simple ID based on hash
    content = result.document.export_to_markdown()
    return KnowledgeSource(
        id=source_id, name=url, type=KnowledgeType.URL, content=content
    )
    
    
def load_from_file(file_name: str, document_data: bytes) -> KnowledgeSource:
    source_id = f"file_{file_name}"
    doc_type = KnowledgeType.DOCUMENT
 
    if file_name.lower().endswith((".txt", ".md")):
         return KnowledgeSource(
            id=source_id,
            name=file_name,
            type=doc_type,
            content=document_data.decode("utf-8"),
        )
 
    converter = create_document_converter()
    buf = BytesIO(document_data)
    source = DocumentStream(name=file_name, stream=buf)
    result = converter.convert(source)
    return KnowledgeSource(
        id=source_id,
        name=file_name,
        type=KnowledgeType.DOCUMENT,
        content=result.document.export_to_markdown(),
    )