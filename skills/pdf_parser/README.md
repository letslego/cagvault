# PDF Parser Skill for Claude

Professional PDF document parsing skill for Claude Code that integrates with the CAG (Cache-Augmented Generation) application.

## Features

✅ **Full PDF Parsing** - Extract text, metadata, structure, and tables  
✅ **Web Content Support** - Parse HTML URLs and convert to Markdown  
✅ **Caching** - Automatic caching for fast re-parsing  
✅ **Structured Output** - Sections, tables, and hierarchy extraction  
✅ **CAG Integration** - Direct integration with knowledge base  
✅ **Search** - Find content within documents  
✅ **Metadata Extraction** - Title, author, pages, dates, etc.  

## Installation

The skill is already included in the CAG project. The required dependencies are:

```bash
pip install docling>=2.64.1
```

(Already installed in your environment)

## Quick Start

### Parse a PDF Document

```python
from skills.pdf_parser import parse_document

result = parse_document('/path/to/document.pdf')
print(f"Document: {result['name']}")
print(f"Pages: {result['metadata']['pages']}")
print(f"Sections: {len(result['sections'])}")
```

### Parse and Store in CAG

```python
from skills.pdf_parser import claude_ingest_pdf

result = claude_ingest_pdf('/path/to/document.pdf')
print(f"Stored with ID: {result['document_id']}")
```

### Search Within Document

```python
from skills.pdf_parser import claude_search_pdf

results = claude_search_pdf('/path/to/document.pdf', 'covenant breach')
print(f"Found {results['matches_found']} matches")
for match in results['results']:
    print(f"Line {match['line_number']}: {match['content']}")
```

### Get Document Structure

```python
from skills.pdf_parser import claude_get_structure

structure = claude_get_structure('/path/to/document.pdf')
print("Table of Contents:")
for section in structure['toc']:
    print(f"{'  ' * (section['level']-1)}- {section['title']}")
```

## Claude Code Usage

In Claude Code, you can use the skill directly:

### Parse PDF for CAG

```
I need you to parse the document 'contracts/loan_agreement.pdf' and extract 
the structure. Show me the sections and how many tables it contains.
```

### Search for Specific Content

```
Search the document 'contracts/loan_agreement.pdf' for all mentions of "covenant" 
and show me the context around each match.
```

### Extract Information

```
Parse 'contracts/loan_agreement.pdf' and give me the metadata including 
number of pages, creation date, and author.
```

### Add to Knowledge Base

```
Ingest 'contracts/loan_agreement.pdf' into the CAG knowledge base so I can 
ask questions about it later.
```

## API Reference

### Core Functions

#### `parse_document(file_path: str) -> Dict`
Parse a PDF and return structured content with metadata.

**Parameters:**
- `file_path` (str): Path to PDF file

**Returns:** Dictionary with:
- `id`: Unique document ID
- `name`: Filename
- `metadata`: Document metadata (pages, size, dates, etc.)
- `sections`: Hierarchical sections extracted from document
- `tables`: Tables found in document
- `content_preview`: First 500 chars of content

---

#### `parse_web_content(url: str) -> Dict`
Parse HTML content from a URL and convert to structured format.

**Parameters:**
- `url` (str): URL to parse

**Returns:** Same structure as `parse_document`

---

#### `search_document(file_path: str, query: str) -> List[Dict]`
Search for text within a document.

**Parameters:**
- `file_path` (str): Path to PDF file
- `query` (str): Search term

**Returns:** List of matches with:
- `line_number`: Line number where match found
- `content`: Text of the line
- `match`: The matched text

---

#### `get_document_metadata(file_path: str) -> Dict`
Extract document metadata.

**Parameters:**
- `file_path` (str): Path to PDF file

**Returns:** Dictionary with:
- `title`: Document title (if available)
- `author`: Author name (if available)
- `pages`: Number of pages
- `file_size`: Size in bytes
- `creation_date`: When created
- `modification_date`: When modified
- `parse_date`: When parsed by skill

---

### CAG Integration Functions

#### `claude_ingest_pdf(file_path: str) -> Dict`
Parse PDF and add to CAG knowledge base.

**Returns:**
```json
{
  "status": "ingested",
  "document_id": "pdf_abc123...",
  "name": "loan_agreement.pdf",
  "type": "document",
  "content_length": 50000,
  "message": "Document successfully stored..."
}
```

---

#### `claude_parse_pdf(file_path: str) -> Dict`
Parse PDF with full information for CAG integration.

**Returns:** Complete document info including summary.

---

#### `claude_search_pdf(file_path: str, query: str) -> Dict`
Search within CAG-stored document.

**Returns:**
```json
{
  "query": "covenant breach",
  "document": "loan_agreement.pdf",
  "matches_found": 5,
  "results": [...]
}
```

---

#### `claude_get_structure(file_path: str) -> Dict`
Extract table of contents and document structure.

**Returns:**
```json
{
  "document": "loan_agreement.pdf",
  "pages": 25,
  "sections": [...],
  "tables": 3,
  "toc": [
    {
      "level": 1,
      "title": "Introduction",
      "subsections": 2
    },
    ...
  ]
}
```

## Examples

### Example 1: Credit Agreement Analysis

```python
from skills.pdf_parser import claude_ingest_pdf, claude_search_pdf

# Ingest the credit agreement
doc = claude_ingest_pdf('credit_agreement.pdf')
print(f"Document ID: {doc['document_id']}")

# Search for covenant clauses
covenants = claude_search_pdf('credit_agreement.pdf', 'covenant')
print(f"Found {covenants['matches_found']} covenant references")
```

### Example 2: Extract and Display Structure

```python
from skills.pdf_parser import claude_get_structure

structure = claude_get_structure('annual_report.pdf')
print(f"Analyzing {structure['document']}")
print(f"Total pages: {structure['pages']}")
print("\nTable of Contents:")
for item in structure['toc']:
    indent = "  " * (item['level'] - 1)
    print(f"{indent}- {item['title']}")
```

### Example 3: CAG Knowledge Base Integration

```python
from skills.pdf_parser import claude_ingest_pdf
from app import st  # Streamlit context

# Ingest multiple documents
docs = [
    'contracts/loan_agreement.pdf',
    'contracts/security_agreement.pdf',
    'contracts/guaranty.pdf'
]

for doc_path in docs:
    result = claude_ingest_pdf(doc_path)
    print(f"✅ Ingested: {result['name']} ({result['document_id']})")

# Now in the Streamlit app, these documents are available for Q&A
```

## Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Parse 10MB PDF | 20-50s | First run, no cache |
| Retrieve cached | <100ms | From cache |
| Search 100 pages | 100-200ms | Linear search |
| Extract metadata | <500ms | Quick scan |
| Parse URL | 5-15s | Depends on network |

## Supported Formats

### Input
- **PDF** (.pdf) - Full support with Docling backend
- **HTML** (via URL) - Converted to Markdown
- **URLs** - Automatically detected

### Output
- **Markdown** - Structured with headers and links
- **JSON** - Structured metadata and content
- **Plain Text** - Raw text extraction

## Caching

The skill automatically caches parsed documents in `.cache/documents/` directory.

Cache keys are generated from:
- File path
- File modification time
- URL content

To clear cache:
```bash
rm -rf .cache/documents/
```

## Troubleshooting

### "File not found"
Check that the file path is absolute or relative to the current working directory.

### "Out of memory" on large PDFs
The skill uses streaming for large files. If issues persist:
```python
from skills.pdf_parser import PDFParserSkill
skill = PDFParserSkill()
# Parsing is streaming-optimized automatically
```

### "Unsupported format"
Only PDF and HTML URLs are supported. Convert documents to PDF first.

### No tables extracted
Not all PDFs have machine-readable tables. Scanned/image PDFs may not extract tables.

## Architecture

```
skills/pdf_parser/
├── __init__.py           # Package exports
├── SKILL.md              # Skill documentation
├── README.md             # This file
├── manifest.json         # Skill manifest
├── pdf_parser.py         # Core parsing logic
└── integration.py        # CAG integration layer
```

## Integration with CAG

The skill integrates seamlessly with CAG:

1. **Document Storage**: Parsed documents stored as `KnowledgeSource` objects
2. **Session Tracking**: Documents tracked via `message_source_ids`
3. **Credit Analyst**: Specialized prompts for contract analysis
4. **KV-Cache**: Efficient context caching for repeated queries

## Development

To extend the skill:

```python
from skills.pdf_parser import PDFParserSkill

class ExtendedParser(PDFParserSkill):
    def extract_clauses(self, doc, pattern):
        # Custom extraction logic
        pass
```

## License

MIT License - See LICENSE.md

## Support

For issues or questions, refer to the main CAG documentation or contact the team.
