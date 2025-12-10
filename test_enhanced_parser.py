#!/usr/bin/env python3
"""
Test Script: Enhanced PDF Parser with Section Memory

Demonstrates:
1. Extracting document sections into memory
2. Accessing section metadata
3. Searching across sections
4. Getting document hierarchy
"""

import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from skills.pdf_parser import (
    extract_document_sections,
    get_document_hierarchy,
    get_section_metadata,
    search_sections,
    get_section_details,
    list_all_documents,
    get_enhanced_parser,
)


def demo_extract_sections(pdf_path: str):
    """Demo: Extract sections from PDF."""
    print("\n" + "="*70)
    print("DEMO 1: Extract Document Sections")
    print("="*70 + "\n")
    
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        print(f"❌ File not found: {pdf_path}")
        return None
    
    print(f"📄 Extracting sections from: {pdf_path.name}\n")
    
    result = extract_document_sections(str(pdf_path))
    
    print(f"✅ Extraction Complete!\n")
    print(f"Document ID:      {result['document_id']}")
    print(f"Document Name:    {result['document_name']}")
    print(f"Pages:            {result['pages']}")
    print(f"Sections Found:   {result['sections_extracted']}\n")
    
    stats = result['statistics']
    print("📊 Statistics:")
    print(f"  Total sections:         {stats.get('total_sections', 0)}")
    print(f"  Total subsections:      {stats.get('total_subsections', 0)}")
    print(f"  Total words:            {stats.get('total_words', 0):,}")
    print(f"  Total content:          {stats.get('total_content', 0):,} chars")
    print(f"  Avg section length:     {stats.get('average_section_length', 0)} words")
    print(f"  Sections with code:     {stats.get('sections_with_code', 0)}")
    print(f"  Sections with tables:   {stats.get('sections_with_tables', 0)}\n")
    
    return result['document_id']


def demo_hierarchy(document_id: str):
    """Demo: Get document hierarchy."""
    print("="*70)
    print("DEMO 2: Document Hierarchy")
    print("="*70 + "\n")
    
    print(f"📑 Getting hierarchy for document: {document_id}\n")
    
    hierarchy_result = get_document_hierarchy(document_id)
    
    print("🌳 Document Structure:\n")
    _print_hierarchy(hierarchy_result['hierarchy'], indent=0)
    
    print(f"\n📊 Hierarchy Statistics:")
    for key, value in hierarchy_result['statistics'].items():
        print(f"  {key}: {value}")


def _print_hierarchy(sections, indent=0):
    """Recursively print section hierarchy."""
    for section in sections:
        prefix = "  " * indent + "├─ "
        title = section['title'][:50] + "..." if len(section['title']) > 50 else section['title']
        print(f"{prefix}{title}")
        print(f"{'  ' * (indent + 1)}  (Level {section['level']}, {section['word_count']} words)")
        
        if section['subsections']:
            _print_hierarchy(section['subsections'], indent + 1)


def demo_search(document_id: str, query: str):
    """Demo: Search sections."""
    print("\n" + "="*70)
    print(f"DEMO 3: Search Sections (query: '{query}')")
    print("="*70 + "\n")
    
    print(f"🔍 Searching document {document_id} for: '{query}'\n")
    
    results = search_sections(document_id, query)
    
    print(f"✅ Search Complete!\n")
    print(f"Query:          {results['query']}")
    print(f"Total matches:  {results['total_matches']}\n")
    
    if results['results']:
        print("📍 Matching Sections:\n")
        for i, result in enumerate(results['results'], 1):
            print(f"{i}. {result['title']}")
            print(f"   Section ID: {result['section_id']}")
            print(f"   Level: {result['level']}")
            print(f"   Matches: {result['match_count']}")
            print()
    else:
        print("❌ No matches found.")


def demo_metadata(document_id: str):
    """Demo: Get section metadata."""
    print("="*70)
    print("DEMO 4: Section Metadata")
    print("="*70 + "\n")
    
    parser = get_enhanced_parser()
    sections = parser.memory.get_document_sections(document_id)
    
    if not sections:
        print(f"❌ No sections found for document {document_id}")
        return
    
    print(f"📊 Sample Metadata for First 3 Sections:\n")
    
    for i, section in enumerate(sections[:3], 1):
        full_id = section.full_id
        metadata = get_section_metadata(full_id)
        
        print(f"{i}. {metadata['title']}")
        print(f"   Full ID:         {full_id}")
        print(f"   Level:           {metadata['level']}")
        print(f"   Word Count:      {metadata['word_count']}")
        print(f"   Content Length:  {metadata['content_length']} chars")
        print(f"   Has Code:        {metadata['has_code']}")
        print(f"   Has Tables:      {metadata['has_tables']}")
        print(f"   Subsections:     {metadata['subsection_count']}")
        print(f"   Created:         {metadata['created_at']}\n")


def demo_all_documents():
    """Demo: List all documents in memory."""
    print("="*70)
    print("DEMO 5: All Documents in Memory")
    print("="*70 + "\n")
    
    all_docs = list_all_documents()
    
    print(f"📚 Documents in Memory: {all_docs['document_count']}\n")
    
    if all_docs['documents']:
        print("Documents:")
        for doc in all_docs['documents']:
            print(f"  • {doc['document_id']}")
            print(f"    Sections: {doc['section_count']}")
            stats = doc['statistics']
            print(f"    Words: {stats.get('total_words', 0):,}")
            print(f"    Tables: {stats.get('sections_with_tables', 0)}")
            print()
    else:
        print("No documents loaded yet.")


def main():
    """Run demonstration."""
    
    if len(sys.argv) < 2:
        print("Usage: python test_enhanced_parser.py <pdf_file_path>\n")
        print("Example: python test_enhanced_parser.py ~/Downloads/Agiliti.pdf\n")
        print("This script demonstrates:")
        print("  1. Extracting sections into memory")
        print("  2. Viewing document hierarchy")
        print("  3. Searching across sections")
        print("  4. Getting section metadata")
        print("  5. Listing documents in memory")
        return
    
    pdf_path = sys.argv[1]
    
    # Extract sections
    document_id = demo_extract_sections(pdf_path)
    
    if not document_id:
        return
    
    # Show hierarchy
    demo_hierarchy(document_id)
    
    # Search
    demo_search(document_id, "agreement")
    
    # Metadata
    demo_metadata(document_id)
    
    # All documents
    demo_all_documents()
    
    print("\n" + "="*70)
    print("✅ Demo Complete!")
    print("="*70 + "\n")
    print("Sections are now in memory and ready for:")
    print("  • Fast lookups (<1ms)")
    print("  • Full-text search")
    print("  • Hierarchical navigation")
    print("  • Metadata analysis")


if __name__ == "__main__":
    main()
