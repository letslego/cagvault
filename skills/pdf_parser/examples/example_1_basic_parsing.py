#!/usr/bin/env python3
"""
Example 1: Basic PDF Parsing
Parse a PDF document and display its information.
"""

from pathlib import Path
from skills.pdf_parser import parse_document


def main():
    """Run basic PDF parsing example."""
    # Replace with your PDF path
    pdf_path = Path("contracts/loan_agreement.pdf")
    
    if not pdf_path.exists():
        print(f"❌ File not found: {pdf_path}")
        print("Please provide a valid PDF file path")
        return
    
    print(f"📄 Parsing: {pdf_path.name}")
    print("-" * 50)
    
    # Parse the document
    result = parse_document(str(pdf_path))
    
    # Display results
    print(f"\n📊 Document Information:")
    print(f"  Name: {result['name']}")
    print(f"  Pages: {result['metadata']['pages']}")
    print(f"  File Size: {result['metadata']['file_size'] / 1024:.1f} KB")
    
    print(f"\n📑 Structure:")
    print(f"  Sections: {len(result['sections'])}")
    print(f"  Tables: {len(result['tables'])}")
    
    if result['sections']:
        print(f"\n📚 Top Sections:")
        for i, section in enumerate(result['sections'][:5], 1):
            title = section.get('title', 'Untitled')
            level = section.get('level', 1)
            print(f"    {i}. {'  ' * (level-1)}{title}")
    
    if result['tables']:
        print(f"\n📋 Tables Found:")
        for i, table in enumerate(result['tables'], 1):
            print(f"    {i}. {len(table.get('rows', []))} rows × {len(table.get('headers', []))} columns")
    
    print(f"\n📝 Content Preview:")
    print(result['content_preview'])
    print("...")


if __name__ == "__main__":
    main()
