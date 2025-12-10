#!/usr/bin/env python3
"""
Example 3: Extract Document Structure
Get the table of contents and hierarchical structure of a PDF.
"""

from pathlib import Path
from skills.pdf_parser import claude_get_structure


def main():
    """Run document structure extraction example."""
    # Replace with your PDF path
    pdf_path = Path("contracts/loan_agreement.pdf")
    
    if not pdf_path.exists():
        print(f"❌ File not found: {pdf_path}")
        return
    
    print(f"📚 Extracting Structure: {pdf_path.name}")
    print("-" * 50)
    
    # Get structure
    structure = claude_get_structure(str(pdf_path))
    
    # Display header info
    print(f"\n📄 Document Info:")
    print(f"  Name: {structure['document']}")
    print(f"  Pages: {structure['pages']}")
    print(f"  Tables: {structure['tables']}")
    
    # Display table of contents
    print(f"\n📑 Table of Contents:")
    if structure['toc']:
        for item in structure['toc']:
            indent = "  " * (item['level'] - 1)
            subsections = f" ({item['subsections']} subsections)" if item['subsections'] > 0 else ""
            print(f"{indent}• {item['title']}{subsections}")
    else:
        print("  No sections found")
    
    # Display raw sections for detailed view
    if structure['sections']:
        print(f"\n🔍 Detailed Sections ({len(structure['sections'])} total):")
        for section in structure['sections'][:5]:  # Show first 5
            level = section.get('level', 1)
            title = section.get('title', 'Untitled')
            content_len = len(section.get('content', ''))
            subs = len(section.get('subsections', []))
            print(f"\n  {'  ' * (level-1)}{'→ ' if subs > 0 else '• '}{title}")
            print(f"     Content: {content_len} chars, Subsections: {subs}")


if __name__ == "__main__":
    main()
