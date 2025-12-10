#!/usr/bin/env python3
"""
Example 2: Search Within PDF
Search for specific content in a PDF document.
"""

from pathlib import Path
from skills.pdf_parser import claude_search_pdf


def main():
    """Run PDF search example."""
    # Replace with your PDF path and search term
    pdf_path = Path("contracts/loan_agreement.pdf")
    search_term = "covenant"
    
    if not pdf_path.exists():
        print(f"❌ File not found: {pdf_path}")
        return
    
    print(f"🔍 Searching for: '{search_term}'")
    print(f"📄 In document: {pdf_path.name}")
    print("-" * 50)
    
    # Search the document
    results = claude_search_pdf(str(pdf_path), search_term)
    
    # Display results
    print(f"\n📊 Search Results:")
    print(f"  Query: {results['query']}")
    print(f"  Document: {results['document']}")
    print(f"  Matches Found: {results['matches_found']}")
    
    if results['matches_found'] > 0:
        print(f"\n📍 Matches:")
        for i, match in enumerate(results['results'], 1):
            print(f"\n  {i}. Line {match['line_number']}")
            print(f"     Content: {match['content'][:100]}...")
            print(f"     Match: '{match['match']}'")
    else:
        print(f"\n   No matches found for '{search_term}'")


if __name__ == "__main__":
    main()
