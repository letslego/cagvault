#!/usr/bin/env python3
"""
Example 4: Integrate with CAG Knowledge Base
Parse PDF and add to CAG knowledge base for Q&A.
"""

from pathlib import Path
from skills.pdf_parser import claude_ingest_pdf


def main():
    """Run CAG integration example."""
    # Replace with your PDF path
    pdf_path = Path("contracts/loan_agreement.pdf")
    
    if not pdf_path.exists():
        print(f"❌ File not found: {pdf_path}")
        return
    
    print(f"🚀 Ingesting PDF into CAG Knowledge Base")
    print(f"📄 Document: {pdf_path.name}")
    print("-" * 50)
    
    # Ingest into CAG
    result = claude_ingest_pdf(str(pdf_path))
    
    # Display confirmation
    print(f"\n✅ Ingestion Status: {result['status'].upper()}")
    print(f"\n📊 Document Details:")
    print(f"  ID: {result['document_id']}")
    print(f"  Name: {result['name']}")
    print(f"  Type: {result['type']}")
    print(f"  Size: {result['content_length']} characters")
    
    print(f"\n💬 Message: {result['message']}")
    
    print(f"\n📝 Next Steps:")
    print(f"  1. Ask questions about this document in the CAG chatbot")
    print(f"  2. Use the document ID '{result['document_id']}' for tracking")
    print(f"  3. Document is now available for credit analyst Q&A")


if __name__ == "__main__":
    main()
