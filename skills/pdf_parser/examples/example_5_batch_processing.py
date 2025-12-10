#!/usr/bin/env python3
"""
Example 5: Batch Processing
Parse multiple PDF documents in a directory.
"""

from pathlib import Path
from skills.pdf_parser import PDFParserSkill


def main():
    """Run batch processing example."""
    # Directory containing PDF files
    pdf_dir = Path("contracts")
    
    if not pdf_dir.exists():
        print(f"❌ Directory not found: {pdf_dir}")
        return
    
    # Find all PDFs
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"❌ No PDF files found in {pdf_dir}")
        return
    
    print(f"📂 Batch Processing PDF Files")
    print(f"📁 Directory: {pdf_dir}")
    print(f"📊 Files found: {len(pdf_files)}")
    print("-" * 50)
    
    # Create skill instance
    skill = PDFParserSkill()
    
    # Process each file
    results = []
    for i, pdf_file in enumerate(pdf_files, 1):
        try:
            print(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_file.name}")
            
            # Parse the PDF
            doc = skill.parse_pdf(str(pdf_file))
            
            # Store result
            results.append({
                'file': pdf_file.name,
                'pages': doc.metadata.pages,
                'size': doc.metadata.file_size,
                'sections': len(doc.sections),
                'tables': len(doc.tables),
                'status': '✅ Success'
            })
            
            print(f"    ✓ {doc.metadata.pages} pages, {len(doc.sections)} sections, {len(doc.tables)} tables")
            
        except Exception as e:
            print(f"    ✗ Error: {e}")
            results.append({
                'file': pdf_file.name,
                'status': f'❌ Error: {str(e)[:50]}'
            })
    
    # Display summary
    print(f"\n\n📈 Summary:")
    print("-" * 50)
    print(f"{'File':<30} {'Pages':<8} {'Sections':<10} {'Tables':<8} {'Status':<20}")
    print("-" * 50)
    
    total_pages = 0
    for result in results:
        file = result['file']
        pages = result.get('pages', 'N/A')
        sections = result.get('sections', 'N/A')
        tables = result.get('tables', 'N/A')
        status = result['status']
        
        if isinstance(pages, int):
            total_pages += pages
        
        print(f"{file:<30} {str(pages):<8} {str(sections):<10} {str(tables):<8} {status:<20}")
    
    print("-" * 50)
    print(f"Total pages processed: {total_pages}")
    print(f"Success rate: {sum(1 for r in results if '✅' in r['status'])}/{len(results)}")


if __name__ == "__main__":
    main()
