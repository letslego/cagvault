#!/usr/bin/env python3
"""
Compare PDF Parsing Output: Claude Skill vs Docling Direct

This script compares the output from:
1. Claude Skill PDF Parser (wrapper with caching and structure extraction)
2. Docling Parser (direct library usage)

Shows differences in output format, structure extraction, and performance.
"""

import time
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import sys

# Add to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from docling.document_converter import (
    DocumentConverter,
    InputFormat,
    PdfFormatOption,
    StandardPdfPipeline,
)
from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from skills.pdf_parser import PDFParserSkill


class PDFParserComparison:
    """Compare Claude Skill vs Docling parser outputs."""
    
    def __init__(self):
        """Initialize both parsers."""
        self.skill = PDFParserSkill()
        self.docling = self._create_docling_converter()
    
    @staticmethod
    def _create_docling_converter() -> DocumentConverter:
        """Create raw Docling converter."""
        return DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=StandardPdfPipeline,
                    backend=DoclingParseV4DocumentBackend,
                ),
            },
        )
    
    def compare_parsing(self, pdf_path: str) -> Dict[str, Any]:
        """
        Parse PDF with both methods and compare outputs.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Comparison dictionary with results from both parsers
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            return {"error": f"File not found: {pdf_path}"}
        
        print(f"\n{'='*70}")
        print(f"📊 COMPARING PDF PARSERS")
        print(f"{'='*70}")
        print(f"📄 Document: {pdf_path.name}")
        print(f"{'='*70}\n")
        
        # Parse with Claude Skill
        print("🔄 Parsing with Claude Skill...")
        skill_start = time.time()
        try:
            skill_result = self.skill.parse_pdf(str(pdf_path))
            skill_time = time.time() - skill_start
            skill_success = True
        except Exception as e:
            skill_result = None
            skill_time = time.time() - skill_start
            skill_success = False
            print(f"❌ Claude Skill Error: {e}")
        
        # Parse with Docling Direct
        print("🔄 Parsing with Docling (direct)...")
        docling_start = time.time()
        try:
            docling_result = self.docling.convert(str(pdf_path))
            docling_time = time.time() - docling_start
            docling_success = True
        except Exception as e:
            docling_result = None
            docling_time = time.time() - docling_start
            docling_success = False
            print(f"❌ Docling Error: {e}")
        
        # Compare results
        return {
            "document": pdf_path.name,
            "file_size": pdf_path.stat().st_size,
            "skill": {
                "success": skill_success,
                "time": skill_time,
                "result": self._format_skill_output(skill_result) if skill_success else None,
                "error": None if skill_success else str(skill_result)
            },
            "docling": {
                "success": docling_success,
                "time": docling_time,
                "result": self._format_docling_output(docling_result) if docling_success else None,
                "error": None if docling_success else str(docling_result)
            }
        }
    
    @staticmethod
    def _format_skill_output(doc) -> Dict[str, Any]:
        """Format Claude Skill output for comparison."""
        return {
            "name": doc.name,
            "id": doc.id,
            "metadata": {
                "pages": doc.metadata.pages,
                "file_size": doc.metadata.file_size,
                "parse_date": doc.metadata.parse_date,
                "format": doc.metadata.format,
            },
            "content": {
                "length": len(doc.content),
                "preview": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
            },
            "structure": {
                "sections": len(doc.sections),
                "top_sections": [
                    {
                        "level": s.get("level", 1),
                        "title": s.get("title", "Untitled"),
                        "content_length": len(s.get("content", ""))
                    }
                    for s in doc.sections[:3]
                ]
            },
            "tables": {
                "count": len(doc.tables),
                "first_table": {
                    "headers": doc.tables[0].get("headers", []) if doc.tables else None,
                    "rows": len(doc.tables[0].get("rows", [])) if doc.tables else None
                } if doc.tables else None
            },
            "cache_path": doc.cache_path
        }
    
    @staticmethod
    def _format_docling_output(result) -> Dict[str, Any]:
        """Format Docling output for comparison."""
        markdown = result.document.export_to_markdown()
        
        return {
            "pages": len(result.document.pages),
            "content": {
                "length": len(markdown),
                "preview": markdown[:200] + "..." if len(markdown) > 200 else markdown
            },
            "document_info": {
                "main_text": len(result.document.main_text) if hasattr(result.document, 'main_text') else "N/A",
                "pages": len(result.document.pages)
            },
            "raw_output_size": len(str(result.document))
        }
    
    def display_comparison(self, comparison: Dict[str, Any]) -> None:
        """Display formatted comparison results."""
        
        if "error" in comparison:
            print(f"❌ Error: {comparison['error']}")
            return
        
        print(f"\n{'='*70}")
        print(f"⏱️  PERFORMANCE COMPARISON")
        print(f"{'='*70}\n")
        
        skill_time = comparison['skill']['time']
        docling_time = comparison['docling']['time']
        
        print(f"Claude Skill Parser:  {skill_time:.2f}s")
        print(f"Docling Direct:      {docling_time:.2f}s")
        print(f"Difference:          {abs(skill_time - docling_time):.2f}s")
        
        if skill_time < docling_time:
            speedup = docling_time / skill_time
            print(f"✅ Claude Skill {speedup:.2f}x faster (likely due to caching)")
        else:
            overhead = skill_time / docling_time
            print(f"📦 Docling {overhead:.2f}x faster (no wrapper overhead)")
        
        # Output comparison
        print(f"\n{'='*70}")
        print(f"📊 OUTPUT COMPARISON")
        print(f"{'='*70}\n")
        
        if comparison['skill']['success']:
            skill_result = comparison['skill']['result']
            print("🎯 Claude Skill Parser Output:")
            print(f"   ✓ Name: {skill_result['name']}")
            print(f"   ✓ Pages: {skill_result['metadata']['pages']}")
            print(f"   ✓ Content length: {skill_result['content']['length']} chars")
            print(f"   ✓ Sections detected: {skill_result['structure']['sections']}")
            print(f"   ✓ Tables detected: {skill_result['tables']['count']}")
            print(f"   ✓ Has caching: {bool(skill_result['cache_path'])}")
            print(f"\n   Top sections found:")
            for section in skill_result['structure']['top_sections']:
                indent = "  " * (section['level'] - 1)
                print(f"     {indent}→ {section['title']} ({section['content_length']} chars)")
        else:
            print(f"❌ Claude Skill Error: {comparison['skill']['error']}")
        
        print()
        
        if comparison['docling']['success']:
            docling_result = comparison['docling']['result']
            print("🔬 Docling Direct Output:")
            print(f"   ✓ Pages: {docling_result['pages']}")
            print(f"   ✓ Content length: {docling_result['content']['length']} chars")
            print(f"   ✓ Document info available: {bool(docling_result['document_info'])}")
            print(f"   ✓ Raw output size: {docling_result['raw_output_size']} bytes")
        else:
            print(f"❌ Docling Error: {comparison['docling']['error']}")
        
        # Detailed comparison
        print(f"\n{'='*70}")
        print(f"🔍 DETAILED ANALYSIS")
        print(f"{'='*70}\n")
        
        if comparison['skill']['success'] and comparison['docling']['success']:
            skill_content = comparison['skill']['result']['content']['length']
            docling_content = comparison['docling']['result']['content']['length']
            
            print(f"Content Length Comparison:")
            print(f"   Claude Skill: {skill_content} chars")
            print(f"   Docling:      {docling_content} chars")
            
            if skill_content == docling_content:
                print(f"   ✅ Same content extraction")
            else:
                diff_pct = abs(skill_content - docling_content) / max(skill_content, docling_content) * 100
                print(f"   ⚠️  Different content length ({diff_pct:.1f}% difference)")
            
            print(f"\nStructure Detection:")
            skill_sections = comparison['skill']['result']['structure']['sections']
            print(f"   Claude Skill detected: {skill_sections} sections")
            print(f"   Docling: Structure extraction in skill wrapper only")
            
            print(f"\nFeatures:")
            print(f"   Claude Skill offers:")
            print(f"      • Automatic caching")
            print(f"      • Section hierarchy extraction")
            print(f"      • Table detection")
            print(f"      • Search capability")
            print(f"      • CAG integration ready")
            print(f"\n   Docling Direct offers:")
            print(f"      • Raw document parsing")
            print(f"      • Multiple export formats")
            print(f"      • Direct access to document objects")
            print(f"      • No caching overhead")
        
        # Recommendations
        print(f"\n{'='*70}")
        print(f"💡 RECOMMENDATIONS")
        print(f"{'='*70}\n")
        
        print("Use Claude Skill PDF Parser when:")
        print("   ✓ You need CAG integration")
        print("   ✓ You want automatic caching")
        print("   ✓ You need section extraction")
        print("   ✓ You're doing repeated parsing (cache helps)")
        print("   ✓ You want search functionality")
        
        print("\nUse Docling Direct when:")
        print("   ✓ You need raw document access")
        print("   ✓ You want multiple export formats")
        print("   ✓ Single-parse operations (no cache benefit)")
        print("   ✓ You need direct document object manipulation")
        print("   ✓ You want minimal wrapper overhead")
    
    def save_comparison_json(self, comparison: Dict[str, Any], output_file: str = "comparison_results.json") -> None:
        """Save comparison results to JSON file."""
        with open(output_file, 'w') as f:
            json.dump(comparison, f, indent=2, default=str)
        print(f"\n✅ Results saved to {output_file}")


def main():
    """Run comparison."""
    
    # Check if PDF path provided
    if len(sys.argv) < 2:
        print("Usage: python compare_parsers.py <pdf_file_path>")
        print("\nExample: python compare_parsers.py contracts/loan_agreement.pdf")
        print("\nThis script compares:")
        print("  1. Claude Skill PDF Parser (with caching & structure)")
        print("  2. Docling Parser (direct library usage)")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    # Run comparison
    comparator = PDFParserComparison()
    comparison = comparator.compare_parsing(pdf_path)
    
    # Display results
    comparator.display_comparison(comparison)
    
    # Save results
    comparator.save_comparison_json(comparison)


if __name__ == "__main__":
    main()
