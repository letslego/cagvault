"""
MCP Tools for Agentic RAG System

Provides specialized tools that the agentic RAG can use during query processing:
- Web search for external context
- Document entity extraction
- Section importance ranking  
- Cross-document relationship discovery
- Fact verification against external sources
"""

import logging
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MCPToolResult:
    """Result from an MCP tool invocation."""
    tool_name: str
    success: bool
    data: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = None


class AgenticMCPTools:
    """MCP-compatible tools for agentic RAG system."""
    
    def __init__(self, mcp_server, parser, search_parser):
        """Initialize tools with MCP server and parsers.
        
        Args:
            mcp_server: ClaudeMCPServer instance
            parser: EnhancedPDFParserSkill instance
            search_parser: SearchableParser instance with NER
        """
        self.mcp_server = mcp_server
        self.parser = parser
        self.search_parser = search_parser
        
    def web_search_augment(
        self,
        query: str,
        document_context: str,
        max_results: int = 3
    ) -> MCPToolResult:
        """Search the web to augment document context with external information.
        
        Use when:
        - Document doesn't contain enough context
        - Need current/external information (e.g., company status, market data)
        - Verifying facts against external sources
        
        Args:
            query: Search query (refined based on document gap)
            document_context: Brief summary of what documents contain
            max_results: Number of web results to retrieve
            
        Returns:
            MCPToolResult with web search findings
        """
        try:
            # Start async web search
            task_id = self.mcp_server.start_web_search(query, max_results)
            
            # Poll for completion (with timeout)
            import time
            timeout = 30
            start = time.time()
            
            while time.time() - start < timeout:
                task = self.mcp_server.get_task(task_id)
                if task and task.status == "done":
                    return MCPToolResult(
                        tool_name="web_search_augment",
                        success=True,
                        data=task.result,
                        metadata={
                            "query": query,
                            "elapsed": time.time() - start,
                            "result_count": task.result.get("search_results", {}).get("count", 0)
                        }
                    )
                elif task and task.status == "error":
                    return MCPToolResult(
                        tool_name="web_search_augment",
                        success=False,
                        data=None,
                        error=task.error
                    )
                time.sleep(0.5)
            
            return MCPToolResult(
                tool_name="web_search_augment",
                success=False,
                data=None,
                error="Web search timed out"
            )
            
        except Exception as e:
            logger.error(f"Web search augment failed: {e}")
            return MCPToolResult(
                tool_name="web_search_augment",
                success=False,
                data=None,
                error=str(e)
            )
    
    def extract_entities_from_context(
        self,
        contexts: List[Dict[str, Any]],
        entity_types: Optional[List[str]] = None
    ) -> MCPToolResult:
        """Extract named entities from retrieved contexts.
        
        Use when:
        - Need to identify key parties, amounts, dates
        - Building structured data from unstructured text
        - Cross-referencing entities across sections
        
        Args:
            contexts: Retrieved context sections
            entity_types: Specific types to extract (MONEY, DATE, ORG, etc.)
            
        Returns:
            MCPToolResult with extracted entities by type
        """
        try:
            all_entities = {}
            entity_types = entity_types or ["MONEY", "DATE", "ORG", "PERSON", "GPE", "PERCENT"]
            
            for ctx in contexts:
                section_id = ctx.get("source_id", "unknown")
                
                # Get entities for this section
                for entity_type in entity_types:
                    entities = self.search_parser.search_engine.search_entities(
                        section_id,
                        entity_type
                    )
                    
                    if entity_type not in all_entities:
                        all_entities[entity_type] = []
                    
                    for entity in entities:
                        all_entities[entity_type].append({
                            "text": entity.text,
                            "type": entity.type,
                            "section_id": entity.section_id,
                            "context": ctx.get("content", "")[:200]
                        })
            
            return MCPToolResult(
                tool_name="extract_entities_from_context",
                success=True,
                data=all_entities,
                metadata={
                    "entity_types": list(all_entities.keys()),
                    "total_entities": sum(len(v) for v in all_entities.values()),
                    "contexts_processed": len(contexts)
                }
            )
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return MCPToolResult(
                tool_name="extract_entities_from_context",
                success=False,
                data=None,
                error=str(e)
            )
    
    def rank_sections_by_importance(
        self,
        document_id: str,
        query_intent: str
    ) -> MCPToolResult:
        """Rank document sections by importance for the query.
        
        Use when:
        - Need to prioritize which sections to read deeply
        - Document has many sections but limited context window
        - Want credit analyst perspective on section relevance
        
        Args:
            document_id: Document to analyze
            query_intent: User's question intent
            
        Returns:
            MCPToolResult with sections ranked by importance
        """
        try:
            # Get all sections with importance scores
            sections = self.parser.memory.get_document_sections(document_id)
            
            if not sections:
                return MCPToolResult(
                    tool_name="rank_sections_by_importance",
                    success=False,
                    data=None,
                    error=f"No sections found for document {document_id}"
                )
            
            # Flatten and score
            flat_sections = self.parser._flatten_sections(sections)
            
            ranked = []
            for section in flat_sections:
                metadata = section.metadata
                score = metadata.importance_score if hasattr(metadata, 'importance_score') else 0.5
                section_type = metadata.section_type if hasattr(metadata, 'section_type') else None
                
                # Boost score based on query intent
                if query_intent:
                    intent_lower = query_intent.lower()
                    title_lower = metadata.title.lower()
                    
                    # Boost if intent keywords in title
                    intent_keywords = intent_lower.split()
                    matching_keywords = sum(1 for kw in intent_keywords if kw in title_lower)
                    score += matching_keywords * 0.1
                    
                    # Boost specific section types based on common intents
                    if "covenant" in intent_lower and section_type == "COVENANT":
                        score += 0.2
                    elif "default" in intent_lower and section_type == "DEFAULT":
                        score += 0.2
                    elif "definition" in intent_lower and section_type == "DEFINITIONS":
                        score += 0.2
                    elif "payment" in intent_lower and section_type == "PAYMENT":
                        score += 0.2
                
                ranked.append({
                    "section_id": metadata.id,
                    "title": metadata.title,
                    "importance_score": min(score, 1.0),
                    "section_type": section_type,
                    "word_count": metadata.word_count,
                    "page_range": metadata.page_range
                })
            
            # Sort by score
            ranked.sort(key=lambda x: x["importance_score"], reverse=True)
            
            return MCPToolResult(
                tool_name="rank_sections_by_importance",
                success=True,
                data=ranked,
                metadata={
                    "document_id": document_id,
                    "total_sections": len(ranked),
                    "query_intent": query_intent
                }
            )
            
        except Exception as e:
            logger.error(f"Section ranking failed: {e}")
            return MCPToolResult(
                tool_name="rank_sections_by_importance",
                success=False,
                data=None,
                error=str(e)
            )
    
    def find_cross_document_relationships(
        self,
        document_ids: List[str],
        relationship_type: str = "references"
    ) -> MCPToolResult:
        """Find relationships between sections across multiple documents.
        
        Use when:
        - Analyzing related agreements (parent/subsidiary, amendment/original)
        - Finding cross-references between documents
        - Understanding document dependencies
        
        Args:
            document_ids: List of document IDs to analyze
            relationship_type: Type of relationship (references, amendments, guarantees)
            
        Returns:
            MCPToolResult with discovered relationships
        """
        try:
            relationships = []
            
            # Get all sections from all documents
            all_sections = {}
            for doc_id in document_ids:
                sections = self.parser.memory.get_document_sections(doc_id)
                if sections:
                    all_sections[doc_id] = self.parser._flatten_sections(sections)
            
            # Look for cross-references
            for doc_id, sections in all_sections.items():
                for section in sections:
                    content_lower = section.content.lower()
                    
                    # Check for references to other documents
                    for other_doc_id in document_ids:
                        if other_doc_id == doc_id:
                            continue
                        
                        # Simple heuristic: look for mentions
                        other_doc_name = other_doc_id.split('_')[-1][:20]
                        
                        if relationship_type == "references":
                            # Look for phrases like "as defined in", "pursuant to"
                            if any(phrase in content_lower for phrase in [
                                "as defined in",
                                "pursuant to",
                                "in accordance with",
                                "subject to",
                                "reference is made to"
                            ]):
                                relationships.append({
                                    "from_document": doc_id,
                                    "from_section": section.metadata.id,
                                    "from_title": section.metadata.title,
                                    "to_document": other_doc_id,
                                    "relationship": "references",
                                    "confidence": 0.7
                                })
                        
                        elif relationship_type == "amendments":
                            if any(word in content_lower for word in ["amend", "modify", "supplement"]):
                                relationships.append({
                                    "from_document": doc_id,
                                    "from_section": section.metadata.id,
                                    "to_document": other_doc_id,
                                    "relationship": "amends",
                                    "confidence": 0.6
                                })
            
            return MCPToolResult(
                tool_name="find_cross_document_relationships",
                success=True,
                data={
                    "relationships": relationships,
                    "relationship_count": len(relationships)
                },
                metadata={
                    "document_count": len(document_ids),
                    "relationship_type": relationship_type
                }
            )
            
        except Exception as e:
            logger.error(f"Cross-document analysis failed: {e}")
            return MCPToolResult(
                tool_name="find_cross_document_relationships",
                success=False,
                data=None,
                error=str(e)
            )
    
    def verify_fact_with_web(
        self,
        claim: str,
        document_source: str
    ) -> MCPToolResult:
        """Verify a factual claim from document against web sources.
        
        Use when:
        - Need to verify current status (company still exists, merger completed)
        - Checking if regulatory requirements still apply
        - Validating numerical data (stock prices, market caps)
        
        Args:
            claim: The factual claim to verify
            document_source: Where in document this claim appears
            
        Returns:
            MCPToolResult with verification status and evidence
        """
        try:
            # Formulate search query
            search_query = f"verify: {claim}"
            
            # Start web search
            task_id = self.mcp_server.start_web_search(search_query, max_results=5)
            
            # Poll for results
            import time
            timeout = 30
            start = time.time()
            
            while time.time() - start < timeout:
                task = self.mcp_server.get_task(task_id)
                if task and task.status == "done":
                    search_results = task.result.get("search_results", {})
                    items = search_results.get("items", [])
                    
                    # Simple verification logic
                    verification_status = "uncertain"
                    evidence = []
                    
                    for item in items:
                        snippet = item.get("snippet", "").lower()
                        claim_keywords = set(claim.lower().split())
                        snippet_keywords = set(snippet.split())
                        
                        overlap = len(claim_keywords & snippet_keywords)
                        if overlap >= len(claim_keywords) * 0.5:
                            evidence.append({
                                "source": item.get("url"),
                                "title": item.get("title"),
                                "snippet": item.get("snippet"),
                                "relevance": overlap / len(claim_keywords)
                            })
                    
                    if evidence:
                        verification_status = "supported"
                    
                    return MCPToolResult(
                        tool_name="verify_fact_with_web",
                        success=True,
                        data={
                            "claim": claim,
                            "verification_status": verification_status,
                            "evidence": evidence,
                            "evidence_count": len(evidence)
                        },
                        metadata={
                            "document_source": document_source,
                            "search_query": search_query
                        }
                    )
                    
                elif task and task.status == "error":
                    return MCPToolResult(
                        tool_name="verify_fact_with_web",
                        success=False,
                        data=None,
                        error=task.error
                    )
                    
                time.sleep(0.5)
            
            return MCPToolResult(
                tool_name="verify_fact_with_web",
                success=False,
                data=None,
                error="Verification timed out"
            )
            
        except Exception as e:
            logger.error(f"Fact verification failed: {e}")
            return MCPToolResult(
                tool_name="verify_fact_with_web",
                success=False,
                data=None,
                error=str(e)
            )
    
    def suggest_follow_up_questions(
        self,
        answer: str,
        sources: List[Dict[str, Any]],
        original_query: str
    ) -> MCPToolResult:
        """Generate intelligent follow-up questions based on answer and sources.
        
        Use when:
        - Answer is provided but may have deeper aspects to explore
        - Sources contain related information not fully addressed
        - Helping user discover what else they should ask
        
        Args:
            answer: The generated answer
            sources: Source contexts used
            original_query: User's original question
            
        Returns:
            MCPToolResult with suggested follow-up questions
        """
        try:
            suggestions = []
            
            # Analyze answer for potential gaps
            answer_lower = answer.lower()
            
            # Common follow-up patterns
            if "covenant" in answer_lower:
                suggestions.append("What are the consequences of breaching this covenant?")
                suggestions.append("How frequently is this covenant tested?")
            
            if "payment" in answer_lower or "interest" in answer_lower:
                suggestions.append("What is the payment schedule or frequency?")
                suggestions.append("Are there any prepayment options or penalties?")
            
            if "default" in answer_lower:
                suggestions.append("What remedies are available to lenders upon default?")
                suggestions.append("Is there a cure period for this default event?")
            
            if "definition" in answer_lower or "means" in answer_lower:
                suggestions.append("How is this term used elsewhere in the document?")
                suggestions.append("Are there any exceptions to this definition?")
            
            # Analyze sources for unexplored topics
            for source in sources[:3]:
                title = source.get("section_title", "")
                if title and title.lower() not in answer_lower:
                    suggestions.append(f"What information is in the '{title}' section?")
            
            # Limit to top 5
            suggestions = list(set(suggestions))[:5]
            
            return MCPToolResult(
                tool_name="suggest_follow_up_questions",
                success=True,
                data={
                    "suggestions": suggestions,
                    "count": len(suggestions)
                },
                metadata={
                    "original_query": original_query,
                    "sources_analyzed": len(sources)
                }
            )
            
        except Exception as e:
            logger.error(f"Follow-up suggestion failed: {e}")
            return MCPToolResult(
                tool_name="suggest_follow_up_questions",
                success=False,
                data=None,
                error=str(e)
            )


def create_agentic_mcp_tools(mcp_server, parser, search_parser) -> AgenticMCPTools:
    """Factory function to create MCP tools for agentic RAG."""
    return AgenticMCPTools(mcp_server, parser, search_parser)
