"""
Claude Agent SDK MCP Tools for CAG Vault

This module uses the Claude Agent SDK to create MCP tools that extend the capabilities
of the Agentic RAG system. These tools are created using the @tool decorator and
exposed via create_sdk_mcp_server.

Tools Available:
- web_search: Search the web for current information
- extract_entities: Extract entities from text using NER
- rank_sections: Rank document sections by importance
- find_cross_references: Find relationships between documents
- verify_facts: Verify facts against web sources
- suggest_followups: Generate follow-up questions

Implementation:
- Tools delegate to AgenticMCPTools class from agentic_mcp_tools.py
- Shared global instance initialized when needed
"""

import asyncio
from typing import Any, Dict, List, Optional
from datetime import datetime
import json
import re

try:
    from claude_agent_sdk import tool, create_sdk_mcp_server
    AGENT_SDK_AVAILABLE = True
except ImportError:
    AGENT_SDK_AVAILABLE = False
    print("Warning: claude-agent-sdk not installed. MCP tools will not be available.")


# ============================================================================
# Global Tools Instance
# ============================================================================

_agentic_tools_instance = None

def get_agentic_tools():
    """Get or create the global AgenticMCPTools instance."""
    global _agentic_tools_instance
    
    if _agentic_tools_instance is None:
        # Lazy import and initialization
        from agentic_mcp_tools import create_agentic_mcp_tools
        from skills.mcp.claude_mcp_server import ClaudeMCPServer
        from skills.pdf_parser.enhanced_parser import get_enhanced_parser
        from skills.pdf_parser.pdf_parser import get_searchable_parser
        
        mcp_server = ClaudeMCPServer()
        parser = get_enhanced_parser()
        search_parser = get_searchable_parser()
        
        _agentic_tools_instance = create_agentic_mcp_tools(mcp_server, parser, search_parser)
    
    return _agentic_tools_instance


# ============================================================================
# Tool 1: Web Search
# ============================================================================

@tool(
    "web_search",
    "Search the web for current information to augment document analysis",
    {
        "query": str,
        "max_results": int,
        "document_context": str  # Brief summary of what documents contain
    }
)
async def web_search(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Search the web for current information.
    
    Triggers: current, latest, recent, status, verify keywords
    """
    try:
        query = args["query"]
        max_results = args.get("max_results", 5)
        document_context = args.get("document_context", "")
        
        # Delegate to agentic_mcp_tools implementation
        tools = get_agentic_tools()
        result = tools.web_search_augment(query, document_context, max_results)
        
        if result.success:
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "success": True,
                        "data": result.data,
                        "metadata": result.metadata,
                        "query": query
                    }, indent=2)
                }]
            }
        else:
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "success": False,
                        "error": result.error,
                        "query": query
                    })
                }],
                "is_error": True
            }
            
    except Exception as e:
        return {
            "content": [{
                "type": "text",
                "text": f"Tool execution error: {str(e)}"
            }],
            "is_error": True
        }


# ============================================================================
# Tool 2: Entity Extraction
# ============================================================================

@tool(
    "extract_entities",
    "Extract named entities from document sections (dates, amounts, organizations, people)",
    {
        "contexts": list,  # List of context dicts with section info
        "entity_types": list  # ["MONEY", "DATE", "ORG", "PERSON", "GPE", "PERCENT"]
    }
)
async def extract_entities(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract entities using spaCy NER.
    
    Triggers: list all, identify, extract, find all
    """
    try:
        contexts = args["contexts"]
        entity_types = args.get("entity_types", ["MONEY", "DATE", "ORG", "PERSON", "GPE", "PERCENT"])
        
        # Delegate to agentic_mcp_tools implementation
        tools = get_agentic_tools()
        result = tools.extract_entities_from_context(contexts, entity_types)
        
        if result.success:
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "success": True,
                        "entities": result.data,
                        "metadata": result.metadata
                    }, indent=2)
                }]
            }
        else:
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "success": False,
                        "error": result.error
                    })
                }],
                "is_error": True
            }
        
    except Exception as e:
        return {
            "content": [{
                "type": "text",
                "text": f"Tool execution error: {str(e)}"
            }],
            "is_error": True
        }


# ============================================================================

# ============================================================================
# Tool 3: Rank Sections by Importance
# ============================================================================

@tool(
    "rank_sections",
    "Rank document sections by importance using credit analyst criteria",
    {
        "contexts": list,  # List of context dicts from retrieval
        "query": str,  # User query for intent-based boosting
        "top_k": int  # Number of top sections to return
    }
)
async def rank_sections(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Rank sections based on credit analyst importance criteria.
    
    Triggers: important, key, main, primary, critical
    """
    try:
        contexts = args["contexts"]
        query = args.get("query", "")
        top_k = args.get("top_k", 10)
        
        # Delegate to agentic_mcp_tools implementation
        tools = get_agentic_tools()
        result = tools.rank_sections_by_importance(contexts, query, top_k)
        
        if result.success:
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "success": True,
                        "ranked_sections": result.data,
                        "metadata": result.metadata
                    }, indent=2)
                }]
            }
        else:
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "success": False,
                        "error": result.error
                    })
                }],
                "is_error": True
            }
    except Exception as e:
        return {
            "content": [{
                "type": "text",
                "text": f"Tool execution error: {str(e)}"
            }],
            "is_error": True
        }
        
        # Credit analyst importance scores
        section_type_scores = {
            "financial": 10.0,
            "risk": 9.0,
            "debt": 8.5,
            "covenant": 8.5,
            "management": 7.5,
            "legal": 7.0,
            "default": 5.0
        }
        
        # Query intent boosting
        query_lower = query.lower()
        intent_boosts = {}
        if any(keyword in query_lower for keyword in ["financial", "revenue", "income", "balance"]):
            intent_boosts["financial"] = 3.0
        if any(keyword in query_lower for keyword in ["risk", "uncertainty", "contingent"]):
            intent_boosts["risk"] = 3.0
        if any(keyword in query_lower for keyword in ["debt", "loan", "credit", "facility"]):
            intent_boosts["debt"] = 2.5
        
        # Rank sections
        rankings = []
        for section in sections:
            title = section.get("title", "").lower()
            content = section.get("content", "")
            
            # Determine section type
            section_type = "default"
            if any(kw in title for kw in ["financial", "income", "balance", "cash flow"]):
                section_type = "financial"
            elif any(kw in title for kw in ["risk", "uncertainty", "contingent"]):
                section_type = "risk"
            elif any(kw in title for kw in ["debt", "loan", "credit", "covenant"]):
                section_type = "debt"
            elif any(kw in title for kw in ["management", "md&a", "discussion"]):
                section_type = "management"
            elif any(kw in title for kw in ["legal", "litigation", "proceeding"]):
                section_type = "legal"
            
            # Calculate score
            base_score = section_type_scores.get(section_type, 5.0)
            boost = intent_boosts.get(section_type, 0.0)
            final_score = base_score + boost
            
            rankings.append({
                "title": section.get("title", "Untitled"),
                "section_type": section_type,
                "score": final_score,
                "content_length": len(content)
            })
        
        # Sort by score descending
        rankings.sort(key=lambda x: x["score"], reverse=True)
        
        return {
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "success": True,
                    "rankings": rankings[:top_k],
                    "total_sections": len(sections),
                    "query_intent": list(intent_boosts.keys())
                }, indent=2)
            }]
        }
        
    except Exception as e:
        return {
            "content": [{
                "type": "text",
                "text": f"Section ranking error: {str(e)}"
            }],
            "is_error": True
        }


# ============================================================================
# Tool 4: Find Cross-Document Relationships
# ============================================================================

@tool(
    "find_cross_references",
    "Find relationships between documents (references, amendments, guarantees)",
    {
        "sections": list,  # List of section dicts
        "query": str  # Search query
    }
)
async def find_cross_references(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Find cross-document relationships.
    
    Triggers: compare, difference, relationship, reference, amendment
    """
    try:
        sections = args["sections"]
        query = args.get("query", "")
        
        # Delegate to agentic_mcp_tools implementation
        tools = get_agentic_tools()
        result = tools.find_cross_document_relationships(sections, query)
        
        if result.success:
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "success": True,
                        "relationships": result.data,
                        "metadata": result.metadata
                    }, indent=2)
                }]
            }
        else:
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "success": False,
                        "error": result.error
                    })
                }],
                "is_error": True
            }
        
    except Exception as e:
        return {
            "content": [{
                "type": "text",
                "text": f"Cross-reference analysis error: {str(e)}"
            }],
            "is_error": True
        }


# ============================================================================
# Tool 5: Verify Facts
# ============================================================================

@tool(
    "verify_facts",
    "Verify factual claims against external web sources",
    {
        "claim": str,
        "document_context": str  # Context from document
    }
)
async def verify_facts(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Verify facts using web search.
    
    Triggers: verify, check, confirm, validate, is this true
    """
    try:
        claim = args["claim"]
        document_context = args.get("document_context", "")
        
        # Delegate to agentic_mcp_tools implementation
        tools = get_agentic_tools()
        result = tools.verify_fact_with_web(claim, document_context)
        
        if result.success:
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "success": True,
                        "verification": result.data,
                        "metadata": result.metadata
                    }, indent=2)
                }]
            }
        else:
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "success": False,
                        "error": result.error
                    })
                }],
                "is_error": True
            }
        
    except Exception as e:
        return {
            "content": [{
                "type": "text",
                "text": f"Fact verification error: {str(e)}"
            }],
            "is_error": True
        }


# ============================================================================
# Tool 6: Suggest Follow-up Questions
# ============================================================================

@tool(
    "suggest_followups",
    "Generate intelligent follow-up questions based on the answer and document content",
    {
        "query": str,
        "answer": str,
        "contexts": list  # Retrieved contexts
    }
)
async def suggest_followups(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate follow-up questions.
    
    Always executes to provide next steps.
    """
    try:
        query = args["query"]
        answer = args["answer"]
        contexts = args.get("contexts", [])
        
        # Delegate to agentic_mcp_tools implementation
        tools = get_agentic_tools()
        result = tools.suggest_follow_up_questions(query, answer, contexts)
        
        if result.success:
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "success": True,
                        "questions": result.data,
                        "metadata": result.metadata
                    }, indent=2)
                }]
            }
        else:
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "success": False,
                        "error": result.error
                    })
                }],
                "is_error": True
            }
        
    except Exception as e:
        return {
            "content": [{
                "type": "text",
                "text": f"Follow-up generation error: {str(e)}"
            }],
            "is_error": True
        }


# ============================================================================
# Create MCP Server
# ============================================================================

def create_cag_mcp_server():
    """
    Create an SDK MCP server with all CAG tools.
    
    Returns:
        McpSdkServerConfig: Server configuration for use with ClaudeAgentOptions
    """
    if not AGENT_SDK_AVAILABLE:
        raise ImportError("claude-agent-sdk not installed. Run: pip install claude-agent-sdk")
    
    return create_sdk_mcp_server(
        name="cag-tools",
        version="1.0.0",
        tools=[
            web_search,
            extract_entities,
            rank_sections,
            find_cross_references,
            verify_facts,
            suggest_followups
        ]
    )


# ============================================================================
# Tool Names for Integration
# ============================================================================

# Tool names as they appear to Claude (prefixed with mcp__servername__)
CAG_TOOL_NAMES = [
    "mcp__cag-tools__web_search",
    "mcp__cag-tools__extract_entities",
    "mcp__cag-tools__rank_sections",
    "mcp__cag-tools__find_cross_references",
    "mcp__cag-tools__verify_facts",
    "mcp__cag-tools__suggest_followups"
]


if __name__ == "__main__":
    # Test the server creation
    try:
        server = create_cag_mcp_server()
        print(f"✅ CAG MCP Server created successfully")
        print(f"   Server name: cag-tools")
        print(f"   Version: 1.0.0")
        print(f"   Tools: {len([web_search, extract_entities, rank_sections, find_cross_references, verify_facts, suggest_followups])}")
        print(f"\nTool names for allowed_tools:")
        for tool_name in CAG_TOOL_NAMES:
            print(f"  - {tool_name}")
    except Exception as e:
        print(f"❌ Error: {e}")
