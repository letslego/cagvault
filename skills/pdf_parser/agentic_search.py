"""
Agentic Search using Claude LLM for Intelligent Document Exploration

Uses Claude as an agent to:
- Understand search intent and context
- Reason about which sections are relevant
- Provide contextual explanations for results
- Handle complex, multi-part queries
"""

import logging
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SearchContext:
    """Context for agentic search."""
    query: str
    document_id: str
    sections: Dict[str, Any]  # section_id -> section data
    search_history: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.search_history is None:
            self.search_history = []


class AgenticSearchEngine:
    """Agentic search using Claude LLM to understand and explore documents."""
    
    def __init__(self, llm=None):
        """Initialize agentic search engine.
        
        Args:
            llm: LangChain LLM instance (e.g., ChatOllama or ChatAnthropic)
        """
        self.llm = llm
        self.search_history: List[Dict[str, Any]] = []
    
    def search(
        self,
        query: str,
        sections: Dict[str, str],
        section_titles: Dict[str, str],
        top_k: int = 5
    ) -> Dict[str, Any]:
        """Search using Claude LLM agent to understand and find relevant sections.
        
        Args:
            query: User search query
            sections: Dict of section_id -> section content
            section_titles: Dict of section_id -> section title
            top_k: Number of results to return
            
        Returns:
            Search results with reasoning and relevant sections
        """
        if not self.llm:
            return self._fallback_search(query, sections, section_titles, top_k)
        
        try:
            # Create context for Claude
            section_summaries = self._create_section_summaries(
                sections, section_titles
            )
            
            # Prompt Claude to reason about the query and find relevant sections
            prompt = self._build_search_prompt(
                query, section_summaries, top_k
            )
            
            # Get response from Claude
            response = self.llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse response to extract section IDs and reasoning
            result = self._parse_search_response(
                response_text, sections, section_titles
            )
            
            # Store in history
            self.search_history.append({
                'query': query,
                'response': result,
                'reasoning': response_text
            })
            
            return result
            
        except Exception as e:
            logger.warning(f"Agentic search failed: {e}. Falling back to keyword search.")
            return self._fallback_search(query, sections, section_titles, top_k)
    
    def _create_section_summaries(
        self, sections: Dict[str, str], section_titles: Dict[str, str]
    ) -> str:
        """Create summaries of sections for Claude to reason about."""
        summaries = []
        for section_id, content in sections.items():
            title = section_titles.get(section_id, "Untitled")
            # Create a brief summary
            preview = content[:200].replace('\n', ' ')
            summaries.append(f"- [{section_id}] {title}: {preview}...")
        
        return "\n".join(summaries[:20])  # Limit to first 20 for context
    
    def _build_search_prompt(
        self, query: str, section_summaries: str, top_k: int
    ) -> str:
        """Build prompt for Claude to search document sections."""
        return f"""You are an intelligent document search agent. Your task is to find the most relevant sections for a user query.

USER QUERY:
{query}

AVAILABLE DOCUMENT SECTIONS:
{section_summaries}

Your task:
1. Understand the user's search intent
2. Identify which sections are most relevant to answering the query
3. Provide top {top_k} section IDs with brief explanations
4. Format your response as JSON with this structure:
{{
    "reasoning": "Your explanation of the search intent and selection",
    "results": [
        {{"section_id": "id", "relevance": "explanation"}},
        ...
    ]
}}

Respond ONLY with valid JSON, no other text."""
    
    def _parse_search_response(
        self, response_text: str, sections: Dict[str, str],
        section_titles: Dict[str, str]
    ) -> Dict[str, Any]:
        """Parse Claude's response to extract search results."""
        import re
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                
                # Build results with full section data
                results = []
                for item in parsed.get('results', []):
                    section_id = item.get('section_id')
                    if section_id in sections:
                        results.append({
                            'section_id': section_id,
                            'title': section_titles.get(section_id, 'Untitled'),
                            'relevance_explanation': item.get('relevance'),
                            'content': sections[section_id]
                        })
                
                return {
                    'reasoning': parsed.get('reasoning', ''),
                    'results': results,
                    'count': len(results)
                }
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"Failed to parse Claude response: {e}")
        
        return self._fallback_search(response_text, sections, section_titles, 5)
    
    def _fallback_search(
        self, query: str, sections: Dict[str, str],
        section_titles: Dict[str, str], top_k: int
    ) -> Dict[str, Any]:
        """Fallback keyword-based search when Claude is unavailable."""
        query_lower = query.lower()
        scores = {}
        
        # Simple keyword matching
        for section_id, content in sections.items():
            score = 0
            if query_lower in content.lower():
                score += 1
            title = section_titles.get(section_id, '').lower()
            if query_lower in title:
                score += 2  # Title matches are weighted higher
            
            if score > 0:
                scores[section_id] = score
        
        # Sort by score and return top-k
        sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = [
            {
                'section_id': section_id,
                'title': section_titles.get(section_id, 'Untitled'),
                'relevance_explanation': 'Contains query terms',
                'content': sections[section_id]
            }
            for section_id, _ in sorted_ids[:top_k]
        ]
        
        return {
            'reasoning': f'Fallback keyword search for: {query}',
            'results': results,
            'count': len(results)
        }
    
    def get_search_history(self) -> List[Dict[str, Any]]:
        """Retrieve search history."""
        return self.search_history
    
    def clear_history(self) -> None:
        """Clear search history."""
        self.search_history.clear()
