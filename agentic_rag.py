"""
Advanced Agentic RAG System

Implements a sophisticated RAG pipeline with:
- Multi-step reasoning for query understanding
- Dynamic retrieval strategy selection
- Context-aware chunk selection
- Answer synthesis with citations
- Self-reflection and answer validation
"""

import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class RetrievalStrategy(Enum):
    """Available retrieval strategies."""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    AGENTIC = "agentic"
    ENTITY_BASED = "entity_based"


@dataclass
class QueryPlan:
    """Represents the agent's plan for answering a query."""
    original_query: str
    intent: str  # What the user is trying to find out
    strategy: RetrievalStrategy
    keywords: List[str]
    entities_to_find: List[str] = field(default_factory=list)
    expected_section_types: List[str] = field(default_factory=list)
    reasoning: str = ""
    

@dataclass
class RetrievedContext:
    """Retrieved context with metadata."""
    content: str
    source_id: str
    relevance_score: float
    section_title: str = ""
    reasoning: str = ""
    entities: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AgenticAnswer:
    """Final answer with full provenance."""
    answer: str
    confidence: float
    sources: List[RetrievedContext]
    reasoning_steps: List[str]
    plan: QueryPlan
    thinking: Optional[str] = None
    validation_notes: str = ""


class AgenticRAG:
    """Advanced RAG system with agent-driven reasoning."""
    
    def __init__(
        self,
        llm,
        parser,
        search_parser,
        enable_self_reflection: bool = True,
        mcp_tools=None
    ):
        """Initialize agentic RAG system.
        
        Args:
            llm: Language model for reasoning and generation
            parser: Enhanced PDF parser with document access
            search_parser: Searchable parser with multiple search strategies
            enable_self_reflection: Whether to validate answers before returning
            mcp_tools: Optional MCP tools for extended capabilities
        """
        self.llm = llm
        self.parser = parser
        self.search_parser = search_parser
        self.enable_self_reflection = enable_self_reflection
        self.mcp_tools = mcp_tools
        self.tool_usage_history = []  # Track which tools were used
        
    def answer_query(
        self,
        query: str,
        document_ids: List[str],
        max_context_length: int = 16000
    ) -> AgenticAnswer:
        """Answer a query using multi-step agentic reasoning.
        
        Steps:
        1. Query Understanding: Decompose and understand user intent
        2. Strategy Selection: Choose best retrieval strategy
        3. Retrieval: Execute retrieval with chosen strategy
        4. Synthesis: Generate answer from retrieved context
        5. Validation: Self-reflect and validate answer quality
        
        Args:
            query: User question
            document_ids: Documents to search
            max_context_length: Maximum context to pass to LLM
            
        Returns:
            AgenticAnswer with complete provenance
        """
        reasoning_steps = []
        
        # Step 1: Understand the query and create a plan
        reasoning_steps.append("Understanding query intent...")
        plan = self._understand_query(query, reasoning_steps)
        
        # Step 1.5: Decide if MCP tools should be used
        tool_decisions = []
        if self.mcp_tools:
            tool_decisions = self._decide_tool_usage(plan, reasoning_steps)
        
        # Step 2: Execute retrieval strategy (potentially with tool augmentation)
        reasoning_steps.append(f"Executing {plan.strategy.value} retrieval...")
        contexts = self._retrieve_context(
            plan, document_ids, max_context_length, reasoning_steps, tool_decisions
        )
        
        # Step 3: Synthesize answer
        reasoning_steps.append("Synthesizing answer from retrieved context...")
        answer, thinking = self._synthesize_answer(query, contexts, plan, reasoning_steps)
        
        # Step 4: Calculate confidence
        confidence = self._calculate_confidence(answer, contexts, plan)
        
        # Create initial answer
        agentic_answer = AgenticAnswer(
            answer=answer,
            confidence=confidence,
            sources=contexts,
            reasoning_steps=reasoning_steps,
            plan=plan,
            thinking=thinking
        )
        
        # Step 5: Self-reflection (optional)
        if self.enable_self_reflection and confidence < 0.8:
            reasoning_steps.append("Validating answer quality...")
            agentic_answer = self._validate_answer(agentic_answer, query)
        
        return agentic_answer
    
    def _understand_query(self, query: str, reasoning_steps: List[str]) -> QueryPlan:
        """Use LLM to understand query intent and create retrieval plan."""
        prompt = f"""Analyze this user query and create a retrieval plan.

Query: "{query}"

Provide a JSON response with:
1. intent: What is the user trying to find out?
2. strategy: Best retrieval strategy (semantic/keyword/hybrid/agentic/entity_based)
3. keywords: Key terms to search for
4. entities_to_find: Named entities that might be relevant (companies, amounts, dates)
5. expected_section_types: Types of sections likely to contain the answer (DEFINITIONS, COVENANTS, DEFAULTS, etc.)
6. reasoning: Brief explanation of your plan

Format:
{{
    "intent": "...",
    "strategy": "semantic",
    "keywords": ["term1", "term2"],
    "entities_to_find": ["entity1", "entity2"],
    "expected_section_types": ["TYPE1", "TYPE2"],
    "reasoning": "..."
}}

Respond ONLY with valid JSON."""

        try:
            response = self.llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse JSON response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                
                # Map string strategy to enum
                strategy_map = {
                    "semantic": RetrievalStrategy.SEMANTIC,
                    "keyword": RetrievalStrategy.KEYWORD,
                    "hybrid": RetrievalStrategy.HYBRID,
                    "agentic": RetrievalStrategy.AGENTIC,
                    "entity_based": RetrievalStrategy.ENTITY_BASED
                }
                
                strategy = strategy_map.get(
                    data.get("strategy", "semantic"),
                    RetrievalStrategy.SEMANTIC
                )
                
                plan = QueryPlan(
                    original_query=query,
                    intent=data.get("intent", query),
                    strategy=strategy,
                    keywords=data.get("keywords", []),
                    entities_to_find=data.get("entities_to_find", []),
                    expected_section_types=data.get("expected_section_types", []),
                    reasoning=data.get("reasoning", "")
                )
                
                reasoning_steps.append(f"Plan: {plan.reasoning}")
                return plan
                
        except Exception as e:
            logger.warning(f"Query understanding failed: {e}. Using default plan.")
        
        # Fallback plan
        return QueryPlan(
            original_query=query,
            intent=query,
            strategy=RetrievalStrategy.SEMANTIC,
            keywords=query.split(),
            reasoning="Fallback to basic semantic search"
        )
    
    def _decide_tool_usage(self, plan: QueryPlan, reasoning_steps: List[str]) -> List[Dict[str, Any]]:
        """Decide which MCP tools to use based on query plan.
        
        Args:
            plan: Query plan with intent and strategy
            reasoning_steps: List to append reasoning to
            
        Returns:
            List of tool decisions with tool names and parameters
        """
        decisions = []
        intent_lower = plan.intent.lower()
        
        # Web search for external context
        if any(keyword in intent_lower for keyword in [
            "current", "latest", "recent", "today", "status", "verify", "confirm"
        ]):
            decisions.append({
                "tool": "web_search_augment",
                "reason": "Query requires current/external information",
                "priority": "high"
            })
            reasoning_steps.append("Will use web search for current information")
        
        # Entity extraction for structured data needs
        if any(keyword in intent_lower for keyword in [
            "list all", "what are the", "identify", "extract", "find all"
        ]) or plan.strategy == RetrievalStrategy.ENTITY_BASED:
            decisions.append({
                "tool": "extract_entities_from_context",
                "reason": "Query needs structured entity extraction",
                "priority": "medium"
            })
            reasoning_steps.append("Will extract entities from contexts")
        
        # Section ranking for large documents
        if any(keyword in intent_lower for keyword in [
            "important", "key", "main", "primary", "critical"
        ]):
            decisions.append({
                "tool": "rank_sections_by_importance",
                "reason": "Query needs importance-based prioritization",
                "priority": "high"
            })
            reasoning_steps.append("Will rank sections by importance")
        
        # Cross-document analysis
        if any(keyword in intent_lower for keyword in [
            "compare", "difference", "relationship", "across", "between documents"
        ]):
            decisions.append({
                "tool": "find_cross_document_relationships",
                "reason": "Query requires cross-document analysis",
                "priority": "medium"
            })
            reasoning_steps.append("Will analyze cross-document relationships")
        
        return decisions
    
    def _retrieve_context(
        self,
        plan: QueryPlan,
        document_ids: List[str],
        max_length: int,
        reasoning_steps: List[str],
        tool_decisions: List[Dict[str, Any]] = None
    ) -> List[RetrievedContext]:
        """Execute retrieval strategy based on plan, with optional tool augmentation."""
        tool_decisions = tool_decisions or []
        contexts = []
        
        # Execute priority tools before retrieval
        web_augmentation = None
        section_rankings = None
        
        for decision in tool_decisions:
            if decision["tool"] == "web_search_augment" and decision.get("priority") == "high":
                if self.mcp_tools:
                    result = self.mcp_tools.web_search_augment(
                        query=plan.original_query,
                        document_context=plan.intent,
                        max_results=3
                    )
                    if result.success:
                        web_augmentation = result.data
                        reasoning_steps.append(f"Web search found {result.metadata.get('result_count', 0)} external sources")
                        self.tool_usage_history.append(result)
            
            elif decision["tool"] == "rank_sections_by_importance" and decision.get("priority") == "high":
                if self.mcp_tools and document_ids:
                    result = self.mcp_tools.rank_sections_by_importance(
                        document_id=document_ids[0],
                        query_intent=plan.intent
                    )
                    if result.success:
                        section_rankings = result.data
                        reasoning_steps.append(f"Ranked {len(section_rankings)} sections by importance")
                        self.tool_usage_history.append(result)
        
        # Normal retrieval with tool enhancements
        
        if plan.strategy == RetrievalStrategy.AGENTIC:
            # Use agentic search
            result = self.search_parser.agentic_search(plan.original_query, top_k=5)
            for res in result.get("results", []):
                contexts.append(RetrievedContext(
                    content=res.get("content", ""),
                    source_id=res.get("section_id", ""),
                    relevance_score=0.9,  # High score for agent-selected
                    section_title=res.get("title", ""),
                    reasoning=res.get("relevance_explanation", "")
                ))
            reasoning_steps.append(f"Agent reasoning: {result.get('reasoning', '')}")
            
        elif plan.strategy == RetrievalStrategy.SEMANTIC:
            # Use semantic search
            result = self.search_parser.semantic_search(plan.original_query, top_k=5)
            for res in result:
                contexts.append(RetrievedContext(
                    content=res.get("content", ""),
                    source_id=res.get("section_id", ""),
                    relevance_score=res.get("score", 0.0),
                    section_title=res.get("title", "")
                ))
            reasoning_steps.append(f"Found {len(contexts)} semantically relevant sections")
            
        elif plan.strategy == RetrievalStrategy.KEYWORD:
            # Use keyword search
            results = self.search_parser.search(' '.join(plan.keywords), top_k=5)
            for res in results:
                contexts.append(RetrievedContext(
                    content=res.get("content", ""),
                    source_id=res.get("section_id", ""),
                    relevance_score=res.get("score", 0.5),
                    section_title=res.get("title", "")
                ))
            reasoning_steps.append(f"Found {len(contexts)} keyword matches")
            
        elif plan.strategy == RetrievalStrategy.ENTITY_BASED:
            # Search by entities
            all_results = []
            for entity_type in plan.entities_to_find:
                for doc_id in document_ids:
                    entities = self.search_parser.search_by_entity_type(doc_id, entity_type)
                    all_results.extend(entities)
            
            # Convert to contexts
            for ent in all_results[:5]:
                contexts.append(RetrievedContext(
                    content=ent.get("in_section", ""),
                    source_id=ent.get("section_id", ""),
                    relevance_score=0.8,
                    section_title="",
                    entities=[{"text": ent.get("text"), "type": ent.get("type")}]
                ))
            reasoning_steps.append(f"Found {len(contexts)} sections with relevant entities")
        
        # Hybrid strategy: combine multiple approaches
        elif plan.strategy == RetrievalStrategy.HYBRID:
            # Get both semantic and keyword results
            semantic_result = self.search_parser.semantic_search(plan.original_query, top_k=3)
            keyword_results = self.search_parser.search(' '.join(plan.keywords), top_k=3)
            
            seen_ids = set()
            for res in semantic_result + keyword_results:
                sid = res.get("section_id", "")
                if sid not in seen_ids:
                    contexts.append(RetrievedContext(
                        content=res.get("content", ""),
                        source_id=sid,
                        relevance_score=res.get("score", 0.5),
                        section_title=res.get("title", "")
                    ))
                    seen_ids.add(sid)
            reasoning_steps.append(f"Hybrid search found {len(contexts)} unique sections")
        
        # Truncate contexts to fit max length
        contexts = self._truncate_contexts(contexts, max_length)
        
        return contexts
    
    def _truncate_contexts(
        self, contexts: List[RetrievedContext], max_length: int
    ) -> List[RetrievedContext]:
        """Truncate contexts to fit within max length."""
        total_length = 0
        truncated = []
        
        # Sort by relevance score
        sorted_contexts = sorted(contexts, key=lambda x: x.relevance_score, reverse=True)
        
        for ctx in sorted_contexts:
            ctx_len = len(ctx.content)
            if total_length + ctx_len <= max_length:
                truncated.append(ctx)
                total_length += ctx_len
            else:
                # Try to fit partial context
                remaining = max_length - total_length
                if remaining > 500:  # Only include if meaningful amount fits
                    ctx.content = ctx.content[:remaining]
                    truncated.append(ctx)
                break
        
        return truncated
    
    def _synthesize_answer(
        self,
        query: str,
        contexts: List[RetrievedContext],
        plan: QueryPlan,
        reasoning_steps: List[str]
    ) -> Tuple[str, str]:
        """Synthesize answer from retrieved contexts using LLM."""
        # Build context string
        context_str = "\n\n".join([
            f"[Source: {ctx.section_title or ctx.source_id}]\n{ctx.content}"
            for ctx in contexts
        ])
        
        prompt = f"""Answer the user's question using the provided document excerpts.

User Question: {query}

Intent: {plan.intent}

Retrieved Context:
{context_str}

Instructions:
1. Answer the question directly and concisely
2. Use information ONLY from the provided context
3. Cite specific sources in your answer using [Source: ...] notation
4. If the context doesn't contain enough information, say so
5. Provide your reasoning process as <thinking>...</thinking>

Your answer:"""

        try:
            response = self.llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Extract thinking if present
            import re
            thinking = ""
            thinking_match = re.search(r'<thinking>(.*?)</thinking>', response_text, re.DOTALL)
            if thinking_match:
                thinking = thinking_match.group(1).strip()
                # Remove thinking from answer
                answer = re.sub(r'<thinking>.*?</thinking>', '', response_text, flags=re.DOTALL).strip()
            else:
                answer = response_text.strip()
            
            reasoning_steps.append(f"Generated answer with {len(contexts)} sources")
            return answer, thinking
            
        except Exception as e:
            logger.error(f"Answer synthesis failed: {e}")
            return f"Error generating answer: {str(e)}", ""
    
    def _calculate_confidence(
        self,
        answer: str,
        contexts: List[RetrievedContext],
        plan: QueryPlan
    ) -> float:
        """Calculate confidence score for the answer."""
        confidence = 0.5  # Base confidence
        
        # Boost for multiple high-quality sources
        if len(contexts) >= 3:
            confidence += 0.2
        
        # Boost for high relevance scores
        avg_relevance = sum(c.relevance_score for c in contexts) / max(len(contexts), 1)
        confidence += avg_relevance * 0.2
        
        # Boost if answer contains citations
        if "[Source:" in answer:
            confidence += 0.1
        
        # Penalty if answer mentions uncertainty
        uncertainty_phrases = [
            "not enough information",
            "unclear",
            "cannot determine",
            "insufficient context"
        ]
        if any(phrase in answer.lower() for phrase in uncertainty_phrases):
            confidence -= 0.3
        
        return min(max(confidence, 0.0), 1.0)
    
    def _validate_answer(
        self, agentic_answer: AgenticAnswer, original_query: str
    ) -> AgenticAnswer:
        """Validate answer quality through self-reflection."""
        validation_prompt = f"""Review this answer for quality and accuracy.

Original Question: {original_query}

Answer: {agentic_answer.answer}

Retrieved Sources: {len(agentic_answer.sources)} documents

Evaluate:
1. Does the answer directly address the question?
2. Is the answer well-supported by the sources?
3. Are there any logical inconsistencies?
4. What could be improved?

Provide feedback in 1-2 sentences."""

        try:
            response = self.llm.invoke(validation_prompt)
            validation_text = response.content if hasattr(response, 'content') else str(response)
            
            agentic_answer.validation_notes = validation_text.strip()
            agentic_answer.reasoning_steps.append(f"Validation: {validation_text[:100]}...")
            
        except Exception as e:
            logger.warning(f"Answer validation failed: {e}")
        
        return agentic_answer


def create_agentic_rag(llm, parser, search_parser, enable_reflection=True, mcp_tools=None) -> AgenticRAG:
    """Factory function to create agentic RAG system.
    
    Args:
        llm: Language model instance
        parser: Enhanced PDF parser
        search_parser: Searchable parser with NER
        enable_reflection: Enable self-reflection
        mcp_tools: Optional MCP tools for extended capabilities
    """
    return AgenticRAG(
        llm=llm,
        parser=parser,
        search_parser=search_parser,
        enable_self_reflection=enable_reflection,
        mcp_tools=mcp_tools
    )
