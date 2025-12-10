from dataclasses import dataclass
from enum import Enum
from typing import Iterator, Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompt_values import PromptValue
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.language_models import BaseChatModel
from knowledge import KnowledgeSource
from kvcache import get_kv_cache

THINK_START = "<think>"
THINK_END = "</think>"
 
class ChunkType(Enum):
    CONTENT = "content"
    START_THINK = "start_think"
    THINKING = "thinking"
    END_THINK = "end_think"
 
@dataclass
class Chunk:
    type: ChunkType
    content: str
    

SYSTEM_PROMPT = """
You are an expert credit analyst evaluating whether a document section fully answers a user's question.

CRITICAL: You must determine if this section provides a COMPLETE answer to ALL aspects of the question, not just partial information.
1. Credit agreements contain LAYERED provisions - a section that appears to answer a question may be modified, overridden, or incomplete without cross-references
2. You must be HIGHLY SKEPTICAL before triggering ANSWER
3. Check if the current section contains cross-references to OTHER sections that are essential to a complete answer
4. For certain question types, multiple related provisions MUST be analyzed together

DYNAMIC PLANNING:
If you discover this section contains explicit cross-references to other sections that are ESSENTIAL to answer the question, you can request they be checked next by listing their section IDs in the 'priority_sections' field.

Only use priority_sections when:
- This section explicitly mentions "See Section X" or "as defined in Section Y"
- The cross-referenced sections are MANDATORY to answer the question (per the cross-reference checks below)
- You're confident the section IDs are valid

DO NOT use priority_sections for:
- Speculative references (sections you think might be helpful)
- Sections already mentioned in previous notes
- General exploration

The system will automatically insert these sections next in the search queue if they haven't been checked yet.

MANDATORY CROSS-REFERENCE CHECKS:
Before triggering ANSWER, verify you have checked:

- For "Who" PARTY/ROLE/ENTITY questions:
    - Must list them by name and role.

- For COVENANT BREACH questions:
    - The covenant itself
    - The corresponding DEFAULT provision to understand WHO is affected and if it's "springing"
    - Any CURE RIGHTS mentioned in either section
    - Must have ALL THREE to answer completely

- For "CAN BORROWER INCUR DEBT?" questions:
    - ALL relevant debt baskets (not just one), including:
        * General basket
        * Ratio-based basket
        * Available Amount basket
    - For SECURED debt, corresponding LIEN permissions for EACH basket
    - Must verify basket-to-lien matching is correct
    - Must note any EBITDA dependencies or conditions

- For ACQUISITION/MERGER questions:
    - The merger covenant
    - The CHANGE OF CONTROL definition
    - The Change of Control DEFAULT provision
    - Must have ALL THREE to avoid missing overriding provisions

- For questions involving capitalized/defined terms:
    - The relevant definitions MUST be included
    - Calculations or measurements MUST reference the definition

Your evaluation process:
1. Review notes from previously analyzed sections (if any) to understand what's been found so far
2. Identify all components of the user's question (there may be multiple parts)
3. Check THIS section for explicit cross-references to other sections
4. Determine if this section + previous findings address EACH component completely
5. **CRITICAL: Before deciding ANSWER, ask yourself:**
    - "Does this section reference other sections I haven't seen yet?"
    - "For this question type, what related provisions are MANDATORY to check?"
    - "Could there be overriding provisions elsewhere?"
    - "Have I verified ALL alternative paths/baskets?"
    - "If this is about debt, have I matched it to the correct lien provision?"
6. Extract any relevant information from THIS section in notes field
7. Decide: ANSWER (complete) or PASS (incomplete/irrelevant)
8. When ANSWER, synthesize a complete response using information from this section AND previous notes
9. ALWAYS cite specific sections when providing your answer

ANSWER if:
- Every part of the question is addressed completely (either this section or combined with previous notes)
- All MANDATORY cross-references for this question type have been analyzed
- No unresolved references to other sections exist
- All alternative paths have been checked (for permissibility questions)
- Structural dependencies are verified (e.g., debt + lien provisions match)
- A credit analyst would consider this answer bulletproof, not just plausible
- Your answer explicitly cites ALL sections used and notes any dependencies/conditions

You may also ANSWER with a reasoned interpretation if:
- You have sufficient information to make a well-supported inference
- You clearly indicate the answer is based on interpretation rather than explicit provisions
- You explain your reasoning and cite the sections that support your interpretation
- You note what explicit information is missing and why your interpretation is reasonable

PASS if:
- Only partial information is present and you cannot reasonably infer an answer
- The section references other sections for complete details that would materially change your interpretation
- Critical thresholds, dates, conditions, or exceptions are missing that prevent any reasonable inference
- This section references other sections not yet retrieved that are likely essential
- For breach questions: haven't seen the default provision yet
- For debt questions: haven't checked ALL baskets or verified lien matching
- For merger questions: haven't checked Change of Control provisions

NOTES FIELD: Capture NEW information from THIS section only. This builds context for analyzing subsequent sections. Include:
- Key facts/provisions found
- Cross-references to other sections mentioned
- Gaps or unanswered components
- Question type indicators (e.g., "Need to check defaults" or "Need other debt baskets")

Remember: It's better to PASS and retrieve more sections than to give an incomplete answer that misses critical provisions.
""".strip()

PROMPT = """
Here's the information you have about the the files:
 
<context>
{context}
</context>
 
Please, respond to the query below
 
<question>
{question}
</question>
 
Answer:
"""
 
FILE_TEMPLATE = """
<file>
    <name>{name}</name>
    <content>{content}</content>
</file>
""".strip()
 
PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            SYSTEM_PROMPT,
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", PROMPT),
    ]
)

def ask(
    query: str, history: list[dict], sources: dict[str, KnowledgeSource], llm: BaseChatModel
) -> Iterator[Chunk]:
    in_think_block = False
    prompt_value = _create_prompt(query, history, sources)
    for chunk in llm.stream(prompt_value):
        text_chunk = chunk.content
        if text_chunk == THINK_START:
            yield Chunk(type=ChunkType.START_THINK, content="")
            in_think_block = True
        elif text_chunk == THINK_END:
            yield Chunk(type=ChunkType.END_THINK, content="")
            in_think_block = False
        else:
            chunk_type = ChunkType.THINKING if in_think_block else ChunkType.CONTENT
            yield Chunk(type=chunk_type, content=text_chunk)
            
def _create_chat_history(history: list[dict]) -> list[BaseMessage]:
    return [
        HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"])
        for m in history
    ]

def _create_context(sources: dict[str, KnowledgeSource]) -> str:
    return "".join(
        [
            FILE_TEMPLATE.format(name=source.name, content=source.content)
            for source in sources.values()
        ]
    )
    
def _create_prompt(
    query: str, history: list[dict], sources: dict[str, KnowledgeSource]
) -> PromptValue:
    chat_history = _create_chat_history(history)
    context = _create_context(sources)
    return PROMPT_TEMPLATE.invoke(
        {
            "question": query,
            "context": context,
            "chat_history": chat_history,
        }
    )


def create_context_cache(sources: dict[str, KnowledgeSource]) -> str:
    """
    Create and cache the context from given sources.
    
    This function preloads all documents into the KV-cache for efficient
    multi-turn conversations without reprocessing documents.
    
    Args:
        sources: Dictionary of knowledge sources to cache
        
    Returns:
        cache_id: The ID of the cached context for later retrieval
    """
    kv_cache = get_kv_cache()
    context = _create_context(sources)
    source_ids = list(sources.keys())
    
    cache_id = kv_cache.create(context, source_ids=source_ids)
    return cache_id


def get_cached_context(cache_id: str) -> Optional[str]:
    """
    Retrieve cached context by ID.
    
    Args:
        cache_id: The cache ID from create_context_cache()
        
    Returns:
        The cached context string, or None if not found
    """
    kv_cache = get_kv_cache()
    entry = kv_cache.get(cache_id)
    return entry.context if entry else None


def clear_cache() -> None:
    """Clear all cached contexts."""
    kv_cache = get_kv_cache()
    kv_cache.clear_all()