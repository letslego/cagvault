from dataclasses import dataclass
from enum import Enum
from typing import Iterator
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompt_values import PromptValue
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.language_models import BaseChatModel
from knowledge import KnowledgeSource

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
You're having a conversation with a user about their files. Try to be helpful and answer their questions.
If you don't know the answer, say that you don't know and try to ask clarifying questions.
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