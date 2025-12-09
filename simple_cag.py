DATABASE = [
    "Working with LLMs is like working with a database. You can ask it to do things, and it will do them.",
    "Generating new ideas is now a mix between human creativity and machine intelligence.",
]

PROMPT = """
Use the following context to answer the question at the end.
 
<context>
{context}
</context>
 
<question>
{question}
</question>
 
Answer:
"""

llm = ChatOllama(
    model="llama3.2:latest",
    verbose=True,
    keep_alive=-1,
)

def ask(query: str):
    message = HumanMessage(PROMPT.format(context="\n".join(DATABASE), question=query))
    return llm.invoke([message])

pprint(ask("How are new ideas generated?"))
pprint(ask("What is like working with LLMs?"))
pprint(ask("What uses humans creativity and machine intelligence?"))