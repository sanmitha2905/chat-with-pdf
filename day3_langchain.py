import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# =============================================
# BLOCK 1 — LLM
# =============================================

# OLD WAY — manual Groq
# client = Groq(api_key=os.getenv("GROQ_API_KEY"))
# response = client.chat.completions.create(
#     model="llama-3.3-70b-versatile",
#     messages=[{"role": "user", "content": "What is AI?"}]
# )
# answer = response.choices[0].message.content

# NEW WAY — LangChain
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)

# call it
response = llm.invoke("What is AI in 2 lines?")

print(f"Type of response: {type(response)}")
print(f"Answer: {response.content}")

from langchain_core.prompts import ChatPromptTemplate

# =============================================
# BLOCK 2 — PROMPT TEMPLATE
# =============================================
print("=" * 50)
print("BLOCK 2: PROMPT TEMPLATE")
print("=" * 50)

# OLD WAY — manual string formatting
# prompt = f"You are an expert in {subject}. Explain {topic} in 3 lines."
# messy, error-prone, hard to reuse

# NEW WAY — LangChain Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in {subject}. Be concise."),
    ("human", "Explain {topic} in exactly 3 lines.")
])

# see what the prompt looks like before sending to AI
formatted = prompt.format_messages(
    subject="Artificial Intelligence",
    topic="RAG"
)

print("Formatted prompt:")
for message in formatted:
    print(f"  Role: {message.type}")
    print(f"  Content: {message.content}")
    print()

# now send to AI
chain = prompt | llm
response = chain.invoke({
    "subject": "Artificial Intelligence",
    "topic": "RAG"
})

print(f"AI Answer: {response.content}")
print()

# reuse same template for different topics
topics = [
    {"subject": "Python", "topic": "list comprehension"},
    {"subject": "databases", "topic": "vector databases"},
]

print("Reusing same template:")
for t in topics:
    response = chain.invoke(t)
    print(f"Topic: {t['topic']}")
    print(f"Answer: {response.content}")
    print()

    from langchain_core.output_parsers import StrOutputParser

# =============================================
# BLOCK 3 — OUTPUT PARSER
# =============================================
print("=" * 50)
print("BLOCK 3: OUTPUT PARSER")
print("=" * 50)

parser = StrOutputParser()

# WITHOUT parser — returns AIMessage object
chain_without_parser = prompt | llm
response1 = chain_without_parser.invoke({
    "subject": "AI",
    "topic": "embeddings"
})
print(f"Without parser type: {type(response1)}")
print(f"Without parser: {response1.content}")
print()

# WITH parser — returns plain string directly
chain_with_parser = prompt | llm | parser
response2 = chain_with_parser.invoke({
    "subject": "AI",
    "topic": "embeddings"
})
print(f"With parser type: {type(response2)}")
print(f"With parser: {response2}")
print()

# WHY THIS MATTERS — use output directly in your code
words = response2.split()
print(f"Word count: {len(words)}")
print(f"First word: {words[0]}")

from langchain_community.document_loaders import TextLoader

# =============================================
# BLOCK 4 — DOCUMENT LOADER
# =============================================
print("=" * 50)
print("BLOCK 4: DOCUMENT LOADER")
print("=" * 50)

# load the text file
loader = TextLoader("sample.txt")
documents = loader.load()

# what does a loaded document look like?
print(f"Number of documents loaded: {len(documents)}")
print()
print(f"Type: {type(documents[0])}")
print()
print(f"Page content: {documents[0].page_content}")
print()
print(f"Metadata: {documents[0].metadata}")

from langchain_text_splitters import RecursiveCharacterTextSplitter

# =============================================
# BLOCK 5 — TEXT SPLITTER
# =============================================
print("=" * 50)
print("BLOCK 5: TEXT SPLITTER")
print("=" * 50)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

# split the document
chunks = splitter.split_documents(documents)

print(f"Original: 1 document")
print(f"After splitting: {len(chunks)} chunks")
print()

# see each chunk
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:")
    print(f"  Content : {chunk.page_content.strip()}")
    print(f"  Length  : {len(chunk.page_content)} characters")
    print(f"  Metadata: {chunk.metadata}")
    print()

# show overlap in action
print("=" * 50)
print("OVERLAP DEMONSTRATION")
print("=" * 50)
print(f"End of Chunk 1  : ...{chunks[0].page_content.strip()[-30:]}")
print(f"Start of Chunk 2: {chunks[1].page_content.strip()[:30]}...")
print()
print("Notice overlapping text between chunks 👆")

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# =============================================
# BLOCK 6 — VECTOR STORE
# =============================================
print("=" * 50)
print("BLOCK 6: VECTOR STORE")
print("=" * 50)

# Step 1 — embedding model
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
print("✅ Embedding model loaded")
print()

# Step 2 — store chunks in ChromaDB
print("Embedding and storing chunks...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./day3_vectordb"
)
print(f"✅ {len(chunks)} chunks embedded and stored")
print()

# Step 3 — search the vectorstore
print("=" * 50)
print("SEARCHING VECTOR STORE")
print("=" * 50)

questions = [
    "What is RAG?",
    "Which tool builds AI web apps?",
    "What city is growing in AI?"
]

for question in questions:
    results = vectorstore.similarity_search(question, k=2)
    print(f"Question: {question}")
    for i, result in enumerate(results):
        print(f"  Match {i+1}: {result.page_content.strip()}")
    print()

# Step 4 — Full RAG with LangChain
print("=" * 50)
print("FULL RAG PIPELINE — LANGCHAIN STYLE")
print("=" * 50)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# retriever — finds relevant chunks automatically
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# prompt
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant.
Answer the question using ONLY the context below.
If answer is not in context say 'I don't have that information.'

Context: {context}"""),
    ("human", "{question}")
])

parser = StrOutputParser()

# manual RAG chain
def rag_chain(question):
    # retrieve relevant chunks
    relevant_chunks = retriever.invoke(question)
    
    # combine chunks into context
    context = "\n".join([chunk.page_content for chunk in relevant_chunks])
    
    # get answer from LLM
    response = rag_prompt | llm | parser
    answer = response.invoke({
        "context": context,
        "question": question
    })
    
    return answer, context

# test it
test_questions = [
    "What is LangChain used for?",
    "What is Hyderabad known for in AI?",
    "What is the capital of France?"  # not in documents!
]

for question in test_questions:
    answer, context = rag_chain(question)
    print(f"Q: {question}")
    print(f"A: {answer}")
    print()

print("=" * 50)
print("✅ Day 3 Complete!")
print("=" * 50)