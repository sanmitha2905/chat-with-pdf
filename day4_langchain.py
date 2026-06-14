import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)
parser = StrOutputParser()

# =============================================
# PART 1 — LCEL DEEP DIVE
# =============================================
print("=" * 50)
print("PART 1: LCEL — LANGCHAIN EXPRESSION LANGUAGE")
print("=" * 50)

# Basic chain — you know this
basic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI tutor."),
    ("human", "{question}")
])

basic_chain = basic_prompt | llm | parser

# invoke = run the chain
answer = basic_chain.invoke({"question": "What is LangChain in 1 line?"})
print(f"Basic chain answer: {answer}")
print()

# Chain with RunnablePassthrough
# passes input through unchanged — useful for debugging
from langchain_core.runnables import RunnablePassthrough

debug_chain = (
    RunnablePassthrough()  # passes input as-is
    | basic_prompt
    | llm
    | parser
)

answer2 = debug_chain.invoke({"question": "What is RAG in 1 line?"})
print(f"Debug chain answer: {answer2}")
print()

# batch invoke — run same chain on multiple inputs at once
questions = [
    {"question": "What is RAG?"},
    {"question": "What is LangChain?"},
    {"question": "What is ChromaDB?"}
]

print("Batch processing 3 questions at once:")
answers = basic_chain.batch(questions)
for q, a in zip(questions, answers):
    print(f"Q: {q['question']}")
    print(f"A: {a}")
    print()

    from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# =============================================
# PART 2 — CONVERSATION MEMORY
# =============================================
print("=" * 50)
print("PART 2: CONVERSATION MEMORY")
print("=" * 50)

# store for keeping chat histories
# key = session_id, value = chat history
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# prompt that includes chat history
memory_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI tutor. Be concise."),
    ("placeholder", "{chat_history}"),  # history goes here
    ("human", "{question}")
])

# chain
memory_chain = memory_prompt | llm | parser

# wrap chain with memory
chain_with_memory = RunnableWithMessageHistory(
    memory_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history"
)

# config — identifies the conversation session
config = {"configurable": {"session_id": "sanmi_session"}}

# have a conversation
print("Starting conversation with memory:")
print()

conversations = [
    "My name is Sanmi and I am learning AI",
    "What is my name?",
    "What am I learning?",
    "Give me one tip for what I am learning"
]

for message in conversations:
    response = chain_with_memory.invoke(
        {"question": message},
        config=config
    )
    print(f"You: {message}")
    print(f"AI : {response}")
    print()

# show what's stored in memory
print("=" * 50)
print("WHAT'S IN MEMORY:")
print("=" * 50)
history = get_session_history("sanmi_session")
for msg in history.messages:
    print(f"{msg.type.upper()}: {msg.content[:60]}...")
print()

from langchain_community.document_loaders import PyPDFLoader

# =============================================
# PART 3 — PDF LOADER
# =============================================
print("=" * 50)
print("PART 3: PDF LOADER")
print("=" * 50)

# load PDF
loader = PyPDFLoader("document.pdf")
pages = loader.load()

print(f"PDF loaded successfully!")
print(f"Number of pages: {len(pages)}")
print()

# look at each page
for i, page in enumerate(pages):
    print(f"Page {i+1}:")
    print(f"  Content preview : {page.page_content[:100].strip()}...")
    print(f"  Metadata        : {page.metadata}")
    print()

# =============================================
# PART 3B — SPLIT PDF INTO CHUNKS
# =============================================
print("=" * 50)
print("PART 3B: SPLIT PDF INTO CHUNKS")
print("=" * 50)

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=40
)

chunks = splitter.split_documents(pages)

print(f"Pages    : {len(pages)}")
print(f"Chunks   : {len(chunks)}")
print(f"Avg size : {sum(len(c.page_content) for c in chunks) // len(chunks)} characters")
print()

print("Sample chunks:")
for i, chunk in enumerate(chunks[:3]):
    print(f"Chunk {i+1}:")
    print(f"  Content : {chunk.page_content.strip()}")
    print(f"  Source  : {chunk.metadata}")
    print()

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
# =============================================
# PART 4 — COMPLETE RAG CHAIN
# =============================================
print("=" * 50)
print("PART 4: COMPLETE RAG CHAIN")
print("=" * 50)
# ---- STEP 3: Create embeddings ----
print("Step 3: Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
print("✅ Embedding model ready")

# ---- STEP 4: Store in ChromaDB ----
print("Step 4: Storing chunks in ChromaDB...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./day4_vectordb"
)
print(f"✅ {len(chunks)} chunks stored in vector database")

# ---- STEP 5: Create retriever ----
print("Step 5: Creating retriever...")
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)
print("✅ Retriever ready")
print()

# ---- STEP 6: Build RAG prompt ----
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say exactly:
'I don't have that information in the document.'

Context:
{context}"""),
    ("human", "{question}")
])

# ---- STEP 7: Helper function to format docs ----
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# ---- STEP 8: Build complete RAG chain ----
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | rag_prompt
    | llm
    | parser
)

print("=" * 50)
print("RAG CHAIN READY — ASKING QUESTIONS")
print("=" * 50)
print()

# ---- STEP 9: Ask questions ----
questions = [
    "What is RAG?",
    "What is ChromaDB used for?",
    "What is Hyderabad known for in AI?",
    "What is the capital of France?",  # not in document
    "What is Streamlit?",
]

for question in questions:
    print(f"Q: {question}")
    answer = rag_chain.invoke(question)
    print(f"A: {answer}")
    print()

print("=" * 50)
print("✅ Day 4 Complete!")
print("You just built a production RAG pipeline!")
print("=" * 50)