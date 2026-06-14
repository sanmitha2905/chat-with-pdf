import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq
import chromadb

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

print("Setting up ChromaDB...")
print()

# =============================================
# PART 1 — Create a Vector Database
# =============================================
print("=" * 50)
print("PART 1: CREATE VECTOR DATABASE")
print("=" * 50)

# create chromadb client — stores data in a folder
chroma_client = chromadb.PersistentClient(path="./my_vectordb")

# create a collection — like a table in normal database
collection = chroma_client.get_or_create_collection(name="my_documents")

print("✅ Vector database created at ./my_vectordb")
print()

# =============================================
# PART 2 — Add documents to the database
# =============================================
print("=" * 50)
print("PART 2: ADD DOCUMENTS")
print("=" * 50)

# these are our "PDF chunks" — in real RAG these come from a PDF
documents = [
    "Python is a programming language used for AI and data science.",
    "LangChain is a framework for building applications with LLMs.",
    "Embeddings convert text into vectors that capture semantic meaning.",
    "Hyderabad is known for its biryani and historic monuments.",
    "RAG stands for Retrieval Augmented Generation.",
    "Machine learning is a subset of artificial intelligence.",
    "ChromaDB is a vector database used to store embeddings.",
    "Streamlit is used to build and deploy AI web applications.",
]

# add to chromadb
# ChromaDB handles embedding automatically!
collection.add(
    documents=documents,
    ids=[f"doc_{i}" for i in range(len(documents))]
)

print(f"✅ Added {len(documents)} documents to vector database")
print(f"Total documents in DB: {collection.count()}")
print()

# =============================================
# PART 3 — Search the database
# =============================================
print("=" * 50)
print("PART 3: SEARCH THE DATABASE")
print("=" * 50)

questions = [
    "What is RAG?",
    "How do I build web apps?",
    "What food is Hyderabad famous for?"
]

for question in questions:
    results = collection.query(
        query_texts=[question],
        n_results=2  # get top 2 matches
    )

    print(f"Question: {question}")
    print(f"Top matches:")
    for i, doc in enumerate(results['documents'][0]):
        distance = results['distances'][0][i]
        similarity = round((1 - distance) * 100, 1)
        print(f"  {i+1}. {doc}")
        print(f"     Similarity: {similarity}%")
    print()

# =============================================
# PART 4 — Full RAG with ChromaDB + Groq
# =============================================
print("=" * 50)
print("PART 4: FULL RAG — CHROMADB + GROQ")
print("=" * 50)

def rag_answer(question):
    # Step 1 — find relevant chunks from vector DB
    results = collection.query(
        query_texts=[question],
        n_results=3
    )
    
    # Step 2 — get the relevant chunks as context
    context_chunks = results['documents'][0]
    context = "\n".join(context_chunks)
    
    # Step 3 — send question + context to Groq
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{
            "role": "system",
            "content": "Answer questions using ONLY the context provided. If the answer is not in the context say 'I don't have that information'."
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }]
    )
    
    return response.choices[0].message.content.strip()

# test it
test_questions = [
    "What is RAG and what does it stand for?",
    "What can I use to build AI web apps?",
    "What is the capital of France?"  # not in our documents!
]

for question in test_questions:
    print(f"Q: {question}")
    print(f"A: {rag_answer(question)}")
    print()

print("=" * 50)
print("✅ Day 2 Complete!")
print("You just built a working RAG system!")
print("=" * 50)