from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma(
    persist_directory="./day4_vectordb",
    embedding_function=embedding_model
)

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)

basic_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=basic_retriever,
    llm=llm
)

question = "What is this document about?"

print("=== BASIC RETRIEVER ===")
basic_results = basic_retriever.invoke(question)
print(f"Number of chunks: {len(basic_results)}")
for i, doc in enumerate(basic_results):
    print(f"\nChunk {i+1}:")
    print(doc.page_content[:200])

print("\n=== MULTI QUERY RETRIEVER ===")
multi_results = multi_query_retriever.invoke(question)
print(f"Number of chunks: {len(multi_results)}")
for i, doc in enumerate(multi_results):
    print(f"\nChunk {i+1}:")
    print(doc.page_content[:200])

compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=basic_retriever
)

print("\n=== CONTEXTUAL COMPRESSION RETRIEVER ===")
compression_results = compression_retriever.invoke("What is RAG?")
print(f"Number of chunks: {len(compression_results)}")
for i, doc in enumerate(compression_results):
    print(f"\nChunk {i+1}:")
    print(doc.page_content)