import streamlit as st
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.chains import RetrievalQA
from langchain_classic.retrievers.document_compressors import LLMChainFilter
from langchain_classic.retrievers import ContextualCompressionRetriever
from dotenv import load_dotenv
import os
import tempfile

load_dotenv()

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Chat with PDF", page_icon="📄")
st.title("📄 Chat with your PDF")
st.markdown("Upload a PDF and ask anything about it!")

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile"
    )
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return llm, embeddings

llm, embeddings = load_models()

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is None:
    st.info("Please upload a PDF to start.")
    st.stop()

# ---------------- SAVE TEMP FILE ----------------
with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
    tmp_file.write(uploaded_file.read())
    tmp_path = tmp_file.name

# ---------------- LOAD & SPLIT ----------------
with st.spinner("Reading your PDF..."):
    try:
        import pdfplumber
        
        text = ""
        with pdfplumber.open(tmp_path) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        
        if text.strip():
            pages = [Document(page_content=text)]
        else:
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
    except:
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(pages)

# ---------------- VECTOR STORE ----------------
with st.spinner("Building knowledge base..."):
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

# ---------------- RETRIEVERS ----------------
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    llm=llm
)

compressor = LLMChainFilter.from_llm(llm)

retriever = ContextualCompressionRetriever(
    base_retriever=multi_query_retriever,
    base_compressor=compressor
)

# ---------------- QA CHAIN ----------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

st.success(f"✅ PDF loaded! {len(chunks)} chunks created. Ask your questions!")

# ---------------- CHAT MEMORY ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- DISPLAY CHAT ----------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ---------------- USER INPUT ----------------
user_question = st.chat_input("Ask something about your PDF...")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})

    with st.chat_message("user"):
        st.write(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = qa_chain.invoke({"query": user_question})
            answer = result["result"]

            st.write(answer)

            with st.expander("📚 Source chunks used"):
                for i, doc in enumerate(result["source_documents"]):
                    st.write(f"**Chunk {i+1}:**")
                    st.write(doc.page_content[:300])
                    st.divider()

    st.session_state.messages.append({"role": "assistant", "content": answer})