# 📄 Chat with PDF — RAG App

An AI-powered app that lets you upload any PDF and ask questions about it using Retrieval Augmented Generation (RAG).

## 🛠️ Tech Stack
- **LLM**: LLaMA 3.3 70B via Groq API
- **Embeddings**: all-MiniLM-L6-v2 (HuggingFace)
- **Vector Store**: FAISS (in-memory)
- **Retrievers**: MultiQuery + Contextual Compression
- **Framework**: LangChain + Streamlit

## ⚙️ How it Works
1. Upload any PDF
2. PDF is split into chunks
3. Chunks are embedded and stored in FAISS
4. You ask a question
5. MultiQuery Retriever finds relevant chunks
6. Contextual Compression filters noise
7. LLaMA 3.3 generates answer from your PDF

## 🔧 Run Locally
```bash
git clone https://github.com/sanmitha2905/chat-with-pdf
cd chat-with-pdf
pip install -r requirements.txt
# Add GROQ_API_KEY in .env file
streamlit run app.py
```

## 📁 Project Structure
rag/

├── app.py              # Main Streamlit app

├── requirements.txt    # Dependencies

├── .env               # API keys (not pushed)

├── .gitignore         # Ignores .env

└── README.md          # This file

## 👩‍💻 Built By
Sanmitha Pusuluri — AI Engineering Student, Hyderabad
