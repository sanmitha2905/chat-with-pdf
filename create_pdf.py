from fpdf import FPDF

pdf = FPDF()
pdf.add_page()
pdf.set_font("Helvetica", size=12)

lines = [
    "Artificial Intelligence Guide",
    "",
    "What is AI?",
    "AI is the simulation of human intelligence by computers.",
    "",
    "What is Machine Learning?", 
    "Machine learning is a subset of AI that learns from data.",
    "",
    "What is RAG?",
    "RAG stands for Retrieval Augmented Generation.",
    "It helps AI answer questions from specific documents.",
    "",
    "What is LangChain?",
    "LangChain is a framework for building LLM applications.",
    "It provides ready-made components like loaders and chains.",
    "",
    "What is ChromaDB?",
    "ChromaDB is a vector database for storing embeddings.",
    "It enables fast similarity search across thousands of chunks.",
    "",
    "What is Streamlit?",
    "Streamlit is used to build and deploy AI web apps quickly.",
    "",
    "About Hyderabad AI Scene",
    "Hyderabad is becoming a major hub for AI companies in India.",
    "Many startups and large companies are hiring AI engineers.",
]

for line in lines:
    pdf.cell(200, 10, text=line, ln=True)

pdf.output("document.pdf")
print("PDF created successfully!")