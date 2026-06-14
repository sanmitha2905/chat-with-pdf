import os
from dotenv import load_dotenv
from groq import Groq
from transformers import pipeline

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def ask_groq(prompt):
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

print()

# =============================================
# TASK 1 — Sentiment Analysis (HuggingFace)
# =============================================
print("=" * 50)
print("TASK 1: SENTIMENT ANALYSIS — HuggingFace")
print("=" * 50)

classifier = pipeline("sentiment-analysis")

reviews = [
    "This AI course is amazing, I love it!",
    "This is the worst app I have ever used.",
    "It is okay, nothing special but works fine."
]

for review in reviews:
    result = classifier(review)
    label = result[0]['label']
    score = round(result[0]['score'] * 100, 1)
    print(f"Review : {review}")
    print(f"Result : {label} — {score}% confident")
    print()

# =============================================
# TASK 2 — Summarization (Groq)
# =============================================
print("=" * 50)
print("TASK 2: SUMMARIZATION — Groq")
print("=" * 50)

long_text = """
Artificial intelligence is rapidly transforming every industry 
across the globe. From healthcare where AI helps doctors diagnose 
diseases earlier, to finance where algorithms detect fraud in 
milliseconds, to education where personalized learning systems 
adapt to each student pace. Tasks which seemed impossible five 
years ago are now routine. However this also brings challenges 
around job displacement, privacy concerns, and need for regulation.
"""

summary = ask_groq(f"""
Summarize this text in exactly 2 sentences. Focus only on key facts.
Text: {long_text}
""")

print(f"Original : {len(long_text.split())} words")
print(f"Summary  : {summary}")
print()

# =============================================
# TASK 3 — Question Answering (Groq)
# =============================================
print("=" * 50)
print("TASK 3: QUESTION ANSWERING — Groq")
print("=" * 50)

context = """
Sanmi is a 20 year old student from Hyderabad who is learning 
Artificial Intelligence. She built a live AI chatbot using Python, 
Groq API and Streamlit. She deployed it on Streamlit Cloud and 
posted it on LinkedIn. Her goal is to get placed as an AI Engineer.
"""

questions = [
    "Where is Sanmi from?",
    "What did Sanmi build?",
    "What is Sanmi's goal?"
]

for question in questions:
    answer = ask_groq(f"""
Answer using ONLY the context below. One line answer only. No extra explanation.
Context: {context}
Question: {question}
""")
    print(f"Q: {question}")
    print(f"A: {answer}")
    print()

# =============================================
# TASK 4 — Zero Shot Classification (Groq)
# =============================================
print("=" * 50)
print("TASK 4: ZERO SHOT CLASSIFICATION — Groq")
print("=" * 50)

texts = [
    "The stock market crashed by 500 points today",
    "Scientists discovered a new treatment for cancer",
    "The new iPhone was released with better camera"
]

categories = ["finance", "healthcare", "technology", "sports"]

for text in texts:
    answer = ask_groq(f"""
Classify this text into exactly ONE of these categories: {categories}
Reply with ONLY the category name. Nothing else.
Text: {text}
""")
    print(f"Text     : {text}")
    print(f"Category : {answer}")
    print()

# =============================================
# TASK 5 — Translation (Groq)
# =============================================
print("=" * 50)
print("TASK 5: TRANSLATION — Groq")
print("=" * 50)

sentences = [
    "I am learning Artificial Intelligence",
    "My goal is to become an AI Engineer",
    "I built a chatbot using Python"
]

for sentence in sentences:
    translated = ask_groq(f"""
Translate this English sentence to Telugu.
Reply with ONLY the translation. Nothing else.
Sentence: {sentence}
""")
    print(f"English : {sentence}")
    print(f"Telugu  : {translated}")
    print()

print("=" * 50)
print("✅ Day 1 Complete!")
print("=" * 50)