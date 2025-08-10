import os
import pdfplumber
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions
import requests
from dotenv import load_dotenv

load_dotenv()

# Load Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Chroma with the new PersistentClient
client = chromadb.PersistentClient(path="./chroma_db")

# Create or get collection
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
collection = client.get_or_create_collection(
    name="study_docs",
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
)

def load_pdf_to_chroma(pdf_path):
    """Extract text from PDF and store in ChromaDB"""
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                embedding = embedding_model.encode([text])[0].tolist()
                doc_id = f"{os.path.basename(pdf_path)}_page_{page_num}"
                collection.add(
                    documents=[text],
                    embeddings=[embedding],
                    ids=[doc_id],
                    metadatas=[{"page": page_num, "source": os.path.basename(pdf_path)}]
                )

def retrieve_context(query, top_k=3):
    """Retrieve top-k relevant chunks from Chroma"""
    query_embedding = embedding_model.encode([query])[0].tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    docs = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        docs.append(f"(Page {meta['page']} - {meta['source']}): {doc}")
    return "\n\n".join(docs)

def query_groq(prompt):
    """Send prompt to Groq LLM with error handling"""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-r1-distill-llama-70b",  # Change if needed
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }

    response = requests.post("https://api.groq.com/openai/v1/chat/completions",
                             json=payload, headers=headers)

    try:
        data = response.json()
    except Exception:
        raise ValueError(f"❌ Failed to parse Groq API response: {response.text}")

    if response.status_code != 200:
        raise ValueError(f"❌ Groq API error {response.status_code}: {data}")

    if "choices" not in data:
        raise ValueError(f"❌ Unexpected Groq API response format: {data}")

    return data["choices"][0]["message"]["content"]

def answer_question(user_question):
    """Retrieve context and send to LLM with StudyBot persona"""
    context = retrieve_context(user_question)

    studybot_prompt = f"""
You are StudyBot, an educational chatbot using Retrieval-Augmented Generation (RAG) to help users learn from unstructured documents (PDFs, textbooks, articles).

Retrieved Context:
{context}

User Question:
{user_question}

Guidelines:
- Retrieve information and cite sources (page numbers, file names).
- Summarize or explain concepts simply.
- Offer quizzes, flashcards, or mnemonics if relevant.
- Keep responses motivational and engaging.
- Ask the user what they’d like to do next (e.g., "Quiz or summary next?").

Now respond to the user.
    """

    return query_groq(studybot_prompt)
