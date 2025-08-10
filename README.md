# ChatBot – AI Study Companion with RAG & Groq LLM

This * chatbot* built with *Retrieval-Augmented Generation (RAG)* to help you learn from unstructured documents like PDFs, textbooks, and articles.  
It uses *ChromaDB* for vector storage, *Sentence Transformers* for embeddings, and *Groq LLM* for fast, high-quality answers — all wrapped in a friendly *Streamlit* UI.  

---

##  Features
- *Upload & Index PDFs* – Extracts text and stores embeddings in a persistent Chroma database.  
- *Question Answering* – Retrieves relevant document sections and answers your questions with citations.  
- *Summaries* – Generate short or detailed summaries of chapters/sections.  
- *Quizzes & Flashcards* – Get multiple-choice and short-answer quizzes to reinforce learning.  


---

## Tech Stack
- *Frontend:* Streamlit  
- *Vector DB:* ChromaDB (PersistentClient API)  
- *Embeddings:* Sentence Transformers (all-MiniLM-L6-v2)  
- *LLM:* Groq API (deepseek-r1-distill-llama-70b)  
- *PDF Parsing:* pdfplumber  

---

##  Project Structure

.
├── RAG_LLM.py              # RAG logic: embedding, ChromaDB, Groq API
├── streamlit_frontend.py   # Streamlit UI
├── requirements.txt        # Dependencies
├── .env                    # Environment variables (GROQ_API_KEY)
└── chroma_db/              # Persistent Chroma database (auto-created)


---

##  Setup Instructions

###  Clone Repository
bash
git clone https://github.com/yourusername/studybot.git
cd studybot


### Create Virtual Environment
bash
conda create -n rag_groq_env python=3.10
conda activate rag_groq_env


###  Install Dependencies
bash
pip install -r requirements.txt


###  Set Groq API Key
Create a .env file in the project root:
env
GROQ_API_KEY=your_groq_api_key_here

You can get your key from: [https://console.groq.com/keys](https://console.groq.com/keys)

###  Run the App
bash
streamlit run streamlit_frontend.py


---

##  Usage
1. *Upload a PDF* – StudyBot will index and store it in ChromaDB.  
2. *Ask a question* – It will retrieve the most relevant sections and answer based on your documents.  
3. *Get summaries* – Ask for a summary of a section or chapter.  
4. *Test yourself* – Ask for quizzes or flashcards.  

---

##  Notes
- First run will create a chroma_db/ folder for persistent storage.  
- Embedding large PDFs may take some time.  
- The Groq API model name can be changed in RAG_LLM.py.  

---