import streamlit as st
from RAG_LLM import load_pdf_to_chroma, answer_question

st.set_page_config(page_title="📚 StudyBot RAG", layout="wide")

st.title("📚 StudyBot - Your AI Study Companion")
st.write("Upload PDFs, ask questions, get summaries, and quiz yourself!")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file is not None:
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.read())
    load_pdf_to_chroma(uploaded_file.name)
    st.success(f"✅ {uploaded_file.name} added to knowledge base!")

# Chat input
user_question = st.text_input("Ask StudyBot a question:")
if st.button("Ask"):
    if not user_question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            answer = answer_question(user_question)
        st.markdown(f"**StudyBot:** {answer}")
