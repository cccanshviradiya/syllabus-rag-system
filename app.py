import streamlit as st
import os
from PyPDF2 import PdfReader

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings

import pytesseract
from pdf2image import convert_from_path

from dotenv import load_dotenv

load_dotenv()  



# ==================== GEMINI CONFIG ====================
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# ==================== OCR CONFIG ====================
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


# ==================== PROMPT ====================
PROMPT = PromptTemplate(
    template="""
    Answer the question using ONLY the given context.
    Respond in the SAME language as the question.
    If the answer is not present, say:
    "Answer is not available in the context."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """,
    input_variables=["context", "question"]
)


# ==================== OCR ====================
def extract_text_with_ocr(pdf_file):
    images = convert_from_path(pdf_file)
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img)
    return text


# ==================== PDF TEXT ====================
def get_pdf_text(pdf_docs):
    text = ""

    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

    if len(text.strip()) < 50:
        st.info("🔍 Scanned PDF detected — using OCR")
        for pdf in pdf_docs:
            text += extract_text_with_ocr(pdf)

    return text


# ==================== CHUNKING ====================
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)


# ==================== EMBEDDINGS ====================
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )


# ==================== VECTOR STORE ====================
def get_vector_store(text_chunks):
    embeddings = load_embeddings()
    db = FAISS.from_texts(text_chunks, embedding=embeddings)
    db.save_local("faiss_index")


# ==================== GEMINI (PRIMARY) ====================
def ask_gemini(context, question):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3
    )
    response = llm.invoke(
        PROMPT.format(context=context, question=question)
    )
    return response.content


# ==================== PHI-3 (FALLBACK) ====================
def ask_phi3(context, question):
    llm = ChatOllama(
        model="phi3",
        temperature=0.3,
        timeout=120
    )
    response = llm.invoke(
        PROMPT.format(context=context, question=question)
    )
    return response.content


# ==================== HYBRID LOGIC ====================
def ask_llm_with_fallback(context, question):
    try:
        return ask_gemini(context, question)
    except Exception:
        st.warning(" Gemini failed. Falling back to local Phi-3.")
        return ask_phi3(context, question)


# ==================== USER QUERY ====================
def user_input(user_question):
    if not os.path.exists("faiss_index"):
        st.warning("Please upload and process PDFs first.")
        return

    embeddings = load_embeddings()

    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = db.similarity_search(user_question, k=2)

    if not docs:
        st.write("Answer is not available in the context.")
        return

    context = "\n\n".join(doc.page_content for doc in docs)

    with st.spinner("Thinking..."):
        answer = ask_llm_with_fallback(context, user_question)

    st.write("### Reply:")
    st.write(answer)


# ==================== STREAMLIT UI ====================
def main():
    st.set_page_config(page_title="Chat PDF")
    st.header(" Syllabus RAG System ")

    user_question = st.text_input("Ask a question from the PDF")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader(
            "Upload PDF files",
            accept_multiple_files=True
        )

        if st.button("Submit & Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF.")
                return

            with st.spinner("Processing PDFs..."):
                raw_text = get_pdf_text(pdf_docs)
                chunks = get_text_chunks(raw_text)
                get_vector_store(chunks)
                st.success(" PDFs processed successfully!")


if __name__ == "__main__":
    main()

