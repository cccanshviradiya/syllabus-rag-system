import streamlit as st
import os
from PyPDF2 import PdfReader
from dotenv import load_dotenv

import google.generativeai as genai

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    
)
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings



# -------------------- CONFIG --------------------
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# -------------------- PDF TEXT --------------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text()
    return text


# -------------------- TEXT CHUNKS --------------------
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    return splitter.split_text(text)


# -------------------- VECTOR STORE --------------------
def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )


    db = FAISS.from_texts(text_chunks, embedding=embeddings)
    db.save_local("faiss_index")


# -------------------- GEMINI ANSWER --------------------
def ask_gemini(context, question):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3
    )

    prompt = PromptTemplate(
        template="""
        Answer the question using the given context.
        Respond in the SAME language as the question.
        If answer not found, say so clearly
        reply"Answer is not available in the context."

        Context:
        {context}

        Question:
        {question}

        Answer:
        """,
        input_variables=["context", "question"]
    )


    response = llm.invoke(
        prompt.format(context=context, question=question)
    )

    return response.content


# -------------------- USER QUERY --------------------
def user_input(user_question):
    embeddings = HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )



    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = db.similarity_search(user_question)
    context = "\n\n".join([doc.page_content for doc in docs])

    answer = ask_gemini(context, user_question)

    st.write("### Reply:")
    st.write(answer)


# -------------------- STREAMLIT UI --------------------
def main():
    st.set_page_config(page_title="Chat PDF")
    st.header("Syllabus RAG System  ")

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
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                chunks = get_text_chunks(raw_text)
                get_vector_store(chunks)
                st.success("PDFs processed successfully!")


if __name__ == "__main__":
    main()
