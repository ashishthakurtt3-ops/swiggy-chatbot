import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Swiggy Report RAG", layout="wide")
st.title("🍔 Swiggy Annual Report AI Assistant")


@st.cache_resource
def prepare_vector_db(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore


def get_answer(vectorstore, question):
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama3-8b-8192",
        temperature=0,
        max_tokens=512
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    retrieved_docs = retriever.invoke(question)

    # keep context short to avoid token limit errors
    context = "\n\n".join([doc.page_content[:500] for doc in retrieved_docs])

    prompt = PromptTemplate.from_template(
        """You are an assistant for the Swiggy Annual Report FY 2023-24.
Use only the context below to answer. If the answer is not in the context, say you don't know based on the report.

Context:
{context}

Question: {question}

Answer:"""
    )

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    return answer, retrieved_docs


uploaded_file = st.sidebar.file_uploader("Step 1: Upload Swiggy Annual Report PDF", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Analyzing the report... please wait."):
        vector_db = prepare_vector_db("temp.pdf")
        st.success("Report Processed Successfully!")

    user_query = st.text_input("Step 2: Ask anything about Swiggy's performance:")

    if user_query:
        with st.spinner("Finding answer..."):
            try:
                answer, source_docs = get_answer(vector_db, user_query)

                st.subheader("💡 AI Answer")
                st.write(answer)

                with st.expander("🔍 View Supporting Context (Source Chunks)"):
                    for i, doc in enumerate(source_docs):
                        st.markdown(f"**Chunk {i+1} (Page {doc.metadata.get('page', 'N/A')}):**")
                        st.info(doc.page_content)
            except Exception as e:
                st.error(f"Error: {str(e)}")
else:
    st.info("Please upload the Swiggy Annual Report PDF in the sidebar to begin.")