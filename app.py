import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

# Load environment variables (for API Key)
load_dotenv()

# --- CONFIGURATION ---
# Get your FREE API key at: https://console.groq.com/
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Swiggy Report RAG", layout="wide")
st.title("🍔 Swiggy Annual Report AI Assistant")

# --- 1. DOCUMENT PROCESSING (Preprocessing) ---
@st.cache_resource
def prepare_vector_db(file_path):
    # Step A: Load the PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    # Step B: Chunking (Keeping context meaningful)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150
    )
    docs = text_splitter.split_documents(documents)
    
    # Step C: Generate Embeddings (Free & Local)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Step D: Store in FAISS Vector Store
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

# --- 2. THE RAG ENGINE ---
def get_answer(vectorstore, question):
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY, 
        model_name="llama3-8b-8192",
        temperature=0  # Zero temperature prevents "creativity" / hallucinations
    )
    
    # Strict instructions for the AI
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer or it's not in the context, just say that you "
        "don't know based on the report. Do not make up information. "
        "Context: {context}"
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    
    return qa_chain({"query": question})

# --- 3. THE UI INTERFACE ---
uploaded_file = st.sidebar.file_uploader("Step 1: Upload Swiggy PDF", type="pdf")

if uploaded_file:
    # Save temp file
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    with st.spinner("Analyzing the report... please wait."):
        vector_db = prepare_vector_db("temp.pdf")
        st.success("Report Processed Successfully!")

    user_query = st.text_input("Step 2: Ask anything about Swiggy's performance:")

    if user_query:
        with st.spinner("Finding answer..."):
            response = get_answer(vector_db, user_query)
            
            st.subheader("💡 AI Answer")
            st.write(response["result"])
            
            # Show "Supporting Context" as required by assignment
            with st.expander("🔍 View Supporting Context (Source Chunks)"):
                for i, doc in enumerate(response["source_documents"]):
                    st.markdown(f"**Chunk {i+1} (Page {doc.metadata['page']}):**")
                    st.info(doc.page_content)
else:
    st.info("Please upload the Swiggy Annual Report PDF in the sidebar to begin.")