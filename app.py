import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

# -----------------------------
# STREAMLIT CONFIG
# -----------------------------
st.set_page_config(page_title="Local RAG App", layout="wide")

# -----------------------------
# LLM (OLLAMA - LOCAL)
# -----------------------------
llm = ChatOllama(model="llama3")

# -----------------------------
# CACHED EMBEDDINGS
# -----------------------------
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = get_embeddings()

# -----------------------------
# SESSION STATE
# -----------------------------
if "db" not in st.session_state:
    st.session_state.db = None

# -----------------------------
# UI
# -----------------------------
st.title("📄 Local RAG App (Ollama + PDF + FAISS)")
st.write("Upload a PDF and ask questions from it.")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

# -----------------------------
# PROCESS PDF
# -----------------------------
if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Reading PDF..."):
        loader = PyPDFLoader("temp.pdf")
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        docs = splitter.split_documents(pages)

        st.session_state.db = FAISS.from_documents(docs, embeddings)

    st.success("✅ PDF processed successfully!")

# -----------------------------
# CHAT SECTION
# -----------------------------
query = st.text_input("Ask a question from your PDF:")

if query:
    if st.session_state.db is None:
        st.warning("⚠️ Please upload and process a PDF first.")
    else:
        with st.spinner("Searching document..."):
            docs = st.session_state.db.similarity_search(query, k=3)

            context = "\n\n".join([d.page_content for d in docs])

            prompt = f"""
You are a precise and helpful assistant.

RULES:
- Use ONLY the context below
- If answer is not in context, say: "Not found in document"

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

            response = llm.invoke(prompt)

        st.subheader("📌 Answer")
        st.write(response.content)