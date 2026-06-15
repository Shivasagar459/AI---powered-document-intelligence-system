from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

print("Loading PDF...")

loader = PyPDFLoader("Cardiac_Arrest_Guide.pdf")
docs = loader.load()

print("Splitting text...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(docs)

print("Creating embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Building FAISS index...")

db = FAISS.from_documents(chunks, embeddings)

db.save_local("faiss_index")

print("✅ FAISS index saved successfully!")
