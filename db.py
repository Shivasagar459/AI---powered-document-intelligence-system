from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

texts = [
    "Cardiac arrest is a sudden loss of heart function",
    "Call emergency services immediately if someone collapses",
    "CPR should be started immediately during cardiac arrest"
]

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.from_texts(texts, embeddings)

db.save_local("faiss_index")

print("FAISS index created successfully")