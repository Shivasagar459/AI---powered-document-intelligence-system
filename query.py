from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "vector_store",
    embeddings,
    allow_dangerous_deserialization=True
)

while True:
    question = input("\nAsk a question: ")

    if question.lower() == "exit":
        break

    docs = db.similarity_search(question, k=3)

    print("\nAnswer from PDF:\n")

    for i, doc in enumerate(docs, 1):
        print(f"Result {i}:")
        print(doc.page_content)
        print("-" * 50)