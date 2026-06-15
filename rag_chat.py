from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

llm = ChatOllama(model="llama3")

while True:
    query = input("\nAsk Question (type 'exit' to stop): ")

    if query.lower() == "exit":
        break

    docs = db.similarity_search(query, k=3)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are a helpful medical assistant.

Use ONLY the context below:

{context}

Question: {query}

Answer clearly:
"""

    response = llm.invoke(prompt)

    print("\nAnswer:\n")
    print(response.content)