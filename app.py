import streamlit as st
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import cohere
import os

co = cohere.Client("05QG3GBZxn4pLhxOLcwF7CREoXCewmRhKhq4ySym")


with open("vector_db.pkl", "rb") as f:
    index, documents = pickle.load(f)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def search_prospectus(query, top_k=3):
    query_vector = embedder.encode([query])
    distances, indices = index.search(query_vector.reshape(1, -1), top_k)
    results = [documents[i] for i in indices[0]]
    return results

def generate_answer(query):
    context = "\n\n".join(search_prospectus(query))
    prompt = f"""
    You are a helpful assistant for college-related queries.
    Use ONLY the context below to answer.

    Context:
    {context}

    Question: {query}
    Answer:
    """

    response = co.chat(
        model="command-xlarge-nightly",
        message=prompt,
        max_tokens=300,
        temperature=0.3
    )
    return response.text.strip()

st.title("📘 Loyola Prospectus Chatbot")

query = st.text_input("Ask a question about the prospectus:")

if query:
    answer = generate_answer(query)
    st.subheader("🤖 Answer:")
    st.write(answer)
