import streamlit as st
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import cohere
import os

st.set_page_config(
    page_title="The Loyola Chatbot",
    page_icon="📘",
    layout="wide"
)

page_bg_img = f"""
<style>
/* Background full page with dim overlay */
[data-testid="stAppViewContainer"]::before {{
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-image: url("https://raw.githubusercontent.com/shanujose03/chatbot/main/images/loy.jpeg");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    opacity: 0.3;  /* adjust darkness: 0.2 = darker, 0.5 = lighter */
    z-index: -1;
}}

/* Transparent header */
[data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
}}

/* Sidebar styling */
[data-testid="stSidebar"] {{
    background-color: rgba(255, 255, 255, 0.85);
}}

/* Add logo top-left */
[data-testid="stToolbar"]::before {{
    content: "";
    position: fixed;
    top: 10px;
    left: 10px;
    width: 120px;
    height: 120px;
    background-image: url("https://raw.githubusercontent.com/shanujose03/chatbot/main/images/log.png");
    background-size: contain;
    background-repeat: no-repeat;
    z-index: 1000;
}}

.chat-bubble {{
    background-color: rgba(255, 255, 255, 0.9);
    border-radius: 15px;
    padding: 15px;
    margin-top: 10px;
    font-size: 16px;
    color: #000000;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

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

st.title("Welcome to Loyola Chatbot")

query = st.text_input("Ask a question:")

if query:
    answer = generate_answer(query)
    st.subheader("🤖 Answer:")
    st.markdown(f"<div class='chat-bubble'>{answer}</div>", unsafe_allow_html=True)

