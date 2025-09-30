import streamlit as st
import os
from openai import OpenAI
import chromadb
from chromadb.config import Settings
from tqdm import tqdm
from typing import List
import glob
from chromadb.utils import embedding_functions

# Title and instructions
st.title("💬 Chatbot with RAG")
st.write(
    "This chatbot uses OpenAI's GPT-3.5 model and retrieves context from two uploaded documents. "
    "To use it, enter your OpenAI API key below. Make sure your documents are in the `docs/` folder."
)


# API key from Streamlit secrets or input
if "API_KEY" in st.secrets:
    openai_api_key = st.secrets["API_KEY"]
else:
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.", icon="🗝️")
        st.stop()

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)


# Load and embed documents into ChromaDB
@st.cache_resource
def load_vectorstore(api_key):
    # Set up ChromaDB in-memory client
    client = chromadb.Client(Settings(anonymized_telemetry=False))
    # Use OpenAI embedding function
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-ada-002"
    )
    collection = client.create_collection(name="docs", embedding_function=openai_ef)

    # Load all .txt files from docs/
    filepaths = glob.glob("docs/*.txt")
    if not filepaths:
        st.warning("No text files found in docs/ folder.")
        return None

    # Read and chunk documents
    chunk_size = 500
    chunk_overlap = 50
    doc_texts = []
    metadatas = []
    ids = []
    idx = 0
    for filepath in filepaths:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
            # Chunk text
            for i in range(0, len(text), chunk_size - chunk_overlap):
                chunk = text[i:i+chunk_size]
                doc_texts.append(chunk)
                metadatas.append({"source": os.path.basename(filepath)})
                ids.append(f"{os.path.basename(filepath)}_{idx}")
                idx += 1
    # Add to Chroma collection
    collection.add(documents=doc_texts, metadatas=metadatas, ids=ids)
    return collection


if openai_api_key.strip():
    vectorstore = load_vectorstore(openai_api_key)
    if vectorstore is None:
        st.stop()
else:
    st.warning("OpenAI API key is missing or invalid.")
    st.stop()

# Session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything…"):
    # Store user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)


    # Retrieve context from ChromaDB
    results = vectorstore.query(query_texts=[prompt], n_results=3)
    context = "\n\n".join(results["documents"][0]) if results["documents"] else ""


    # Build message history with strict context use
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer ONLY using the information provided in the CONTEXT below. "
                "If the answer is not in the context, reply with 'I don't know based on the provided information.'\n"
                f"CONTEXT:\n{context}"
            )
        },
        *st.session_state.messages,
        {"role": "user", "content": prompt}
    ]

    # Generate response
    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        stream=True,
    )

    # Display and store assistant response
    with st.chat_message("assistant"):
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})