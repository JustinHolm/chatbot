import tiktoken
import streamlit as st
import os
from openai import OpenAI
import time

# Fix sqlite3 version issue for ChromaDB
import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import chromadb
from chromadb.config import Settings
from tqdm import tqdm
from typing import List
import glob
from chromadb.utils import embedding_functions

# Title and instructions
st.title("💬 Dragedal Family History Chatbot")
st.write(
    "This chatbot helps you research families, people, and events in Dragedal using historical documents. "
    "To use it, enter your OpenAI API key below."
)


# API key from Streamlit secrets or input
try:
    if "API_KEY" in st.secrets:
        openai_api_key = st.secrets["API_KEY"]
    else:
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.", icon="🗝️")
            st.stop()
except:
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.", icon="🗝️")
        st.stop()

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)


# Load and embed documents into ChromaDB
@st.cache_resource
def load_vectorstore(api_key):
    # Use tiktoken to count tokens for OpenAI embedding model
    encoding = tiktoken.encoding_for_model("text-embedding-ada-002")
    # Set up ChromaDB in-memory client
    client = chromadb.Client(Settings(anonymized_telemetry=False))
    # Use OpenAI embedding function
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-ada-002"
    )
    import uuid
    collection_name = f"docs_{uuid.uuid4().hex[:8]}"
    collection = client.create_collection(name=collection_name, embedding_function=openai_ef)

    # Load all .txt files from docs/
    filepaths = glob.glob("docs/*.txt")
    if not filepaths:
        st.warning("No text files found in docs/ folder.")
        return None

    # Read and chunk documents with optimized chunking for speed
    chunk_size = 800  # Balanced size for speed and token limits
    chunk_overlap = 150
    all_docs, all_metas, all_ids = [], [], []
    idx = 0
    
    with st.spinner("Loading historical documents..."):
        for filepath in filepaths:
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
                filename = os.path.basename(filepath)
                
                # Split into chunks
                for i in range(0, len(text), chunk_size - chunk_overlap):
                    chunk = text[i:i+chunk_size]
                    if len(chunk.strip()) > 50:  # Skip very short chunks
                        all_docs.append(chunk)
                        all_metas.append({"source": filename})
                        all_ids.append(f"{filename}_{idx}")
                        idx += 1
        
        # Add documents to ChromaDB with intelligent batching for speed
        if all_docs:
            total_chunks = len(all_docs)
            progress_bar = st.progress(0)
            processed = 0
            
            # Process in smart batches based on token count
            i = 0
            while i < total_chunks:
                batch_docs = []
                batch_metas = []
                batch_ids = []
                batch_tokens = 0
                max_batch_tokens = 150000  # Conservative but faster than one-by-one
                
                # Build batch until we hit token limit
                while i < total_chunks and len(batch_docs) < 25:  # Max 25 docs per batch
                    doc = all_docs[i]
                    doc_tokens = len(encoding.encode(doc))
                    
                    # If this doc would exceed limit, process current batch
                    if batch_tokens + doc_tokens > max_batch_tokens and len(batch_docs) > 0:
                        break
                    
                    # Skip extremely large documents
                    if doc_tokens > 10000:
                        st.warning(f"Skipping very large document chunk ({doc_tokens} tokens)")
                        i += 1
                        continue
                    
                    batch_docs.append(doc)
                    batch_metas.append(all_metas[i])
                    batch_ids.append(all_ids[i])
                    batch_tokens += doc_tokens
                    i += 1
                
                # Process the batch
                if batch_docs:
                    try:
                        collection.add(documents=batch_docs, metadatas=batch_metas, ids=batch_ids)
                        processed += len(batch_docs)
                        progress_bar.progress(min(processed / total_chunks, 1.0))
                        
                    except Exception as e:
                        # If batch fails, try smaller batches
                        st.warning(f"Large batch failed, trying smaller batches...")
                        for doc, meta, doc_id in zip(batch_docs, batch_metas, batch_ids):
                            try:
                                collection.add(documents=[doc], metadatas=[meta], ids=[doc_id])
                                processed += 1
                            except Exception as e2:
                                st.warning(f"Skipped problematic document: {str(e2)[:50]}...")
                                processed += 1
                        progress_bar.progress(min(processed / total_chunks, 1.0))
            
            progress_bar.empty()
            st.success(f"✅ Successfully loaded {processed} text segments from {len(filepaths)} historical documents.")
    
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
if prompt := st.chat_input("Ask about families, people, or events in Dragedal..."):
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
                "You are a helpful genealogical and historical research assistant. Answer questions about the families, people, and events in Dragedal using ONLY the information provided in the CONTEXT below. "
                "When discussing people, include details about their family relationships, dates, locations, and significant events when available. "
                "If asked about families, provide information about family members, their connections, and their history in Dragedal. "
                "If the answer is not in the context, reply with 'I don't have that information in the provided documents about Dragedal.'\n\n"
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