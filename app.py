"""
app.py
------
Streamlit chat UI that
1. Loads the FAISS index built by store_vector.py
2. Wraps it in a ConversationalRetrievalChain (LangChain) backed by Gemini-Pro
3. Lets you chat and see answers + sources

Run locally:   streamlit run app.py
Deploy:        put GOOGLE_API_KEY in .streamlit/secrets.toml
"""

from pathlib import Path
import os
import streamlit as st
from dotenv import load_dotenv

# ── 1. Load Google Gemini API key ──────────────────────────────────────────
load_dotenv()  # local .env

try:  # Streamlit Cloud → st.secrets
    google_key = st.secrets["GOOGLE_API_KEY"]
except (AttributeError, KeyError):
    google_key = os.getenv("GOOGLE_API_KEY")

if not google_key:
    st.error("🔑  GOOGLE_API_KEY missing – add to .env or Streamlit Secrets")
    st.stop()

os.environ["GOOGLE_API_KEY"] = google_key  # required by google-generativeai SDK

# ── 2. Import LangChain objects (Gemini + FAISS + utilities) ──────────────
try:  # new package names (≥0.1.x)
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.memory import ConversationBufferMemory
    from langchain.chains import ConversationalRetrievalChain
except ModuleNotFoundError:  # fallback for older monolith install
    from langchain.chat_models import ChatGoogleGenerativeAI
    from langchain.vectorstores import FAISS
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.memory import ConversationBufferMemory
    from langchain.chains import ConversationalRetrievalChain

# ── 3. Load the FAISS vector index ────────────────────────────────────────
DB_PATH = Path("faiss_index")
if not DB_PATH.exists():
    st.error("❌  faiss_index not found – run store_vector.py first.")
    st.stop()

embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(
    str(DB_PATH),
    embed,
    allow_dangerous_deserialization=True,  # remove after using safe serialization
)

# ── 4. Build the Gemini-powered RAG chain ─────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash",   # <= the *exact* string from list_models
    temperature=0.2,
    max_output_tokens=1024,    
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={"k": 4}),
    memory=memory,
)

# ── 5. Streamlit UI ───────────────────────────────────────────────────────
st.set_page_config(page_title="✈️ Live-Wire RAG Bot (Gemini)", page_icon="🛩️")
st.title("✈️ Live-Wire RAG Bot — Gemini-powered")

# Display past messages
for role, msg in st.session_state.get("messages", []):
    with st.chat_message(role):
        st.markdown(msg)

# Chat input
if prompt := st.chat_input("Ask me about any flight, e.g. 'Where is DLH2AX?'"):
    st.session_state.setdefault("messages", []).append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run the RAG chain
    response = qa_chain({"question": prompt})
    answer   = response["answer"]
    sources  = response.get("source_documents", [])

    st.session_state["messages"].append(("assistant", answer))
    with st.chat_message("assistant"):
        st.markdown(answer)
        if sources:
            with st.expander("sources"):
                for doc in sources:
                    st.write("• " + doc.page_content)
