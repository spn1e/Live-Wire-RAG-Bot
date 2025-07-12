"""
app.py
------
Streamlit chat UI that
1. Loads the FAISS index built by store_vector.py
2. Wraps it in a ConversationalRetrievalChain
3. Lets you chat and see answers + sources
Run locally:   streamlit run app.py
Deploy note:   add keys in .streamlit/secrets.toml or Streamlit Cloud ▸ Settings ▸ Secrets
"""

from pathlib import Path
import os

import streamlit as st
from dotenv import load_dotenv

# ── 1. Load secrets / environment variables ────────────────────────────────
# • Local dev: .env (never commit!)
# • Streamlit Cloud: st.secrets
load_dotenv()
if "OPENAI_API_KEY" in st.secrets:            # Cloud
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:                                         # Local
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

# ── 2. Import LangChain objects (handle old vs new package names) ──────────
try:  # Newer split-package layout (≥0.1.x)
    from langchain_openai import ChatOpenAI
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.memory import ConversationBufferMemory
    from langchain.chains import ConversationalRetrievalChain
except ModuleNotFoundError:  # Older monolith layout (≤0.0.352)
    from langchain.chat_models import ChatOpenAI
    from langchain.vectorstores import FAISS
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.memory import ConversationBufferMemory
    from langchain.chains import ConversationalRetrievalChain

# ── 3. Load the FAISS index ────────────────────────────────────────────────
DB_PATH = Path("faiss_index")
if not DB_PATH.exists():
    st.error("❌  faiss_index not found – run store_vector.py first.")
    st.stop()

embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# If you built the index with default pickle serialization, allow it:
db = FAISS.load_local(
    str(DB_PATH),
    embed,
    allow_dangerous_deserialization=True,   # set False once you switch to safe serialization
)

# ── 4. Build the RAG chain ────────────────────────────────────────────────
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={"k": 4}),
    memory=memory,
)

# ── 5. Streamlit UI ────────────────────────────────────────────────────────
st.set_page_config(page_title="✈️ Live-Wire RAG Bot", page_icon="🛩️")
st.title("✈️ Live-Wire RAG Bot")

# Show past messages
for role, msg in st.session_state.get("messages", []):
    with st.chat_message(role):
        st.markdown(msg)

# Chat input
if prompt := st.chat_input("Ask me about any flight, e.g. 'Where is DLH2AX?'"):
    st.session_state.setdefault("messages", []).append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run the chain
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
