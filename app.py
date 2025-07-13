%%writefile app.py
"""
app.py
------
Streamlit RAG UI powered by Google Gemini-Flash and FAISS.
Run: streamlit run app.py
"""

import os
import sys
import streamlit as st
from dotenv import load_dotenv

# ── 1. Load API Key ─────────────────────────────────────────────────────────
load_dotenv()  # loads /content/drive/MyDrive/colab_secrets/livewire.env
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
if not GOOGLE_API_KEY:
    st.error("Missing GOOGLE_API_KEY in environment")
    st.stop()
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# ── 2. Imports (new embeddings) ────────────────────────────────────────────
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings   # <-- new
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# ── 3. Load FAISS index ────────────────────────────────────────────────────
embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db    = FAISS.load_local(
    "faiss_index",
    embed,
    allow_dangerous_deserialization=True,
)

# ── 4. Build the RAG chain ─────────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash",
    temperature=0.2,
    max_output_tokens=1024,
)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={"k": 4}),
    memory=memory,
)

# ── 5. Streamlit UI ────────────────────────────────────────────────────────
st.set_page_config(page_title="✈️ Flight RAG Bot", page_icon="🛩️")
st.title("✈️ Live-Wire Flight RAG Bot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# display past messages
for role, msg in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(msg)

# chat input
if prompt := st.chat_input("Ask about any flight, e.g. 'Where is DLH2AX?'"):
    st.session_state.messages.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    response = qa_chain({"question": prompt})
    answer   = response["answer"]
    st.session_state.messages.append(("assistant", answer))

    with st.chat_message("assistant"):
        st.markdown(answer)
        if docs := response.get("source_documents", []):
            with st.expander("Sources"):
                for d in docs:
                    st.write("• " + d.page_content)
