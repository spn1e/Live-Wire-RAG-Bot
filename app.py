"""
app.py
------
Streamlit chat UI that:
1. Loads the FAISS index built in store_vector.py
2. Wraps it in a ConversationalRetrievalChain (LangChain)
3. Lets you chat and see answers + sources
Run:  streamlit run app.py
"""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# ── 1. Env + keys ──────────────────────────────────────────────────────────
load_dotenv()
os.environ["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"] = os.getenv("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", "")

# ── 2. Choose the right imports for your LangChain version ─────────────────
try:
    # Newer split-package layout (>=0.1.x)
    from langchain_openai import ChatOpenAI
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.memory import ConversationBufferMemory
    from langchain.chains import ConversationalRetrievalChain
except ModuleNotFoundError:
    # Older monolith layout (<=0.0.352)
    from langchain.chat_models import ChatOpenAI
    from langchain.vectorstores import FAISS
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.memory import ConversationBufferMemory
    from langchain.chains import ConversationalRetrievalChain

# ── 3. Load vector DB ──────────────────────────────────────────────────────
DB_PATH = Path("faiss_index")
if not DB_PATH.exists():
    st.error("faiss_index not found - run store_vector.py first.")
    st.stop()

embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(str(DB_PATH), embed)

# ── 4. Build the RAG chain ────────────────────────────────────────────────
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={"k": 4}),
    memory=memory,
)

# ── 5. Streamlit UI ────────────────────────────────────────────────────────
st.set_page_config(page_title="✈️ Live-Wire RAG Bot", page_icon="🛩️")
st.title("✈️ Live-Wire RAG Bot")

# Chat history stored in Streamlit session_state
if "messages" not in st.session_state:
    st.session_state.messages = []

# display past messages
for role, message in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(message)

# input box
if prompt := st.chat_input("Ask me about any flight, e.g. 'Where is DLH2AX?'"):
    # save user prompt
    st.session_state.messages.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # run RAG chain
    response = qa_chain({"question": prompt})

    answer = response["answer"]
    sources = response.get("source_documents", [])

    # save assistant response
    st.session_state.messages.append(("assistant", answer))

    # display assistant response
    with st.chat_message("assistant"):
        st.markdown(answer)

        # expandable citations
        if sources:
            with st.expander("sources"):
                for doc in sources:
                    st.write("• " + doc.page_content)

