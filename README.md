# ✈️ Live‑Wire Flight RAG Bot

<p align="center">
  <a href="https://live-wire-rag-bot-sfcqesnb6ssjtzbx62e7w5.streamlit.app/" target="_blank"><img src="https://img.shields.io/badge/Live%20Demo-Streamlit-success?logo=streamlit" alt="Live demo badge"></a>
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="MIT License">
</p>

A minimal **Retrieval‑Augmented Generation (RAG)** chatbot that answers questions about *live* aircraft positions worldwide.

- **Data source:** real‑time state vectors from the OpenSky Network  
- **Vector store:** FAISS built from natural‑language sentences (one per aircraft)  
- **LLM:** Google Gemini 1.5 Flash via LangChain  
- **Interface:** Streamlit chat UI with clickable source documents

Ask things like:

> *“Where is BAW123 right now?”*  
> *“Show me the altitude of the aircraft over Paris with callsign starting ‘AF’.”*

---

## ✨ Features

| Category | Details |
|----------|---------|
| **Live data ingest** | `live_fetch.py` pulls `/api/states/all` every **60 s** and stores JSON locally |
| **Vector index** | `store_vector.py` converts each aircraft row into a sentence → embeds with **all‑MiniLM‑L6‑v2** → saves a **FAISS** index |
| **Conversational RAG** | LangChain `ConversationalRetrievalChain` (top‑4 docs) + **Gemini Flash** (1024 tokens) |
| **Source transparency** | Each answer can expand a “Sources” panel showing the raw sentences retrieved |
| **Simple deployment** | No databases or servers—everything runs from the local file system or Streamlit Community Cloud |

> **Note:** The Streamlit app reads the *existing* `faiss_index/` at start‑up.  
> Run `live_fetch.py` and `store_vector.py` again whenever you need fresher data.

---

## 🚀 Quick Start

```bash
# 1 · Clone
git clone https://github.com/your‑username/live‑wire‑rag‑bot.git
cd live‑wire‑rag‑bot

# 2 · Python env
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3 · Environment variables
cp .env.example .env               # or create .env manually
# Inside .env, set:
# GOOGLE_API_KEY=<your‑gemini‑key>
# OPENSKY_CLIENT_ID=<optional>
# OPENSKY_CLIENT_SECRET=<optional>

# 4 · Fetch data & build index
python live_fetch.py               # downloads live_data.json (⏳ 60 s loop)
python store_vector.py             # builds FAISS under ./faiss_index

# 5 · Run the chatbot
streamlit run app.py
```

Open <http://localhost:8501> and start asking flight questions!

---

## 🏗️ Project Layout

```
live‑wire‑rag‑bot/
├── app.py            # Streamlit UI & RAG chain
├── live_fetch.py     # 60‑second polling of OpenSky → live_data.json
├── store_vector.py   # Builds FAISS index from JSON
├── requirements.txt  # Python deps (Streamlit, LangChain, Gemini, FAISS…)
├── apt.txt           # System deps for Streamlit Cloud (none required yet)
├── .env.example      # Template for API keys
└── README.md
```

---

## 🌐 Deploying on Streamlit Community Cloud

1. **Pre‑build the index locally** (`live_fetch.py` & `store_vector.py`) and commit *both* `live_data.json` and the `faiss_index/` folder to Git.
2. Push to GitHub → New app on <https://share.streamlit.io/>.
3. Set **Main file** to `app.py` and add `GOOGLE_API_KEY` as a secret.
4. Click **Deploy**. Updates require re‑running step 1 and pushing fresh files.

---

## 🛠️ Tech Stack

- **Python 3.10+**, **Streamlit 1.34+**
- **LangChain** (`google-genai`, `huggingface`, `community`)
- **Google Gemini 1.5 Flash** (chat completion)
- **FAISS‑CPU** for vector similarity search
- **sentence‑transformers/all‑MiniLM‑L6‑v2** embeddings
- **OpenSky Network REST API** for live ADS‑B state vectors

---

## 📝 Environment Variables

| Name | Required | Purpose |
|------|----------|---------|
| `GOOGLE_API_KEY` | ✅ | Access Google Gemini |
| `OPENSKY_CLIENT_ID` & `OPENSKY_CLIENT_SECRET` | optional | Higher request quota; otherwise the app uses anonymous access |

---

## 🤝 Contributing

Pull requests are welcome—especially for:

- Automated nightly refresh of the FAISS index
- Mapping visualisations (e.g., altitudes on a leaflet map)
- Additional retrieval pipelines or model options

Please follow conventional commits and include clear, focused changes.

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for full text.

---

## 🙏 Acknowledgements

- **OpenSky Network** for providing open aircraft tracking data  
- **HuggingFace & Sentence‑Transformers** for the embedding model  
- **LangChain** & **Streamlit** communities for fantastic tooling

---

*Happy plane spotting! 🛩️*
