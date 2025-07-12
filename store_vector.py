"""
store_vector.py
---------------
Reads live_data.json, turns each flight row into a readable sentence,
embeds that text with Sentence-Transformers, and saves a FAISS index
under ./faiss_index.

Run:  python store_vector.py
"""

import json
from pathlib import Path

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# 1 ── Load the latest JSON ----------------------------------------------------
DATA_FILE = Path("live_data.json")
if not DATA_FILE.exists():
    raise FileNotFoundError(
        "live_data.json missing – run live_fetch.py for a minute first."
    )

raw = json.loads(DATA_FILE.read_text(encoding="utf-8"))
states = raw.get("states", [])

if not states:
    raise ValueError("No aircraft rows found – API may have hiccuped.")

# 2 ── Turn each row into plain-English text ----------------------------------
def row_to_sentence(row):
    try:
        callsign   = row[1].strip() or "UNKNOWN"
        country    = row[2] or "unknown country"
        lon, lat   = row[5], row[6]
        altitude   = row[7]
        sentence = (
            f"Callsign {callsign} from {country} at "
            f"latitude {lat:.4f} longitude {lon:.4f} "
            f"altitude {altitude:.0f} metres."
        )
        return sentence
    except Exception:
        return "corrupted record"

sentences = [row_to_sentence(r) for r in states]

print(f"[INFO] Built {len(sentences)} sentences from live_data.json")

# 3 ── Build embeddings --------------------------------------------------------
print("[INFO] Loading embedding model (this may take ~20 s first time)…")
embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4 ── Build the FAISS index ---------------------------------------------------
index = FAISS.from_texts(sentences, embed)
index.save_local("faiss_index")
print("[INFO] Saved FAISS index to ./faiss_index")

