
"""
store_vector.py
---------------
Reads live_data.json → builds FAISS index under ./faiss_index.
Run: python store_vector.py
"""

import json
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_FILE = Path("live_data.json")
if not DATA_FILE.exists():
    raise FileNotFoundError("live_data.json missing – run live_fetch.py first")

raw    = json.loads(DATA_FILE.read_text())
states = raw.get("states", [])
if not states:
    raise ValueError("No aircraft rows found in live_data.json")

def row_to_sentence(r):
    try:
        cs = (r[1] or "").strip() or "UNKNOWN"
        co = r[2] or "unknown country"
        lon, lat, alt = r[5], r[6], r[7]
        return (f"Callsign {cs} from {co} at "
                f"latitude {lat:.4f}, longitude {lon:.4f}; "
                f"altitude {alt:.0f} m.")
    except:
        return "corrupted record"

sentences = [row_to_sentence(r) for r in states]
print(f"[INFO] Built {len(sentences)} sentences")

print("[INFO] Loading embedding model…")
embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


index = FAISS.from_texts(sentences, embed)
index.save_local("faiss_index")
print("[INFO] Saved FAISS index → ./faiss_index")
