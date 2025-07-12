"""
live_fetch.py – OAuth-2 version
"""
import json, os, re, time, requests
from datetime import datetime, timezone
from dotenv import load_dotenv

# ── 0. Config from .env ─────────────────────────────────────────────
load_dotenv()

CLIENT_ID     = os.getenv("OPENSKY_CLIENT_ID", "").strip()
CLIENT_SECRET = os.getenv("OPENSKY_CLIENT_SECRET", "").strip()
RAW_INTERVAL  = os.getenv("FETCH_INTERVAL_SECS", "60")
INTERVAL      = int(re.search(r"\d+", RAW_INTERVAL).group()) if re.search(r"\d+", RAW_INTERVAL) else 60

TOKEN_URL   = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
API_URL     = "https://opensky-network.org/api/states/all"

token_cache = {"access_token": None, "expires_at": 0}

# ── 1. Helper: get or refresh token ─────────────────────────────────
def get_token() -> str:
    # If client ID/secret are missing, return "" so we work anonymously
    if not (CLIENT_ID and CLIENT_SECRET):
        return ""
    # Re-use token until 60 s before expiry
    if time.time() < token_cache["expires_at"] - 60:
        return token_cache["access_token"]

    resp = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        },
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    token_cache["access_token"] = data["access_token"]
    token_cache["expires_at"]  = time.time() + int(data["expires_in"])
    return token_cache["access_token"]

# ── 2. Main loop ────────────────────────────────────────────────────
print(f"[INFO] Fetch loop every {INTERVAL}s "
      f"({'OAuth-2' if CLIENT_ID else 'anonymous'})")

while True:
    try:
        headers = {}
        token = get_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"

        r = requests.get(API_URL, headers=headers, timeout=15)
        r.raise_for_status()
        payload = r.json()
        payload["fetched_at_utc"] = datetime.now(timezone.utc).isoformat()

        with open("live_data.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print(f"[{datetime.utcnow().strftime('%H:%M:%S')}] "
              f"Saved {len(payload.get('states', []))} aircraft ➜ live_data.json")

    except Exception as e:
        print(f"[WARN] fetch failed → {e!r}")

    time.sleep(INTERVAL)
