%%writefile live_fetch.py
"""
live_fetch.py
-------------
Pulls live aircraft data from the OpenSky Network every 60 s and
writes it to live_data.json. Supports anonymous or OAuth2 client-credentials.
"""

import json, os, time, requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
CLIENT_ID     = os.getenv("OPENSKY_CLIENT_ID", "").strip()
CLIENT_SECRET = os.getenv("OPENSKY_CLIENT_SECRET", "").strip()
SLEEP_SECONDS = 60

TOKEN_URL = ("https://auth.opensky-network.org/auth/realms/"
             "opensky-network/protocol/openid-connect/token")
API_URL   = "https://opensky-network.org/api/states/all"
_token_cache = {"value": None, "exp": 0}

def _get_token():
    if not CLIENT_ID or not CLIENT_SECRET:
        return ""
    if time.time() < _token_cache["exp"] - 30:
        return _token_cache["value"]
    resp = requests.post(
        TOKEN_URL,
        data={
            "grant_type":"client_credentials",
            "client_id":CLIENT_ID,
            "client_secret":CLIENT_SECRET
        },
        timeout=10,
    )
    resp.raise_for_status()
    tk = resp.json()
    _token_cache["value"] = tk["access_token"]
    _token_cache["exp"]   = time.time() + int(tk["expires_in"])
    return _token_cache["value"]

def fetch_loop():
    while True:
        try:
            headers = {}
            tok = _get_token()
            if tok:
                headers["Authorization"] = f"Bearer {tok}"
            r = requests.get(API_URL, headers=headers, timeout=15)
            r.raise_for_status()
            data = r.json()
            data["fetched_at_utc"] = datetime.utcnow().isoformat()
            with open("live_data.json", "w") as f:
                json.dump(data, f, indent=2)
            print(f"[{datetime.utcnow().strftime('%H:%M:%S')}] "
                  f"Saved {len(data.get('states', []))} rows")
        except Exception as e:
            print("WARN:", e)
        time.sleep(SLEEP_SECONDS)

if __name__ == "__main__":
    fetch_loop()
