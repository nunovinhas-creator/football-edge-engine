#!/bin/bash

echo "=== Sprint 4 - API Explorer ==="

mkdir -p src/api
mkdir -p src/tools
mkdir -p data/raw

cat > src/api/client.py << 'EOPY'
import requests

from src.config.settings import API_KEY, BASE_URL


class BzzoiroClient:

    def __init__(self):
        self.headers = {
            "Authorization": f"Token {API_KEY}"
        }

    def get(self, endpoint):

        url = f"{BASE_URL}/{endpoint}"

        r = requests.get(
            url,
            headers=self.headers,
            timeout=30
        )

        r.raise_for_status()

        return r.json()
EOPY


cat > src/tools/explorer.py << 'EOPY'
import json
import sys
from pathlib import Path

from src.api.client import BzzoiroClient


client = BzzoiroClient()

endpoint = sys.argv[1]

print(f"Downloading {endpoint}...")

data = client.get(endpoint)

Path("data/raw").mkdir(parents=True, exist_ok=True)

outfile = f"data/raw/{endpoint.replace('/','_')}.json"

with open(outfile, "w", encoding="utf8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f"Saved -> {outfile}")

if isinstance(data, dict):
    print()
    print("FIELDS:")
    for k in data.keys():
        print("-", k)
EOPY

echo
echo "Sprint 4 installed."
