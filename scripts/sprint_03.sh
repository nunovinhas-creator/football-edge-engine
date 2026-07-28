#!/bin/bash

echo "== Football Edge Engine - Sprint 3 =="

mkdir -p src/config
mkdir -p src/api
mkdir -p src/tools

cat > src/config/settings.py << 'EOPY'
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("BZZ_API_KEY")
BASE_URL = "https://sports.bzzoiro.com/football/api/v2"
EOPY

cat > src/api/client.py << 'EOPY'
import requests

from src.config.settings import API_KEY, BASE_URL


class BzzoiroClient:

    def __init__(self):
        self.headers = {
            "Authorization": f"Token {API_KEY}"
        }

    def get(self, endpoint: str):

        url = f"{BASE_URL}/{endpoint}"

        response = requests.get(
            url,
            headers=self.headers,
            timeout=30
        )

        response.raise_for_status()

        return response.json()
EOPY

cat > src/tools/test_connection.py << 'EOPY'
from src.api.client import BzzoiroClient


def main():

    client = BzzoiroClient()

    print("Testing Bzzoiro API...")

    data = client.get("events/?limit=1")

    print("Connection successful!")

    print(data)


if __name__ == "__main__":
    main()
EOPY

cat > .env.example << 'EOPY'
BZZ_API_KEY=YOUR_API_KEY
EOPY

echo
echo "Sprint 3 completed."
echo
echo "Now create your .env file:"
echo
echo "cp .env.example .env"
echo
echo "Edit .env and place your API key."
