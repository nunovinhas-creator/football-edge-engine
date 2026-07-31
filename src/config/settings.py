from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = (
    os.getenv("BSD_API_KEY")
    or os.getenv("BZZ_API_KEY")
    or os.getenv("BZZOIRO_API_KEY")
    or os.getenv("API_KEY")
)

BASE_URL = "https://sports.bzzoiro.com/api/v2"
