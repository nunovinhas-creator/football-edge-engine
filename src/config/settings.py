from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = (
    os.getenv("BSD_API_KEY")
    or os.getenv("BZZ_API_KEY")
    or os.getenv("BZZOIRO_API_KEY")
    or os.getenv("API_KEY")
)

BSD_ROOT_URL = "https://sports.bzzoiro.com"

BASE_URL = f"{BSD_ROOT_URL}/api/v2"
