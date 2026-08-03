from dotenv import load_dotenv
import os

load_dotenv()

SUPPORTED_API_KEY_ENV_VARS = (
    "BSD_API_KEY",
    "BZZ_API_KEY",
    "BZZOIRO_API_KEY",
    "API_KEY",
)

API_KEY = (
    os.getenv("BSD_API_KEY")
    or os.getenv("BZZ_API_KEY")
    or os.getenv("BZZOIRO_API_KEY")
    or os.getenv("API_KEY")
)

BSD_ROOT_URL = "https://sports.bzzoiro.com"

BASE_URL = f"{BSD_ROOT_URL}/api/v2"


class MissingAPIKeyError(RuntimeError):
    """Raised when no BSD API key is configured via environment or .env."""


def require_api_key():
    """Return the configured BSD API key, or raise a friendly error if missing."""

    if not API_KEY:
        raise MissingAPIKeyError(
            "BSD API key not configured. Set one of the following "
            "environment variables: "
            f"{', '.join(SUPPORTED_API_KEY_ENV_VARS)}. "
            "You can also place it in a .env file at the project root "
            "(e.g. BSD_API_KEY=your_key_here)."
        )

    return API_KEY
