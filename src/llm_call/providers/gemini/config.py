import os

GEMINI_MODEL_DEF = "gemini-3.8-flash"


def get_api_key() -> str:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "Missing GEMINI_API_KEY. Create ENV variable with GEMINI_API_KEY=..."
        )
    return key


def get_model(default: str = GEMINI_MODEL_DEF) -> str:
    return os.getenv("GEMINI_MODEL", default)
