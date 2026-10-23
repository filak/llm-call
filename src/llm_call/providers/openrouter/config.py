import os

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL_DEF = "google/gemma-4-31b-it"
OPENROUTER_MODELS = ("google/gemma-4-31b-it", "google/gemma-4-26b-a4b-it")


def get_api_key() -> str:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "Missing OPENROUTER_API_KEY. "
            "Create ENV variable with OPENROUTER_API_KEY=..."
        )
    return key


def get_model(default: str = OPENROUTER_MODEL_DEF) -> str:
    return os.getenv("OPENROUTER_MODEL", default)
