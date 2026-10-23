from typing import Callable

from . import gemini, openrouter

PROVIDERS: dict[str, Callable[..., object]] = {
    "gemini": gemini.call,
    "openrouter": openrouter.call,
}
