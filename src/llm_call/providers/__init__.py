from typing import Callable

from . import gemini

PROVIDERS: dict[str, Callable[..., object]] = {
    "gemini": gemini.call,
}
