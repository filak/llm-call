from __future__ import annotations

from typing import Any, Literal, Optional, TypedDict


class TokenUsage(TypedDict):
    total_tokens: Optional[int]
    request_tokens: Optional[int]
    response_tokens: Optional[int]


class CallResponse(TypedDict):
    response: str
    usage: TokenUsage


def is_valid_temperature(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and 0.0 <= value <= 2.0
    )


def validate_options(
    response_format: str, timeout: Optional[float], temperature: float
) -> None:
    if response_format not in ("text", "json"):
        raise ValueError("response_format must be 'text' or 'json'")
    if timeout is not None and timeout <= 0:
        raise ValueError("timeout must be a positive number of seconds")
    if not is_valid_temperature(temperature):
        raise ValueError("temperature must be a number from 0.0 to 2.0")


def sum_counts(*counts: Optional[int]) -> Optional[int]:
    """Return the sum of the known counts, or None when no count is known."""
    known = [count for count in counts if count is not None]
    return sum(known) if known else None


def build_response(
    text: str,
    usage: TokenUsage,
    response_format: Literal["text", "json"],
) -> str | CallResponse:
    """Return text with usage first, or a JSON-serializable response envelope."""
    if response_format == "json":
        return CallResponse(response=text, usage=usage)

    counts = ", ".join(
        f"{name}={count if count is not None else 'unavailable'}"
        for name, count in usage.items()
    )
    return f"Tokens: {counts}\n{text}"
