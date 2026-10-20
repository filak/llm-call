from __future__ import annotations

from typing import Literal, Optional, Sequence, TypedDict

import httpx
from google import genai
from google.genai import types
from google.genai.errors import APIError

from .config import get_api_key, get_model
from ...files import load_parts

_client: Optional[genai.Client] = None


class TokenUsage(TypedDict):
    prompt_token_count: Optional[int]
    candidates_token_count: Optional[int]
    total_token_count: Optional[int]
    thoughts_token_count: Optional[int]
    cached_content_token_count: Optional[int]
    tool_use_prompt_token_count: Optional[int]


class CallResponse(TypedDict):
    response: str
    usage: TokenUsage


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=get_api_key())
    return _client


def call(
    prompt: str,
    files: Optional[Sequence[str]] = None,
    model: Optional[str] = None,
    system: Optional[str] = None,
    include_file_name: bool = False,
    response_format: Literal["text", "json"] = "text",
    timeout: Optional[float] = None,
) -> str | CallResponse:
    """Return text with usage first, or a JSON-serializable response envelope.

    timeout is the request timeout in seconds. None means no client timeout.
    """
    if response_format not in ("text", "json"):
        raise ValueError("response_format must be 'text' or 'json'")
    if timeout is not None and timeout <= 0:
        raise ValueError("timeout must be a positive number of seconds")
    contents = [
        prompt,
        *load_parts(list(files or []), include_file_name=include_file_name),
    ]
    config = None
    if system or timeout is not None:
        config = types.GenerateContentConfig(
            system_instruction=system or None,
            http_options=(
                types.HttpOptions(timeout=int(timeout * 1000))
                if timeout is not None
                else None
            ),
        )

    try:
        response = get_client().models.generate_content(
            model=model or get_model(),
            contents=contents,
            config=config,
        )
    except APIError as exc:
        raise RuntimeError(f"Gemini API Error: {exc}") from exc
    except httpx.TimeoutException as exc:
        raise RuntimeError(f"Gemini request timed out after {timeout} s") from exc
    except httpx.TransportError as exc:
        raise RuntimeError(f"Gemini connection error: {exc}") from exc

    metadata = response.usage_metadata
    usage = TokenUsage(
        prompt_token_count=getattr(metadata, "prompt_token_count", None),
        candidates_token_count=getattr(metadata, "candidates_token_count", None),
        total_token_count=getattr(metadata, "total_token_count", None),
        thoughts_token_count=getattr(metadata, "thoughts_token_count", None),
        cached_content_token_count=getattr(
            metadata, "cached_content_token_count", None
        ),
        tool_use_prompt_token_count=getattr(
            metadata, "tool_use_prompt_token_count", None
        ),
    )
    text = response.text or ""
    if response_format == "json":
        return CallResponse(response=text, usage=usage)

    counts = ", ".join(
        f"{name}={count if count is not None else 'unavailable'}"
        for name, count in usage.items()
    )
    return f"Tokens: {counts}\n{text}"
