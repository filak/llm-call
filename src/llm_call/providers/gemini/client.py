from __future__ import annotations

from typing import Literal, Optional, Sequence

import httpx
from google import genai
from google.genai import types
from google.genai.errors import APIError

from .config import get_api_key, get_model
from ..response import (
    CallResponse,
    TokenUsage,
    build_response,
    sum_counts,
    validate_options,
)
from ...files import load_parts

__all__ = ["CallResponse", "TokenUsage", "call", "get_client"]

_client: Optional[genai.Client] = None


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
    temperature: float = 0.0,
) -> str | CallResponse:
    """Return text with usage first, or a JSON-serializable response envelope.

    timeout is the request timeout in seconds. None means no client timeout.
    """
    validate_options(response_format, timeout, temperature)
    contents = [
        prompt,
        *load_parts(list(files or []), include_file_name=include_file_name),
    ]
    config = types.GenerateContentConfig(
        system_instruction=system or None,
        temperature=temperature,
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
        total_tokens=getattr(metadata, "total_token_count", None),
        request_tokens=sum_counts(
            getattr(metadata, "prompt_token_count", None),
            getattr(metadata, "tool_use_prompt_token_count", None),
        ),
        # Thinking tokens are billed as output, so they count as response tokens.
        response_tokens=sum_counts(
            getattr(metadata, "candidates_token_count", None),
            getattr(metadata, "thoughts_token_count", None),
        ),
    )
    return build_response(response.text or "", usage, response_format)
