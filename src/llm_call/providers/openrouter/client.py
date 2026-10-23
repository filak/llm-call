from __future__ import annotations

import base64
from typing import Any, Dict, List, Literal, Optional, Sequence

import httpx

from .config import OPENROUTER_BASE_URL, OPENROUTER_MODELS, get_api_key, get_model
from ..response import CallResponse, TokenUsage, build_response, validate_options
from ...files import load_part

IMAGE_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}

_client: Optional[httpx.Client] = None


def get_client() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(
            base_url=OPENROUTER_BASE_URL,
            headers={"Authorization": f"Bearer {get_api_key()}"},
        )
    return _client


def _data_url(mime_type: str, data: bytes) -> str:
    if mime_type not in IMAGE_MIME_TYPES:
        raise ValueError("Use a JPEG, PNG, or WebP image.")
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _build_content(
    prompt: str, files: Sequence[str], include_file_name: bool
) -> List[Dict[str, Any]]:
    """Append text file contents to the prompt; send images as data URLs."""
    texts = [prompt]
    images: List[Dict[str, Any]] = []
    for path in files:
        parts = load_part(path, include_file_name=include_file_name)
        if not any("inline_data" in part for part in parts):
            texts.extend(part["text"] for part in parts)
            continue
        # Image: keep the optional filename label next to its image.
        for part in parts:
            if "text" in part:
                images.append({"type": "text", "text": part["text"]})
            else:
                inline = part["inline_data"]
                url = _data_url(inline["mime_type"], inline["data"])
                images.append({"type": "image_url", "image_url": {"url": url}})
    return [{"type": "text", "text": "\n\n".join(texts)}, *images]


def _error_detail(response: httpx.Response) -> str:
    try:
        error = response.json().get("error")
    except ValueError:
        return response.text
    if isinstance(error, dict):
        return str(error.get("message") or error)
    return str(error or response.text)


def _usage(data: Optional[Dict[str, Any]]) -> TokenUsage:
    data = data or {}
    # completion_tokens already includes reasoning tokens.
    return TokenUsage(
        total_tokens=data.get("total_tokens"),
        request_tokens=data.get("prompt_tokens"),
        response_tokens=data.get("completion_tokens"),
    )


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
    model = model or get_model()
    if model not in OPENROUTER_MODELS:
        raise ValueError(
            f"Unsupported OpenRouter model '{model}'. "
            f"Supported: {', '.join(OPENROUTER_MODELS)}"
        )
    content = _build_content(prompt, files or [], include_file_name)
    messages: List[Dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})

    try:
        response = get_client().post(
            "/chat/completions",
            json={"model": model, "messages": messages, "temperature": temperature},
            timeout=timeout,
        )
    except httpx.TimeoutException as exc:
        raise RuntimeError(f"OpenRouter request timed out after {timeout} s") from exc
    except httpx.TransportError as exc:
        raise RuntimeError(f"OpenRouter connection error: {exc}") from exc

    if response.is_error:
        raise RuntimeError(
            f"OpenRouter API Error: {response.status_code} {_error_detail(response)}"
        )
    try:
        result = response.json()
    except ValueError as exc:
        raise RuntimeError(f"OpenRouter API Error: invalid JSON: {exc}") from exc
    if result.get("error"):
        raise RuntimeError(
            f"OpenRouter API Error: {response.status_code} {_error_detail(response)}"
        )

    choices = result.get("choices") or [{}]
    text = (choices[0].get("message") or {}).get("content") or ""
    return build_response(text, _usage(result.get("usage")), response_format)
