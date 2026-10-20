import json
from pathlib import Path
from typing import Any, Dict

ALLOWED_KEYS = {"prompt", "system", "model", "files", "include_file_name"}
REQUIRED_KEYS = ("prompt", "system")


class PayloadError(ValueError):
    """Raised when a payload file cannot be read or is not valid."""


def load_payload(path: str) -> Dict[str, Any]:
    """Read a UTF-8 JSON payload file (BOM optional) and validate it."""
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise PayloadError(
            f"cannot read payload file '{path}': {exc.strerror or exc}"
        ) from exc
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise PayloadError(f"payload file '{path}' is not valid UTF-8") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PayloadError(
            f"invalid JSON in payload file '{path}': {exc.msg} "
            f"(line {exc.lineno}, column {exc.colno})"
        ) from exc
    return validate_payload(data)


def _is_non_empty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_payload(data: Any) -> Dict[str, Any]:
    """Return the payload unchanged, or raise PayloadError for invalid content."""
    if not isinstance(data, dict):
        raise PayloadError("payload must be a JSON object")
    unknown = sorted(set(data) - ALLOWED_KEYS)
    if unknown:
        raise PayloadError(f"unknown payload key(s): {', '.join(unknown)}")
    for key in REQUIRED_KEYS:
        if key not in data:
            raise PayloadError(f"payload is missing required key '{key}'")
    for key in ("prompt", "system", "model"):
        if key in data and not _is_non_empty_str(data[key]):
            raise PayloadError(f"payload key '{key}' must be a non-empty string")
    if "files" in data:
        files = data["files"]
        if not isinstance(files, list) or not all(map(_is_non_empty_str, files)):
            raise PayloadError(
                "payload key 'files' must be a list of non-empty path strings"
            )
    if "include_file_name" in data and type(data["include_file_name"]) is not bool:
        raise PayloadError("payload key 'include_file_name' must be true or false")
    return data
