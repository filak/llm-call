import json
import mimetypes
import pathlib
from typing import Any, Dict, List

TEXT_SUFFIXES = {".txt", ".html", ".htm"}
JSON_SUFFIXES = {".json"}
IMAGE_SUFFIXES = {".png", ".jpg", ".webp"}

ALLOWED_SUFFIXES = TEXT_SUFFIXES | JSON_SUFFIXES | IMAGE_SUFFIXES


def load_part(path: str, include_file_name: bool = False) -> List[Dict[str, Any]]:
    file_path = pathlib.Path(path)

    if not file_path.exists():
        raise FileNotFoundError(file_path)
    if file_path.is_dir():
        raise ValueError(f"Expected file, got directory: {file_path}")

    suffix = file_path.suffix.lower()

    if suffix not in ALLOWED_SUFFIXES:
        raise ValueError(
            f"Unsupported file format '{suffix}'. "
            f"Allowed formats: {', '.join(sorted(ALLOWED_SUFFIXES))}"
        )

    # Process text files
    if suffix in TEXT_SUFFIXES:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        content = f"File {file_path.name}:\n{text}" if include_file_name else text
        return [{"text": content}]

    # Process JSON files
    if suffix in JSON_SUFFIXES:
        payload = json.loads(file_path.read_bytes())
        json_text = json.dumps(payload, indent=2, ensure_ascii=False)
        content = (
            f"JSON {file_path.name}:\n{json_text}" if include_file_name else json_text
        )
        return [{"text": content}]

    # Process Image files
    mime_type, _ = mimetypes.guess_type(str(file_path))
    parts = []

    if include_file_name:
        parts.append({"text": f"Attachment filename: {file_path.name}"})

    parts.append(
        {
            "inline_data": {
                "mime_type": mime_type or "application/octet-stream",
                "data": file_path.read_bytes(),
            }
        }
    )
    return parts


def load_parts(
    paths: List[str], include_file_name: bool = False
) -> List[Dict[str, Any]]:
    parts = []
    if paths:
        for path in paths:
            parts.extend(load_part(path, include_file_name=include_file_name))
    return parts
