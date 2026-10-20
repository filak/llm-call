import argparse
import json
from pathlib import Path
from typing import Any, Optional

from .payload import PayloadError, load_payload
from .providers import PROVIDERS
from .providers.gemini.config import GEMINI_MODEL_DEF


def _pick(console: Optional[Any], payload: Optional[Any], default: Any) -> Any:
    """Return the first value that is not None: console, payload, default."""
    if console is not None:
        return console
    if payload is not None:
        return payload
    return default


def _error_message(exc: Exception) -> str:
    if isinstance(exc, FileNotFoundError):
        return f"file not found: {exc.filename or exc}"
    if isinstance(exc, OSError) and exc.filename is not None:
        return f"{exc.strerror or type(exc).__name__}: {exc.filename}"
    return str(exc)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="llm-call",
        description="Call an LLM provider API with prompt + files",
    )
    parser.add_argument(
        "provider",
        choices=sorted(PROVIDERS),
        help="LLM provider",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="Prompt text (optional with --payload)",
    )
    parser.add_argument(
        "--file",
        action="append",
        dest="files",
        default=None,
        help="File to attach (repeatable): html, image (PNG, JPEG, WEBP), json, txt",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"Model (default depends on provider; gemini: {GEMINI_MODEL_DEF})",
    )
    parser.add_argument(
        "--system",
        default=None,
        help="System instruction",
    )
    parser.add_argument(
        "--include-file-name",
        action=argparse.BooleanOptionalAction,
        dest="include_file_name",
        default=None,
        help="Include original file names with attached contents",
    )
    parser.add_argument(
        "--payload",
        metavar="PATH",
        default=None,
        help="JSON file with request parameters; console arguments override it",
    )
    parser.add_argument(
        "--response",
        choices=("text", "json"),
        dest="response_format",
        default="text",
        help="Output format (default: text); includes token usage",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        default=None,
        help="Write output to a file instead of standard output",
    )
    # Intermixed parsing allows the prompt after options: gemini --payload x prompt
    args = parser.parse_intermixed_args()

    payload = {}
    if args.payload is not None:
        try:
            payload = load_payload(args.payload)
        except PayloadError as exc:
            parser.error(str(exc))
    elif args.prompt is None:
        parser.error("the following arguments are required: prompt")

    try:
        result = PROVIDERS[args.provider](
            _pick(args.prompt, payload.get("prompt"), None),
            files=_pick(args.files, payload.get("files"), []),
            model=_pick(args.model, payload.get("model"), None),
            system=_pick(args.system, payload.get("system"), None),
            include_file_name=_pick(
                args.include_file_name, payload.get("include_file_name"), False
            ),
            response_format=args.response_format,
        )
        if args.response_format == "json":
            text = json.dumps(result, ensure_ascii=False)
        else:
            text = result
        if args.output is not None:
            Path(args.output).write_text(text + "\n", encoding="utf-8")
        else:
            print(text)
    except (RuntimeError, OSError, ValueError) as exc:
        parser.exit(1, f"{parser.prog}: error: {_error_message(exc)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
