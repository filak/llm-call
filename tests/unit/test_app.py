import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from google.genai import types
from google.genai.errors import APIError

from llm_call import cli
from llm_call.files import load_part, load_parts
from llm_call.providers.gemini import client
from llm_call.providers.gemini.config import GEMINI_MODEL_DEF, get_api_key, get_model


def test_get_api_key_reads_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert get_api_key() == "test-key"


def test_get_api_key_requires_value(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="Missing GEMINI_API_KEY"):
        get_api_key()


def test_get_model_uses_default(monkeypatch):
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    assert get_model() == GEMINI_MODEL_DEF


def test_load_part_reads_text_file(tmp_path):
    file_path = tmp_path / "notes.txt"
    file_path.write_text("hello world\n", encoding="utf-8")

    # Passing include_file_name=True ensures "File notes.txt:" header is present
    result = load_part(str(file_path), include_file_name=True)

    # Access the first element since load_part returns a list
    assert result[0]["text"].startswith("File notes.txt:")


def test_load_part_default_omits_filename(tmp_path):
    file_path = tmp_path / "notes.txt"
    file_path.write_text("hello world\n", encoding="utf-8")

    result = load_part(str(file_path))
    assert result[0]["text"] == "hello world\n"


def test_load_parts_returns_empty_for_none():
    assert load_parts([]) == []


def test_cli_passes_arguments_to_call(monkeypatch, capsys):
    mocked_call = Mock(return_value="Tokens: total_token_count=12\nresult")
    monkeypatch.setitem(cli.PROVIDERS, "gemini", mocked_call)
    monkeypatch.setattr(
        "sys.argv",
        ["llm-call", "gemini", "hello", "--file", "note.txt", "--model", "model"],
    )

    assert cli.main() == 0
    assert capsys.readouterr().out == "Tokens: total_token_count=12\nresult\n"
    mocked_call.assert_called_once_with(
        "hello",
        files=["note.txt"],
        model="model",
        system=None,
        include_file_name=False,
        response_format="text",
    )


def test_load_part_unsupported_extension(tmp_path):
    # Test a completely unsupported extension (e.g., .pdf)
    unsupported_file = tmp_path / "document.pdf"
    unsupported_file.write_text("dummy content", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file format '.pdf'"):
        load_part(str(unsupported_file))


def test_load_part_mismatched_extension_png(tmp_path):
    # A text file saved with a .png extension will be treated as an image
    # by extension validation, loading as binary inline_data
    fake_png = tmp_path / "fake_image.png"
    fake_png.write_text("this is plain text content", encoding="utf-8")

    result = load_part(str(fake_png))

    # Expect inline_data payload because of .png extension
    assert len(result) == 1
    assert "inline_data" in result[0]
    assert result[0]["inline_data"]["mime_type"] == "image/png"
    assert result[0]["inline_data"]["data"] == b"this is plain text content"


@pytest.fixture
def generate_content(monkeypatch):
    generate = Mock(
        return_value=types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=types.Content(
                        parts=[types.Part(text="Příliš žluťoučký\nresult")]
                    )
                )
            ],
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=10,
                candidates_token_count=20,
                thoughts_token_count=5,
                cached_content_token_count=4,
                tool_use_prompt_token_count=2,
                total_token_count=37,
            ),
        )
    )
    monkeypatch.setattr(
        client,
        "get_client",
        lambda: SimpleNamespace(models=SimpleNamespace(generate_content=generate)),
    )
    return generate


def test_call_json_preserves_usage_and_request(generate_content, monkeypatch):
    load = Mock(return_value=[{"text": "attached"}])
    monkeypatch.setattr(client, "load_parts", load)
    result = client.call(
        "hello",
        files=["notes.txt"],
        model="test-model",
        system="instruction",
        include_file_name=True,
        response_format="json",
    )
    assert result == {
        "response": "Příliš žluťoučký\nresult",
        "usage": {
            "prompt_token_count": 10,
            "candidates_token_count": 20,
            "total_token_count": 37,
            "thoughts_token_count": 5,
            "cached_content_token_count": 4,
            "tool_use_prompt_token_count": 2,
        },
    }
    assert json.loads(json.dumps(result)) == result
    load.assert_called_once_with(["notes.txt"], include_file_name=True)
    request = generate_content.call_args.kwargs
    assert request["model"] == "test-model"
    assert request["contents"] == ["hello", {"text": "attached"}]
    assert request["config"].system_instruction == "instruction"
    assert request["config"].response_mime_type is None


def test_call_text_starts_with_usage(generate_content):
    result = client.call("hello")
    header, text = result.split("\n", 1)
    assert header.startswith("Tokens: ")
    assert "total_token_count=37" in header
    assert "thoughts_token_count=5" in header
    assert text == "Příliš žluťoučký\nresult"


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        types.GenerateContentResponseUsageMetadata(
            prompt_token_count=0,
            total_token_count=0,
        ),
    ],
)
def test_call_handles_missing_usage_and_empty_text(generate_content, metadata):
    generate_content.return_value = types.GenerateContentResponse(
        usage_metadata=metadata,
    )
    result = client.call("hello", response_format="json")
    assert result["response"] == ""
    assert result["usage"]["total_token_count"] == (0 if metadata else None)
    assert result["usage"]["candidates_token_count"] is None
    assert "candidates_token_count=unavailable" in client.call("hello")


def test_call_rejects_invalid_format_before_request(generate_content):
    with pytest.raises(ValueError, match="response_format"):
        client.call("hello", response_format="xml")
    generate_content.assert_not_called()


def test_call_wraps_api_errors(generate_content):
    generate_content.side_effect = APIError(500, {"message": "failed"})
    with pytest.raises(RuntimeError, match="Gemini API Error"):
        client.call("hello", response_format="json")


def test_call_wraps_connection_error(generate_content):
    generate_content.side_effect = httpx.ConnectError("certificate verify failed")
    with pytest.raises(RuntimeError, match="Gemini connection error"):
        client.call("hello")


def test_call_passes_timeout_in_milliseconds(generate_content):
    client.call("hello", timeout=2.5)
    config = generate_content.call_args.kwargs["config"]
    assert config.http_options.timeout == 2500
    assert config.system_instruction is None


def test_call_without_timeout_sends_no_config(generate_content):
    client.call("hello")
    assert generate_content.call_args.kwargs["config"] is None


def test_call_rejects_invalid_timeout_before_request(generate_content):
    with pytest.raises(ValueError, match="timeout"):
        client.call("hello", timeout=0)
    generate_content.assert_not_called()


def test_call_wraps_timeout_error(generate_content):
    generate_content.side_effect = httpx.ReadTimeout("timed out")
    with pytest.raises(RuntimeError, match="timed out after 5 s"):
        client.call("hello", timeout=5)


def test_cli_json_is_one_document(monkeypatch, capsys):
    result = {
        "response": "Příliš žluťoučký\nresult",
        "usage": {"total_token_count": 37},
    }
    mocked_call = Mock(return_value=result)
    monkeypatch.setitem(cli.PROVIDERS, "gemini", mocked_call)
    monkeypatch.setattr(
        "sys.argv", ["llm-call", "gemini", "hello", "--response", "json"]
    )
    assert cli.main() == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == result
    assert "Příliš" in captured.out
    assert captured.err == ""
    assert mocked_call.call_args.kwargs["response_format"] == "json"


def test_cli_invalid_format_does_not_call_api(monkeypatch, capsys):
    mocked_call = Mock()
    monkeypatch.setitem(cli.PROVIDERS, "gemini", mocked_call)
    monkeypatch.setattr(
        "sys.argv", ["llm-call", "gemini", "hello", "--response", "xml"]
    )
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    mocked_call.assert_not_called()
    assert capsys.readouterr().out == ""


def test_cli_error_stays_on_stderr(monkeypatch, capsys):
    monkeypatch.setitem(
        cli.PROVIDERS, "gemini", Mock(side_effect=RuntimeError("failed"))
    )
    monkeypatch.setattr(
        "sys.argv", ["llm-call", "gemini", "hello", "--response", "json"]
    )
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "failed" in captured.err


def test_cli_output_writes_text_file(monkeypatch, capsys, tmp_path):
    output = tmp_path / "out.txt"
    monkeypatch.setitem(cli.PROVIDERS, "gemini", Mock(return_value="Příliš\nresult"))
    monkeypatch.setattr(
        "sys.argv", ["llm-call", "gemini", "hello", "--output", str(output)]
    )
    assert cli.main() == 0
    assert output.read_text(encoding="utf-8") == "Příliš\nresult\n"
    assert capsys.readouterr().out == ""


def test_cli_output_writes_json_file(monkeypatch, capsys, tmp_path):
    output = tmp_path / "out.json"
    result = {"response": "result", "usage": {"total_token_count": 3}}
    monkeypatch.setitem(cli.PROVIDERS, "gemini", Mock(return_value=result))
    monkeypatch.setattr(
        "sys.argv",
        ["llm-call", "gemini", "hello", "--response", "json", "--output", str(output)],
    )
    assert cli.main() == 0
    assert json.loads(output.read_text(encoding="utf-8")) == result
    assert capsys.readouterr().out == ""


def test_cli_output_missing_directory_fails(monkeypatch, capsys, tmp_path):
    output = tmp_path / "missing" / "out.txt"
    monkeypatch.setitem(cli.PROVIDERS, "gemini", Mock(return_value="result"))
    monkeypatch.setattr(
        "sys.argv", ["llm-call", "gemini", "hello", "--output", str(output)]
    )
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "out.txt" in captured.err


@pytest.mark.parametrize("response_format", ["text", "json"])
def test_html_response_round_trip(generate_content, tmp_path, response_format):
    prompt = "Clean and normalize the attached HTML using the given rules."
    system = "Return only valid cleaned HTML, without Markdown fences."
    source = "<h1>Článek</h1><p>infor-<br>mation &amp; text</p>"
    cleaned = '<h1>Článek</h1>\n<p class="body">information &amp; text</p>'
    attachment = tmp_path / "input.html"
    attachment.write_text(source, encoding="utf-8")
    generate_content.return_value.candidates[0].content.parts = [
        types.Part(text=cleaned)
    ]

    result = client.call(
        prompt,
        files=[str(attachment)],
        system=system,
        response_format=response_format,
    )

    request = generate_content.call_args.kwargs
    assert request["contents"] == [prompt, {"text": source}]
    assert request["config"].system_instruction == system
    if response_format == "json":
        decoded = json.loads(json.dumps(result))
        assert decoded["response"] == cleaned
        assert decoded["usage"]["total_token_count"] == 37
    else:
        header, body = result.split("\n", 1)
        assert header.startswith("Tokens: ")
        assert "total_token_count=37" in header
        assert body == cleaned


@pytest.mark.parametrize("response_format", ["text", "json"])
def test_cli_html_response_round_trip(
    generate_content, tmp_path, monkeypatch, capsys, response_format
):
    cleaned = '<h1>Článek</h1>\n<p title="Example">Text &amp; more.</p>'
    attachment = tmp_path / "input.html"
    attachment.write_text("<h1>Článek</h1><p>Text</p>", encoding="utf-8")
    generate_content.return_value.candidates[0].content.parts = [
        types.Part(text=cleaned)
    ]
    monkeypatch.setattr(
        "sys.argv",
        [
            "llm-call",
            "gemini",
            "Clean HTML",
            "--file",
            str(attachment),
            "--system",
            "Return only HTML",
            "--response",
            response_format,
        ],
    )

    assert cli.main() == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    if response_format == "json":
        result = json.loads(captured.out)
        assert result["response"] == cleaned
        assert result["usage"]["total_token_count"] == 37
    else:
        header, body = captured.out.split("\n", 1)
        assert header.startswith("Tokens: ")
        assert body == cleaned + "\n"


@pytest.mark.parametrize("suffix", [".html", ".htm", ".HTML", ".HTM"])
@pytest.mark.parametrize("include_file_name", [False, True])
def test_load_html_as_complete_text(tmp_path, suffix, include_file_name):
    html = "<h1>Článek</h1><p>" + "Žluťoučký &amp; text. " * 1500 + "</p>END"
    attachment = tmp_path / ("article" + suffix)
    attachment.write_text(html, encoding="utf-8")
    result = load_part(str(attachment), include_file_name=include_file_name)
    prefix = f"File {attachment.name}:\n" if include_file_name else ""
    assert result == [{"text": prefix + html}]


@pytest.mark.parametrize("include_file_name", [False, True])
def test_load_text_does_not_truncate(tmp_path, include_file_name):
    text = "x" * 20000 + "END"
    attachment = tmp_path / "long.txt"
    attachment.write_text(text, encoding="utf-8")
    prefix = "File long.txt:\n" if include_file_name else ""
    assert load_part(str(attachment), include_file_name=include_file_name) == [
        {"text": prefix + text}
    ]


@pytest.mark.parametrize("include_file_name", [False, True])
def test_load_json_does_not_truncate(tmp_path, include_file_name):
    payload = {"text": "Ž" * 20000 + "END"}
    attachment = tmp_path / "long.json"
    attachment.write_text(json.dumps(payload), encoding="utf-8")
    text = load_part(str(attachment), include_file_name=include_file_name)[0]["text"]
    if include_file_name:
        prefix, text = text.split("\n", 1)
        assert prefix == "JSON long.json:"
    assert json.loads(text) == payload


def write_payload(tmp_path, data=None, raw=None, bom=False):
    path = tmp_path / "request.json"
    content = raw if raw is not None else json.dumps(data).encode("utf-8")
    path.write_bytes((b"\xef\xbb\xbf" if bom else b"") + content)
    return str(path)


@pytest.fixture
def mocked_cli_call(monkeypatch):
    mocked_call = Mock(return_value="result")
    monkeypatch.setitem(cli.PROVIDERS, "gemini", mocked_call)
    return mocked_call


def run_cli(monkeypatch, *argv):
    monkeypatch.setattr("sys.argv", ["llm-call", "gemini", *argv])
    return cli.main()


def run_cli_error(monkeypatch, capsys, *argv):
    with pytest.raises(SystemExit) as exc:
        run_cli(monkeypatch, *argv)
    captured = capsys.readouterr()
    assert captured.out == ""
    return exc.value.code, captured.err


FULL_PAYLOAD = {
    "prompt": "Vyčisti přiložené HTML.",
    "system": "Vrať jen platné HTML. Žluťoučký kůň.",
    "model": "payload-model",
    "files": ["input.html", "data/notes.txt"],
    "include_file_name": True,
}


def test_cli_payload_minimal(tmp_path, monkeypatch, mocked_cli_call):
    path = write_payload(tmp_path, {"prompt": "hello", "system": "be brief"})
    assert run_cli(monkeypatch, "--payload", path) == 0
    mocked_cli_call.assert_called_once_with(
        "hello",
        files=[],
        model=None,
        system="be brief",
        include_file_name=False,
        response_format="text",
    )


@pytest.mark.parametrize("bom", [False, True])
def test_cli_payload_complete_unicode(tmp_path, monkeypatch, mocked_cli_call, bom):
    raw = json.dumps(FULL_PAYLOAD, ensure_ascii=False).encode("utf-8")
    path = write_payload(tmp_path, raw=raw, bom=bom)
    assert run_cli(monkeypatch, "--payload", path) == 0
    mocked_cli_call.assert_called_once_with(
        FULL_PAYLOAD["prompt"],
        files=FULL_PAYLOAD["files"],
        model="payload-model",
        system=FULL_PAYLOAD["system"],
        include_file_name=True,
        response_format="text",
    )


@pytest.mark.parametrize(
    "argv, key, expected",
    [
        (["Console prompt"], "prompt", "Console prompt"),
        (["--system", "Console system"], "system", "Console system"),
        (["--model", "console-model"], "model", "console-model"),
        (["--file", "a.txt"], "files", ["a.txt"]),
        (["--file", "a.txt", "--file", "b.png"], "files", ["a.txt", "b.png"]),
        (["--no-include-file-name"], "include_file_name", False),
    ],
)
def test_cli_console_overrides_payload(
    tmp_path, monkeypatch, mocked_cli_call, argv, key, expected
):
    path = write_payload(tmp_path, FULL_PAYLOAD)
    assert run_cli(monkeypatch, "--payload", path, *argv) == 0
    args, kwargs = mocked_cli_call.call_args
    merged = {"prompt": args[0], **kwargs}
    for name in FULL_PAYLOAD:
        assert merged[name] == (expected if name == key else FULL_PAYLOAD[name])


def test_cli_include_file_name_overrides_payload_false(
    tmp_path, monkeypatch, mocked_cli_call
):
    data = {"prompt": "p", "system": "s", "include_file_name": False}
    path = write_payload(tmp_path, data)
    assert run_cli(monkeypatch, "--payload", path, "--include-file-name") == 0
    assert mocked_cli_call.call_args.kwargs["include_file_name"] is True


def test_cli_payload_with_json_response_format(
    tmp_path, monkeypatch, mocked_cli_call, capsys
):
    mocked_cli_call.return_value = {"response": "ok", "usage": {}}
    path = write_payload(tmp_path, {"prompt": "p", "system": "s"})
    assert run_cli(monkeypatch, "--payload", path, "--response", "json") == 0
    assert json.loads(capsys.readouterr().out) == {"response": "ok", "usage": {}}
    assert mocked_cli_call.call_args.kwargs["response_format"] == "json"


def test_cli_without_payload_keeps_defaults(monkeypatch, mocked_cli_call):
    assert run_cli(monkeypatch, "hello") == 0
    mocked_cli_call.assert_called_once_with(
        "hello",
        files=[],
        model=None,
        system=None,
        include_file_name=False,
        response_format="text",
    )


def test_cli_requires_prompt_without_payload(monkeypatch, capsys, mocked_cli_call):
    code, err = run_cli_error(monkeypatch, capsys)
    assert code == 2
    assert "required: prompt" in err
    mocked_cli_call.assert_not_called()


@pytest.mark.parametrize(
    "data, message",
    [
        (["prompt"], "must be a JSON object"),
        ("prompt", "must be a JSON object"),
        (None, "must be a JSON object"),
        ({"system": "s"}, "missing required key 'prompt'"),
        ({"prompt": "p"}, "missing required key 'system'"),
        ({"prompt": "", "system": "s"}, "'prompt' must be a non-empty string"),
        ({"prompt": "  ", "system": "s"}, "'prompt' must be a non-empty string"),
        ({"prompt": 1, "system": "s"}, "'prompt' must be a non-empty string"),
        ({"prompt": "p", "system": ""}, "'system' must be a non-empty string"),
        ({"prompt": "p", "system": ["s"]}, "'system' must be a non-empty string"),
        ({"prompt": "p", "system": "s", "model": ""}, "'model' must be"),
        ({"prompt": "p", "system": "s", "model": 5}, "'model' must be"),
        ({"prompt": "p", "system": "s", "model": None}, "'model' must be"),
        ({"prompt": "p", "system": "s", "files": "a.txt"}, "'files' must be"),
        ({"prompt": "p", "system": "s", "files": [""]}, "'files' must be"),
        ({"prompt": "p", "system": "s", "files": [1]}, "'files' must be"),
        (
            {"prompt": "p", "system": "s", "include_file_name": "true"},
            "'include_file_name' must be true or false",
        ),
        (
            {"prompt": "p", "system": "s", "include_file_name": 1},
            "'include_file_name' must be true or false",
        ),
        (
            {"prompt": "p", "system": "s", "temperature": 1, "extra": 2},
            "unknown payload key(s): extra, temperature",
        ),
    ],
)
def test_cli_invalid_payload_does_not_call_api(
    tmp_path, monkeypatch, capsys, mocked_cli_call, data, message
):
    # Console overrides must not hide an invalid payload.
    path = write_payload(tmp_path, data)
    code, err = run_cli_error(
        monkeypatch,
        capsys,
        "--payload",
        path,
        "--system",
        "x",
        "--model",
        "x",
        "--file",
        "x",
    )
    assert code == 2
    assert message in err
    mocked_cli_call.assert_not_called()


@pytest.mark.parametrize(
    "raw, message",
    [
        (b'{"prompt": "p", "system": ', "invalid JSON in payload file"),
        (b"", "invalid JSON in payload file"),
        (b'{"prompt": "\xff", "system": "s"}', "is not valid UTF-8"),
    ],
)
def test_cli_unparsable_payload_does_not_call_api(
    tmp_path, monkeypatch, capsys, mocked_cli_call, raw, message
):
    path = write_payload(tmp_path, raw=raw)
    code, err = run_cli_error(monkeypatch, capsys, "--payload", path)
    assert code == 2
    assert message in err
    mocked_cli_call.assert_not_called()


@pytest.mark.parametrize("name", ["missing.json", "."])
def test_cli_unreadable_payload_does_not_call_api(
    tmp_path, monkeypatch, capsys, mocked_cli_call, name
):
    code, err = run_cli_error(monkeypatch, capsys, "--payload", str(tmp_path / name))
    assert code == 2
    assert "cannot read payload file" in err
    mocked_cli_call.assert_not_called()


def test_cli_payload_relative_files_use_cwd(tmp_path, monkeypatch, generate_content):
    workdir = tmp_path / "work"
    (workdir / "data").mkdir(parents=True)
    (workdir / "data" / "input.html").write_text("<p>Článek</p>", encoding="utf-8")
    payload_dir = tmp_path / "payloads"
    payload_dir.mkdir()
    data = {"prompt": "p", "system": "s", "files": ["data/input.html"]}
    path = write_payload(payload_dir, data)
    monkeypatch.chdir(workdir)
    assert run_cli(monkeypatch, "--payload", path) == 0
    contents = generate_content.call_args.kwargs["contents"]
    assert contents == ["p", {"text": "<p>Článek</p>"}]


def test_cli_missing_attachment_is_clean_error(
    tmp_path, monkeypatch, capsys, generate_content
):
    missing = tmp_path / "missing.html"
    code, err = run_cli_error(monkeypatch, capsys, "hello", "--file", str(missing))
    assert code == 1
    assert err.startswith("llm-call: error: ")
    assert "missing.html" in err
    assert "Traceback" not in err
    generate_content.assert_not_called()


@pytest.mark.parametrize(
    "argv, message",
    [
        ([], "required: provider"),
        (["openai", "hello"], "invalid choice: 'openai'"),
        (["--payload", "request.json"], "required: provider"),
    ],
)
def test_cli_requires_valid_provider(
    monkeypatch, capsys, mocked_cli_call, argv, message
):
    monkeypatch.setattr("sys.argv", ["llm-call", *argv])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert message in capsys.readouterr().err
    mocked_cli_call.assert_not_called()


def test_cli_prompt_after_payload_option(tmp_path, monkeypatch, mocked_cli_call):
    path = write_payload(tmp_path, {"prompt": "p", "system": "s"})
    monkeypatch.setattr(
        "sys.argv", ["llm-call", "gemini", "--payload", path, "Console prompt"]
    )
    assert cli.main() == 0
    assert mocked_cli_call.call_args.args[0] == "Console prompt"
