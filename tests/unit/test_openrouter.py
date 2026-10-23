import base64
import json
from unittest.mock import Mock

import httpx
import pytest

from llm_call import cli
from llm_call.providers.openrouter import client
from llm_call.providers.openrouter.config import (
    OPENROUTER_MODEL_DEF,
    get_api_key,
    get_model,
)

USAGE_KEYS = ["total_tokens", "request_tokens", "response_tokens"]

COMPLETION = {
    "choices": [
        {
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "Příliš žluťoučký\nresult"},
        }
    ],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 25,
        "total_tokens": 35,
        "completion_tokens_details": {"reasoning_tokens": 5},
        "prompt_tokens_details": {"cached_tokens": 4},
    },
}


class FakeOpenRouter:
    def __init__(self):
        self.requests = []
        self.response = httpx.Response(200, json=COMPLETION)
        self.error = None

    def handler(self, request):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.response

    @property
    def body(self):
        return json.loads(self.requests[-1].content)


@pytest.fixture
def api(monkeypatch):
    fake = FakeOpenRouter()
    http = httpx.Client(
        base_url="https://openrouter.test/api/v1",
        transport=httpx.MockTransport(fake.handler),
    )
    monkeypatch.setattr(client, "get_client", lambda: http)
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    return fake


def test_get_api_key_reads_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    assert get_api_key() == "test-key"


def test_get_api_key_requires_value(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="Missing OPENROUTER_API_KEY"):
        get_api_key()


def test_get_model_uses_default(monkeypatch):
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    assert get_model() == OPENROUTER_MODEL_DEF == "google/gemma-4-31b-it"


def test_call_json_matches_gemini_envelope(api, monkeypatch):
    load = Mock(return_value=[{"text": "attached"}])
    monkeypatch.setattr(client, "load_part", load)
    result = client.call(
        "hello",
        files=["notes.txt"],
        model="google/gemma-4-26b-a4b-it",
        system="instruction",
        include_file_name=True,
        response_format="json",
    )
    assert result == {
        "response": "Příliš žluťoučký\nresult",
        "usage": {"total_tokens": 35, "request_tokens": 10, "response_tokens": 25},
    }
    assert list(result["usage"]) == USAGE_KEYS
    assert json.loads(json.dumps(result)) == result
    load.assert_called_once_with("notes.txt", include_file_name=True)
    assert api.requests[-1].url.path.endswith("/chat/completions")
    assert api.body == {
        "model": "google/gemma-4-26b-a4b-it",
        "messages": [
            {"role": "system", "content": "instruction"},
            {
                "role": "user",
                "content": [{"type": "text", "text": "hello\n\nattached"}],
            },
        ],
        "temperature": 0.0,
    }


def test_call_text_starts_with_usage(api):
    result = client.call("hello")
    header, text = result.split("\n", 1)
    assert header == "Tokens: total_tokens=35, request_tokens=10, response_tokens=25"
    assert text == "Příliš žluťoučký\nresult"
    assert api.body["model"] == OPENROUTER_MODEL_DEF
    assert api.body["messages"][0]["role"] == "user"


def test_call_sends_image_as_data_url(api, tmp_path):
    image = tmp_path / "photo.png"
    image.write_bytes(b"\x89PNG data")
    client.call("describe", files=[str(image)])
    parts = api.body["messages"][0]["content"]
    encoded = base64.b64encode(b"\x89PNG data").decode("ascii")
    assert parts == [
        {"type": "text", "text": "describe"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encoded}"},
        },
    ]


def test_call_appends_text_files_to_prompt(api, tmp_path):
    image = tmp_path / "photo.JPEG"
    image.write_bytes(b"jpeg")
    page = tmp_path / "page.html"
    page.write_text("<p>Článek</p>", encoding="utf-8")
    data = tmp_path / "data.json"
    data.write_text('{"a": 1}', encoding="utf-8")
    client.call("Clean", files=[str(page), str(image), str(data)])
    encoded = base64.b64encode(b"jpeg").decode("ascii")
    assert api.body["messages"][0]["content"] == [
        {"type": "text", "text": 'Clean\n\n<p>Článek</p>\n\n{\n  "a": 1\n}'},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
        },
    ]


def test_call_keeps_image_label_next_to_image(api, tmp_path):
    note = tmp_path / "note.txt"
    note.write_text("text", encoding="utf-8")
    image = tmp_path / "photo.webp"
    image.write_bytes(b"webp")
    client.call("hello", files=[str(note), str(image)], include_file_name=True)
    parts = api.body["messages"][0]["content"]
    assert parts[0] == {"type": "text", "text": "hello\n\nFile note.txt:\ntext"}
    assert parts[1] == {"type": "text", "text": "Attachment filename: photo.webp"}
    assert parts[2]["image_url"]["url"].startswith("data:image/webp;base64,")
    assert len(parts) == 3


def test_call_rejects_unsupported_image_type_before_request(api, monkeypatch):
    part = {"inline_data": {"mime_type": "application/octet-stream", "data": b"x"}}
    monkeypatch.setattr(client, "load_part", Mock(return_value=[part]))
    with pytest.raises(ValueError, match="Use a JPEG, PNG, or WebP image."):
        client.call("hello", files=["image.png"])
    assert api.requests == []


def test_call_handles_missing_usage_and_empty_text(api):
    api.response = httpx.Response(200, json={"choices": []})
    result = client.call("hello", response_format="json")
    assert result == {"response": "", "usage": dict.fromkeys(USAGE_KEYS)}
    assert "response_tokens=unavailable" in client.call("hello")


def test_call_passes_temperature(api):
    client.call("hello", temperature=1.2)
    assert api.body["temperature"] == 1.2


def test_call_rejects_unsupported_model_before_request(api):
    with pytest.raises(ValueError, match="Unsupported OpenRouter model 'other'"):
        client.call("hello", model="other")
    assert api.requests == []


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"response_format": "xml"}, "response_format"),
        ({"timeout": 0}, "timeout"),
        ({"temperature": 2.1}, "temperature"),
        ({"temperature": -0.1}, "temperature"),
    ],
)
def test_call_rejects_invalid_options_before_request(api, kwargs, message):
    with pytest.raises(ValueError, match=message):
        client.call("hello", **kwargs)
    assert api.requests == []


def test_call_passes_timeout(api):
    client.call("hello", timeout=2.5)
    assert api.requests[-1].extensions["timeout"]["read"] == 2.5


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(401, json={"error": {"message": "No auth", "code": 401}}),
        httpx.Response(502, text="No auth"),
        httpx.Response(200, json={"error": {"message": "No auth", "code": 401}}),
    ],
)
def test_call_wraps_api_errors(api, response):
    api.response = response
    with pytest.raises(RuntimeError, match="OpenRouter API Error: .*No auth"):
        client.call("hello")


def test_call_wraps_connection_error(api):
    api.error = httpx.ConnectError("certificate verify failed")
    with pytest.raises(RuntimeError, match="OpenRouter connection error"):
        client.call("hello")


def test_call_wraps_timeout_error(api):
    api.error = httpx.ReadTimeout("timed out")
    with pytest.raises(RuntimeError, match="timed out after 5 s"):
        client.call("hello", timeout=5)


def test_cli_openrouter_json(api, monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv", ["llm-call", "openrouter", "hello", "--response", "json"]
    )
    assert cli.main() == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out)["response"] == "Příliš žluťoučký\nresult"


def test_cli_openrouter_unsupported_model(api, monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv", ["llm-call", "openrouter", "hello", "--model", "other"]
    )
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1
    assert "Unsupported OpenRouter model" in capsys.readouterr().err
    assert api.requests == []
