# llm-call

A compact API wrapper and CLI for calling a LLM model provider.

This project uses LLM-AI coding tools.

The current project status is: in development

[![CodeQL](https://github.com/filak/llm-call/actions/workflows/codeql.yml/badge.svg)](https://github.com/filak/llm-call/security/code-scanning)

## Install

Use Python 3.12 or later and [uv](https://docs.astral.sh/uv/).

Install the CLI from a source checkout:

```bash
uv tool install .
```

For development, create the project environment instead:

```bash
uv sync
```

```bash
uv sync --upgrade
```

## Setup

For the `gemini` provider, set a Gemini API key as an environment variable: GEMINI_API_KEY

```bash
# CMD: setx /M GEMINI_API_KEY "your-api-key"
# PowerShell: $env:GEMINI_API_KEY = "your-api-key"
# POSIX shell: export GEMINI_API_KEY="your-api-key"
```

For the `openrouter` provider, set an OpenRouter API key as an environment variable: OPENROUTER_API_KEY

```bash
# CMD: setx /M OPENROUTER_API_KEY "your-api-key"
# PowerShell: $env:OPENROUTER_API_KEY = "your-api-key"
# POSIX shell: export OPENROUTER_API_KEY="your-api-key"
```

## Providers

Give the provider as the first argument.
The available providers are `gemini` and `openrouter`.
Other providers can be added in `src/llm_call/providers/`.

All providers give the same output format and the same token usage fields:

- `total_tokens`: the total token count that the provider reports.
- `request_tokens`: the tokens in the request (prompt, system instruction, and attachments).
- `response_tokens`: the tokens in the response. This count includes thinking (reasoning) tokens.

If a provider does not supply a token count, the output shows `unavailable` (text)
or `null` (JSON).

Text output example:

```text
Tokens: total_tokens=35, request_tokens=10, response_tokens=25
<response text>
```

JSON output example:

```json
{"response": "<response text>", "usage": {"total_tokens": 35, "request_tokens": 10, "response_tokens": 25}}
```

### Temperature

All providers use a temperature of `0.0` by default.
To use a different value, give `--temperature` or the payload key `temperature`.
The value must be a number from `0.0` to `2.0`.

The `openrouter` provider supports only these models:

- `google/gemma-4-31b-it` (default)
- `google/gemma-4-26b-a4b-it`

If you give a different model, the command shows an error and does not call the API.

The `openrouter` provider sends attachments as follows:

- It adds the contents of each text, HTML, and JSON file to the end of the prompt.
  It puts an empty line between the prompt and each file.
- It sends each image as a separate base64 data URL after the prompt.
  The image must be JPEG, PNG, or WebP.

## Commands

```bash
uv run llm-call --help
```

```bash
usage: llm-call [-h] [--file FILES] [--model MODEL] [--system SYSTEM]
                [--temperature TEMPERATURE]
                [--include-file-name | --no-include-file-name]
                [--payload PATH] [--response {text,json}] [--output PATH]
                {gemini,openrouter} [prompt]

Call an LLM provider API with prompt + files

positional arguments:
  {gemini,openrouter}   LLM provider
  prompt                Prompt text (optional with --payload)

options:
  -h, --help            show this help message and exit
  --file FILES          File to attach (repeatable): html, image (PNG, JPEG,
                        WEBP), json, txt
  --model MODEL         Model (default depends on provider; gemini:
                        gemini-3.8-flash; openrouter: google/gemma-4-31b-it,
                        supported: google/gemma-4-31b-it,
                        google/gemma-4-26b-a4b-it)
  --system SYSTEM       System instruction
  --temperature TEMPERATURE
                        Sampling temperature 0.0-2.0 (default: 0.0)
  --include-file-name, --no-include-file-name
                        Include original file names with attached contents
  --payload PATH        JSON file with request parameters; console arguments
                        override it
  --response {text,json}
                        Output format (default: text); includes token usage
  --output PATH         Write output to a file instead of standard output
```

Examples:

```bash
uv run llm-call gemini "hello"
uv run llm-call gemini "Explain this file" --file README.txt
uv run llm-call gemini "hello" --output result.txt
uv run llm-call openrouter "hello"
uv run llm-call openrouter "hello" --model google/gemma-4-26b-a4b-it
```

Give the `provider` argument first. It is always necessary.
If you do not use `--payload`, you must give the `prompt` argument.

### Output file

If you use `--output PATH`, the command writes the output to the file.
The command does not write the output to standard output.
The command writes the file in UTF-8 and replaces an existing file.
The command resolves a relative path from the current working directory.
The parent directory must exist.
If the command cannot write the file, it shows an error and stops with exit code 1.

## Payload file

Use `--payload PATH` to read request parameters from a JSON file.

Example `request.json`:

```json
{
  "prompt": "Clean and normalize the attached HTML.",
  "system": "Return only valid HTML.",
  "model": "gemini-3.8-flash",
  "files": ["input.html"],
  "include_file_name": true,
  "temperature": 0.0
}
```

Required fields:

- `prompt`: a non-empty string.
- `system`: a non-empty string.

Optional fields:

- `model`: a non-empty string.
- `files`: a list of non-empty path strings.
- `include_file_name`: a JSON boolean (`true` or `false`).
- `temperature`: a JSON number from `0.0` to `2.0`. The default is `0.0`.

Encode the payload file as UTF-8. A byte order mark (BOM) is optional.
The command rejects unknown fields.

The command validates the complete payload before it calls the API.
If the payload file is missing, unreadable, not valid UTF-8, not valid JSON,
or contains an invalid value, the command shows an error and stops with exit code 2.
If an attachment is missing or unreadable, the command shows an error and stops with exit code 1.

### Precedence

The command selects each value in this order:

1. The console argument.
2. The payload value.
3. The default value.

A `--file` argument replaces the complete payload `files` list.
The command does not merge the two lists.

`provider`, `--response`, and `--output` are console options only. Do not put them in the payload file.

### Paths

The command resolves relative attachment paths from the current working directory.
It does not resolve them from the location of the payload file.
This is the same behavior as for `--file`.

### Boolean overrides

- Use `--include-file-name` to set `include_file_name` to `true`.
- Use `--no-include-file-name` to set `include_file_name` to `false`.

### Payload examples

```bash
uv run llm-call gemini --payload request.json
uv run llm-call gemini --payload request.json "Summarize the attached HTML."
uv run llm-call gemini --payload request.json --model other-model-name
uv run llm-call gemini --payload request.json --file page1.html --file page2.html
uv run llm-call gemini --payload request.json --no-include-file-name
uv run llm-call gemini --payload request.json --response json
```

## Environment variables

The `gemini` provider uses these variables:

- `GEMINI_API_KEY` or `GOOGLE_API_KEY`
- `GEMINI_MODEL` (optional, defaults to `gemini-3.8-flash`)

The `openrouter` provider uses these variables:

- `OPENROUTER_API_KEY`
- `OPENROUTER_MODEL` (optional, defaults to `google/gemma-4-31b-it`)

The command reads attached text and JSON files as text. It sends other file
types as inline binary data.


## Development

Run the following commands from the repository root.
Install the locked Python dependencies:

```bash
uv sync --locked
```

Testing:

```bash
uv run pytest
```
