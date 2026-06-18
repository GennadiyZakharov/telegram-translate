# Telegram chat export translator

This project translates Telegram chat history exports (HTML format) from Russian to English
using a local LLM via an OpenAI-compatible API (e.g. `llama.cpp` or `Ollama`),
while keeping the surrounding HTML structure intact.

It assumes the message content to translate is inside:

```html
<div class="text">...</div>
```

or any `div` whose class list includes `text`, such as:

```html
<div class="text bold">...</div>
```

## What it does

- Reads one HTML file or a whole directory of HTML files recursively.
- Finds all `div.text` blocks.
- Sends message text to a local LLM in batches, including speaker names for context.
- Rewrites only those text blocks in the output HTML.
- Skips blocks without Cyrillic by default, which saves time.
- Transliterates speaker names from Cyrillic to Latin in the output.

## Requirements

- Python 3.12+
- A running OpenAI-compatible LLM server, e.g. `llama-server` from `llama.cpp`:

```bash
model="Qwen/Qwen2.5-14B-Instruct-GGUF:Q5_K_M"

~/apps/llama.cpp/llama-server \
    --host 0.0.0.0 \
    --port 8100 \
    -hf "$model"
```

The server must expose a `/v1/chat/completions` endpoint with JSON schema support
(`response_format: { type: "json_object", schema: ... }`).

## Install

```bash
uv sync
source .venv/bin/activate
```

## Basic usage

Translate one file (output defaults to `chat.translated.html`):

```bash
python telegram_translate.py chat.html
```

Specify an explicit output file:

```bash
python telegram_translate.py chat.html translated_chat.html
```

Translate a whole directory recursively (output defaults to `./export_html.translated/`):

```bash
python telegram_translate.py ./export_html
```

Specify an explicit output directory:

```bash
python telegram_translate.py ./export_html ./translated_html --overwrite
```

## Useful options

```bash
--model Qwen/Qwen2.5-14B-Instruct-GGUF:Q5_K_M
```
Model name as reported by the server's `/v1/models` endpoint.
The script checks that the requested model is available before starting.

```bash
--api-url http://localhost:8100/v1/chat/completions
```
URL of the OpenAI-compatible chat completions endpoint (default shown above).

```bash
--batch-size 48
```
How many messages are sent in one LLM request.
Smaller values are safer for weaker or slower local models.

```bash
--force-all
```
Translate **all** `div.text` blocks, even when they do not contain Cyrillic.

```bash
--overwrite
```
Overwrite output files if they already exist.

```bash
--timeout 180
```
HTTP timeout in seconds. Increase this for large batches or slow hardware.

```bash
--glossary glossary.tsv
```
Read approved phrase translations from a tab-separated glossary file.
By default, the script reads `glossary.tsv` from the project root if it exists.

The file must have three tab-separated columns:

```tsv
phrase	translation	comment
телега	Telegram	The name of the messenger where this conversation takes place
```

The glossary is added to the LLM system prompt. The model is instructed to use
the provided translation for matching phrases; the comment explains the context.

```bash
--debug
```
Print the full request payload and LLM responses for each batch.

## Suggested models

For RU → EN chat translation, strong instruction-following models work best.
A good starting point with `llama-server`:

- `Qwen/Qwen2.5-14B-Instruct-GGUF:Q5_K_M`
- `Qwen/Qwen2.5-7B-Instruct-GGUF:Q5_K_M`
- `bartowski/gemma-3-12b-it-GGUF:Q5_K_M`

Try a smaller model first if speed matters more than nuance.

## Notes on quality

Chat translation is tricky because short messages depend heavily on context.
This script translates **independent message batches**, not the whole conversation statefully.
That is usually good enough, but tiny messages like:

- `угу`
- `ага`
- `да ну`
- `и я за тебя)`

may sometimes need manual review.
