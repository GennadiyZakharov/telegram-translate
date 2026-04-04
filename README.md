# Telegram chat export translator with Ollama

This project translates Telegram/Messenger exported HTML files from Russian to English 
using a local Ollama model while keeping the surrounding HTML structure intact.

It assumes the message content to translate is inside:

```html
<div class="text">...</div>
```

or any `div` whose class list includes `text`, such as:

```html
<div class="text bold">...</div>
```

## What it does

- Recursively reads one HTML file or a whole directory of HTML files.
- Finds all `div.text` blocks.
- Sends message text to a local Ollama model in batches.
- Rewrites only those text blocks in the output HTML.
- Skips blocks without Cyrillic by default, which saves time.

## Requirements

- Python 3.12+
- Ollama installed and running locally
- A downloaded model, for example:

```bash
ollama pull qwen2.5:14b
```

Ollama exposes a local API on `http://localhost:11434/api/chat`, 
supports structured outputs via a JSON schema, and supports `keep_alive` 
so the model can stay loaded between requests.

## Install

```bash
uv sync
source .venv/bin/activate
```

## Basic usage

Translate one file:

```bash
python translate_messenger_html.py chat.html translated_chat.html
```

Translate a whole directory recursively:
```bash
python translate_messenger_html.py ./export_html ./translated_html --overwrite
```

## Useful options

```bash
--batch-size 24
```
How many messages are sent in one LLM request.
Smaller values are safer for weaker local models.

```bash
--force-all
```
Translate **all** `div.text` blocks, even when they do not contain Cyrillic.

```bash
--overwrite
```
Overwrite output files if they already exist.

```bash
--keep-alive 10m
```
Ask Ollama to keep the model loaded in memory for faster repeated requests.

## Suggested models

For RU → EN chat translation, strong instruction-following models are usually the safest choice locally. 
In Ollama, a good starting point is something like:

- `qwen2.5:14b`
- `qwen2.5:7b`
- `gemma3`
- `gpt-oss`

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