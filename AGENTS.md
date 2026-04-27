# AI Agents Information

This document provides information for AI agents interacting with or contributing to the **Telegram-translate** project.

## Project Overview

The **Telegram-translate** project is designed to translate Telegram chat history exports (HTML format) 
from Russian to English. 
It uses local Large Language Models (LLMs) via an OpenAI-compatible API (specifically targeting `llama.cpp` or `Ollama`).

### Key Features

- **HTML Parsing:** Uses `BeautifulSoup` to find and replace text within `<div class="text">` blocks.
- **Batch Processing:** Translates messages in batches to maintain conversation context and improve efficiency.
- **Speaker Awareness:** Transliterates and includes speaker names in the LLM prompt to help the model understand context.
- **Cyrillic Detection:** Automatically identifies messages that need translation by checking for Cyrillic characters.
- **Schema-Based Output:** Uses JSON schemas to ensure the LLM returns structured and predictable data.

## Interaction Guidelines
1.  **API Compatibility:** The script expects an OpenAI-compatible endpoint. 
    By default, it looks for `llama-server` at `http://localhost:8000/v1/chat/completions`.
2.  **Prompt Engineering:** The system prompt is defined in `DEFAULT_SYSTEM_PROMPT`. 
    It instructs the model to be a precise translator and strictly follow a JSON schema.
3.  Dependency management is handled by `uv`.

## Technical Details
- **Main Entry Point:** `telegram_translate.py`
- **Translator Class:** `LlamaCppTranslator` handles all communication with the LLM.
- **Validation:** Includes checks to ensure the LLM doesn't return the speaker's name as the translation 
  and that the output is indeed in English (no Cyrillic).

## How to Run (for Agents)

To test a translation, run:

```bash
bash test_run.sh
```

It runs translation of the test messages from the `test/messages.html`,
and saves results in the `messages.translated.html` file.

If the local LLM is not running, stop and ask me to fix it.

