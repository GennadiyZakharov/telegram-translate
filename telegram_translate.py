#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import requests
from bs4 import BeautifulSoup

LLAMA_CPP_DEFAULT_URL = "http://localhost:8000/v1/chat/completions"
DEFAULT_SYSTEM_PROMPT = """You are a careful literary translator.
Translate Russian chat messages into natural, fluent English.
Rules:
- Preserve the original meaning and tone.
- Keep emojis, repeated punctuation, and casual style where sensible.
- Do not add explanations.
- Return only the JSON requested by the schema.
- If a message is already in English or does not need translation, return it unchanged.
- Keep line breaks if they matter.
"""

CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
LEAD_TRAIL_WS_RE = re.compile(r"^(\s*)(.*?)(\s*)$", re.DOTALL)


@dataclass
class MessageItem:
    key: str
    text: str
    leading_ws: str
    trailing_ws: str


class LlamaCppTranslator:
    def __init__(
        self,
        model: str,
        api_url: str = LLAMA_CPP_DEFAULT_URL,
        timeout: int = 300,
        temperature: float = 0.0,
        retries: int = 3,
        debug: bool = False,
    ):
        self.model = model
        self.api_url = api_url
        self.timeout = timeout
        self.temperature = temperature
        self.retries = retries
        self.debug = debug

    def translate_batch(self, texts: List[str]) -> List[str]:
        schema = {
            "type": "object",
            "properties": {
                "translations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "translated_text": {"type": "string"},
                        },
                        "required": ["id", "translated_text"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["translations"],
            "additionalProperties": False,
        }

        numbered = [{"id": i, "text": t} for i, t in enumerate(texts)]
        user_prompt = (
            "Translate the following chat messages from Russian to English. "
            "Return JSON strictly matching this schema. "
            "Keep the same number of items and the same ids.\n\n"
            f"Schema:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
            f"Messages:\n{json.dumps(numbered, ensure_ascii=False, indent=2)}"
        )

        payload = {
            "model": self.model,
            "stream": False,
            "temperature": self.temperature,
            "response_format": {
                "type": "json_object",
                "schema": schema,
            },
            "messages": [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        }

        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                print("Running translation for a message batch of size", len(texts), "with model", self.model, "attempt", attempt,"...")
                if self.debug:
                    print(f"Request payload:\n{json.dumps(payload, ensure_ascii=False, indent=2)}")
                response = requests.post(self.api_url, json=payload, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                if self.debug:
                    print(f"Model response:\n{content}")
                parsed = json.loads(content)
                items = parsed["translations"]
                result = [None] * len(texts)
                for item in items:
                    idx = item["id"]
                    if not 0 <= idx < len(texts):
                        raise ValueError(f"Invalid translation id returned: {idx}")
                    result[idx] = item["translated_text"]
                if any(v is None for v in result):
                    raise ValueError("Model returned an incomplete translation batch")
                return result  # type: ignore[return-value]
            except (requests.RequestException, KeyError, IndexError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt == self.retries:
                    break
                time.sleep(min(2**attempt, 8))

        raise RuntimeError(f"Failed to translate batch after {self.retries} attempts: {last_error}")


def extract_inner_text(raw_text: str) -> tuple[str, str, str]:
    match = LEAD_TRAIL_WS_RE.match(raw_text)
    if not match:
        return "", raw_text, ""
    return match.group(1), match.group(2), match.group(3)


def needs_translation(text: str, force_all: bool = False) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if force_all:
        return True
    return bool(CYRILLIC_RE.search(stripped))


def iter_message_divs(soup: BeautifulSoup):
    for div in soup.find_all("div"):
        classes = div.get("class", [])
        if "text" in classes:
            yield div


def replace_div_text(div, new_text: str) -> None:
    div.clear()
    div.append(new_text)


def translate_html_file(
    input_path: Path,
    output_path: Path,
    translator: LlamaCppTranslator,
    batch_size: int,
    force_all: bool,
    overwrite: bool,
) -> dict:
    if output_path.exists() and not overwrite:
        return {
            "file": str(input_path),
            "output": str(output_path),
            "status": "skipped_existing",
            "translated_blocks": 0,
        }

    html = input_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")

    divs = list(iter_message_divs(soup))
    candidates: list[tuple[object, MessageItem]] = []
    translated_blocks = 0

    for idx, div in enumerate(divs):
        raw_text = div.get_text(separator="", strip=False)
        leading_ws, core_text, trailing_ws = extract_inner_text(raw_text)
        if not needs_translation(core_text, force_all=force_all):
            continue
        key = f"{input_path.name}::{idx}"
        candidates.append(
            (
                div,
                MessageItem(
                    key=key,
                    text=core_text,
                    leading_ws=leading_ws,
                    trailing_ws=trailing_ws,
                ),
            )
        )

    for i in range(0, len(candidates), batch_size):
        batch = candidates[i : i + batch_size]
        src_texts = [item.text for _, item in batch]
        translated = translator.translate_batch(src_texts)
        for (div, item), out_text in zip(batch, translated):
            replace_div_text(div, f"{item.leading_ws}{out_text}{item.trailing_ws}")
            translated_blocks += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(str(soup), encoding="utf-8")

    return {
        "file": str(input_path),
        "output": str(output_path),
        "status": "ok",
        "translated_blocks": translated_blocks,
    }


def discover_html_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(p for p in input_path.rglob("*.html") if p.is_file())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate Messenger-exported HTML files from Russian to English using a local llama.cpp server."
    )
    parser.add_argument("input", type=Path, help="Input HTML file or directory")
    parser.add_argument("output", type=Path, nargs="?", help="Output HTML file or directory")
    parser.add_argument("--model", default="ggml-org/gemma-3-1b-it-GGUF", help="Model name (e.g., from llama-server -hf)")
    parser.add_argument("--api-url", default=LLAMA_CPP_DEFAULT_URL, help="llama.cpp v1/chat/completions URL")
    parser.add_argument("--batch-size", type=int, default=12, help="Messages per LLM request")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout in seconds")
    parser.add_argument(
        "--force-all",
        action="store_true",
        help="Translate all <div class=\"text\"> blocks, even if they do not contain Cyrillic",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Display model request payload and answers",
    )
    return parser.parse_args()


def map_output_path(input_root: Path, output_root: Path, current_file: Path) -> Path:
    if input_root.is_file():
        return output_root
    relative = current_file.relative_to(input_root)
    return output_root / relative


def main() -> int:
    args = parse_args()

    input_path: Path = args.input
    output_path: Path = args.output

    if not input_path.exists():
        print(f"Input path does not exist: {input_path}", file=sys.stderr)
        return 2

    if output_path is None:
        if input_path.is_file():
            # e.g., messages.html -> messages.translated.html
            output_path = input_path.with_suffix(".translated" + input_path.suffix)
        else:
            # e.g., test/ -> test.translated/
            output_path = input_path.with_name(input_path.name + ".translated")

    files = discover_html_files(input_path)
    if not files:
        print("No HTML files found.", file=sys.stderr)
        return 2

    translator = LlamaCppTranslator(
        model=args.model,
        api_url=args.api_url,
        timeout=args.timeout,
        debug=args.debug,
    )

    overall_translated = 0

    for file_path in files:
        target = map_output_path(input_path, output_path, file_path)
        result = translate_html_file(
            input_path=file_path,
            output_path=target,
            translator=translator,
            batch_size=args.batch_size,
            force_all=args.force_all,
            overwrite=args.overwrite,
        )
        overall_translated += result["translated_blocks"]
        print(
            f"[{result['status']}] {result['file']} -> {result['output']} | "
            f"translated={result['translated_blocks']}"
        )

    print(f"Done. Translated blocks: {overall_translated}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
