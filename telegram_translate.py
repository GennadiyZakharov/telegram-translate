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
DEFAULT_SYSTEM_PROMPT = """You are a literary translator, translating a history of chat between two users.
Translate chat messages from Russian into English.
Rules:
- Return ENGLISH translation of the original message text.
- If a message is already in English or does not need translation, return it unchanged.
- Do not add explanations.
- Return JSON strictly matching this schema.
- Preserve the original meaning and tone.
- Keep emojis, repeated punctuation, and casual style where sensible.
- If emoji contain non-latin symbols, substitute it by corresponding latin symbols. 
"""

CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
LEAD_TRAIL_WS_RE = re.compile(r"^(\s*)(.*?)(\s*)$", re.DOTALL)

TRANSLIT_MAP = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "yo", "ж": "zh",
    "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o",
    "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "kh", "ц": "ts",
    "ч": "ch", "ш": "sh", "щ": "shch", "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu",
    "я": "ya",
    "А": "A", "Б": "B", "В": "V", "Г": "G", "Д": "D", "Е": "E", "Ё": "Yo", "Ж": "Zh",
    "З": "Z", "И": "I", "Й": "Y", "К": "K", "Л": "L", "М": "M", "Н": "N", "О": "O",
    "П": "P", "Р": "R", "С": "S", "Т": "T", "У": "U", "Ф": "F", "Х": "Kh", "Ц": "Ts",
    "Ч": "Ch", "Ш": "Sh", "Щ": "Shch", "Ъ": "", "Ы": "Y", "Ь": "", "Э": "E", "Ю": "Yu",
    "Я": "Ya",
}


def transliterate(text: str) -> str:
    return "".join(TRANSLIT_MAP.get(c, c) for c in text)


@dataclass(slots=True)
class MessageItem:
    key: str
    text: str
    leading_ws: str
    trailing_ws: str
    speaker: str = ""


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

    def list_models(self) -> List[str]:
        # Typical OpenAI-compatible /v1/models endpoint
        # The api_url is /v1/chat/completions, so models URL should be /v1/models
        models_url = self.api_url.replace("/chat/completions", "/models")
        try:
            if self.debug:
                print(f"Querying models from {models_url}...")
            response = requests.get(models_url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            # /v1/models returns data which is a list of model objects
            models = [m["id"] for m in data.get("data", [])]
            if not models:
                print("No LLMs are available on the server.")
                return []
            return models
        except requests.RequestException as exc:
            print(f"Error: Server is not responding or returned an error: {exc}")
            return []
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            print(f"Error parsing server response: {exc}")
            return []

    def translate_batch(self, texts: List[str], speakers: List[str]) -> List[str]:
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

        numbered = [{"id": i, "speaker": p, "text": t} for i, (p, t) in enumerate(zip(speakers, texts))]
        user_prompt = (
            "Translate the following chat messages from Russian to English. "
            "Use the 'speaker' field to understand who is typing the message and maintain consistent translation."
            "Keep the same number of items and the same ids.\n\n"
            "Do not put speaker name in the translation. Use it only to maintain context.\n\n"
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
        attempt = 0
        result: List[str] = ["Not translated - error"] * len(texts)
        while attempt < self.retries:
            try:
                if self.debug:
                    print("Running translation for a message batch of size", len(texts), "with model", self.model, "attempt", attempt,"...")
                response = requests.post(self.api_url, json=payload, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                items = parsed["translations"]
                result = ["Not translated - no matching LLM response"]*len(texts)
                for item in items:
                    idx = item["id"]
                    if not (0 <= idx < len(texts)):
                        print(f"Warning: Unexpected translation id returned: {idx}. Message: {item['translated_text']}")
                        continue
                    translated_text = item["translated_text"]
                    if CYRILLIC_RE.search(translated_text):
                        raise ValueError(f"Non-latin character detected in translation for id {idx}: {translated_text}")
                    result[idx] = translated_text

                if self.debug:
                    print("Debug - side-by-side:")
                    for idx, (orig, trans, pers) in enumerate(zip(texts, result, speakers)):
                        print(f"id: {idx} (person: {pers})")
                        print(f"{orig}\n{trans}\n")
                    print("----------------------------------------------------------------")

                return result  # type: ignore[return-value]

            except (requests.RequestException, KeyError, IndexError, ValueError, json.JSONDecodeError) as exc:
                print(f"Error during translation attempt {attempt + 1}/{self.retries}: {exc}")
                attempt += 1
                time.sleep(min(2**attempt, 8))
                continue

        return result


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

    candidates: list[tuple[object, MessageItem]] = []
    translated_blocks = 0
    current_speaker = ""

    # Try to find header name
    header_name_div = soup.find("div", class_="page_header").find("div", class_="text bold") if soup.find("div", class_="page_header") else None
    if header_name_div:
        current_speaker = transliterate(header_name_div.get_text(strip=True))

    # Iterate through all divs to find names and text
    for idx, div in enumerate(soup.find_all("div")):
        classes = div.get("class", [])
        if "from_name" in classes:
            name_text = div.get_text(strip=True)
            current_speaker = transliterate(name_text)
            continue

        if "text" in classes:
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
                        speaker=current_speaker,
                    ),
                )
            )

    for i in range(0, len(candidates), batch_size):
        batch = candidates[i : i + batch_size]
        src_texts = [item.text for _, item in batch]
        src_speakers = [item.speaker for _, item in batch]
        print(f"Translating batch {i} with {len(src_texts)} messages...")
        translated = translator.translate_batch(src_texts, src_speakers)
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
    parser.add_argument("input", type=Path, nargs="?", help="Input HTML file or directory")
    parser.add_argument("output", type=Path, nargs="?", help="Output HTML file or directory")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M", help="Model name (e.g., from llama-server -hf)")
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
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Query the server for available models and exit",
    )
    return parser.parse_args()


def map_output_path(input_root: Path, output_root: Path, current_file: Path) -> Path:
    if input_root.is_file():
        return output_root
    relative = current_file.relative_to(input_root)
    return output_root / relative


def main() -> int:
    args = parse_args()

    translator = LlamaCppTranslator(
        model=args.model,
        api_url=args.api_url,
        timeout=args.timeout,
        debug=args.debug,
    )

    if args.list_models:
        models = translator.list_models()
        if models:
            print("Available models:")
            for m in models:
                print(f" - {m}")
            return 0
        return 1

    input_path: Path | None = args.input
    output_path: Path | None = args.output

    if input_path is None:
        parser = argparse.ArgumentParser(
            description="Translate Messenger-exported HTML files from Russian to English using a local llama.cpp server."
        )
        # We need to re-parse or just show help if no input provided and not list-models
        print("Error: the following arguments are required: input", file=sys.stderr)
        return 2

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

    # Check if server is running and model is available
    available_models = translator.list_models()
    if not available_models:
        print("Error: Could not connect to the LLM server or no models available.", file=sys.stderr)
        return 1
    
    if args.model not in available_models:
        print(f"Error: Requested model '{args.model}' is not available on the server.", file=sys.stderr)
        print(f"Available models: {', '.join(available_models)}", file=sys.stderr)
        return 1

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
