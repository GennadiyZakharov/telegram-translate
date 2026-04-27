#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests
from bs4 import BeautifulSoup

LLAMA_CPP_DEFAULT_URL = "http://localhost:8000/v1/chat/completions"
DEFAULT_SYSTEM_PROMPT = """You are a precise translator for Telegram chat history.
Your goal is to translate messages from Russian to English.

Rules:
1. Each 'translated_text' must be the English translation of the corresponding Russian 'text'.
2. Maintain the same 'id' for each message."
3. If the message is already in English, or is just a link/emoji/number, return it as is.
4. NEVER return the 'speaker' name as the translation.
5. NEVER explain your work or add any text outside the JSON structure.
6. Strictly follow the provided JSON schema.
7. Preserve the original tone and casual style (emojis, punctuation).
"""

NOT_TRANSLATED_MESSAGE = "Not translated - error"

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
    """A translator class that uses a llama.cpp server's OpenAI-compatible API."""

    def __init__(
        self,
        model: str,
        api_url: str = LLAMA_CPP_DEFAULT_URL,
        timeout: int = 300,
        temperature: float = 0.0,
        retries: int = 3,
        debug: bool = False,
    ) -> None:
        self.model = model
        self.api_url = api_url
        self.timeout = timeout
        self.temperature = temperature
        self.retries = retries
        self.debug = debug

    def list_models(self) -> List[str]:
        """Fetch the list of available models from the server."""
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
        """Translate a batch of messages, maintaining conversation context."""
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


        results: List[str] = [""] * len(texts)
        attempt = 0

        while attempt < self.retries:
            # Identify indices that still need translation
            pending_indices = [i for i, res in enumerate(results) if res == ""]
            if not pending_indices: # Everything has been translated
                break

            # Running translation of the full batch to maintain a consistent dialog
            numbered = [{"id": i, "speaker": p, "text": t} for i, (p, t) in enumerate(zip(speakers, texts))]
            user_prompt = (
                "Translate the following chat messages from Russian to English.\n\n"              
                f"Schema:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
                f"Messages to translate:\n{json.dumps(numbered, ensure_ascii=False, indent=2)}"
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

            try:
                if self.debug:
                    print(f"Running translation for {len(texts)} messages (attempt {attempt + 1}/{self.retries}) with model {self.model}...")
                
                response = requests.post(self.api_url, json=payload, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                items = parsed["translations"]
                
                new_translations_count = 0
                for item in items:
                    idx = item.get("id")
                    if idx is None:
                        continue
                    if results[idx] != "":  # We already have this message translated, skip it
                        continue
                    if not (0 <= idx < len(texts)):
                        print(f"Warning: Unexpected translation id returned: {idx}. Message: {item.get('translated_text')}")
                        continue
                    translated_text = item.get("translated_text", "")
                    
                    # Validation: check if LLM returned Cyrillic
                    if CYRILLIC_RE.search(translated_text):
                        if self.debug:
                            print(f"Non-latin character detected in translation for id {idx}: {translated_text}")
                        continue
                    
                    # Validation: check if LLM just returned the speaker's name -
                    if translated_text.strip().lower() == speakers[idx].strip().lower() and translated_text.strip() != "":
                         if self.debug:
                            print(f"LLM returned speaker name instead of translation for id {idx}: {translated_text}")
                         continue

                    results[idx] = translated_text
                    new_translations_count += 1

                if self.debug:
                    print(f"Successfully translated {new_translations_count} messages in this attempt.")
                
                # If we translated everything we asked for, we are done
                if new_translations_count == len(pending_indices):
                    break
                
                # If we got some but not all, we'll retry the rest in the next iteration
                attempt += 1

            except (requests.RequestException, KeyError, IndexError, ValueError, json.JSONDecodeError) as exc:
                print(f"Error during translation attempt {attempt + 1}/{self.retries}: {exc}")
                attempt += 1
                if attempt < self.retries:
                    time.sleep(min(2**attempt, 8))
                continue

        if self.debug:
            print("Debug - translated messages:")
            for idx, (orig, trans, pers) in enumerate(zip(texts, results, speakers)):
                print(f"id: {idx} (person: {pers})")
                print(f"{orig}\n{trans}\n")
            print("----------------------------------------------------------------")

        return results


def extract_inner_text(raw_text: str) -> Tuple[str, str, str]:
    """Extract leading whitespace, core text, and trailing whitespace."""
    match = LEAD_TRAIL_WS_RE.match(raw_text)
    if not match:
        return "", raw_text, ""
    return match.group(1), match.group(2), match.group(3)


def needs_translation(text: str, force_all: bool = False) -> bool:
    """Check if the text needs translation (contains Cyrillic or force_all is True)."""
    stripped = text.strip()
    if not stripped:
        return False
    if force_all:
        return True
    return bool(CYRILLIC_RE.search(stripped))


def replace_div_text(div: Any, new_text: str) -> None:
    """Replace the text content of a BeautifulSoup tag."""
    div.clear()
    div.append(new_text)


def translate_html_file(
    input_path: Path,
    output_path: Path,
    translator: LlamaCppTranslator,
    batch_size: int,
    force_all: bool,
    overwrite: bool,
) -> Dict[str, Any]:
    """Translate all eligible text blocks in an HTML file."""
    if output_path.exists() and not overwrite:
        return {
            "file": str(input_path),
            "output": str(output_path),
            "status": "skipped_existing",
            "translated_blocks": 0,
        }

    html = input_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")

    candidates: List[Tuple[Any, MessageItem]] = []
    translated_blocks = 0
    current_speaker = ""

    # Try to find header name
    header_name_div = soup.find("div", class_="page_header").find("div", class_="text bold") if soup.find("div", class_="page_header") else None
    if header_name_div:
        current_speaker = transliterate(header_name_div.get_text(strip=True))

    # Iterate through all divs to find names and text
    for idx, div in enumerate(soup.find_all("div")):
        classes = div.get("class", [])
        if not isinstance(classes, list):
            classes = [classes] if classes else []
        if "from_name" in classes:
            name_text = div.get_text(strip=True)
            current_speaker = transliterate(name_text)
            continue

        if "text" in classes:
            # We use get_text with separator="" and strip=False to preserve whitespace between tags
            # but BeautifulSoup's get_text doesn't actually take a separator as a positional argument in all versions, 
            # and separator="" is default in some contexts.
            raw_text = div.get_text(strip=False)
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


def discover_html_files(input_path: Path) -> List[Path]:
    """Find all HTML files in the given path (file or directory)."""
    if input_path.is_file():
        return [input_path]
    return sorted(p for p in input_path.rglob("*.html") if p.is_file())


def parse_args(return_parser: bool = False) -> argparse.Namespace | argparse.ArgumentParser:
    """Parse command-line arguments."""
    # Try to fetch available models for the help message
    available_models_str = ""
    models_url = LLAMA_CPP_DEFAULT_URL.replace("/chat/completions", "/models")
    try:
        # We don't have the translator yet, but we can do a quick request
        # Default URL is LLAMA_CPP_DEFAULT_URL
        response = requests.get(models_url, timeout=2)
        if response.status_code == 200:
            data = response.json()
            models = [m["id"] for m in data.get("data", [])]
            if models:
                available_models_str = "\n\nAvailable models on the server:\n" + "\n".join(f"  - {m}" for m in models)
    except Exception:
        pass

    if available_models_str == "":
        available_models_str = f"\n\nWARNING: No available LLMs found on the server {models_url}"

    parser = argparse.ArgumentParser(
        description="Translate Messenger-exported HTML files from Russian to English using a local llama.cpp server." + available_models_str,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", type=Path, nargs="?", help="Input HTML file or directory")
    parser.add_argument("output", type=Path, nargs="?", help="Output HTML file or directory")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M", help="Model name (e.g., from llama-server -hf)")
    parser.add_argument("--api-url", default=LLAMA_CPP_DEFAULT_URL, help="llama.cpp v1/chat/completions URL")
    parser.add_argument("--batch-size", type=int, default=24, help="Messages per LLM request")
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
    if return_parser:
        return parser
    return parser.parse_args()


def map_output_path(input_root: Path, output_root: Path, current_file: Path) -> Path:
    """Map an input file path to its corresponding output path."""
    if input_root.is_file():
        return output_root
    relative = current_file.relative_to(input_root)
    return output_root / relative


def main() -> int:
    args = parse_args()

    input_path: Path | None = args.input
    output_path: Path | None = args.output

    if input_path is None:
        # We need to show help if no input provided
        parse_args(return_parser=True).print_help()
        print("\nError: the following arguments are required: input", file=sys.stderr)
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

    overall_translated = 0

    translator = LlamaCppTranslator(
        model=args.model,
        api_url=args.api_url,
        timeout=args.timeout,
        debug=args.debug,
    )

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
