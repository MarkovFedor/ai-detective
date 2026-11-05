"""
llm.py — участник 3: вызов LLM для режимов A, B, C.

Провайдер по умолчанию — OpenRouter (бесплатные модели, OpenAI-совместимый API).
Альтернативы: Groq, Gemini (через OpenAI-совместимый endpoint), Ollama (локально).

Переменные окружения:
    LLM_API_KEY     — ключ провайдера (обязательно)
    LLM_BASE_URL    — base_url (по умолчанию OpenRouter)
    LLM_MODEL       — имя модели (по умолчанию бесплатная Llama 3.3 70B)

Примеры:
    # OpenRouter
    $env:LLM_API_KEY = "sk-or-v1-..."
    $env:LLM_BASE_URL = "https://openrouter.ai/api/v1"
    $env:LLM_MODEL = "meta-llama/llama-3.3-70b-instruct:free"

    # Groq
    $env:LLM_API_KEY = "gsk_..."
    $env:LLM_BASE_URL = "https://api.groq.com/openai/v1"
    $env:LLM_MODEL = "llama-3.3-70b-versatile"

    # Gemini через OpenAI-совместимый endpoint
    $env:LLM_API_KEY = "AIza..."
    $env:LLM_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
    $env:LLM_MODEL = "gemini-2.5-flash"

    # Ollama (локально)
    $env:LLM_API_KEY = "ollama"
    $env:LLM_BASE_URL = "http://localhost:11434/v1"
    $env:LLM_MODEL = "qwen2.5:7b"

Запуск:
    python -m src.llm --mode A --out results/A.jsonl
    python -m src.llm --mode B --out results/B.jsonl
    python -m src.llm --mode C --out results/C.jsonl


"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from openai import OpenAI
from pydantic import ValidationError

from src.context_builder import ContextPack, build_context, load_dossier
from src.schemas import ModelAnswer, parse_model_answer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PROMPTS = ROOT / "prompts"
RESULTS = ROOT / "results"

# Значения по умолчанию: OpenRouter, бесплатная Llama 3.3 70B
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openrouter/free"


def load_prompt_template() -> str:
    """Читает prompts/system.txt. Проверяет наличие плейсхолдеров."""
    path = PROMPTS / "system.txt"
    if not path.exists():
        raise FileNotFoundError(f"Нет файла промпта: {path}")
    text = path.read_text(encoding="utf-8")
    if "{question}" not in text or "{context}" not in text:
        raise ValueError("В промпте должны быть {question} и {context}")
    return text


def build_prompt(template: str, question: str, pack: ContextPack) -> str:
    """Подставляет вопрос и контекст в шаблон."""
    return template.replace("{question}", question).replace("{context}", pack.text)


def make_client() -> OpenAI:
    """Создаёт OpenAI-совместимый клиент из переменных окружения."""
    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Не задан LLM_API_KEY. Установите переменную окружения:\n"
            '  $env:LLM_API_KEY = "sk-or-v1-..."'
        )
    base_url = os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL)
    return OpenAI(api_key=api_key, base_url=base_url)


def call_llm(
    client: OpenAI,
    model: str,
    prompt: str,
    pack: ContextPack,
    max_retries: int = 2,
) -> tuple[ModelAnswer | None, dict[str, Any]]:
    """
    Один вызов LLM с retry при невалидном JSON.

    Возвращает (ответ, метаданные). При ошибке — (None, {"error": ...}).
    """
    meta: dict[str, Any] = {
        "tokens_in": None,
        "tokens_out": None,
        "tokens_total": None,
        "time_sec": None,
        "raw": None,
        "error": None,
    }

    last_error: str | None = None
    for attempt in range(1, max_retries + 2):  # 1 попытка + max_retries
        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            meta["time_sec"] = round(time.time() - t0, 3)
            meta["raw"] = resp.choices[0].message.content

            if resp.usage:
                meta["tokens_in"] = resp.usage.prompt_tokens
                meta["tokens_out"] = resp.usage.completion_tokens
                meta["tokens_total"] = resp.usage.total_tokens

            obj = json.loads(meta["raw"])
            ans = parse_model_answer(obj, pack.claim_ids)
            meta["error"] = None
            return ans, meta

        except json.JSONDecodeError as e:
            last_error = f"JSONDecodeError: {e}"
            meta["error"] = last_error
            if attempt <= max_retries:
                time.sleep(1.0 * attempt)
                continue
            return None, meta

        except (ValidationError, ValueError) as e:
            last_error = f"{type(e).__name__}: {e}"
            meta["error"] = last_error
            if attempt <= max_retries:
                time.sleep(1.0 * attempt)
                continue
            return None, meta

        except Exception as e:
            # Сетевые ошибки, 401, 429 и т.п. — retry с бэкоффом
            last_error = f"{type(e).__name__}: {e}"
            meta["error"] = last_error
            if attempt <= max_retries:
                time.sleep(2.0 ** attempt)
                continue
            return None, meta

    # Сюда не должны попадать, но на всякий случай
    meta["error"] = last_error or "unknown"
    return None, meta


def run_mode(
    mode: str,
    out_path: Path,
    model: str | None = None,
    data_dir: Path = DATA,
    limit: int | None = None,
) -> None:
    """Прогоняет один режим по всем вопросам и пишет JSONL."""
    template = load_prompt_template()
    client = make_client()
    model = model or os.environ.get("LLM_MODEL", DEFAULT_MODEL)

    documents, claims, questions = load_dossier(data_dir)
    if limit is not None:
        questions = questions[:limit]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(questions)

    with out_path.open("w", encoding="utf-8") as f:
        for i, q in enumerate(questions, start=1):
            pack = build_context(mode, q["question"], documents, claims)
            prompt = build_prompt(template, q["question"], pack)
            ans, meta = call_llm(client, model, prompt, pack)

            record: dict[str, Any] = {
                "question_id": q["question_id"],
                "mode": mode,
                "status": ans.status if ans else None,
                "answer": ans.answer if ans else None,
                "evidence_ids": ans.evidence_ids if ans else [],
                "counterevidence_ids": ans.counterevidence_ids if ans else [],
                "confidence": ans.confidence if ans else None,
                "explanation": ans.explanation if ans else None,
                **meta,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            status = "OK" if meta["error"] is None else f"ERR ({meta['error']})"
            print(f"[{i}/{total}] {q['question_id']} {status}")

    print(f"\nГотово: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Вызов LLM в режимах A/B/C")
    ap.add_argument("--mode", choices=["A", "B", "C"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=None,
                    help="Переопределить LLM_MODEL из окружения")
    ap.add_argument("--limit", type=int, default=None,
                    help="Прогнать только первые N вопросов (для отладки)")
    args = ap.parse_args()

    run_mode(args.mode, args.out, model=args.model, limit=args.limit)


if __name__ == "__main__":
    main()
