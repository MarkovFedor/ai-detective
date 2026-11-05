"""
stress_retry.py — точечный перезапуск пропущенных записей стресс-теста.

Читает существующий results/stress_deletion.jsonl, находит, каких
(question_id, mode) не хватает, и прогоняет только их. Дописывает в файл,
не перезаписывая уже полученные ответы.

Запуск:
    python -m src.stress_retry --test 2
    python -m src.stress_retry --test 2 --question Q05 --mode C
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.context_builder import load_dossier, remove_claim
from src.llm import build_prompt, call_llm, load_prompt_template, make_client
from src.stress import (
    DEFAULT_MODEL,
    MODES,
    RESULTS,
    _pick_claim_to_remove,
    build_context,
)

TEST_FILES = {
    "1": RESULTS / "stress_permutation.jsonl",
    "2": RESULTS / "stress_deletion.jsonl",
    "3": RESULTS / "stress_contradiction.jsonl",
}


def load_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def existing_keys(records: list[dict]) -> set[tuple[str, str]]:
    return {(r["question_id"], r["mode"]) for r in records}


def run_one(
    q: dict,
    mode: str,
    documents: list[dict],
    claims: list[dict],
    model: str,
    removed_claim: str | None = None,
) -> dict:
    template = load_prompt_template()
    client = make_client()
    pack = build_context(mode, q["question"], documents, claims)
    prompt = build_prompt(template, q["question"], pack)
    ans, meta = call_llm(client, model, prompt, pack)

    rec = {
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
    if removed_claim is not None:
        rec["removed_claim"] = removed_claim
        rec["expected_status"] = "insufficient"
    return rec


def retry_deletion(
    question_id: str | None = None,
    mode: str | None = None,
    model: str | None = None,
) -> None:
    """Дозаписывает пропущенные (question_id, mode) в stress_deletion.jsonl."""
    path = TEST_FILES["2"]
    existing = load_existing(path)
    done = existing_keys(existing)

    documents, claims, questions = load_dossier()
    targets = [q for q in questions if q["qtype"] in ("multi_hop", "temporal")]
    if question_id is not None:
        targets = [q for q in targets if q["question_id"] == question_id]
    modes = (mode,) if mode else MODES

    # Строим список пропущенных пар
    to_run = []
    for q in targets:
        cid = _pick_claim_to_remove(q, claims)
        if cid is None:
            continue
        for m in modes:
            if (q["question_id"], m) in done:
                continue
            to_run.append((q, m, cid))

    if not to_run:
        print("Все записи уже есть, дозаписывать нечего.")
        return

    print(f"Будет дозаписано {len(to_run)} записей:")
    for q, m, cid in to_run:
        print(f"  {q['question_id']} {m} (removed_claim={cid})")

    model = model or os.environ.get("LLM_MODEL", DEFAULT_MODEL)
    new_records = []
    for q, m, cid in to_run:
        new_docs, new_claims, side_effects = remove_claim(documents, claims, cid)
        if side_effects:
            print(f"  {q['question_id']} {m}: удаление {cid} задело {side_effects}, пропуск")
            continue
        rec = run_one(q, m, new_docs, new_claims, model, removed_claim=cid)
        status = "OK" if rec["error"] is None else f"ERR ({rec['error']})"
        print(f"  {q['question_id']} {m}: {status}")
        new_records.append(rec)

    # Дописываем в файл (append, не перезапись)
    with path.open("a", encoding="utf-8") as f:
        for r in new_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[appended] {path} (+{len(new_records)} записей)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Дозапись пропущенных записей стресс-теста")
    ap.add_argument("--test", choices=["2"], required=True,
                    help="Пока поддерживается только тест 2 (deletion)")
    ap.add_argument("--question", default=None, help="Только этот question_id")
    ap.add_argument("--mode", default=None, choices=list(MODES), help="Только этот режим")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    if args.test == "2":
        retry_deletion(question_id=args.question, mode=args.mode, model=args.model)


if __name__ == "__main__":
    main()