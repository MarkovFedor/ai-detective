import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PROMPT = ROOT / "prompts" / "system.txt"


def test_prompt_does_not_leak_gold():
    qs = [json.loads(l) for l in (DATA / "questions.jsonl").open(encoding="utf-8")]
    gold_qids = {q["question_id"] for q in qs}
    gold_claims = set()
    for q in qs:
        gold_claims |= set(q["gold_evidence_ids"]) | set(q["gold_counterevidence_ids"])

    text = PROMPT.read_text(encoding="utf-8")
    example_qids = set(re.findall(r'"question_id": "(Q\d+)"', text))
    example_claims = set(re.findall(r"\bC\d{2}\b", text))

    assert not (gold_qids & example_qids), \
        f"эталонные qid в промпте: {sorted(gold_qids & example_qids)}"
    assert not (gold_claims & example_claims), \
        f"эталонные claims в промпте: {sorted(gold_claims & example_claims)}"


def test_prompt_has_all_statuses():
    text = PROMPT.read_text(encoding="utf-8")
    for status in ("entailed", "contradicted", "both", "insufficient"):
        assert f'"status": "{status}"' in text, f"в промпте нет примера {status}"


def test_prompt_has_placeholders():
    text = PROMPT.read_text(encoding="utf-8")
    assert "{question}" in text
    assert "{context}" in text