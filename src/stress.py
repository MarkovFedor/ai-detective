"""
stress.py — участник 3: три стресс-теста из задания.

Тест 1. Перестановка документов (random_state=42).
Тест 2. Удаление посылки — для multi_hop и temporal вопросов.
Тест 3. Добавление противоречия — сценарии читаются из data/stress_scenarios.json.

Каждый тест меняет РОВНО ОДИН фактор. Исходные данные не модифицируются.

Запуск:
    python -m src.stress --test 1
    python -m src.stress --test 2
    python -m src.stress --test 3 --scenario concierge
    python -m src.stress --test 3 --scenario admin
    python -m src.stress --test 3 --scenario camera
    python -m src.stress --test all

Выход:
    results/stress_permutation.jsonl
    results/stress_deletion.jsonl
    results/stress_contradiction.jsonl
    results/stress_comparison.csv
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from src.context_builder import (
    add_claim,
    build_context,
    load_dossier,
    remove_claim,
    shuffle_documents,
)
from src.evaluation import evaluate_mode, load_gold, load_jsonl
from src.llm import build_prompt, call_llm, load_prompt_template, make_client

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
SCENARIOS_PATH = DATA / "stress_scenarios.json"
MODES = ("A", "B", "C")
DEFAULT_MODEL = os.environ.get("LLM_MODEL", "openrouter/free")



# Загрузка и валидация сценариев

REQUIRED_SCENARIO_FIELDS = {"title", "why", "expected_status", "question_ids", "document", "claim"}
REQUIRED_DOC_FIELDS = {"document_id", "title", "source_type", "created_at", "author", "text"}
REQUIRED_CLAIM_FIELDS = {"claim_id", "text", "document_id", "evidence", "status"}
VALID_STATUSES = {"entailed", "contradicted", "both", "insufficient"}


def load_scenarios(path: Path = SCENARIOS_PATH) -> dict[str, dict]:
    """Читает сценарии из JSON и валидирует структуру."""
    if not path.exists():
        raise FileNotFoundError(f"Нет файла сценариев: {path}")
    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    # Убираем служебные ключи вроде "_comment"
    scenarios = {k: v for k, v in data.items() if not k.startswith("_")}
    if not scenarios:
        raise ValueError(f"В {path} нет ни одного сценария")

    for name, spec in scenarios.items():
        missing = REQUIRED_SCENARIO_FIELDS - set(spec)
        if missing:
            raise ValueError(f"{name}: не хватает полей {missing}")
        missing = REQUIRED_DOC_FIELDS - set(spec["document"])
        if missing:
            raise ValueError(f"{name}.document: не хватает полей {missing}")
        missing = REQUIRED_CLAIM_FIELDS - set(spec["claim"])
        if missing:
            raise ValueError(f"{name}.claim: не хватает полей {missing}")
        if spec["expected_status"] not in VALID_STATUSES:
            raise ValueError(
                f"{name}: expected_status должен быть одним из {VALID_STATUSES}, "
                f"получено {spec['expected_status']!r}"
            )
        if not isinstance(spec["question_ids"], list) or not spec["question_ids"]:
            raise ValueError(f"{name}: question_ids должен быть непустым списком")

    return scenarios


def validate_scenarios_against_dossier(
    scenarios: dict[str, dict],
    documents: list[dict],
    claims: list[dict],
    questions: list[dict],
) -> None:
    """
    Проверяет, что:
      - document_id и claim_id не пересекаются с существующими;
      - question_ids существуют в questions.jsonl;
      - evidence дословно входит в текст нового документа.
    """
    doc_ids = {d["document_id"] for d in documents}
    claim_ids = {c["claim_id"] for c in claims}
    qids = {q["question_id"] for q in questions}

    for name, spec in scenarios.items():
        new_doc = spec["document"]
        new_claim = spec["claim"]

        if new_doc["document_id"] in doc_ids:
            raise ValueError(
                f"{name}: document_id {new_doc['document_id']} уже есть в досье"
            )
        if new_claim["claim_id"] in claim_ids:
            raise ValueError(
                f"{name}: claim_id {new_claim['claim_id']} уже есть в claims"
            )
        if new_claim["document_id"] != new_doc["document_id"]:
            raise ValueError(
                f"{name}: claim.document_id != document.document_id"
            )
        unknown_q = set(spec["question_ids"]) - qids
        if unknown_q:
            raise ValueError(f"{name}: вопросы {unknown_q} отсутствуют в questions.jsonl")
        if new_claim["evidence"] not in new_doc["text"]:
            raise ValueError(
                f"{name}: evidence не входит дословно в текст документа"
            )



# Общий запуск тестов

def run_questions(
    documents: list[dict],
    claims: list[dict],
    questions: list[dict],
    modes: tuple[str, ...] = MODES,
    model: str | None = None,
) -> list[dict]:
    template = load_prompt_template()
    client = make_client()
    model = model or DEFAULT_MODEL

    records = []
    total = len(questions) * len(modes)
    n = 0
    for q in questions:
        for mode in modes:
            n += 1
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
            records.append(rec)
            status = "OK" if meta["error"] is None else f"ERR ({meta['error']})"
            print(f"[{n}/{total}] {q['question_id']} {mode} {status}", flush=True)

    return records


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[saved] {path}")



# Тест 1. Перестановка документов

def test_permutation(model: str | None = None) -> list[dict]:
    documents, claims, questions = load_dossier()
    shuffled = shuffle_documents(documents, random_state=42)
    print(f"\nТест 1. Перестановка документов")
    print(f"Порядок до:    {[d['document_id'] for d in documents]}")
    print(f"Порядок после: {[d['document_id'] for d in shuffled]}")

    records = run_questions(shuffled, claims, questions, model=model)
    save_jsonl(records, RESULTS / "stress_permutation.jsonl")
    return records



# Тест 2. Удаление посылки

def _pick_claim_to_remove(q: dict, claims: list[dict]) -> str | None:
    gold_ev = q.get("gold_evidence_ids") or []
    claims_by_id = {c["claim_id"]: c for c in claims}
    for cid in gold_ev:
        if cid in claims_by_id:
            return cid
    return None


def test_deletion(model: str | None = None) -> list[dict]:
    documents, claims, questions = load_dossier()
    targets = [q for q in questions if q["qtype"] in ("multi_hop", "temporal")]
    print(f"\nТест 2. Удаление посылки")
    print(f"Вопросов для теста: {len(targets)}")

    records = []
    for q in targets:
        cid = _pick_claim_to_remove(q, claims)
        if cid is None:
            print(f"  {q['question_id']}: нет gold evidence, пропуск")
            continue
        new_docs, new_claims, side_effects = remove_claim(documents, claims, cid)
        if side_effects:
            print(f"  {q['question_id']}: удаление {cid} задело {side_effects}, пропуск")
            continue
        print(f"  {q['question_id']}: удалён {cid}")
        recs = run_questions(new_docs, new_claims, [q], model=model)
        for r in recs:
            r["removed_claim"] = cid
            r["expected_status"] = "insufficient"
        records.extend(recs)

    save_jsonl(records, RESULTS / "stress_deletion.jsonl")
    return records



# Тест 3. Добавление противоречия

def test_contradiction(
    scenario: str,
    scenarios: dict[str, dict] | None = None,
    model: str | None = None,
) -> list[dict]:
    scenarios = scenarios or load_scenarios()
    if scenario not in scenarios:
        raise ValueError(
            f"Неизвестный сценарий: {scenario}. Доступны: {sorted(scenarios)}"
        )
    spec = scenarios[scenario]

    documents, claims, questions = load_dossier()
    validate_scenarios_against_dossier(
        {scenario: spec}, documents, claims, questions
    )

    new_docs, new_claims = add_claim(
        documents, claims,
        new_claim=spec["claim"],
        new_document=spec["document"],
    )
    targets = [q for q in questions if q["question_id"] in spec["question_ids"]]

    print(f"\nТест 3. Добавление противоречия: {scenario}")
    print(f"Причина: {spec['why']}")
    print(f"Ожидаемый статус: {spec['expected_status']}")
    print(f"Затронутые вопросы: {spec['question_ids']}")

    records = run_questions(new_docs, new_claims, targets, model=model)
    for r in records:
        r["scenario"] = scenario
        r["added_claim"] = spec["claim"]["claim_id"]
        r["expected_status"] = spec["expected_status"]

    save_jsonl(records, RESULTS / "stress_contradiction.jsonl")
    return records



# Сборка итоговой таблицы

def build_comparison_table(
    baseline_records: dict[str, dict],
    stress_records: list[dict],
    gold: dict[str, dict],
    test_name: str,
) -> pd.DataFrame:
    rows = []
    for mode in MODES:
        base_pred = baseline_records.get(mode, {})
        stress_pred = {r["question_id"]: r for r in stress_records if r["mode"] == mode}
        common_qids = set(base_pred) & set(stress_pred)
        if not common_qids:
            continue
        base_subset = {q: base_pred[q] for q in common_qids}
        stress_subset = {q: stress_pred[q] for q in common_qids}

        m_base = evaluate_mode({q: gold[q] for q in common_qids}, base_subset)
        m_stress = evaluate_mode({q: gold[q] for q in common_qids}, stress_subset)

        changed_status = sum(
            1 for q in common_qids
            if base_subset[q].get("status") != stress_subset[q].get("status")
        )
        changed_evidence = sum(
            1 for q in common_qids
            if set(base_subset[q].get("evidence_ids") or [])
            != set(stress_subset[q].get("evidence_ids") or [])
        )

        rows.append({
            "test": test_name,
            "mode": mode,
            "n_questions": len(common_qids),
            "status_accuracy_base": m_base["status_accuracy"],
            "status_accuracy_stress": m_stress["status_accuracy"],
            "evidence_f1_base": m_base["evidence_f1"],
            "evidence_f1_stress": m_stress["evidence_f1"],
            "changed_status": changed_status,
            "changed_evidence": changed_evidence,
        })

    return pd.DataFrame(rows)


def load_baseline() -> dict[str, dict]:
    baseline = {}
    for mode in MODES:
        path = RESULTS / f"{mode}.jsonl"
        if not path.exists():
            print(f"[warn] нет базовой линии {path}, пропуск")
            continue
        baseline[mode] = {r["question_id"]: r for r in load_jsonl(path)}
    return baseline



def main() -> None:
    ap = argparse.ArgumentParser(description="Три стресс-теста")
    ap.add_argument("--test", choices=["1", "2", "3", "all"], required=True)
    ap.add_argument("--scenario", default="concierge",
                    help="Имя сценария для теста 3 (см. data/stress_scenarios.json)")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    scenarios = load_scenarios()
    gold = load_gold()
    baseline = load_baseline()
    all_tables = []

    if args.test in ("1", "all"):
        recs = test_permutation(model=args.model)
        if baseline:
            all_tables.append(build_comparison_table(baseline, recs, gold, "permutation"))

    if args.test in ("2", "all"):
        recs = test_deletion(model=args.model)
        if baseline:
            all_tables.append(build_comparison_table(baseline, recs, gold, "deletion"))

    if args.test in ("3", "all"):
        recs = test_contradiction(args.scenario, scenarios=scenarios, model=args.model)
        if baseline:
            all_tables.append(build_comparison_table(baseline, recs, gold, "contradiction"))

    if all_tables:
        combined = pd.concat(all_tables, ignore_index=True)
        out = RESULTS / "stress_comparison.csv"
        combined.to_csv(out, index=False)
        print(f"\nSaved {out}")
        print(combined.to_string(index=False))


if __name__ == "__main__":
    main()