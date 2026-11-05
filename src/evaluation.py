"""
evaluation.py — участник 3: метрики по трём режимам.

Читает:
    data/questions.jsonl         — эталоны (gold_status, gold_evidence_ids, ...)
    results/{A,B,C}.jsonl        — ответы LLM (см. src/llm.py)

Считает (раздел 10 задания):
    — accuracy логического статуса;
    — evidence precision, recall, F1;
    — counterevidence recall;
    — точность распознавания both;
    — точность распознавания insufficient;
    — доля уверенных ответов при недостатке информации
      (confidence > 0.7 при gold_status = insufficient);
    — число токенов и время ответа.

Сохраняет:
    results/comparison.csv       — таблица A | B | C
    results/per_question.csv     — детализация по вопросам

Запуск:
    python -m src.evaluation
    python -m src.evaluation --results results --out results/comparison.csv
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

MODES = ("A", "B", "C")
MODE_NAMES = {"A": "Полный контекст",
              "B": "BM25 top-15", "C": "p-adic context"}

# Порог уверенности для метрики confident-on-insufficient
CONFIDENCE_THRESHOLD = 0.7


# Загрузка

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Нет файла: {path}")
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_gold(data_dir: Path = DATA) -> dict[str, dict]:
    rows = load_jsonl(data_dir / "questions.jsonl")
    return {r["question_id"]: r for r in rows}


def load_run(results_dir: Path, mode: str) -> dict[str, dict]:
    path = results_dir / f"{mode}.jsonl"
    rows = load_jsonl(path)
    return {r["question_id"]: r for r in rows}


# Метрики

def evidence_prf(pred_ids: list[str], gold_ids: list[str]) -> tuple[float, float, float]:
    """Precision, recall, F1 по множествам claim_id"""
    p, g = set(pred_ids or []), set(gold_ids or [])
    if not g and not p:
        return 1.0, 1.0, 1.0
    if not p or not g:
        return 0.0, 0.0, 0.0
    tp = len(p & g)
    precision = tp / len(p)
    recall = tp / len(g)
    f1 = 2 * precision * recall / \
        (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def counterevidence_recall(pred_ids: list[str], gold_ids: list[str]) -> float | None:
    """Recall по counterevidence_ids. None — если эталон не требует counterevidence"""
    g = set(gold_ids or [])
    if not g:
        return None
    p = set(pred_ids or [])
    return len(p & g) / len(g)


def _mean(xs: list[float | None]) -> float:
    xs = [x for x in xs if x is not None]
    return float(sum(xs) / len(xs)) if xs else 0.0


def evaluate_mode(gold: dict[str, dict], pred: dict[str, dict]) -> dict[str, Any]:
    """Рассчет всех метрик для одного режима"""
    qids = sorted(set(gold) & set(pred))
    missing = set(gold) - set(pred)

    status_correct = 0
    ev_precisions, ev_recalls, ev_f1s = [], [], []
    ce_recalls: list[float | None] = []

    both_total = both_correct = 0
    insuf_total = insuf_correct = 0
    insuf_confident = 0  # уверенные ответы при gold_status = insufficient

    tokens_total = 0
    time_total = 0.0
    tokens_count = 0
    time_count = 0

    per_question = []

    for qid in qids:
        g = gold[qid]
        p = pred[qid]

        gold_status = g["gold_status"]
        pred_status = p.get("status")
        is_correct_status = pred_status == gold_status
        status_correct += int(is_correct_status)

        # Evidence P/R/F1 — только если есть эталонные evidence
        gold_ev = g.get("gold_evidence_ids") or []
        if gold_ev:
            ep, er, ef = evidence_prf(p.get("evidence_ids") or [], gold_ev)
            ev_precisions.append(ep)
            ev_recalls.append(er)
            ev_f1s.append(ef)

        # Counterevidence recall — только для both
        gold_ce = g.get("gold_counterevidence_ids") or []
        ce_r = counterevidence_recall(
            p.get("counterevidence_ids") or [], gold_ce)
        if ce_r is not None:
            ce_recalls.append(ce_r)

        # Both / insufficient accuracy
        if gold_status == "both":
            both_total += 1
            both_correct += int(pred_status == "both")
        if gold_status == "insufficient":
            insuf_total += 1
            insuf_correct += int(pred_status == "insufficient")
            conf = p.get("confidence")
            if conf is not None and conf > CONFIDENCE_THRESHOLD:
                insuf_confident += 1

        # Токены и время
        if p.get("tokens_total") is not None:
            tokens_total += p["tokens_total"]
            tokens_count += 1
        if p.get("time_sec") is not None:
            time_total += p["time_sec"]
            time_count += 1

        per_question.append({
            "question_id": qid,
            "mode": p.get("mode"),
            "qtype": g.get("qtype"),
            "gold_status": gold_status,
            "pred_status": pred_status,
            "status_correct": is_correct_status,
            "gold_answer": g.get("gold_answer"),
            "pred_answer": p.get("answer"),
            "gold_evidence_ids": ",".join(gold_ev),
            "pred_evidence_ids": ",".join(p.get("evidence_ids") or []),
            "evidence_precision": ep if gold_ev else None,
            "evidence_recall": er if gold_ev else None,
            "evidence_f1": ef if gold_ev else None,
            "gold_counterevidence_ids": ",".join(gold_ce),
            "pred_counterevidence_ids": ",".join(p.get("counterevidence_ids") or []),
            "counterevidence_recall": ce_r,
            "confidence": p.get("confidence"),
            "tokens_total": p.get("tokens_total"),
            "time_sec": p.get("time_sec"),
            "error": p.get("error"),
        })

    n = len(qids) or 1
    insuf_total_safe = insuf_total or 1

    return {
        "n_questions": len(qids),
        "n_missing_in_run": len(missing),
        "status_accuracy": status_correct / n,
        "evidence_precision": _mean(ev_precisions),
        "evidence_recall": _mean(ev_recalls),
        "evidence_f1": _mean(ev_f1s),
        "counterevidence_recall": _mean(ce_recalls) if ce_recalls else None,
        "both_accuracy": (both_correct / both_total) if both_total else None,
        "insufficient_accuracy": (insuf_correct / insuf_total) if insuf_total else None,
        "confident_on_insufficient_rate": (insuf_confident / insuf_total_safe) if insuf_total else None,
        "avg_tokens": (tokens_total / tokens_count) if tokens_count else 0.0,
        "avg_time_sec": (time_total / time_count) if time_count else 0.0,
        "_per_question": per_question,
    }


# Сборка таблиц

def build_comparison(data_dir: Path, results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    gold = load_gold(data_dir)
    per_mode: dict[str, dict[str, Any]] = {}
    all_per_question: list[dict] = []

    for mode in MODES:
        pred = load_run(results_dir, mode)
        metrics = evaluate_mode(gold, pred)
        per_mode[mode] = metrics
        all_per_question.extend(metrics["_per_question"])

    metric_order = [
        "n_questions",
        "n_missing_in_run",
        "status_accuracy",
        "evidence_precision",
        "evidence_recall",
        "evidence_f1",
        "counterevidence_recall",
        "both_accuracy",
        "insufficient_accuracy",
        "confident_on_insufficient_rate",
        "avg_tokens",
        "avg_time_sec",
    ]

    comparison = pd.DataFrame({
        "metric": metric_order,
        **{m: [per_mode[m].get(k) for k in metric_order] for m in MODES},
    })

    # Имена режимов добаылены в отдельную колонку
    comparison.insert(1, "description", [
                      _metric_description(k) for k in metric_order])

    per_question = pd.DataFrame(all_per_question)
    return comparison, per_question


def _metric_description(metric: str) -> str:
    return {
        "n_questions": "Число вопросов, для которых есть ответ модели",
        "n_missing_in_run": "Вопросы эталона, которых нет в прогоне",
        "status_accuracy": "Доля верных логических статусов",
        "evidence_precision": "Точность evidence_ids",
        "evidence_recall": "Полнота evidence_ids",
        "evidence_f1": "F1 по evidence_ids",
        "counterevidence_recall": "Полнота counterevidence_ids (только для both)",
        "both_accuracy": "Доля верных both среди gold_status = both",
        "insufficient_accuracy": "Доля верных insufficient среди gold_status = insufficient",
        "confident_on_insufficient_rate": f"Доля ответов с confidence > {CONFIDENCE_THRESHOLD} при gold_status = insufficient",
        "avg_tokens": "Среднее число токенов на вопрос",
        "avg_time_sec": "Среднее время ответа, сек",
    }.get(metric, "")


# Печать и сохранение

def print_comparison(df: pd.DataFrame) -> None:
    with pd.option_context("display.max_colwidth", 60, "display.width", 160):
        print(df.to_string(index=False))


def print_status_breakdown(per_question: pd.DataFrame) -> None:
    """Матрица gold_status × pred_status для каждого режима"""
    for mode in MODES:
        sub = per_question[per_question["mode"] == mode]
        if sub.empty:
            continue
        print(f"\nМатрица статусов, режим {mode} ({MODE_NAMES[mode]})")
        matrix = pd.crosstab(
            sub["gold_status"], sub["pred_status"].fillna("None"),
            rownames=["gold"], colnames=["pred"], dropna=False,
        )
        print(matrix.to_string())
        counts = Counter(sub["gold_status"])
        print("Эталонных статусов:", dict(counts))


def print_failures(per_question: pd.DataFrame, limit: int = 5) -> None:
    """Печатает до 5 ошибок статуса на каждый режим"""
    for mode in MODES:
        sub = per_question[(per_question["mode"] == mode) &
                           (~per_question["status_correct"])]
        if sub.empty:
            print(f"\n[{mode}] ошибок статуса нет")
            continue
        print(
            f"\n[{mode}] первые {min(limit, len(sub))} ошибок статуса из {len(sub)}:")
        for _, r in sub.head(limit).iterrows():
            print(f"  {r['question_id']} ({r['qtype']}): "
                  f"gold={r['gold_status']} pred={r['pred_status']} "
                  f"answer={r['pred_answer']!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Метрики трёх режимов")
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--out", type=Path, default=RESULTS / "comparison.csv")
    ap.add_argument("--per-question", type=Path,
                    default=RESULTS / "per_question.csv")
    args = ap.parse_args()

    comparison, per_question = build_comparison(args.data, args.results)

    print_comparison(comparison)
    print_status_breakdown(per_question)
    print_failures(per_question)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.out, index=False)
    per_question.to_csv(args.per_question, index=False)
    print(f"\nSaved {args.out}")
    print(f"Saved {args.per_question}")


if __name__ == "__main__":
    main()
