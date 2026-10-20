"""
retrieval.py — участник 2: BM25-поиск атомарных утверждений (режимы B и C).

Корпус для BM25 — поле `text` утверждений из claims.jsonl (один «документ»
BM25 = одно утверждение). Запрос — текст вопроса.

Предобработка (одинакова для корпуса и запроса):
  1. нижний регистр, ё → е;
  2. токены — последовательности букв/цифр (латиница тоже: RAKITIN-S, SEMENOV-P);
  3. удаление стоп-слов;
  4. лемматизация pymorphy3 (rank-bm25 сам ничего не нормализует,
     без лемм «сейф» и «сейфа» — разные слова).

Порядок выдачи детерминирован: при равных score выше идёт утверждение,
стоящее раньше в claims.jsonl. Поэтому перестановка документов (стресс-тест 1)
не меняет состав top-15.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pymorphy3
from rank_bm25 import BM25Okapi

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
TOP_K = 15

STOPWORDS = frozenset(
    """
    и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по
    только ее мне было вот от меня еще нет о из ему теперь когда даже ну ли если уже
    или ни быть был него до вас нибудь опять уж вам ведь там потом себя ничего ей может
    они тут где есть надо ней для мы тебя их чем была сам чтоб без будто чего раз тоже
    себе под будет ж тогда кто этот того потому этого какой совсем ним здесь этом один
    почти мой тем чтобы нее сейчас были куда зачем всех никогда можно при об другой хоть
    после над больше тот через эти нас про всего них какая много разве эту моя свою этой
    перед иногда лучше чуть том нельзя такой им более всегда конечно всю между именно
    """.split()
)

_TOKEN_RE = re.compile(r"[a-zа-я0-9]+")
_morph = pymorphy3.MorphAnalyzer()


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@lru_cache(maxsize=50_000)
def _lemma(word: str) -> str:
    return _morph.parse(word)[0].normal_form.replace("ё", "е")


def tokenize(text: str) -> list[str]:
    words = _TOKEN_RE.findall(text.lower().replace("ё", "е"))
    return [_lemma(w) for w in words if w not in STOPWORDS]


@dataclass(frozen=True)
class RetrievedClaim:
    rank: int          # 1..k
    claim_id: str
    score: float
    claim: dict


class ClaimRetriever:
    """BM25 по атомарным утверждениям. Индекс строится один раз на набор claims."""

    def __init__(self, claims: list[dict]):
        if not claims:
            raise ValueError("пустой список утверждений")
        self.claims = list(claims)
        self._bm25 = BM25Okapi([tokenize(c["text"]) for c in self.claims])

    def search(self, query: str, k: int = TOP_K) -> list[RetrievedClaim]:
        if k <= 0:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(self.claims)), key=lambda i: (-scores[i], i))
        return [
            RetrievedClaim(rank=r, claim_id=self.claims[i]["claim_id"],
                           score=float(scores[i]), claim=self.claims[i])
            for r, i in enumerate(order[:k], start=1)
        ]

    def top_ids(self, query: str, k: int = TOP_K) -> list[str]:
        return [r.claim_id for r in self.search(query, k)]


def gold_recall_at_k(claims: list[dict], questions: list[dict], k: int = TOP_K) -> list[dict]:
    """Какая доля эталонных evidence/counterevidence попала в top-k (для отчёта)."""
    retriever = ClaimRetriever(claims)
    rows = []
    for q in questions:
        top = set(retriever.top_ids(q["question"], k))
        gold = q["gold_evidence_ids"] + q["gold_counterevidence_ids"]
        missed = [c for c in gold if c not in top]
        rows.append({
            "question_id": q["question_id"],
            "qtype": q.get("qtype"),
            "gold": len(gold),
            "found": len(gold) - len(missed),
            "recall": (len(gold) - len(missed)) / len(gold) if gold else None,
            "missed": missed,
        })
    return rows


if __name__ == "__main__":
    claims = load_jsonl(DATA_DIR / "claims.jsonl")
    questions = load_jsonl(DATA_DIR / "questions.jsonl")
    rows = gold_recall_at_k(claims, questions)
    total = sum(r["gold"] for r in rows)
    found = sum(r["found"] for r in rows)
    for r in rows:
        print(f'{r["question_id"]:4} {r["qtype"]:18} {r["found"]}/{r["gold"]}  missed={r["missed"]}')
    print(f"\nRecall@{TOP_K} эталонных утверждений: {found}/{total} = {found / total:.2f}")
