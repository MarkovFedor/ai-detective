"""
context_builder.py — участник 2: упаковка контекста для режимов A, B, C.

A — полный контекст: все документы в фиксированном порядке (порядок списка
    documents). Чтобы модель могла ссылаться на claim_id, сразу после
    дословного evidence в тексте ставится метка [Cxx]. Сам текст не меняется.
B — плоский retrieval: BM25 top-15 утверждений списком «Cxx: текст» по убыванию score.
C — p-adic: ТЕ ЖЕ 15 утверждений, сгруппированные по ветвям [a0,a1,a2],
    внутри ветви — по времени факта.

Контроль справедливости: B и C строятся из одного вызова retriever.search(),
поэтому набор claim_id у них совпадает по построению (проверяется в тестах).

Функции не читают файлы сами — им передаются documents и claims. Это нужно для
стресс-тестов: перестановка документов, удаление посылки и добавление
противоречия меняют только входные списки (helpers внизу файла).
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from pathlib import Path

from src.padic import ClaimCode, assign_codes, branch_label, group_by_branch
from src.retrieval import DATA_DIR, TOP_K, ClaimRetriever, RetrievedClaim, load_jsonl

MODES = ("A", "B", "C")
MODE_NAMES = {"A": "Полный контекст", "B": "BM25 top-15", "C": "p-adic context"}


@dataclass
class ContextPack:
    mode: str
    text: str
    claim_ids: list[str]                                   # что модели доступно для ссылок
    retrieved: list[RetrievedClaim] = field(default_factory=list)  # пусто для A
    branches: list[tuple[tuple[int, int, int], list[str]]] = field(default_factory=list)  # только C


def load_dossier(data_dir: str | Path = DATA_DIR) -> tuple[list[dict], list[dict], list[dict]]:
    data_dir = Path(data_dir)
    return (load_jsonl(data_dir / "documents.jsonl"),
            load_jsonl(data_dir / "claims.jsonl"),
            load_jsonl(data_dir / "questions.jsonl"))


# --------------------------------------------------------------------------- режим A

def mark_claims_in_text(text: str, claims_of_doc: list[dict]) -> tuple[str, list[str]]:
    """Вставляет ' [Cxx]' сразу после первого вхождения evidence. Возвращает текст и
    claim_id, которые удалось разметить (evidence мог быть вырезан стресс-тестом)."""
    inserts: dict[int, list[str]] = {}
    for c in claims_of_doc:
        pos = text.find(c["evidence"])
        if pos >= 0:
            inserts.setdefault(pos + len(c["evidence"]), []).append(c["claim_id"])
    out, prev = [], 0
    for end in sorted(inserts):
        out.append(text[prev:end])
        out.append("".join(f" [{cid}]" for cid in inserts[end]))
        prev = end
    out.append(text[prev:])
    marked = [cid for end in sorted(inserts) for cid in inserts[end]]
    return "".join(out), marked


def build_full_context(documents: list[dict], claims: list[dict], mark_claims: bool = True) -> ContextPack:
    by_doc: dict[str, list[dict]] = {}
    for c in claims:
        by_doc.setdefault(c["document_id"], []).append(c)

    parts = [f"ДОСЬЕ: {len(documents)} документов."]
    if mark_claims:
        parts.append("Метка [Cxx] стоит сразу после фрагмента, который соответствует утверждению Cxx; "
                     "ссылайтесь на эти идентификаторы.")
    available: list[str] = []
    for d in documents:
        text = d["text"]
        if mark_claims:
            text, marked = mark_claims_in_text(text, by_doc.get(d["document_id"], []))
            available.extend(marked)
        parts.append(
            f"\n=== {d['document_id']} | {d['title']} | {d['source_type']} | "
            f"{d['created_at']} | {d['author']} ===\n{text}"
        )
    return ContextPack(mode="A", text="\n".join(parts), claim_ids=available)


# --------------------------------------------------------------------------- режимы B и C

def _claim_line(r: RetrievedClaim) -> str:
    return f"{r.claim_id}: {r.claim['text']}"


def build_flat_context(retrieved: list[RetrievedClaim]) -> ContextPack:
    lines = [f"УТВЕРЖДЕНИЯ ({len(retrieved)}), найденные поиском по вопросу:"]
    lines += [_claim_line(r) for r in retrieved]
    return ContextPack(mode="B", text="\n".join(lines),
                       claim_ids=[r.claim_id for r in retrieved], retrieved=retrieved)


def build_padic_context(retrieved: list[RetrievedClaim], codes: dict[str, ClaimCode]) -> ContextPack:
    by_id = {r.claim_id: r for r in retrieved}
    branches = group_by_branch(list(by_id), codes)
    lines = [
        f"УТВЕРЖДЕНИЯ ({len(retrieved)}), найденные поиском по вопросу, сгруппированы по ветвям "
        "[тип источника, логический статус, время относительно события]; внутри ветви — по времени:"
    ]
    for path, ids in branches:
        lines.append(f"\nВЕТВЬ {branch_label(path)}")
        lines += [_claim_line(by_id[cid]) for cid in ids]
    return ContextPack(mode="C", text="\n".join(lines),
                       claim_ids=[cid for _, ids in branches for cid in ids],
                       retrieved=retrieved, branches=branches)


# --------------------------------------------------------------------------- общий вход

def build_context(mode: str, question: str, documents: list[dict], claims: list[dict],
                  k: int = TOP_K, retriever: ClaimRetriever | None = None) -> ContextPack:
    mode = mode.upper()
    if mode not in MODES:
        raise ValueError(f"режим должен быть одним из {MODES}, получено {mode!r}")
    if mode == "A":
        return build_full_context(documents, claims)
    retriever = retriever or ClaimRetriever(claims)
    retrieved = retriever.search(question, k)
    if mode == "B":
        return build_flat_context(retrieved)
    return build_padic_context(retrieved, assign_codes(claims, documents))


def build_all_modes(question: str, documents: list[dict], claims: list[dict],
                    k: int = TOP_K) -> dict[str, ContextPack]:
    """A, B, C за один вызов; B и C — из одного и того же результата BM25."""
    retrieved = ClaimRetriever(claims).search(question, k)
    return {
        "A": build_full_context(documents, claims),
        "B": build_flat_context(retrieved),
        "C": build_padic_context(retrieved, assign_codes(claims, documents)),
    }


# --------------------------------------------------------------------------- стресс-тесты
# Каждая функция меняет ровно один фактор и возвращает НОВЫЕ списки (исходные не трогает).

def shuffle_documents(documents: list[dict], random_state: int = 42) -> list[dict]:
    """Тест 1: перестановка документов."""
    shuffled = list(documents)
    random.Random(random_state).shuffle(shuffled)
    return shuffled


def remove_claim(documents: list[dict], claims: list[dict], claim_id: str
                 ) -> tuple[list[dict], list[dict], list[str]]:
    """Тест 2: удаление посылки. Утверждение убирается из claims (режимы B/C), а его
    evidence вырезается из текста документа (режим A). Третье значение — другие
    claim_id, чей evidence тоже исчез из текста (их надо проверить вручную)."""
    target = next((c for c in claims if c["claim_id"] == claim_id), None)
    if target is None:
        raise KeyError(f"утверждение {claim_id} не найдено")
    new_docs = copy.deepcopy(documents)
    for d in new_docs:
        if d["document_id"] == target["document_id"]:
            d["text"] = d["text"].replace(target["evidence"], "", 1)
    new_claims = [c for c in claims if c["claim_id"] != claim_id]
    texts = {d["document_id"]: d["text"] for d in new_docs}
    side_effects = [c["claim_id"] for c in new_claims if c["evidence"] not in texts[c["document_id"]]]
    return new_docs, new_claims, side_effects


def add_claim(documents: list[dict], claims: list[dict], new_claim: dict,
              new_document: dict | None = None) -> tuple[list[dict], list[dict]]:
    """Тест 3: добавление противоречия. Если утверждение из нового источника —
    передайте new_document; evidence обязан дословно входить в его текст."""
    new_docs = copy.deepcopy(documents)
    if new_document is not None:
        if any(d["document_id"] == new_document["document_id"] for d in new_docs):
            raise ValueError(f"документ {new_document['document_id']} уже есть")
        new_docs.append(copy.deepcopy(new_document))
    if any(c["claim_id"] == new_claim["claim_id"] for c in claims):
        raise ValueError(f"claim_id {new_claim['claim_id']} уже есть")
    doc = next((d for d in new_docs if d["document_id"] == new_claim["document_id"]), None)
    if doc is None:
        raise KeyError(f"документ {new_claim['document_id']} не найден")
    if new_claim["evidence"] not in doc["text"]:
        raise ValueError(f"evidence {new_claim['claim_id']} не входит дословно в {doc['document_id']}")
    return new_docs, list(claims) + [dict(new_claim)]


if __name__ == "__main__":
    import sys

    qid = sys.argv[1] if len(sys.argv) > 1 else "Q09"
    mode = sys.argv[2] if len(sys.argv) > 2 else "C"
    documents, claims, questions = load_dossier()
    q = next(q for q in questions if q["question_id"] == qid)
    pack = build_context(mode, q["question"], documents, claims)
    print(f"{qid}: {q['question']}\nрежим {pack.mode}, утверждений: {len(pack.claim_ids)}, "
          f"символов: {len(pack.text)}\n")
    print(pack.text if mode.upper() != "A" else pack.text[:3000] + "\n…")
