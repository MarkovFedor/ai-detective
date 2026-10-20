"""
padic.py — участник 2: p-adic (p = 3) организация утверждений.

Путь утверждения [a0, a1, a2], код x = a0 + a1·3 + a2·3².

    уровень | 0                        | 1                        | 2
    --------+--------------------------+--------------------------+----------------------
    a0      | системный/документальный | свидетельство            | правило/интерпретация
    a1      | установленный факт       | сообщение/предположение  | спорное/исключение
    a2      | до события               | во время события         | после/неизвестно

Расстояние: k — число совпадающих НАЧАЛЬНЫХ уровней, d3(x, y) = 3^(-k),
при полном совпадении пути d = 0. Это ровно 3-адическая норма разности кодов:
|x − y|_3 = 3^(−v3(x − y)), потому что a0 — младшая 3-ичная цифра.
abs(x − y) НЕ используется (запрещено заданием и не является p-adic метрикой).

Правила разметки
----------------
a0 — по source_type документа, из которого взято утверждение;
     утверждения со статусом inferred (вывод из нескольких записей) → 2
     («интерпретация»), независимо от документа.
a1 — по status утверждения: direct → 0; reported, inferred → 1; disputed → 2;
     пункты-исключения из регламента (RULE_EXCEPTIONS) → 2.
a2 — по ВРЕМЕНИ ФАКТА, а не по created_at документа: все показания написаны
     18–19 мая, но описывают ночь 16 мая. Время берётся из текста утверждения
     (HH:MM, «DD.MM HH:MM»), для утверждений без часов — из TIME_OVERRIDES.
     Событие = ночь 16.05 00:00–07:00 (до конца ночной смены).
     Утверждения без привязки ко времени (правила, общие сведения) → 2.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

P = 3
DEPTH = 3
Path3 = tuple[int, int, int]

LEVEL_NAMES: dict[int, dict[int, str]] = {
    0: {0: "системный/документальный", 1: "свидетельство", 2: "правило/интерпретация"},
    1: {0: "установленный факт", 1: "сообщение/предположение", 2: "спорное/исключение"},
    2: {0: "до события", 1: "во время события", 2: "после/неизвестно"},
}

SOURCE_TYPE_TO_A0: dict[str, int] = {
    "incident_report": 0,
    "timeline": 0,
    "system_log": 0,
    "press_release": 0,
    "audit_report": 0,
    "statement": 1,
    "chat_log": 1,
    "regulation": 2,
}

STATUS_TO_A1: dict[str, int] = {
    "direct": 0,
    "reported": 1,
    "inferred": 1,
    "disputed": 2,
}

# Пункт 6.4 регламента: EMERGENCY_OVERRIDE не даёт доступа к сейфу — исключение из
# общего правила «режим открывает все двери».
RULE_EXCEPTIONS: frozenset[str] = frozenset({"C24"})

EVENT_START = datetime(2026, 5, 16, 0, 0)
EVENT_END = datetime(2026, 5, 16, 7, 0)
INCIDENT_YEAR = 2026

# claim_id -> (время факта ISO или None, фаза a2, обоснование).
# Только для утверждений, где время нельзя надёжно извлечь из текста.
TIME_OVERRIDES: dict[str, tuple[str | None, int, str]] = {
    "C17": ("2026-05-15T19:00", 1, "интервал 15.05 19:00–16.05 07:00 накрывает событие"),
    "C27": ("2026-05-16T02:30", 1, "пропуск оставлен в комнате на время обхода 02:30"),
    "C28": ("2026-05-16T02:30", 1, "относится к ночной смене / обходу"),
    "C29": ("2026-05-15T17:30", 0, "ключ убран в шкафчик после закрытия сейфа в 17:30"),
    "C32": (None, 1, "«в выходные» — накрывает ночь события, часов нет"),
    "C33": (None, 1, "«в ночь с 15 на 16 мая», часов нет"),
    "C38": ("2026-05-16T02:30", 1, "касается заявленного обхода 02:30"),
}

_DATE_TIME_RE = re.compile(r"(\d{1,2})\.(\d{2})(?:\.\d{4})?\s+(\d{1,2}):(\d{2})")
_TIME_RE = re.compile(r"\b(\d{1,2}):(\d{2})\b")


# --------------------------------------------------------------------------- коды и расстояние

def validate_path(path) -> Path3:
    path = tuple(int(a) for a in path)
    if len(path) != DEPTH or any(a not in (0, 1, 2) for a in path):
        raise ValueError(f"путь должен быть из {DEPTH} цифр 0..2, получено {path}")
    return path  # type: ignore[return-value]


def encode(path) -> int:
    """[a0, a1, a2] -> x = a0 + a1·3 + a2·9."""
    a0, a1, a2 = validate_path(path)
    return a0 + a1 * P + a2 * P ** 2


def decode(x: int) -> Path3:
    if not 0 <= x < P ** DEPTH:
        raise ValueError(f"код вне диапазона 0..{P ** DEPTH - 1}: {x}")
    return (x % P, (x // P) % P, (x // P ** 2) % P)


def common_prefix_len(p1, p2) -> int:
    p1, p2 = validate_path(p1), validate_path(p2)
    k = 0
    while k < DEPTH and p1[k] == p2[k]:
        k += 1
    return k


def padic_distance(p1, p2) -> float:
    """d3 по путям: 1, 1/3, 1/9 или 0 (полное совпадение)."""
    k = common_prefix_len(p1, p2)
    return 0.0 if k == DEPTH else float(P) ** (-k)


def padic_valuation(n: int, p: int = P) -> int:
    if n == 0:
        raise ValueError("v_p(0) = +inf")
    n, v = abs(n), 0
    while n % p == 0:
        n //= p
        v += 1
    return v


def padic_norm(n: int, p: int = P) -> float:
    return 0.0 if n == 0 else float(p) ** (-padic_valuation(n, p))


def padic_distance_codes(x: int, y: int) -> float:
    """Та же метрика через коды: |x − y|_3 (не abs!)."""
    return padic_norm(x - y)


# --------------------------------------------------------------------------- разметка утверждений

def extract_fact_time(claim: dict, document: dict) -> datetime | None:
    """Время, к которому относится факт (а не время написания документа)."""
    text = claim["text"]
    m = _DATE_TIME_RE.search(text)
    if m:
        day, month, hh, mm = map(int, m.groups())
        return datetime(INCIDENT_YEAR, month, day, hh, mm)
    m = _TIME_RE.search(text)
    if not m:
        return None
    hh, mm = map(int, m.groups())
    # Досье описывает ночь 15→16 мая: вечер — пятница 15.05, ночь — суббота 16.05,
    # утро (08–12) — день написания документа (переписка в понедельник 18.05).
    if hh >= 12:
        day = datetime(INCIDENT_YEAR, 5, 15)
    elif hh < 8:
        day = datetime(INCIDENT_YEAR, 5, 16)
    else:
        day = datetime.fromisoformat(document["created_at"])
    return day.replace(hour=hh, minute=mm, second=0, microsecond=0)


def time_phase(t: datetime | None) -> int:
    if t is None:
        return 2
    if t < EVENT_START:
        return 0
    if t <= EVENT_END:
        return 1
    return 2


@dataclass(frozen=True)
class ClaimCode:
    claim_id: str
    path: Path3
    code: int
    fact_time: datetime | None

    @property
    def branch(self) -> str:
        return branch_label(self.path)


def claim_path(claim: dict, document: dict) -> tuple[Path3, datetime | None]:
    source_type = document["source_type"]
    if source_type not in SOURCE_TYPE_TO_A0:
        raise ValueError(f"{claim['claim_id']}: неизвестный source_type {source_type!r}")
    status = claim["status"]
    if status not in STATUS_TO_A1:
        raise ValueError(f"{claim['claim_id']}: неизвестный status {status!r}")

    a0 = 2 if status == "inferred" else SOURCE_TYPE_TO_A0[source_type]
    a1 = 2 if claim["claim_id"] in RULE_EXCEPTIONS else STATUS_TO_A1[status]

    override = TIME_OVERRIDES.get(claim["claim_id"])
    if override is not None:
        iso, a2, _reason = override
        fact_time = datetime.fromisoformat(iso) if iso else None
    else:
        fact_time = extract_fact_time(claim, document)
        a2 = time_phase(fact_time)
    return (a0, a1, a2), fact_time


def assign_codes(claims: list[dict], documents: list[dict]) -> dict[str, ClaimCode]:
    docs = {d["document_id"]: d for d in documents}
    result: dict[str, ClaimCode] = {}
    for c in claims:
        if c["document_id"] not in docs:
            raise KeyError(f"{c['claim_id']}: документ {c['document_id']} не найден")
        path, t = claim_path(c, docs[c["document_id"]])
        result[c["claim_id"]] = ClaimCode(c["claim_id"], path, encode(path), t)
    return result


def branch_label(path) -> str:
    a0, a1, a2 = validate_path(path)
    return (f"[{a0},{a1},{a2}] — {LEVEL_NAMES[0][a0]} · "
            f"{LEVEL_NAMES[1][a1]} · {LEVEL_NAMES[2][a2]}")


def _time_key(code: ClaimCode) -> tuple:
    # сначала по времени факта, утверждения без времени — в конец ветви
    return (code.fact_time is None, code.fact_time or datetime.max, code.claim_id)


def group_by_branch(claim_ids: list[str], codes: dict[str, ClaimCode]) -> list[tuple[Path3, list[str]]]:
    """Ветви в порядке обхода 3-ичного дерева (a0, затем a1, затем a2), внутри — по времени."""
    branches: dict[Path3, list[ClaimCode]] = {}
    for cid in claim_ids:
        branches.setdefault(codes[cid].path, []).append(codes[cid])
    return [
        (path, [c.claim_id for c in sorted(branches[path], key=_time_key)])
        for path in sorted(branches)
    ]


def distance_matrix(claim_ids: list[str], codes: dict[str, ClaimCode]) -> list[list[float]]:
    return [[padic_distance(codes[a].path, codes[b].path) for b in claim_ids] for a in claim_ids]


if __name__ == "__main__":
    from src.retrieval import DATA_DIR, load_jsonl

    documents = load_jsonl(DATA_DIR / "documents.jsonl")
    claims = load_jsonl(DATA_DIR / "claims.jsonl")
    codes = assign_codes(claims, documents)
    by_id = {c["claim_id"]: c for c in claims}
    for path, ids in group_by_branch(list(codes), codes):
        print(f"\nВЕТВЬ {branch_label(path)}   (x = {encode(path)})")
        for cid in ids:
            t = codes[cid].fact_time
            print(f"  {cid} {t:%d.%m %H:%M}" if t else f"  {cid}   —   ", "|", by_id[cid]["text"])
