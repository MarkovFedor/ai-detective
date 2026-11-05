"""
test_llm_validation.py — обязательные тесты 4–8 из задания.

Пункты:
  4. невалидный JSON LLM;
  5. несуществующий evidence_id;
  6. both без counterevidence_ids;
  7. insufficient с выдуманным фактом;
  8. недоступность LLM.

Тесты используют мок-клиент, поэтому не требуют реального API-ключа
и работают без сети. Каждый тест изолирован: мок возвращает один
подготовленный ответ или бросает одно исключение.

Запуск:
    pytest tests/test_llm_validation.py -v
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from src.llm import call_llm
from src.schemas import ModelAnswer, parse_model_answer


# ------------------------------------------------------------------
# Вспомогательные объекты
# ------------------------------------------------------------------

class _FakePack:
    """Минимальный ContextPack: call_llm использует только claim_ids."""
    def __init__(self, claim_ids: list[str]):
        self.claim_ids = claim_ids


def _fake_response(content: str, prompt_tokens: int = 100, completion_tokens: int = 20):
    """Собирает объект, похожий на ChatCompletion, для мока."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def _make_client(responses: list):
    """
    Клиент-мок. responses — список:
      - str — вернуть как content;
      - Exception-инстанс — бросить;
    Каждый вызов chat.completions.create берёт следующий элемент.
    """
    client = MagicMock()
    iterator = iter(responses)

    def _create(**kwargs):
        item = next(iterator)
        if isinstance(item, Exception):
            raise item
        return _fake_response(item)

    client.chat.completions.create.side_effect = _create
    return client


def _valid_payload(**over) -> dict:
    base = {
        "question_id": "Q01",
        "status": "entailed",
        "answer": "17:30",
        "evidence_ids": ["C05"],
        "counterevidence_ids": [],
        "confidence": 0.9,
        "explanation": "C05 прямо указывает время.",
    }
    base.update(over)
    return base


# ------------------------------------------------------------------
# Тест 4. Невалидный JSON LLM
# ------------------------------------------------------------------

def test_invalid_json_retried_then_error():
    """
    Пункт 4. Если LLM возвращает невалидный JSON, call_llm делает retry.
    Если и второй ответ невалиден — возвращает (None, error='JSONDecodeError: ...').
    """
    pack = _FakePack(["C05"])
    client = _make_client([
        "это не JSON",                 # попытка 1
        "{ тоже не json",              # попытка 2
        "и это не json",               # попытка 3 (max_retries=2 → 3 попытки)
    ])

    ans, meta = call_llm(client, "fake-model", "prompt", pack, max_retries=2)

    assert ans is None
    assert meta["error"] is not None
    assert "JSONDecodeError" in meta["error"]
    # Было ровно 3 попытки
    assert client.chat.completions.create.call_count == 3


def test_invalid_json_then_valid_on_retry():
    """
    Пункт 4 (позитивный сценарий): первый ответ невалиден, второй — валиден.
    call_llm должен вернуть ответ, а не ошибку.
    """
    pack = _FakePack(["C05"])
    valid_json = (
        '{"question_id":"Q01","status":"entailed","answer":"17:30",'
        '"evidence_ids":["C05"],"counterevidence_ids":[],'
        '"confidence":0.9,"explanation":"C05 прямо указывает время."}'
    )
    client = _make_client([
        "не json",       # попытка 1 — падает
        valid_json,      # попытка 2 — ок
    ])

    ans, meta = call_llm(client, "fake-model", "prompt", pack, max_retries=2)

    assert ans is not None
    assert ans.status == "entailed"
    assert ans.evidence_ids == ["C05"]
    assert meta["error"] is None


# ------------------------------------------------------------------
# Тест 5. Несуществующий evidence_id
# ------------------------------------------------------------------

def test_nonexistent_evidence_id_rejected():
    """
    Пункт 5. Если evidence_id нет в context_ids, parse_model_answer
    бросает ValueError.
    """
    payload = _valid_payload(evidence_ids=["C99"])
    with pytest.raises(ValueError) as exc:
        parse_model_answer(payload, available_claim_ids=["C05", "C06"])
    assert "C99" in str(exc.value)


def test_nonexistent_counterevidence_id_rejected():
    """
    Пункт 5 (расширение): counterevidence_id тоже должен быть из контекста.
    """
    payload = _valid_payload(
        status="both",
        answer="конфликт",
        evidence_ids=["C05"],
        counterevidence_ids=["C77"],
    )
    with pytest.raises(ValueError) as exc:
        parse_model_answer(payload, available_claim_ids=["C05", "C06"])
    assert "C77" in str(exc.value)


def test_llm_call_with_nonexistent_evidence_returns_error():
    """
    Пункт 5 на уровне call_llm: если LLM вернула C99, а в контексте только C05,
    call_llm должен вернуть (None, error='ValueError: ...') после retry.
    """
    pack = _FakePack(["C05"])
    bad_json = (
        '{"question_id":"Q01","status":"entailed","answer":"17:30",'
        '"evidence_ids":["C99"],"counterevidence_ids":[],'
        '"confidence":0.9,"explanation":"C99 нет в контексте."}'
    )
    client = _make_client([bad_json, bad_json, bad_json])

    ans, meta = call_llm(client, "fake-model", "prompt", pack, max_retries=2)

    assert ans is None
    assert meta["error"] is not None
    assert "C99" in meta["error"]


# ------------------------------------------------------------------
# Тест 6. both без counterevidence_ids
# ------------------------------------------------------------------

def test_both_without_counterevidence_raises():
    """
    Пункт 6. status='both' требует непустой counterevidence_ids.
    """
    payload = _valid_payload(
        status="both",
        answer="конфликт",
        evidence_ids=["C05"],
        counterevidence_ids=[],
    )
    with pytest.raises(ValidationError) as exc:
        ModelAnswer(**payload)
    assert "counterevidence" in str(exc.value).lower()


# ------------------------------------------------------------------
# Тест 7. insufficient с выдуманным фактом
# ------------------------------------------------------------------

def test_insufficient_with_answer_raises():
    """
    Пункт 7. status='insufficient' требует answer=null.
    """
    payload = _valid_payload(
        status="insufficient",
        answer="Семёнов вынес прототип",  # выдуманный факт
    )
    with pytest.raises(ValidationError) as exc:
        ModelAnswer(**payload)
    assert "insufficient" in str(exc.value).lower()


def test_insufficient_with_null_answer_ok():
    """
    Пункт 7 (позитивный): insufficient с answer=null — валидно.
    """
    payload = _valid_payload(
        status="insufficient",
        answer=None,
        evidence_ids=["C14"],
    )
    ans = ModelAnswer(**payload)
    assert ans.status == "insufficient"
    assert ans.answer is None


# ------------------------------------------------------------------
# Тест 8. Недоступность LLM
# ------------------------------------------------------------------

def test_llm_unavailable_returns_error():
    """
    Пункт 8. Если клиент бросает исключение (сеть, 500, таймаут),
    call_llm должен вернуть (None, error='...'), а не упасть.
    """
    pack = _FakePack(["C05"])
    client = _make_client([
        ConnectionError("network unreachable"),
        ConnectionError("network unreachable"),
        ConnectionError("network unreachable"),
    ])

    ans, meta = call_llm(client, "fake-model", "prompt", pack, max_retries=2)

    assert ans is None
    assert meta["error"] is not None
    assert "ConnectionError" in meta["error"]


def test_llm_auth_error_returns_error():
    """
    Пункт 8 (расширение). 401 от провайдера — тоже ошибка, не падение.
    """
    pack = _FakePack(["C05"])

    class _AuthError(Exception):
        pass

    client = _make_client([
        _AuthError("401 Unauthorized"),
        _AuthError("401 Unauthorized"),
        _AuthError("401 Unauthorized"),
    ])

    ans, meta = call_llm(client, "fake-model", "prompt", pack, max_retries=2)

    assert ans is None
    assert "401" in meta["error"]