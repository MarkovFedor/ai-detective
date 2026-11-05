"""
schemas.py — участник 3: Pydantic-схема ответа LLM и валидация ссылок.

Схема соответствует формату из задания:

    {
      "question_id": "Q12",
      "status": "entailed | contradicted | both | insufficient",
      "answer": "Краткий ответ либо null",
      "evidence_ids": ["C04", "C17"],
      "counterevidence_ids": [],
      "confidence": 0.78,
      "explanation": "Краткая проверяемая связь между доказательствами"
    }

Правила (из задания):
  1. status — одно из четырёх значений;
  2. answer — строка или null;
  3. evidence_ids, counterevidence_ids — списки claim_id;
  4. confidence ∈ [0, 1];
  5. explanation — строка;
  6. при both заполнены evidence_ids и counterevidence_ids;
  7. при insufficient answer = null;
  8. все идентификаторы должны существовать в claims.jsonl;
  9. explanation не вводит новых фактов — только ссылки на доступные основания.

Модуль не вызывает LLM. Он используется в src/llm.py после парсинга JSON
и в тестах tests/test_schemas.py.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# Допустимые значения статуса. Literal даёт статическую проверку и
# автоматическое сообщение об ошибке при неверном значении.
Status = Literal["entailed", "contradicted", "both", "insufficient"]


class ModelAnswer(BaseModel):
    """Ответ LLM на один вопрос по досье."""

    question_id: str = Field(min_length=1)
    status: Status
    answer: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    counterevidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str = Field(min_length=1)

    # Часть 1
    # Нормализация отдельных полей

    @field_validator("answer", mode="after")
    @classmethod
    def _strip_answer(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip()
        return v if v else None

    @field_validator("explanation", mode="after")
    @classmethod
    def _strip_explanation(cls, v: str) -> str:
        return v.strip()

    @field_validator("evidence_ids", "counterevidence_ids", mode="after")
    @classmethod
    def _dedupe_ids(cls, v: list[str]) -> list[str]:
        """Убираем дубли, сохраняя порядок. Пустые строки отбрасываем."""
        seen: set[str] = set()
        out: list[str] = []
        for x in v:
            x = str(x).strip()
            if not x or x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    # Часть 2
    # Согласованность полей (правила из задания)

    @model_validator(mode="after")
    def _check_consistency(self) -> "ModelAnswer":
        # both: оба списка непустые
        if self.status == "both":
            if not self.evidence_ids:
                raise ValueError("status=both требует непустой evidence_ids")
            if not self.counterevidence_ids:
                raise ValueError(
                    "status=both требует непустой counterevidence_ids")

        # insufficient: answer = null, ссылки допустимы (какой информации не хватает)
        if self.status == "insufficient":
            if self.answer is not None:
                raise ValueError("status=insufficient требует answer=null")

        # entailed: должен быть хотя бы один подтверждающий claim
        if self.status == "entailed":
            if not self.evidence_ids:
                raise ValueError(
                    "status=entailed требует непустой evidence_ids")
            if self.counterevidence_ids:
                raise ValueError(
                    "status=entailed не должен содержать counterevidence_ids")

        # contradicted: должен быть хотя бы один опровергающий claim
        if self.status == "contradicted":
            if not self.evidence_ids and not self.counterevidence_ids:
                raise ValueError(
                    "status=contradicted требует evidence_ids или counterevidence_ids"
                )

        return self

    # Часть 3
    # Проверка ссылок против переданного контекста

    def check_evidence_in_context(self, available_claim_ids: list[str] | set[str]) -> None:
        """
        Все evidence_ids и counterevidence_ids должны присутствовать в контексте,
        который реально видела LLM. available_claim_ids — это ContextPack.claim_ids
        из src/context_builder.py.

        Бросает ValueError со списком несуществующих id.
        """
        allowed = set(available_claim_ids)
        unknown = (set(self.evidence_ids) | set(
            self.counterevidence_ids)) - allowed
        if unknown:
            raise ValueError(
                f"идентификаторы отсутствуют в контексте: {sorted(unknown)}"
            )

    # Часть 4.
    # Удобные свойства

    @property
    def all_ids(self) -> list[str]:
        """Объединение evidence_ids и counterevidence_ids без дублей."""
        seen: set[str] = set()
        out: list[str] = []
        for x in self.evidence_ids + self.counterevidence_ids:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out


# Вспомогательные функции для llm и тестов

def validate_evidence_ids(
    ans: ModelAnswer,
    available_claim_ids: list[str] | set[str],
) -> ModelAnswer:
    """
    Обёртка над ModelAnswer.check_evidence_in_context: возвращает тот же объект,
    чтобы удобно было писать `ans = validate_evidence_ids(ModelAnswer(**obj), pack.claim_ids)`.
    """
    ans.check_evidence_in_context(available_claim_ids)
    return ans


def parse_model_answer(
    obj: dict,
    available_claim_ids: list[str] | set[str] | None = None,
) -> ModelAnswer:
    """
    Парсит сырой dict из LLM в ModelAnswer и (опционально) проверяет ссылки
    против контекста. Используется в src/llm.py после json.loads.

    Если available_claim_ids не передан — только структурная валидация.
    """
    ans = ModelAnswer(**obj)
    if available_claim_ids is not None:
        ans.check_evidence_in_context(available_claim_ids)
    return ans
