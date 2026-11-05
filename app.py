"""
app.py — участник 3: Streamlit-стенд «AI-детектив».

Функции (по заданию):
  — загрузка досье;
  — выбор вопроса;
  — выбор режима A, B или C;
  — просмотр ответа и confidence;
  — просмотр доказательств и контрдоказательств;
  — перестановка документов;
  — удаление выбранной посылки;
  — добавление подготовленного противоречия;
  — просмотр p-adic ветвей;
  — сравнение сохранённых результатов режимов.

Запуск:
    streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.context_builder import (
    add_claim,
    build_context,
    load_dossier,
    remove_claim,
    shuffle_documents,
)
from src.padic import assign_codes, branch_label, group_by_branch
from src.stress import load_scenarios

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

MODES = ("A", "B", "C")
MODE_NAMES = {
    "A": "Полный контекст",
    "B": "BM25 top-15",
    "C": "p-adic context",
}


# ------------------------------------------------------------------
# Загрузка данных (кэшируется)
# ------------------------------------------------------------------

@st.cache_data
def load_dossier_cached():
    documents, claims, questions = load_dossier(DATA)
    return documents, claims, questions


@st.cache_data
def load_results_cached():
    """Читает results/{A,B,C}.jsonl и comparison.csv, если есть."""
    runs = {}
    for mode in MODES:
        path = RESULTS / f"{mode}.jsonl"
        if path.exists():
            with path.open(encoding="utf-8") as f:
                runs[mode] = {json.loads(l)["question_id"]: json.loads(l)
                              for l in f if l.strip()}
        else:
            runs[mode] = {}

    comparison = None
    cmp_path = RESULTS / "comparison.csv"
    if cmp_path.exists():
        comparison = pd.read_csv(cmp_path)

    stress = None
    stress_path = RESULTS / "stress_comparison.csv"
    if stress_path.exists():
        stress = pd.read_csv(stress_path)

    return runs, comparison, stress


@st.cache_data
def load_scenarios_cached():
    try:
        return load_scenarios(DATA / "stress_scenarios.json")
    except FileNotFoundError:
        return {}


# ------------------------------------------------------------------
# Боковая панель: режим, вопрос, стресс-действия
# ------------------------------------------------------------------

def render_sidebar(documents, claims, questions):
    st.sidebar.header("Параметры")

    mode = st.sidebar.radio(
        "Режим контекста",
        options=MODES,
        format_func=lambda m: f"{m} — {MODE_NAMES[m]}",
        index=0,
    )

    qid = st.sidebar.selectbox(
        "Вопрос",
        options=[q["question_id"] for q in questions],
        format_func=lambda q: f"{q} — {next(x['question'] for x in questions if x['question_id'] == q)[:40]}...",
    )
    question = next(q for q in questions if q["question_id"] == qid)

    st.sidebar.divider()
    st.sidebar.subheader("Стресс-действия")
    stress_action = st.sidebar.radio(
        "Что применить",
        options=["—", "Перестановка документов",
                 "Удаление посылки", "Добавление противоречия"],
    )

    stress_state = {"action": "none", "payload": None}

    if stress_action == "Перестановка документов":
        if st.sidebar.button("Применить перестановку (random_state=42)"):
            stress_state = {"action": "shuffle", "payload": None}
            st.sidebar.success("Документы переставлены")

    elif stress_action == "Удаление посылки":
        gold_ev = question.get("gold_evidence_ids") or []
        if gold_ev:
            cid = st.sidebar.selectbox("Какой claim удалить", options=gold_ev)
            if st.sidebar.button("Применить удаление"):
                stress_state = {"action": "remove", "payload": cid}
                st.sidebar.success(f"Удалён {cid}")
        else:
            st.sidebar.info("У этого вопроса нет gold evidence")

    elif stress_action == "Добавление противоречия":
        scenarios = load_scenarios_cached()
        if scenarios:
            name = st.sidebar.selectbox("Сценарий", options=list(scenarios))
            spec = scenarios[name]
            st.sidebar.caption(f"Ожидаемый статус: {spec['expected_status']}")
            if st.sidebar.button("Применить сценарий"):
                stress_state = {"action": "add", "payload": name}
                st.sidebar.success(f"Добавлен {spec['claim']['claim_id']}")
        else:
            st.sidebar.warning("data/stress_scenarios.json не найден")

    return mode, question, stress_state


# ------------------------------------------------------------------
# Применение стресс-действия к данным
# ------------------------------------------------------------------

def apply_stress(documents, claims, stress_state):
    """Возвращает (documents, claims, note) после стресс-действия."""
    action = stress_state["action"]
    if action == "none":
        return documents, claims, None

    if action == "shuffle":
        return shuffle_documents(documents, random_state=42), claims, "Документы переставлены"

    if action == "remove":
        cid = stress_state["payload"]
        new_docs, new_claims, side = remove_claim(documents, claims, cid)
        note = f"Удалён {cid}"
        if side:
            note += f" (задело: {side})"
        return new_docs, new_claims, note

    if action == "add":
        scenarios = load_scenarios_cached()
        spec = scenarios.get(stress_state["payload"])
        if spec is None:
            return documents, claims, "Сценарий не найден"
        new_docs, new_claims = add_claim(
            documents, claims,
            new_claim=spec["claim"],
            new_document=spec["document"],
        )
        return new_docs, new_claims, f"Добавлен {spec['claim']['claim_id']}"

    return documents, claims, None


# ------------------------------------------------------------------
# Основной экран
# ------------------------------------------------------------------

def render_main(mode, question, documents, claims, runs):
    qid = question["question_id"]

    st.header(f"{qid}: {question['question']}")
    st.caption(
        f"Тип: `{question.get('qtype')}` · "
        f"Gold: `{question.get('gold_status')}` · "
        f"Режим: `{mode}`"
    )

    # Контекст
    pack = build_context(mode, question["question"], documents, claims)

    # Сохранённый ответ
    rec = runs.get(mode, {}).get(qid)
    if rec:
        render_saved_answer(rec)
    else:
        st.info(
            f"Нет сохранённого ответа для {qid} / {mode}. Запустите `python -m src.llm --mode {mode}`.")

    # Evidence и counterevidence
    render_evidence(question, claims)

    # p-adic ветви
    if mode == "C":
        render_padic_branches(pack, claims)

    # Контекст (свёрнутый)
    with st.expander("Показать контекст, переданный LLM"):
        st.text(pack.text[:5000] + ("\n…" if len(pack.text) > 5000 else ""))


def render_saved_answer(rec: dict):
    st.subheader("Ответ модели")
    c1, c2, c3 = st.columns(3)
    c1.metric("Статус", rec.get("status") or "—")
    c2.metric("Confidence", f"{rec.get('confidence'):.2f}" if rec.get(
        "confidence") is not None else "—")
    c3.metric("Токены", rec.get("tokens_total") or "—")

    st.write("**Answer:**", rec.get("answer")
             if rec.get("answer") is not None else "null")
    st.write("**Explanation:**", rec.get("explanation") or "—")

    if rec.get("error"):
        st.error(f"Ошибка: {rec['error']}")

    ev = rec.get("evidence_ids") or []
    ce = rec.get("counterevidence_ids") or []
    st.write("**Evidence:**", ", ".join(ev) if ev else "—")
    st.write("**Counterevidence:**", ", ".join(ce) if ce else "—")


def render_evidence(question, claims):
    st.subheader("Эталонные доказательства")
    claims_by_id = {c["claim_id"]: c for c in claims}

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Evidence (gold):**")
        for cid in question.get("gold_evidence_ids") or []:
            c = claims_by_id.get(cid)
            if c:
                st.markdown(f"- `{cid}`: {c['text']}")
            else:
                st.markdown(f"- `{cid}`: (нет в claims)")

    with col2:
        st.write("**Counterevidence (gold):**")
        ce = question.get("gold_counterevidence_ids") or []
        if not ce:
            st.caption("—")
        for cid in ce:
            c = claims_by_id.get(cid)
            if c:
                st.markdown(f"- `{cid}`: {c['text']}")
            else:
                st.markdown(f"- `{cid}`: (нет в claims)")


def render_padic_branches(pack, claims):
    st.subheader("p-adic ветви (режим C)")
    codes = assign_codes(claims, load_dossier_cached()[0])
    by_id = {c["claim_id"]: c for c in claims}
    branches = group_by_branch(pack.claim_ids, codes)
    for path, ids in branches:
        with st.expander(f"ВЕТВЬ {branch_label(path)} ({len(ids)})"):
            for cid in ids:
                st.markdown(f"- `{cid}`: {by_id[cid]['text']}")


# ------------------------------------------------------------------
# Экран сравнения
# ------------------------------------------------------------------

def render_comparison(runs, comparison, stress):
    st.header("Сравнение режимов")

    # Таблица метрик
    if comparison is not None:
        st.subheader("Метрики (results/comparison.csv)")
        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Нет results/comparison.csv. Запустите `python -m src.evaluation`.")

    # Таблица ответов по вопросам
    st.subheader("Ответы по вопросам")
    rows = []
    for qid in sorted({q for mode in MODES for q in runs.get(mode, {})}):
        for mode in MODES:
            rec = runs.get(mode, {}).get(qid)
            rows.append({
                "question_id": qid,
                "mode": mode,
                "status": rec.get("status") if rec else "—",
                "answer": rec.get("answer") if rec else "—",
                "confidence": rec.get("confidence") if rec else None,
                "tokens": rec.get("tokens_total") if rec else None,
                "time_sec": rec.get("time_sec") if rec else None,
                "error": rec.get("error") if rec else "нет прогона",
            })
    if rows:
        st.dataframe(pd.DataFrame(rows),
                     use_container_width=True, hide_index=True)
    else:
        st.info("Нет сохранённых прогонов в results/.")

    # Стресс-тесты
    if stress is not None:
        st.subheader("Стресс-тесты (results/stress_comparison.csv)")
        st.dataframe(stress, use_container_width=True, hide_index=True)
    else:
        st.caption(
            "Нет results/stress_comparison.csv. Запустите `python -m src.stress --test all`.")


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="AI-детектив · Nimbus Labs",
        page_icon="🔍",
        layout="wide",
    )
    st.title("🔍 AI-детектив · Пропавший прототип Aurora-7")
    st.caption("ЛР6 · Автоматическая обработка текстов · интерактивный стенд")

    documents, claims, questions = load_dossier_cached()
    runs, comparison, stress = load_results_cached()

    mode, question, stress_state = render_sidebar(documents, claims, questions)
    new_docs, new_claims, note = apply_stress(documents, claims, stress_state)

    if note:
        st.warning(f"Стресс-действие: {note}")

    tab_main, tab_compare = st.tabs(["Стенд", "Сравнение"])

    with tab_main:
        render_main(mode, question, new_docs, new_claims, runs)

    with tab_compare:
        render_comparison(runs, comparison, stress)


if __name__ == "__main__":
    main()
