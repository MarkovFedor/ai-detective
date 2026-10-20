"""Тесты BM25 и сборки контекста (участник 2)."""
import pytest

from src.context_builder import (add_claim, build_all_modes, build_context, build_full_context,
                                 load_dossier, remove_claim, shuffle_documents)
from src.retrieval import ClaimRetriever, tokenize


@pytest.fixture(scope="module")
def dossier():
    return load_dossier()


def test_tokenize_lemmatizes():
    assert tokenize("Сейфа") == tokenize("сейф")
    assert "rakitin" in tokenize("badge=RAKITIN-S")


def test_bm25_returns_at_most_15(dossier):
    _, claims, questions = dossier
    r = ClaimRetriever(claims)
    for q in questions:
        res = r.search(q["question"])
        assert len(res) <= 15
        assert len({x.claim_id for x in res}) == len(res)


def test_bm25_k_larger_than_corpus(dossier):
    _, claims, _ = dossier
    assert len(ClaimRetriever(claims[:5]).search("сейф", k=15)) == 5


def test_bm25_scores_descending(dossier):
    _, claims, questions = dossier
    res = ClaimRetriever(claims).search(questions[0]["question"])
    scores = [x.score for x in res]
    assert scores == sorted(scores, reverse=True)


def test_bm25_direct_question_finds_gold(dossier):
    _, claims, questions = dossier
    q = next(q for q in questions if q["question_id"] == "Q01")
    assert set(q["gold_evidence_ids"]) <= set(ClaimRetriever(claims).top_ids(q["question"]))


def test_modes_b_and_c_get_same_claim_ids(dossier):
    documents, claims, questions = dossier
    for q in questions:
        packs = build_all_modes(q["question"], documents, claims)
        assert set(packs["B"].claim_ids) == set(packs["C"].claim_ids)
        assert len(packs["B"].claim_ids) == len(packs["C"].claim_ids) <= 15
        # отдельные вызовы build_context тоже дают одинаковый набор
        b = build_context("B", q["question"], documents, claims)
        c = build_context("C", q["question"], documents, claims)
        assert set(b.claim_ids) == set(c.claim_ids)


def test_mode_c_has_branch_headers_and_only_packaging_differs(dossier):
    documents, claims, questions = dossier
    packs = build_all_modes(questions[8]["question"], documents, claims)
    assert "ВЕТВЬ [" in packs["C"].text
    b_lines = {l for l in packs["B"].text.splitlines() if l[:1] == "C"}
    c_lines = {l for l in packs["C"].text.splitlines() if l[:1] == "C"}
    assert b_lines == c_lines   # строки утверждений идентичны, меняется только порядок/группы


def test_mode_a_contains_all_documents_in_order(dossier):
    documents, claims, _ = dossier
    pack = build_full_context(documents, claims)
    positions = [pack.text.index(f"=== {d['document_id']} |") for d in documents]
    assert positions == sorted(positions)
    assert set(pack.claim_ids) == {c["claim_id"] for c in claims}


def test_unknown_mode(dossier):
    documents, claims, _ = dossier
    with pytest.raises(ValueError):
        build_context("D", "вопрос", documents, claims)


def test_shuffle_is_deterministic_and_keeps_content(dossier):
    documents, claims, _ = dossier
    s1, s2 = shuffle_documents(documents, 42), shuffle_documents(documents, 42)
    assert [d["document_id"] for d in s1] == [d["document_id"] for d in s2]
    assert [d["document_id"] for d in s1] != [d["document_id"] for d in documents]
    assert sorted(d["document_id"] for d in s1) == sorted(d["document_id"] for d in documents)


def test_shuffle_does_not_change_retrieval(dossier):
    documents, claims, questions = dossier
    q = questions[9]["question"]
    a = build_context("C", q, documents, claims)
    b = build_context("C", q, shuffle_documents(documents), claims)
    assert a.text == b.text


def test_remove_claim(dossier):
    documents, claims, _ = dossier
    new_docs, new_claims, side = remove_claim(documents, claims, "C13")
    assert "C13" not in {c["claim_id"] for c in new_claims}
    assert "C13" not in build_full_context(new_docs, new_claims).claim_ids
    assert side == []
    assert len(new_claims) == len(claims) - 1
    # исходные данные не изменились
    assert any(c["claim_id"] == "C13" for c in claims)


def test_add_claim_requires_literal_evidence(dossier):
    documents, claims, _ = dossier
    doc = {"document_id": "D11", "title": "t", "source_type": "statement",
           "created_at": "2026-05-19T12:00:00", "author": "x", "text": "Я видел его в 02:40."}
    bad = {"claim_id": "C41", "text": "...", "document_id": "D11", "evidence": "нет такого", "status": "reported"}
    with pytest.raises(ValueError):
        add_claim(documents, claims, bad, doc)
    ok = dict(bad, evidence="Я видел его в 02:40.")
    new_docs, new_claims = add_claim(documents, claims, ok, doc)
    assert len(new_docs) == len(documents) + 1 and new_claims[-1]["claim_id"] == "C41"
