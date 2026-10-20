"""Тесты p-adic модуля (участник 2)."""
import itertools

import pytest

from src.context_builder import load_dossier
from src.padic import (assign_codes, common_prefix_len, decode, encode, padic_distance,
                       padic_distance_codes, group_by_branch)

ALL_PATHS = list(itertools.product(range(3), repeat=3))


def test_encode_formula():
    assert encode([1, 2, 1]) == 1 + 2 * 3 + 1 * 9 == 16
    assert encode([0, 0, 0]) == 0 and encode([2, 2, 2]) == 26


def test_encode_decode_roundtrip():
    assert sorted(encode(p) for p in ALL_PATHS) == list(range(27))
    for p in ALL_PATHS:
        assert decode(encode(p)) == p


@pytest.mark.parametrize("bad", [[0, 0], [0, 0, 3], [-1, 0, 0], [0, 0, 0, 0]])
def test_invalid_path(bad):
    with pytest.raises(ValueError):
        encode(bad)


def test_same_path_distance_zero():
    for p in ALL_PATHS:
        assert padic_distance(p, p) == 0


def test_differs_at_level_0():
    # разный тип источника — максимальное расстояние, остальное неважно
    assert padic_distance([0, 1, 2], [1, 1, 2]) == 1
    assert padic_distance([0, 0, 0], [2, 0, 0]) == 1


def test_differs_at_level_1():
    assert padic_distance([1, 0, 2], [1, 2, 2]) == pytest.approx(1 / 3)


def test_differs_at_level_2():
    assert padic_distance([1, 2, 0], [1, 2, 1]) == pytest.approx(1 / 9)


def test_prefix_not_suffix():
    # совпадает время, но не источник -> всё равно 1 (важен общий НАЧАЛЬНЫЙ отрезок)
    assert common_prefix_len([0, 2, 1], [1, 2, 1]) == 0
    assert padic_distance([0, 2, 1], [1, 2, 1]) == 1


def test_not_absolute_difference():
    # коды 0 и 9 рядом в p-adic смысле ([0,0,0] и [0,0,1]), а abs дал бы 9
    assert padic_distance([0, 0, 0], [0, 0, 1]) == pytest.approx(1 / 9)
    # коды 0 и 1 далеко ([0,0,0] и [1,0,0]), хотя abs = 1 — минимальный
    assert padic_distance([0, 0, 0], [1, 0, 0]) == 1


def test_path_distance_equals_3adic_norm_of_codes():
    for p, q in itertools.product(ALL_PATHS, repeat=2):
        assert padic_distance(p, q) == pytest.approx(padic_distance_codes(encode(p), encode(q)))


def test_ultrametric_inequality():
    for x, y, z in itertools.product(ALL_PATHS, repeat=3):
        assert padic_distance(x, z) <= max(padic_distance(x, y), padic_distance(y, z)) + 1e-12


def test_symmetry():
    for p, q in itertools.product(ALL_PATHS, repeat=2):
        assert padic_distance(p, q) == padic_distance(q, p)


# ---------- разметка реального досье

@pytest.fixture(scope="module")
def codes():
    documents, claims, _ = load_dossier()
    return assign_codes(claims, documents)


def test_every_claim_has_valid_path(codes):
    _, claims, _ = load_dossier()
    assert set(codes) == {c["claim_id"] for c in claims}
    for c in codes.values():
        assert c.code == encode(c.path)


@pytest.mark.parametrize("cid, path", [
    ("C12", (0, 0, 1)),  # журнал, факт, ночь события
    ("C05", (0, 0, 0)),  # служебная записка, факт, до события
    ("C26", (1, 1, 1)),  # показание Семёнова про ночь
    ("C39", (1, 2, 1)),  # спорное свидетельство (как в примере задания)
    ("C22", (2, 0, 2)),  # правило, вне времени
    ("C24", (2, 2, 2)),  # исключение из правила
    ("C38", (2, 1, 1)),  # вывод -> интерпретация
    ("C35", (1, 0, 2)),  # переписка в понедельник -> после события
])
def test_known_paths(codes, cid, path):
    assert codes[cid].path == path


def test_branches_sorted_by_tree_then_time(codes):
    ids = ["C15", "C12", "C24", "C06", "C13", "C11"]
    branches = group_by_branch(ids, codes)
    assert [p for p, _ in branches] == sorted(p for p, _ in branches)
    assert dict(branches)[(0, 0, 1)] == ["C12", "C13", "C15"]
    assert dict(branches)[(0, 0, 0)] == ["C06", "C11"]
