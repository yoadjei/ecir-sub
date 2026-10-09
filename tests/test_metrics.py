import itertools
import math
import random

import pytest

from src.utils import metrics as M

try:
    import ir_measures
    from ir_measures import nDCG
    HAS_IR_MEASURES = True
except ImportError:
    HAS_IR_MEASURES = False


REL = {"d1": 1, "d2": 1, "d3": 0, "d9": 1}
RANK = ["d3", "d1", "u1", "d2", "u2", "u3", "d7", "u4", "u5", "u6", "d9"]


def test_judged_at_k():
    assert M.judged_at_k(RANK, REL, 10) == pytest.approx(0.3)
    assert M.judged_at_k([], REL, 10) == 0.0


@pytest.mark.skipif(not HAS_IR_MEASURES, reason="ir_measures not installed")
def test_ndcg_matches_ir_measures_on_standard_case():
    qrels = {"q": REL}
    run = {"q": {d: float(len(RANK) - i) for i, d in enumerate(RANK)}}
    ref = ir_measures.calc_aggregate([nDCG @ 10], qrels, run)[nDCG @ 10]
    assert M.ndcg_at_k(RANK, REL, 10) == pytest.approx(ref, abs=1e-6)


def test_ndcg_zero_when_no_positives():
    assert M.ndcg_at_k(RANK, {"d3": 0}, 10) == 0.0


def test_condensed_removes_unjudged_then_scores():
    cond = M.condensed_ndcg(RANK, REL, 10)
    expected_dcg = 1 / math.log2(3) + 1 / math.log2(4) + 1 / math.log2(5)
    ideal = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert cond == pytest.approx(expected_dcg / ideal)


def _brute_force_optimistic(ranking, rel, k):
    """max ndcg over every 0/1 labelling of the unjudged docs in the top k."""
    top = ranking[:k]
    unjudged = [i for i, d in enumerate(top) if d not in rel]
    positives = [g for g in rel.values() if g > 0]
    best = 0.0
    for labels in itertools.product([0, 1], repeat=len(unjudged)):
        gains = [float(rel.get(d, 0)) for d in top]
        for i, label in zip(unjudged, labels):
            gains[i] = float(label)
        ideal = sorted(positives + [1] * sum(labels), reverse=True)[:k]
        idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
        if idcg > 0:
            best = max(best, sum(g / math.log2(i + 2) for i, g in enumerate(gains)) / idcg)
    return best


def test_all_relevant_adds_unjudged_in_topk_as_relevant_and_recomputes_ideal():
    val = M.all_relevant_ndcg(RANK, REL, 10)
    gains = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sum(1 / math.log2(i + 2) for i in range(10))
    assert val == pytest.approx(dcg / ideal)


def test_optimistic_fixed_uses_original_qrels_ideal():
    val = M.optimistic_fixed_ndcg(RANK, REL, 10)
    gains = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sum(1 / math.log2(i + 2) for i in range(3))
    assert val == pytest.approx(dcg / ideal)


def test_optimistic_is_not_below_lower_when_all_relevant_would_be():
    rel = {"a": 1, "b": 1, "c": 1, "n": 0}
    ranking = ["a", "b", "c", "n"] + [f"u{i}" for i in range(6)]
    assert M.ndcg_at_k(ranking, rel, 10) == pytest.approx(1.0)
    assert M.all_relevant_ndcg(ranking, rel, 10) < 1.0
    assert M.optimistic_ndcg(ranking, rel, 10) == pytest.approx(1.0)


def test_optimistic_labels_everything_when_the_only_positive_is_buried():
    rel = {"p": 1}
    ranking = [f"u{i}" for i in range(9)] + ["p"]
    val = M.optimistic_ndcg(ranking, rel, 10)
    assert val == pytest.approx(1.0)
    assert val > M.ndcg_at_k(ranking, rel, 10)
    assert val == pytest.approx(_brute_force_optimistic(ranking, rel, 10))


def test_optimistic_zero_when_no_ideal_can_be_positive():
    assert M.optimistic_ndcg(["d3", "d7"], {"d3": 0, "d7": 0}, 10) == 0.0
    assert M.optimistic_ndcg([], {"d1": 1}, 10) == 0.0


def test_optimistic_matches_brute_force_and_bounds_lower_on_random_cases():
    rng = random.Random(7)
    for case in range(200):
        k = rng.choice([3, 5, 10])
        n_unjudged = rng.randint(0, min(6, k))
        n_judged = rng.randint(0, k - n_unjudged)
        pool = {f"j{i}": rng.choice([0, 1, 1, 2]) for i in range(rng.randint(1, 8))}
        judged_in_run = rng.sample(sorted(pool), min(n_judged, len(pool)))
        ranking = judged_in_run + [f"u{i}" for i in range(n_unjudged)]
        rng.shuffle(ranking)
        val = M.optimistic_ndcg(ranking, pool, k)
        assert val == pytest.approx(_brute_force_optimistic(ranking, pool, k)), case
        assert val >= M.ndcg_at_k(ranking, pool, k) - 1e-9, case


def test_expected_ndcg_reduces_to_ndcg_for_hard_labels():
    probs = {d: float(r) for d, r in REL.items()}
    assert M.expected_ndcg(RANK, probs, 10) == pytest.approx(M.ndcg_at_k(RANK, REL, 10))


def test_expected_ndcg_with_fractional_gains():
    probs = {"d1": 0.5, "d2": 1.0}
    rank = ["d1", "d2"]
    dcg = 0.5 / math.log2(2) + 1.0 / math.log2(3)
    ideal = 1.0 / math.log2(2) + 0.5 / math.log2(3)
    assert M.expected_ndcg(rank, probs, 10) == pytest.approx(dcg / ideal)
