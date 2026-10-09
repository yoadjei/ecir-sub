"""metric functions, no io. a ranking is a list of docids with rank 1 first.

rel maps docid to an integer grade, 0 is judged non relevant and a docid
missing from the dict is unjudged.
"""
import math


def _dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def _ideal_dcg(grades, k: int) -> float:
    positives = sorted((float(g) for g in grades if g > 0), reverse=True)
    return _dcg(positives[:k])


def judged_at_k(ranking: list[str], rel: dict[str, int], k: int) -> float:
    """share of the top k that has any judgement, relevant or not."""
    return sum(1 for d in ranking[:k] if d in rel) / k


def ndcg_at_k(ranking: list[str], rel: dict[str, int], k: int) -> float:
    """trec_eval style ndcg, unjudged counts as non relevant, ideal from all judged docs.

    0.0 when the query has no positive judgement, same as trec_eval."""
    gains = [float(rel.get(d, 0)) for d in ranking[:k]]
    idcg = _ideal_dcg(rel.values(), k)
    return _dcg(gains) / idcg if idcg > 0 else 0.0


def condensed_ndcg(ranking: list[str], rel: dict[str, int], k: int) -> float:
    """sakai's condensed ndcg, drop unjudged docs then score the top k of what is left."""
    judged_only = [d for d in ranking if d in rel]
    return ndcg_at_k(judged_only, rel, k)


def _with_unjudged_relevant(ranking: list[str], rel: dict[str, int], k: int) -> dict[str, int]:
    """rel plus every unjudged doc in the top k at grade 1."""
    augmented = dict(rel)
    for d in ranking[:k]:
        if d not in augmented:
            augmented[d] = 1
    return augmented


def all_relevant_ndcg(ranking: list[str], rel: dict[str, int], k: int) -> float:
    """every unjudged doc in the top k counts as relevant (grade 1), ideal recomputed.

    not an upper bound, the ideal can grow faster than the run's dcg."""
    augmented = _with_unjudged_relevant(ranking, rel, k)
    gains = [float(augmented.get(d, 0)) for d in ranking[:k]]
    idcg = _ideal_dcg(augmented.values(), k)
    return _dcg(gains) / idcg if idcg > 0 else 0.0


def optimistic_fixed_ndcg(ranking: list[str], rel: dict[str, int], k: int) -> float:
    """all unjudged top k docs relevant but the ideal kept from the original qrels, can exceed 1."""
    augmented = _with_unjudged_relevant(ranking, rel, k)
    gains = [float(augmented.get(d, 0)) for d in ranking[:k]]
    idcg = _ideal_dcg(rel.values(), k)
    return _dcg(gains) / idcg if idcg > 0 else 0.0


def optimistic_ndcg(ranking: list[str], rel: dict[str, int], k: int) -> float:
    """exact maximum ndcg over every 0/1 labelling of the unjudged docs in the top k.

    for a fixed number s of newly relevant docs the best ones are the s highest ranked,
    so only s = 0 .. number of unjudged needs checking. s = 0 is the standard ndcg."""
    gains = [float(rel.get(d, 0)) for d in ranking[:k]]
    unjudged = [i for i, d in enumerate(ranking[:k]) if d not in rel]
    grades = list(rel.values())
    best = 0.0
    for s in range(len(unjudged) + 1):
        if s > 0:
            gains[unjudged[s - 1]] = 1.0
            grades.append(1)
        idcg = _ideal_dcg(grades, k)
        if idcg > 0:
            best = max(best, _dcg(gains) / idcg)
    return best


def expected_ndcg(ranking: list[str], prob: dict[str, float], k: int) -> float:
    """ndcg with fractional gains, the gain is the probability of relevance.

    ideal is the dcg of the k largest probabilities, docs missing from prob get gain 0."""
    gains = [float(prob.get(d, 0.0)) for d in ranking[:k]]
    idcg = _ideal_dcg(prob.values(), k)
    return _dcg(gains) / idcg if idcg > 0 else 0.0
