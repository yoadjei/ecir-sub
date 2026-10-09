import numpy as np
import pytest

from src.utils import bootstrap as B


def test_bootstrap_ci_contains_mean_and_is_deterministic():
    x = np.arange(100, dtype=float)
    m1, lo1, hi1 = B.bootstrap_ci(x, n_resamples=2000, seed=13)
    m2, lo2, hi2 = B.bootstrap_ci(x, n_resamples=2000, seed=13)
    assert (m1, lo1, hi1) == (m2, lo2, hi2)
    assert lo1 < m1 < hi1
    assert m1 == pytest.approx(49.5)


def test_bootstrap_diff_ci_excludes_zero_for_separated_groups():
    a = np.full(50, 0.8)
    b = np.full(50, 0.2)
    d, lo, hi = B.bootstrap_diff_ci(a, b, n_resamples=1000, seed=13)
    assert d == pytest.approx(0.6)
    assert lo > 0


def test_group_means_diff_ci_excludes_zero_and_is_deterministic():
    group_a = [np.full(30, 0.8), np.full(30, 0.6)]
    group_b = [np.full(30, 0.2), np.full(30, 0.4)]
    r1 = B.bootstrap_group_means_diff(group_a, group_b, n_resamples=500, seed=13)
    r2 = B.bootstrap_group_means_diff(group_a, group_b, n_resamples=500, seed=13)
    assert r1 == r2
    obs, lo, hi = r1
    assert obs == pytest.approx(0.4)
    assert lo > 0


def test_paired_randomisation_p_small_for_clear_difference():
    rng = np.random.default_rng(0)
    x = rng.normal(0.5, 0.05, 60)
    y = x - 0.2
    assert B.paired_randomisation_p(x, y, n_perm=2000, seed=13) < 0.01


def test_paired_randomisation_p_large_for_no_difference():
    rng = np.random.default_rng(0)
    x = rng.normal(0.5, 0.05, 60)
    y = x + rng.normal(0, 0.05, 60)
    assert B.paired_randomisation_p(x, y, n_perm=2000, seed=13) > 0.05


def test_holm_adjustment():
    adj = B.holm([0.01, 0.04, 0.03])
    # sorted: 0.01*3 = 0.03, 0.03*2 = 0.06, 0.04*1 = 0.04, made monotone: 0.03, 0.06, 0.06
    assert adj == pytest.approx([0.03, 0.06, 0.06])
