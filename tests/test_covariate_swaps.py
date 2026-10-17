from itertools import combinations, permutations
from math import comb

import numpy as np
import pytest

from covariate_swaps import (
    CorrectionGame,
    ExactGame,
    additive_values,
    enumerate_game,
    fit_ols,
    make_benchmark,
    permutation_shapley,
)


def test_unanimity_separates_shapley_from_banzhaf():
    values = np.zeros(8)
    values[-1] = 1
    exact = ExactGame(values, np.ones(8))
    np.testing.assert_allclose(exact.shapley(), 1 / 3)
    np.testing.assert_allclose(exact.banzhaf(), 1 / 4)
    nodes, weights = np.polynomial.legendre.leggauss(3)
    integral = exact.marginal_path((nodes + 1) / 2) @ (weights / 2)
    np.testing.assert_allclose(integral, exact.shapley())


def test_exact_shapley_matches_independent_permutations():
    game = make_benchmark("aligned", n=32, players=4)
    exact = enumerate_game(game)
    draws = []
    for order in permutations(range(4)):
        mask = 0
        row = np.zeros(4)
        for j in order:
            row[j] = exact.values[mask | (1 << j)] - exact.values[mask]
            mask |= 1 << j
        draws.append(row)
    np.testing.assert_allclose(exact.shapley(), np.mean(draws, axis=0))
    assert exact.shapley().sum() == pytest.approx(exact.values[-1] - exact.values[0])
    estimate, se = permutation_shapley(game, permutations=400, seed=29)
    assert np.all(np.abs(estimate - exact.shapley()) < 5 * se)
    assert estimate.sum() == pytest.approx(exact.values[-1] - exact.values[0])


def test_bernoulli_mixes_fixed_budget_distributions():
    exact = ExactGame(np.array([2, -1, 4, 7, 0, 3, -2, 9]), np.ones(8))
    p = np.array([0, 0.1, 0.5, 0.9, 1])
    mean, sd = exact.bernoulli(p)
    for j, probability in enumerate(p):
        weights = np.array(
            [
                probability ** m.bit_count() * (1 - probability) ** (3 - m.bit_count())
                for m in range(8)
            ]
        )
        expected = weights @ exact.values
        assert mean[j] == pytest.approx(expected)
        assert sd[j] ** 2 == pytest.approx(weights @ (exact.values - expected) ** 2)
    assert [len(v) for v in exact.fixed_size()] == [1, 3, 3, 1]


def test_finite_normal_equation_identity_and_paired_score():
    game = make_benchmark("diffuse", n=32, players=4)
    beta, _ = fit_ols(game.proxy, game.outcome)
    gram = game.proxy.T @ game.proxy
    residual = game.outcome - game.proxy @ beta
    contributions = game.additive_scores()["paired_score"]
    for mask in range(16):
        delta = game.design(mask) - game.proxy
        q = delta.T @ residual - game.proxy.T @ delta @ beta - delta.T @ delta @ beta
        change = game.proxy.T @ delta + delta.T @ game.proxy + delta.T @ delta
        refit, _ = fit_ols(game.design(mask), game.outcome)
        np.testing.assert_allclose(
            np.linalg.solve(gram + change, q), refit - beta, atol=1e-13
        )
        score = sum(contributions[j] for j in range(4) if mask & (1 << j))
        assert score == pytest.approx(np.linalg.solve(gram, q)[game.target], abs=1e-13)


def test_zero_error_and_single_active_player_are_additive():
    zero = enumerate_game(make_benchmark("diffuse", rmse=0, players=3))
    np.testing.assert_allclose(zero.values, zero.values[0])
    np.testing.assert_allclose(zero.shapley(), 0, atol=1e-14)
    game = make_benchmark("concentrated", players=4)
    exact = enumerate_game(game)
    np.testing.assert_allclose(
        exact.values,
        additive_values(exact.values[0], game.additive_scores()["singleton"]),
        atol=1e-13,
    )


def test_partial_validation_endpoint_ignores_unknown_rows():
    full = make_benchmark("diffuse", n=32, players=4)
    verified = np.full_like(full.verified, np.nan)
    rows = list(full.players[0])
    verified[rows] = full.verified[rows]
    game = CorrectionGame(full.proxy, verified, full.outcome, (full.players[0],), 1)
    exact = enumerate_game(game)
    assert exact.values[-1] == full.value(1)
    np.testing.assert_allclose(exact.shapley(), [full.value(1) - full.value(0)])


def test_intermediate_rank_failure_is_reported():
    proxy = np.array([[1, 0], [1, 1], [1, 0]], dtype=float)
    verified = np.array([[1, 1], [1, 0], [1, 0]], dtype=float)
    game = CorrectionGame(proxy, verified, np.arange(3), ((0,), (1,)), 1)
    with pytest.raises(ValueError, match="coalition 2.*rank-deficient"):
        enumerate_game(game)


def test_matched_rmse_and_amplitude_scaling():
    games = [
        make_benchmark(s, players=4) for s in ("diffuse", "concentrated", "aligned")
    ]
    for game in games:
        np.testing.assert_array_equal(game.verified, games[0].verified)
        np.testing.assert_array_equal(game.outcome, games[0].outcome)
        assert game.players == games[0].players
        assert np.sqrt(
            np.mean((game.proxy[:, 2] - game.verified[:, 2]) ** 2)
        ) == pytest.approx(0.75)
    errors = []
    for amplitude in (0.001, 0.0005):
        game = make_benchmark("diffuse", rmse=amplitude, players=3)
        exact = enumerate_game(game)
        approx = additive_values(
            exact.values[0], game.additive_scores()["paired_score"]
        )
        errors.append(np.linalg.norm(exact.values - approx))
    assert 3.8 < errors[0] / errors[1] < 4.2


def test_invalid_inputs_and_enumeration_limit():
    game = make_benchmark("diffuse", players=3)
    with pytest.raises(ValueError, match="disjoint"):
        CorrectionGame(game.proxy, game.verified, game.outcome, ((0,), (0,)), 1)
    with pytest.raises(ValueError, match="mask"):
        game.design(8)
    with pytest.raises(ValueError, match="at most"):
        enumerate_game(make_benchmark("diffuse", players=17))
    with pytest.raises(ValueError, match="probabilities"):
        enumerate_game(game).bernoulli(np.array([1.1]))


def test_path_dispersion_is_invariant_to_a_large_offset():
    values = np.array([0.0, 1.0, 3.0, 4.0])
    exact = ExactGame(values, np.ones(4))
    shifted = ExactGame(values + 1e12, np.ones(4))
    p = np.array([0.0, 0.17, 0.5, 0.82, 1.0])
    _, expected_sd = exact.bernoulli(p)
    _, shifted_sd = shifted.bernoulli(p)
    np.testing.assert_allclose(shifted_sd, expected_sd, atol=1e-14)
    constant = ExactGame(np.full(8, 1e12), np.ones(8))
    np.testing.assert_array_equal(constant.bernoulli(p)[1], np.zeros(5))


@pytest.mark.parametrize("n_rows", [3, 8, 100])
def test_hidden_correction_interaction_counterexample(n_rows):
    proxy = np.zeros((n_rows, 3))
    proxy[:, 0] = 1
    proxy[-2, 1] = 1
    proxy[-1, 2] = 1
    outcome = np.zeros(n_rows)
    outcome[-2:] = 1
    verified = proxy.copy()
    verified[-2:, 2] = [1, 2]
    game = CorrectionGame(proxy, verified, outcome, ((n_rows - 2,), (n_rows - 1,)), 1)
    exact = enumerate_game(game)
    np.testing.assert_allclose(exact.values, [1, 0, 1, 0.5], atol=1e-14)
    interaction = exact.values[3] - exact.values[1] - exact.values[2] + exact.values[0]
    assert interaction == pytest.approx(0.5)
    additive = additive_values(exact.values[0], game.additive_scores()["singleton"])
    assert additive[-1] < 0.25 < exact.values[-1]
    no_error = CorrectionGame(proxy, proxy, outcome, game.players, 1)
    np.testing.assert_allclose(enumerate_game(no_error).values, 1)
    np.testing.assert_array_equal(proxy[:-2], verified[:-2])


def test_random_pilot_miss_probability_by_enumeration():
    n_rows = 8
    special = {n_rows - 2, n_rows - 1}
    for sample_size in range(n_rows + 1):
        pilots = list(combinations(range(n_rows), sample_size))
        misses = sum(not special.intersection(pilot) for pilot in pilots)
        expected = (n_rows - sample_size) * (n_rows - sample_size - 1)
        expected /= n_rows * (n_rows - 1)
        assert misses / len(pilots) == pytest.approx(expected)
    assert comb(98, 10) / comb(100, 10) == pytest.approx(89 / 110)
