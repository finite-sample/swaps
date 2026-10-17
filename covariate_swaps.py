"""Finite covariate replacements and their cooperative-game diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

MAX_EXACT_PLAYERS = 16


def fit_ols(design: np.ndarray, outcome: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit a unique OLS coefficient vector and return the design condition number."""
    beta, _, rank, singular = np.linalg.lstsq(design, outcome, rcond=None)
    if rank != design.shape[1]:
        raise ValueError("correction produces a rank-deficient regression design")
    return beta, float(singular[0] / singular[-1])


@dataclass
class CorrectionGame:
    """Replace declared, disjoint row sets while keeping outcomes and other rows fixed.

    Both designs include the intercept and every regressor. Only verified-design
    rows belonging to players are used; other rows may be NaN. Column ``target``
    selects the coefficient. Players may be individual rows or correction batches.
    """

    proxy: np.ndarray
    verified: np.ndarray
    outcome: np.ndarray
    players: tuple[tuple[int, ...], ...]
    target: int

    def __post_init__(self) -> None:
        self.proxy = np.array(self.proxy, dtype=float, copy=True)
        self.verified = np.array(self.verified, dtype=float, copy=True)
        self.outcome = np.array(self.outcome, dtype=float, copy=True)
        if self.proxy.ndim != 2 or self.verified.shape != self.proxy.shape:
            raise ValueError("proxy and verified designs must have the same 2D shape")
        n, p = self.proxy.shape
        if (
            not isinstance(self.target, (int, np.integer))
            or not 0 <= self.target < p
            or self.outcome.shape != (n,)
        ):
            raise ValueError("invalid target column or outcome shape")
        if not self.players or any(not rows for rows in self.players):
            raise ValueError("players must be nonempty row sets")
        flat = [i for rows in self.players for i in rows]
        if any(not isinstance(i, (int, np.integer)) or not 0 <= i < n for i in flat):
            raise ValueError("player row indices must be integers within the dataset")
        if len(set(flat)) != len(flat):
            raise ValueError("players must be disjoint and contain no duplicate rows")
        if not (
            np.isfinite(self.proxy).all()
            and np.isfinite(self.outcome).all()
            and np.isfinite(self.verified[flat]).all()
        ):
            raise ValueError("observed inputs and verified player rows must be finite")
        self.players = tuple(tuple(int(i) for i in rows) for rows in self.players)
        fit_ols(self.proxy, self.outcome)

    @property
    def size(self) -> int:
        return len(self.players)

    def design(self, mask: int) -> np.ndarray:
        if not isinstance(mask, (int, np.integer)) or not 0 <= mask < 2**self.size:
            raise ValueError("mask must identify a subset of the declared players")
        result = self.proxy.copy()
        for player, rows in enumerate(self.players):
            if mask & (1 << player):
                index = list(rows)
                result[index] = self.verified[index]
        return result

    def value(self, mask: int) -> float:
        return float(fit_ols(self.design(mask), self.outcome)[0][self.target])

    def additive_scores(self) -> dict[str, np.ndarray]:
        """Single-repair, finite paired-score, and feature-derivative contributions."""
        beta, _ = fit_ols(self.proxy, self.outcome)
        residual = self.outcome - self.proxy @ beta
        gram = self.proxy.T @ self.proxy
        direction = np.linalg.solve(gram, np.eye(gram.shape[0])[:, self.target])
        paired, derivative, singleton = [], [], []
        for j, rows in enumerate(self.players):
            index = list(rows)
            x = self.proxy[index]
            delta = self.verified[index] - x
            linear = delta.T @ residual[index] - x.T @ (delta @ beta)
            finite = linear - delta.T @ (delta @ beta)
            derivative.append(float(direction @ linear))
            paired.append(float(direction @ finite))
            singleton.append(self.value(1 << j) - beta[self.target])
        return {
            "singleton": np.array(singleton),
            "paired_score": np.array(paired),
            "feature_derivative": np.array(derivative),
        }


@dataclass
class ExactGame:
    """Values indexed by binary coalition mask; no Monte Carlo approximation."""

    values: np.ndarray
    condition_numbers: np.ndarray

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=float)
        self.condition_numbers = np.asarray(self.condition_numbers, dtype=float)
        length = self.values.size
        if (
            self.values.ndim != 1
            or length < 2
            or length & (length - 1)
            or self.condition_numbers.shape != self.values.shape
            or not np.isfinite(self.values).all()
            or not np.isfinite(self.condition_numbers).all()
        ):
            raise ValueError("finite values and conditions must cover all 2**m subsets")

    @property
    def size(self) -> int:
        return self.values.size.bit_length() - 1

    @property
    def counts(self) -> np.ndarray:
        return np.array([mask.bit_count() for mask in range(len(self.values))])

    def fixed_size(self) -> list[np.ndarray]:
        counts = self.counts
        return [self.values[counts == k] for k in range(self.size + 1)]

    def marginal_by_size(self) -> np.ndarray:
        """Row i, column k: i's mean marginal change after k other corrections."""
        result = np.zeros((self.size, self.size))
        counts = self.counts
        masks = np.arange(len(self.values))
        for i in range(self.size):
            absent = masks[(masks & (1 << i)) == 0]
            changes = self.values[absent | (1 << i)] - self.values[absent]
            for k in range(self.size):
                result[i, k] = changes[counts[absent] == k].mean()
        return result

    def shapley(self) -> np.ndarray:
        return self.marginal_by_size().mean(axis=1)

    def banzhaf(self) -> np.ndarray:
        weights = np.array([math.comb(self.size - 1, k) for k in range(self.size)])
        return self.marginal_by_size() @ weights / 2 ** (self.size - 1)

    def bernoulli(self, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Exact mean and SD when every player is independently corrected with p."""
        probabilities = np.asarray(probabilities, dtype=float)
        if (
            probabilities.ndim != 1
            or not np.isfinite(probabilities).all()
            or np.any((probabilities < 0) | (probabilities > 1))
        ):
            raise ValueError("probabilities must be a finite vector in [0, 1]")
        groups = [group - self.values[0] for group in self.fixed_size()]
        means = np.array([group.mean() for group in groups])
        seconds = np.array([np.mean(group**2) for group in groups])
        weights = np.array(
            [
                [
                    math.comb(self.size, k) * p**k * (1 - p) ** (self.size - k)
                    for k in range(self.size + 1)
                ]
                for p in probabilities
            ]
        ).reshape(len(probabilities), self.size + 1)
        mean = weights @ means
        return mean + self.values[0], np.sqrt(
            np.maximum(0, weights @ seconds - mean**2)
        )

    def marginal_path(self, probabilities: np.ndarray) -> np.ndarray:
        probabilities = np.asarray(probabilities, dtype=float)
        self.bernoulli(probabilities)
        weights = np.array(
            [
                [
                    math.comb(self.size - 1, k) * p**k * (1 - p) ** (self.size - 1 - k)
                    for k in range(self.size)
                ]
                for p in probabilities
            ]
        ).reshape(len(probabilities), self.size)
        return self.marginal_by_size() @ weights.T


def enumerate_game(game: CorrectionGame) -> ExactGame:
    if game.size > MAX_EXACT_PLAYERS:
        raise ValueError(
            f"exact enumeration accepts at most {MAX_EXACT_PLAYERS} players"
        )
    values = np.empty(2**game.size)
    conditions = np.empty_like(values)
    for mask in range(len(values)):
        try:
            beta, condition = fit_ols(game.design(mask), game.outcome)
        except ValueError as error:
            raise ValueError(f"coalition {mask}: {error}") from error
        values[mask] = beta[game.target]
        conditions[mask] = condition
    return ExactGame(values, conditions)


def additive_values(base: float, contributions: np.ndarray) -> np.ndarray:
    """Evaluate an additive approximation in the same binary-mask ordering."""
    values = np.array([base])
    for contribution in contributions:
        values = np.concatenate([values, values + contribution])
    return values


def permutation_shapley(
    game: CorrectionGame, permutations: int = 1000, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Shapley estimates and per-coordinate Monte Carlo SEs from independent orders.

    SEs describe numerical integration error, not uncertainty about a population
    coefficient. Every sampled order telescopes to the full-player endpoint.
    """
    if permutations < 2:
        raise ValueError("at least two independent permutations are required")
    rng = np.random.default_rng(seed)
    draws = np.empty((permutations, game.size))
    base = game.value(0)
    for draw in draws:
        mask, previous = 0, base
        for player in rng.permutation(game.size):
            mask |= 1 << int(player)
            current = game.value(mask)
            draw[player] = current - previous
            previous = current
    return draws.mean(axis=0), draws.std(axis=0, ddof=1) / math.sqrt(permutations)


def make_benchmark(
    scenario: str, seed: int = 2024, rmse: float = 0.75, n: int = 96, players: int = 8
) -> CorrectionGame:
    """Construct matched-error proxies for the same simulated confounded experiment."""
    if n < 16 or players < 2 or players > n or not np.isfinite(rmse) or rmse < 0:
        raise ValueError("invalid benchmark size, player count or error magnitude")
    rng = np.random.default_rng(seed)
    confounder, control, assignment_noise, noise = rng.normal(size=(4, n))
    treatment = (0.8 * confounder + 0.3 * control + assignment_noise > 0).astype(float)
    outcome = 0.5 * treatment + confounder + 0.3 * control + 0.5 * noise
    verified = np.column_stack([np.ones(n), treatment, confounder, control])
    batches = tuple(
        tuple(map(int, a)) for a in np.array_split(rng.permutation(n), players)
    )
    error = rng.normal(size=n)
    if scenario == "concentrated":
        error[np.setdiff1d(np.arange(n), batches[0])] = 0
    elif scenario == "aligned":
        error = treatment - treatment.mean()
    elif scenario != "diffuse":
        raise ValueError(f"unknown scenario: {scenario}")
    error *= rmse / np.sqrt(np.mean(error**2))
    proxy = verified.copy()
    proxy[:, 2] += error
    return CorrectionGame(proxy, verified, outcome, batches, target=1)
