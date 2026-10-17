"""Check an OLS substitution identity and two distinct one-step approximations."""

import json
from itertools import combinations

import numpy as np


def changes(x, y, delta):
    a = x.T @ x
    beta = np.linalg.solve(a, x.T @ y)
    residual = y - x @ beta
    b = x.T @ delta + delta.T @ x + delta.T @ delta
    first = delta.T @ residual - x.T @ delta @ beta
    score = first - delta.T @ delta @ beta
    exact = np.linalg.solve(a + b, score)
    refit = np.linalg.lstsq(x + delta, y, rcond=None)[0] - beta
    np.testing.assert_allclose(exact, refit, atol=1e-10, rtol=1e-10)
    return exact, np.linalg.solve(a, first), np.linalg.solve(a, score)


def main():
    rng = np.random.default_rng(2026)
    max_identity_error = 0.0
    for _ in range(100):
        x = np.column_stack((np.ones(40), rng.normal(size=(40, 3))))
        y = rng.normal(size=40)
        delta = np.zeros_like(x)
        delta[:, 2:] = rng.normal(size=(40, 2))
        delta[rng.random(40) < 0.5] = 0
        exact, _, _ = changes(x, y, delta)
        refit = np.linalg.lstsq(x + delta, y, rcond=None)[0]
        baseline = np.linalg.lstsq(x, y, rcond=None)[0]
        max_identity_error = max(
            max_identity_error, float(np.max(np.abs(exact - (refit - baseline))))
        )

    # Prespecified deterministic stress example, not a performance benchmark.
    treatment = np.repeat([0.0, 1.0], 6)
    proxy = np.tile([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0], 2) + treatment
    x = np.column_stack((np.ones(12), treatment, proxy))
    y = 0.2 * treatment + proxy + np.tile([0.2, -0.3, 0.1], 4)
    direction = np.zeros_like(x)
    direction[[0, 1, 6, 7], 2] = [1.0, -1.0, -1.0, 1.0]
    scale_results = []
    for scale in [0.001, 0.1, 1.0, 3.0]:
        exact, amplitude, paired_score = changes(x, y, scale * direction)
        scale_results.append(
            {
                "error_scale": scale,
                "exact_treatment_change": float(exact[1]),
                "feature_derivative": float(amplitude[1]),
                "paired_score_one_step": float(paired_score[1]),
            }
        )

    all_subsets = []
    for subset in combinations([0, 1, 6, 7], 2):
        delta = np.zeros_like(x)
        delta[list(subset)] = direction[list(subset)]
        exact, _, paired_score = changes(x, y, delta)
        singleton_sum = 0.0
        for unit in subset:
            single = np.zeros_like(x)
            single[unit] = direction[unit]
            singleton_sum += changes(x, y, single)[0][1]
        all_subsets.append(
            {
                "subset": subset,
                "exact": float(exact[1]),
                "sum_singletons": float(singleton_sum),
                "paired_score_one_step": float(paired_score[1]),
            }
        )
    assert max(abs(row["exact"] - row["sum_singletons"]) for row in all_subsets) > 0.01
    print(
        json.dumps(
            {
                "identity_checks": 100,
                "max_identity_error": max_identity_error,
                "scale_results": scale_results,
                "all_two_record_subsets": all_subsets,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
