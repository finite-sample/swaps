"""Decisive checks of the original ab1c17f algorithm and simulation claims."""

import importlib.util
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "original_swaps", ROOT / "original_snapshot" / "replicate_final.py"
)
ORIGINAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORIGINAL)


def coalition_check():
    x, proxy, w, y, vi = ORIGINAL.dgp(30, 5, "ugly", np.random.default_rng(2026))
    m = len(vi)
    values = {}
    for mask in range(2**m):
        selected = np.array([bool(mask & (1 << j)) for j in range(m)])
        corrected = proxy.copy()
        corrected[vi[selected]] = x[vi[selected]]
        values[mask] = ORIGINAL.ols(corrected, w, y)[0]
    shapley = np.zeros(m)
    banzhaf = np.zeros(m)
    for j in range(m):
        for mask in range(2**m):
            if mask & (1 << j):
                continue
            size = mask.bit_count()
            marginal = values[mask | (1 << j)] - values[mask]
            shapley[j] += marginal / (m * math.comb(m - 1, size))
            banzhaf[j] += marginal / 2 ** (m - 1)
    endpoint_gap = values[2**m - 1] - values[0]
    np.testing.assert_allclose(shapley.sum(), endpoint_gap, atol=1e-12)
    assert abs(banzhaf.sum() - endpoint_gap) > 1e-5
    approximate = ORIGINAL.compute_sim(
        x, proxy, w, y, vi, B=20000, rng=np.random.default_rng(2027)
    )
    return {
        "n": 30,
        "validation_size": m,
        "endpoint_gap": endpoint_gap,
        "exact_shapley": shapley.tolist(),
        "exact_uniform_coalition_banzhaf": banzhaf.tolist(),
        "original_algorithm_20000_draws": approximate.tolist(),
        "shapley_sum": shapley.sum(),
        "banzhaf_sum": banzhaf.sum(),
        "banzhaf_efficiency_error": banzhaf.sum() - endpoint_gap,
        "mc_max_error_to_banzhaf": np.max(np.abs(approximate - banzhaf)),
        "mc_max_error_to_shapley": np.max(np.abs(approximate - shapley)),
    }


def dgp_check():
    x, proxy, w, y, _ = ORIGINAL.dgp(200000, 500, "ugly", np.random.default_rng(2028))
    ugly_oracle = ORIGINAL.ols(x, w, y)[0]
    projection = np.linalg.lstsq(
        np.column_stack((np.ones(len(x)), proxy, w)), x, rcond=None
    )[0]
    x, proxy, w, y, _ = ORIGINAL.dgp(
        200000, 500, "heterogeneous", np.random.default_rng(2029)
    )
    design = np.column_stack((np.ones(len(x)), proxy, w))
    linear_mu = design @ np.linalg.lstsq(design, x, rcond=None)[0]
    true_mu = proxy / (1 + (0.1 + 1.5 * np.abs(w[:, 0])) ** 2)
    return {
        "ugly_oracle_coefficient": ugly_oracle,
        "ugly_linear_calibration_coefficients": projection.tolist(),
        "heterogeneous_linear_conditional_mean_rmse": np.sqrt(
            np.mean((linear_mu - true_mu) ** 2)
        ),
        "heterogeneous_linear_projection_corrected_coefficient": ORIGINAL.ols(
            linear_mu, w, y
        )[0],
        "heterogeneous_true_conditional_mean_corrected_coefficient": ORIGINAL.ols(
            true_mu, w, y
        )[0],
    }


def calibration_uncertainty_check():
    estimates, reported_ses = [], []
    rng = np.random.default_rng(2030)
    for _ in range(300):
        x, proxy, w, y, vi = ORIGINAL.dgp(2000, 500, "heterogeneous", rng)
        design = np.column_stack((np.ones(len(x)), proxy, w))
        gamma = np.linalg.lstsq(design[vi], x[vi], rcond=None)[0]
        coefficient, se = ORIGINAL.ols(design @ gamma, w, y)
        estimates.append(coefficient)
        reported_ses.append(se)
    estimates = np.array(estimates)
    reported_ses = np.array(reported_ses)
    return {
        "replications": 300,
        "mean_estimate": estimates.mean(),
        "empirical_standard_deviation": estimates.std(ddof=1),
        "mean_reported_hc1_se": reported_ses.mean(),
        "coverage_using_reported_hc1": np.mean(
            np.abs(estimates - 1) <= 1.96 * reported_ses
        ),
    }


def main():
    result = {
        "coalition_check": coalition_check(),
        "dgp_check": dgp_check(),
        "calibration_uncertainty": calibration_uncertainty_check(),
        "set_membership_bug": {
            "original_set_input": np.isin(np.array([1, 2, 3]), {1, 3}).tolist(),
            "array_input": np.isin(np.array([1, 2, 3]), [1, 3]).tolist(),
        },
    }
    rendered = json.dumps(result, indent=2)
    (ROOT / "verification_results.json").write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
