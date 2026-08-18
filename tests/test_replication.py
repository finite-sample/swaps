from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import replicate_final as replication


def test_configuration_rejects_intractable_exact_enumeration() -> None:
    args = argparse.Namespace(
        fast=False,
        outdir="generated",
        clusters=24,
        cluster_size=None,
        seed=None,
    )

    with pytest.raises(ValueError, match="at most 22"):
        replication.configuration(args)
    with pytest.raises(ValueError, match="at most 22"):
        replication.exact_subset_sums(np.zeros(23))


def test_scenarios_match_accuracy_but_not_contribution_concentration() -> None:
    experiments = {
        scenario: replication.make_experiment(scenario)
        for scenario in replication.SCENARIOS
    }
    metrics = [replication.prediction_metrics(value) for value in experiments.values()]

    assert np.allclose(metrics, metrics[0], atol=1e-12)
    assert np.isclose(metrics[0][0], replication.TARGET_RMSE)

    diffuse = experiments["diffuse"].contributions
    concentrated = experiments["concentrated"].contributions
    assert np.isclose(diffuse.sum(), replication.TOTAL_DISTORTION)
    assert np.isclose(concentrated.sum(), replication.TOTAL_DISTORTION)
    assert np.std(concentrated) > np.std(diffuse) + 0.005


def test_cluster_contributions_sum_to_the_ate_gap() -> None:
    for scenario in replication.SCENARIOS:
        experiment = replication.make_experiment(scenario)
        gold_ate = replication.difference_in_means(
            experiment.gold, experiment.treatment
        )
        proxy_ate = replication.difference_in_means(
            experiment.proxy, experiment.treatment
        )
        assert np.isclose(experiment.contributions.sum(), gold_ate - proxy_ate)


def test_fixed_budget_mean_path_is_endpoint_interpolation() -> None:
    experiment = replication.make_experiment("concentrated")
    distributions = replication.raw_swap_distributions(experiment)
    proxy_ate = replication.difference_in_means(experiment.proxy, experiment.treatment)
    gold_ate = replication.difference_in_means(experiment.gold, experiment.treatment)
    clusters = len(experiment.contributions)

    for size, distribution in enumerate(distributions):
        expected = proxy_ate + (size / clusters) * (gold_ate - proxy_ate)
        assert np.isclose(np.mean(distribution), expected)


def test_scaled_swap_distribution_is_design_unbiased() -> None:
    for scenario in replication.SCENARIOS:
        experiment = replication.make_experiment(scenario)
        distributions = replication.scaled_swap_distributions(experiment)
        gold_ate = replication.difference_in_means(
            experiment.gold, experiment.treatment
        )
        for size in range(1, len(distributions)):
            assert np.isclose(np.mean(distributions[size]), gold_ate)


def test_exact_distribution_matches_finite_population_variance() -> None:
    experiment = replication.make_experiment("concentrated")
    raw = replication.raw_swap_distributions(experiment)
    scaled = replication.scaled_swap_distributions(experiment)
    size = 5

    assert np.isclose(
        np.var(raw[size]),
        replication.finite_population_variance(
            experiment.contributions, size, scaled=False
        ),
    )
    assert np.isclose(
        np.var(scaled[size]),
        replication.finite_population_variance(
            experiment.contributions, size, scaled=True
        ),
    )


def test_subset_enumeration_has_every_fixed_size_coalition() -> None:
    values = np.array([0.1, 0.2, 0.4, 0.8])
    distributions = replication.exact_subset_sums(values)

    for size, distribution in enumerate(distributions):
        assert len(distribution) == math.comb(len(values), size)
    assert np.allclose(np.sort(distributions[2]), [0.3, 0.5, 0.6, 0.9, 1.0, 1.2])


def test_same_mean_can_hide_different_swap_distributions() -> None:
    diffuse = replication.make_experiment("diffuse")
    concentrated = replication.make_experiment("concentrated")
    diffuse_distribution = replication.raw_swap_distributions(diffuse)[5]
    concentrated_distribution = replication.raw_swap_distributions(concentrated)[5]

    assert np.isclose(diffuse_distribution.mean(), concentrated_distribution.mean())
    assert np.isclose(np.std(diffuse_distribution), 0.0, atol=1e-12)
    assert np.std(concentrated_distribution) > 0.01


def test_cli_writes_all_declared_outputs(tmp_path: Path) -> None:
    command = [
        sys.executable,
        "replicate_final.py",
        "--fast",
        "--outdir",
        str(tmp_path),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)

    expected = {
        "fig1_same_mean.png",
        "fig2_budget_distributions.png",
        "macros.tex",
        "metadata.json",
        "table_budgets.csv",
        "table_budgets.tex",
        "table_scenarios.csv",
        "table_scenarios.tex",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected
