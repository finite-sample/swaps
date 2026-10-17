import csv
import json
from pathlib import Path

import numpy as np
import pytest

from covariate_swaps import CorrectionGame, fit_ols
from replicate_archives import METHODS, hc0_scale, load_design, row_scores


@pytest.mark.parametrize("target", [1, 2])
def test_sufficient_statistic_singletons_match_direct_refits(target):
    rng = np.random.default_rng(93)
    x = np.column_stack([np.ones(40), rng.normal(size=(40, 3))])
    z = x.copy()
    z[:, 2:] += rng.normal(size=(40, 2))
    y = x @ np.array([1, 2, 3, 4]) + rng.normal(size=40)
    game = CorrectionGame(x, z, y, tuple((i,) for i in range(40)), target)
    expected = game.additive_scores()
    actual = row_scores(x, z, y, target)
    for method in METHODS:
        np.testing.assert_allclose(actual[method], expected[method], atol=1e-13)


def test_sufficient_statistic_singletons_reject_rank_loss():
    x = np.array([[1.0, 0], [1, 0], [1, 1]])
    z = np.array([[1.0, 0], [1, 0], [1, 0]])
    with pytest.raises(ValueError, match="singular"):
        row_scores(x, z, np.array([0, 1, 2]), 1)


def test_no_correction_has_zero_scores():
    rng = np.random.default_rng(17)
    x = np.column_stack([np.ones(30), rng.normal(size=(30, 2))])
    y = rng.normal(size=30)
    for scores in row_scores(x, x, y, 1).values():
        np.testing.assert_array_equal(scores, 0)


def test_hc0_scale_matches_sandwich_matrix():
    rng = np.random.default_rng(11)
    x = np.column_stack([np.ones(25), rng.normal(size=(25, 2))])
    y = rng.normal(size=25)
    residual = y - x @ fit_ols(x, y)[0]
    bread = np.linalg.inv(x.T @ x)
    meat = x.T @ np.diag(residual**2) @ x
    assert hc0_scale(x, y, 1) == pytest.approx(np.sqrt((bread @ meat @ bread)[1, 1]))


def test_housing_scaling_is_fixed_across_corrections(tmp_path):
    rng = np.random.default_rng(2)
    raw = rng.normal(size=(50, 6))
    path = tmp_path / "housing.csv"
    np.savetxt(
        path,
        raw,
        delimiter=",",
        header="y,income,proxy_nightlights,proxy_roadlength,"
        "reference_nightlights,reference_roadlength",
        comments="",
    )
    x, z, y = load_design(path, "housing")
    for indices in ([], [0, 1, 5], list(range(50))):
        corrected = x.copy()
        corrected[indices] = z[indices]
        unscaled = np.column_stack([np.ones(50), raw[:, 1:4]])
        unscaled[indices, 2:] = raw[indices, 4:]
        expected = fit_ols(unscaled, y)[0][1] * raw[:, 1].std()
        assert fit_ols(corrected, y)[0][1] == pytest.approx(expected)


def test_saved_archive_results_cover_plan_and_arithmetic():
    directory = Path(__file__).resolve().parents[1] / "generated/archives"
    metadata = json.loads((directory / "metadata.json").read_text())
    assert metadata["draws"] == 200
    assert metadata["seed"] == 20261017
    assert json.loads((directory / "failures.json").read_text()) == []
    endpoints = json.loads((directory / "endpoints.json").read_text())
    with (directory / "sampled_sets.csv").open() as source:
        rows = list(csv.DictReader(source))
    with (directory / "summary.csv").open() as source:
        summary = list(csv.DictReader(source))
    for dataset, n in (("housing", 46418), ("reddit", 613)):
        selected = [r for r in rows if r["dataset"] == dataset]
        assert len(selected) == 1201
        assert endpoints[dataset]["rows"] == n
        for result in (r for r in summary if r["dataset"] == dataset):
            group = [
                r for r in selected if r["corrected_rows"] == result["corrected_rows"]
            ]
            assert len(group) == (1 if int(result["corrected_rows"]) == n else 200)
            errors = np.array(
                [float(r[result["method"]]) - float(r["coefficient"]) for r in group]
            )
            assert float(result["rmse"]) == pytest.approx(np.sqrt(np.mean(errors**2)))
    with (directory / "preparation_audit.csv").open() as source:
        audit = next(csv.DictReader(source))
    assert int(audit["overlapping_ids"]) == 0
    assert int(audit["training_rows"]) == 2413
