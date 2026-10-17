import csv
import json
from pathlib import Path

import numpy as np
import pytest

from covariate_swaps import make_benchmark

GENERATED = Path(__file__).resolve().parents[1] / "generated" / "covariates"


def rows(filename):
    with (GENERATED / filename).open() as source:
        return list(csv.DictReader(source))


def test_full_benchmark_has_every_declared_design():
    metadata = json.loads((GENERATED / "metadata.json").read_text())
    assert metadata["n"] == 96
    assert metadata["players"] == 8
    assert not metadata["fast"]
    assert json.loads((GENERATED / "failures.json").read_text()) == []
    actual = {
        (int(r["seed"]), float(r["rmse"]), r["scenario"]) for r in rows("benchmark.csv")
    }
    expected = {
        (s, a, p)
        for s in range(20)
        for a in (0.05, 0.25, 0.75, 1.5)
        for p in ("diffuse", "concentrated", "aligned")
    }
    assert actual == expected
    assert len(rows("benchmark.csv")) == 240


def test_saved_coalitions_agree_with_refits_and_attribution_endpoints():
    coalitions = rows("coalitions.csv")
    attribution = rows("attributions.csv")
    for scenario in ("diffuse", "concentrated", "aligned"):
        selected = [r for r in coalitions if r["scenario"] == scenario]
        assert [int(r["mask"]) for r in selected] == list(range(256))
        game = make_benchmark(scenario)
        for mask in (0, 1, 73, 142, 255):
            assert float(selected[mask]["value"]) == pytest.approx(game.value(mask))
        shapley_sum = sum(
            float(r["shapley"]) for r in attribution if r["scenario"] == scenario
        )
        gap = float(selected[-1]["value"]) - float(selected[0]["value"])
        assert shapley_sum == pytest.approx(gap)
        if scenario == "concentrated":
            np.testing.assert_allclose(
                [float(r["singleton"]) for r in selected],
                [float(r["value"]) for r in selected],
                atol=1e-13,
            )
