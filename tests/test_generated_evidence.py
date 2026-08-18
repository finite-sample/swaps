from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "generated"


def read_rows(filename: str) -> list[dict[str, str]]:
    with (GENERATED / filename).open(encoding="utf-8") as source:
        return list(csv.DictReader(source))


def find_row(
    rows: list[dict[str, str]], scenario: str, validated_clusters: int | None = None
) -> dict[str, str]:
    return next(
        row
        for row in rows
        if row["scenario"] == scenario
        and (
            validated_clusters is None
            or int(row["validated_clusters"]) == validated_clusters
        )
    )


def test_committed_evidence_uses_full_exact_configuration() -> None:
    metadata = json.loads((GENERATED / "metadata.json").read_text(encoding="utf-8"))

    assert metadata["clusters"] == 20
    assert metadata["cluster_size"] == 50
    assert metadata["seed"] == 2024
    assert metadata["target_ate"] == 0.10
    assert metadata["enumeration"] == "exact fixed-size subsets"


def test_generated_results_show_same_accuracy_and_different_fragility() -> None:
    scenarios = read_rows("table_scenarios.csv")
    budgets = read_rows("table_budgets.csv")
    diffuse = find_row(scenarios, "diffuse")
    concentrated = find_row(scenarios, "concentrated")
    diffuse_five = find_row(budgets, "diffuse", 5)
    concentrated_five = find_row(budgets, "concentrated", 5)

    assert np.isclose(float(diffuse["rmse"]), float(concentrated["rmse"]))
    assert np.isclose(float(diffuse["endpoint_gap"]), 0.04)
    assert np.isclose(float(concentrated["endpoint_gap"]), 0.04)
    assert float(diffuse_five["sd"]) < 1e-12
    assert float(concentrated_five["sd"]) > 0.04


def test_manuscript_inputs_are_generated_artifacts() -> None:
    manuscript = (ROOT / "validation_swaps.tex").read_text(encoding="utf-8")
    required = {
        "macros.tex",
        "table_budgets.tex",
        "table_scenarios.tex",
        "fig1_same_mean.png",
        "fig2_budget_distributions.png",
    }

    for filename in required:
        assert f"generated/{filename}" in manuscript
        assert (GENERATED / filename).is_file()
