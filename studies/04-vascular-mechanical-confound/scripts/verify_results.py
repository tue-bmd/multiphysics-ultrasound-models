"""Verify the stored outputs used by RESULTS.md."""
from __future__ import annotations

import json
import math
from pathlib import Path

from vmconf.provenance import code_hashes

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def reject_nonstandard_constant(value):
    raise ValueError(f"non-standard JSON constant: {value}")


def load(path):
    with path.open() as f:
        return json.load(f, parse_constant=reject_nonstandard_constant)


def check_provenance(path, blob, expected_hashes, require_current_vmconf=False):
    provenance = blob["provenance"]
    assert "porovasc_path" not in provenance, path
    assert provenance.get("porovasc_package") == "porovasc", path
    assert provenance["network"]["params"] is not None, path
    assert provenance["porovasc_sha256"] == expected_hashes["porovasc_sha256"], path
    assert len(provenance["vmconf_sha256"]) == 64, path
    int(provenance["vmconf_sha256"], 16)
    if require_current_vmconf:
        assert provenance["vmconf_sha256"] == expected_hashes["vmconf_sha256"], path
    encoded = json.dumps(blob)
    assert "/Users/" not in encoded and "/sessions/" not in encoded, path


def main():
    expected_hashes = code_hashes()
    ceus_files = sorted(RESULTS.glob("ceus_d30_seed*.json"))
    inference_files = sorted(RESULTS.glob("inference_d30_seed*.json"))
    assert len(ceus_files) == 3
    assert len(inference_files) == 3

    for path in ceus_files:
        blob = load(path)
        check_provenance(path, blob, expected_hashes)
        rows = sorted(blob["table"]["rows"], key=lambda row: row["s"])
        assert all(a["s"] < b["s"] for a, b in zip(rows, rows[1:])), path
        assert all(a["auc"] < b["auc"] for a, b in zip(rows, rows[1:])), path

    for path in inference_files:
        blob = load(path)
        check_provenance(path, blob, expected_hashes, require_current_vmconf=True)
        summary = blob["summary"]
        repetitions = blob["posterior_by_repetition"]
        n_reps = blob["provenance"]["settings"]["reps"]
        assert len(blob["observations"]) == n_reps, path
        assert set(repetitions) == set(summary), path
        assert all(len(rows) == n_reps for rows in repetitions.values()), path
        assert all(
            len(data["swe"]) == 48 and len(data["ceus"]) == 1
            for data in blob["observations"]
        ), path
        assert all(
            math.isclose(
                summary["joint_free"]["s"][key],
                summary["swe"]["s"][key],
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            for key in summary["swe"]["s"]
        ), path
        assert summary["joint_shared"]["s"]["width_pct"] < summary["swe"]["s"][
            "width_pct"
        ], path
        assert len(blob["offset_sweep"]) == 6, path
        assert len(blob["offset_sweep_by_repetition"]) == 6, path

    sweep_files = sorted((RESULTS / "true_value_sweep").glob("*.json"))
    assert len(sweep_files) >= 81
    combinations = set()
    reported_sweep_files = []
    for path in sweep_files:
        blob = load(path)
        check_provenance(path, blob, expected_hashes, require_current_vmconf=True)
        truth = blob["truth"]
        seed = blob["provenance"]["network"]["params"]["seed"]
        if truth["s"] in (0.7, 0.8, 0.9):
            reported_sweep_files.append(path)
            combinations.add((truth["mu"], truth["eta"], truth["s"], seed))
        assert len(blob["observations"]) == 40, path
        assert all(len(rows) == 40 for rows in blob["posterior_by_repetition"].values()), path
    assert {row[0] for row in combinations} == {1500.0, 2000.0, 2500.0}
    assert {row[1] for row in combinations} == {0.5, 1.0, 1.5}
    assert len(reported_sweep_files) == 81
    assert {row[2] for row in combinations} == {0.7, 0.8, 0.9}
    assert {row[3] for row in combinations} == {1, 2, 3}

    sweep_summary = load(RESULTS / "true_value_sweep_summary.json")
    assert sweep_summary["n_files"] == 81
    assert sweep_summary["n_networks"] == 3
    assert sweep_summary["repetitions_per_file"] == 40
    assert set(sweep_summary["source_files"]) == {
        path.name for path in reported_sweep_files
    }
    assert len(sweep_summary["combinations"]) == 27
    assert all(
        math.isfinite(row["metrics"][quantity]["interval_width_reduction_pct"])
        for row in sweep_summary["combinations"]
        for quantity in ("mu", "eta")
    )

    diagnostics = load(RESULTS / "diagnostics.json")
    check_provenance(RESULTS / "diagnostics.json", diagnostics, expected_hashes)
    assert all(
        row["retained_weight"] <= 1 for row in diagnostics["poiseuille_bracket"]
    )
    print("stored results verified")


if __name__ == "__main__":
    main()
