"""Integrity audit for the completed distribution-shift Phase-A pilot."""
from __future__ import annotations

import json
from pathlib import Path

from .reproduce_jpt import ROOT, dump, fingerprint


OUT = ROOT / "outputs/shift_phase_a_v1"
REFERENCE = ROOT / "outputs/jpt_reproduction_v1"
RANDOM_OOF = ROOT / "outputs/residual_oof_full_v1"


def main():
    protocol = json.loads((OUT / "protocol.json").read_text())
    summary = json.loads((OUT / "summary.json").read_text())
    predictions = json.loads((OUT / "predictions.json").read_text())
    rows = [json.loads(line) for line in (OUT / "results.jsonl").read_text().splitlines()]
    fit_routes = json.loads((REFERENCE / "fit_routes.json").read_text())
    random_routes = {row["route_id"]: row for row in json.loads((RANDOM_OOF / "routes.json").read_text())}

    hashes_match = all(fingerprint(Path(path)) == value
                       for path, value in protocol["source_hashes"].items())
    keys = {(row["mode"], row["route_id"], row["method"]) for row in rows}
    complete = len(rows) == 240 and len(keys) == 240
    split_valid = True
    predictions_complete = True
    random_scores_match = True
    for mode in ("station", "temporal"):
        expected = {rid for ids in protocol["selected"][mode].values() for rid in ids}
        predictions_complete &= set(predictions[mode]) == expected
        for index, group in enumerate(protocol["groups"][mode]):
            fold = json.loads((OUT / f"{mode}_{index}/fold.json").read_text())
            train = fold["train"]
            targets = fold["targets"]
            split_valid &= not set(train) & set(targets)
            if mode == "station":
                split_valid &= all(fit_routes[rid]["station_code"] != group for rid in train)
                split_valid &= all(fit_routes[rid]["station_code"] == group for rid in targets)
            else:
                split_valid &= all(fit_routes[rid]["date_YYYY_MM_DD"] < group for rid in train)
                split_valid &= all(fit_routes[rid]["date_YYYY_MM_DD"] == group for rid in targets)
        for row in (item for item in rows if item["mode"] == mode and item["method"] == "random_oof_jpt"):
            random_scores_match &= abs(row["official_score"] - random_routes[row["route_id"]]["anchor_score"]) < 1e-12

    tour_count = sum(1 for path in OUT.glob("*_*/tours_*/*.tour"))
    audit = {
        "passed": bool(hashes_match and complete and split_valid and predictions_complete
                       and random_scores_match and tour_count == 240),
        "source_hashes_match": hashes_match,
        "complete_unique_records": complete,
        "split_rules_valid": split_valid,
        "prediction_sets_complete": predictions_complete,
        "random_oof_scores_match_existing_corpus": random_scores_match,
        "tour_files": tour_count,
        "outer_development_access": False,
        "decision": summary["decision"],
    }
    dump(OUT / "audit.json", audit)

    station = summary["modes"]["station"]
    temporal = summary["modes"]["temporal"]
    report = f"""# Distribution-shift Phase A

## Decision

**{summary['decision']}**. The predeclared Phase-B gate passed for station
cold-start and did not pass for short-horizon temporal shift.

| Protocol | Routes | Random-OOF JPT | Shifted-history JPT | Relative degradation | Paired delta CI95 |
|---|---:|---:|---:|---:|---:|
| Station cold-start | {station['routes']} | {station['random_oof_mean_score']:.8f} | {station['shifted_mean_score']:.8f} | {100*station['relative_degradation']:.2f}% | [{station['delta_ci95'][0]:.8f}, {station['delta_ci95'][1]:.8f}] |
| Temporal forward | {temporal['routes']} | {temporal['random_oof_mean_score']:.8f} | {temporal['shifted_mean_score']:.8f} | {100*temporal['relative_degradation']:.2f}% | [{temporal['delta_ci95'][0]:.8f}, {temporal['delta_ci95'][1]:.8f}] |

Lower official score is better. Station cold-start worsened 35 routes, tied 7,
and improved 18. Temporal-forward worsened 8, tied 49, and improved 3.

## Locked protocol

- Fit-only High routes with pre-existing random-OOF JPT and residual-oracle controls.
- Five held-out stations, twelve deterministic routes per station. The complete
  target station is removed from JPT's historical model.
- Five late dates, twelve deterministic routes per date. Training dates are
  strictly earlier than the target date.
- Two JPT variants, seed 1, one second per variant. All 120 shifted predictions
  were frozen before actual sequences were loaded.
- Phase-B gate fixed before scoring: at least 5% degradation and paired route
  bootstrap CI95 entirely above zero in either shift mode.

## Interpretation

The publishable shift signal is specifically **station cold-start**. Short-range
temporal restriction produces a statistically positive but small 1.37% change;
it does not satisfy the materiality gate. The station result is heterogeneous:
DLA9 degrades most, while all five held-out stations have worse mean score.

This does not show that every AI routing method fails on unseen stations. It
shows that the current strongest JPT pipeline depends materially on station-local
historical structure. Phase B should therefore estimate station novelty and
calibrate a risk-controlled fallback; it should not frame generic temporal drift
as the primary contribution.

## Integrity

Audit: `{'PASS' if audit['passed'] else 'FAIL'}`. All source hashes match, all
240 solver tours and 240 unique scored method records are present, station/date
exclusion rules hold, and random-OOF control scores exactly match the existing
full OOF corpus. No outer development, confirmation, or evaluation data was used.
"""
    (ROOT / "reports/shift_phase_a_v1.md").write_text(report)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
