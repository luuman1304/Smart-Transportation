"""Integrity audit for the 12-station cold-start expansion."""
from __future__ import annotations

import json
from pathlib import Path

from .reproduce_jpt import ROOT, dump, fingerprint


OUT = ROOT / "outputs/station_shift_expansion_v1"
REFERENCE = ROOT / "outputs/jpt_reproduction_v1"
RANDOM_OOF = ROOT / "outputs/residual_oof_full_v1"


def main():
    protocol = json.loads((OUT / "protocol.json").read_text())
    summary = json.loads((OUT / "summary.json").read_text())
    predictions = json.loads((OUT / "predictions.json").read_text())["station"]
    rows = [json.loads(line) for line in (OUT / "results.jsonl").read_text().splitlines()]
    fit_routes = json.loads((REFERENCE / "fit_routes.json").read_text())
    random_rows = {row["route_id"]: row for row in json.loads((RANDOM_OOF / "routes.json").read_text())}
    expected = {route_id for ids in protocol["selected"]["station"].values() for route_id in ids}
    keys = {(row["route_id"], row["method"]) for row in rows}

    split_valid = True
    for index, group in enumerate(protocol["groups"]["station"]):
        fold = json.loads((OUT / f"station_{index}/fold.json").read_text())
        split_valid &= fold["group"] == group
        split_valid &= not set(fold["train"]) & set(fold["targets"])
        split_valid &= all(fit_routes[route_id]["station_code"] != group for route_id in fold["train"])
        split_valid &= all(fit_routes[route_id]["station_code"] == group for route_id in fold["targets"])

    random_match = all(
        abs(row["official_score"] - random_rows[row["route_id"]]["anchor_score"]) < 1e-12
        for row in rows if row["method"] == "random_oof_jpt"
    )
    source_hashes_match = all(
        fingerprint(Path(path)) == value for path, value in protocol["source_hashes"].items()
    )
    audit = {
        "passed": bool(
            source_hashes_match and len(expected) == 144 and set(predictions) == expected
            and len(rows) == 288 and len(keys) == 288 and split_valid and random_match
            and sum(1 for _ in OUT.glob("station_*/tours_*/*.tour")) == 288
        ),
        "source_hashes_match": source_hashes_match,
        "target_routes": len(expected), "scored_records": len(rows),
        "tour_files": sum(1 for _ in OUT.glob("station_*/tours_*/*.tour")),
        "split_rules_valid": bool(split_valid),
        "predictions_complete": set(predictions) == expected,
        "random_oof_scores_match_existing_corpus": random_match,
        "outer_development_access": False,
        "decision": summary["decision"],
    }
    dump(OUT / "audit.json", audit)
    print(json.dumps(audit, indent=2))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
