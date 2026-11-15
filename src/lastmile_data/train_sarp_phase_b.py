"""Station-cold-start risk prediction with group-held-out conformal abstention.

The random-OOF score is an evaluation label only; it is never an inference
feature. Both the five-station pilot and the locked 17-station expansion use
the same implementation.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import roc_auc_score

from .baselines import nearest_neighbor, travel_time, two_opt
from .research_utils import atomic_dump, normalized_zones
from .reproduce_jpt import ROOT, fingerprint
from .research_utils import compressed_zones, directed_edges, overlap


PILOT_SOURCE = ROOT / "outputs/shift_phase_a_v1"
EXPANSION_SOURCE = ROOT / "outputs/station_shift_expansion_v1"
ALPHA = 0.10
RISK_TOLERANCE = 0.0
SEED = 42
FEATURE_NAMES = [
    "stop_count", "zone_count", "missing_zone_fraction", "mean_stops_per_zone",
    "max_stops_per_zone", "zone_blocks_per_stop", "repeated_zone_blocks",
    "matrix_mean", "matrix_cv", "matrix_asymmetry", "shifted_to_nn_travel",
    "shifted_to_nn2opt_travel", "edge_overlap_nn", "edge_overlap_nn2opt",
    "lat_span", "lng_span", "package_count_per_stop", "windowed_package_fraction",
    "service_seconds_per_stop",
]


def finite_sample_upper_quantile(residuals, alpha=ALPHA):
    """Split-conformal one-sided quantile with finite-sample ceiling rank."""
    if not residuals or not 0 < alpha < 1:
        raise ValueError("need residuals and 0 < alpha < 1")
    ordered = sorted(float(value) for value in residuals)
    rank = min(len(ordered), math.ceil((len(ordered) + 1) * (1 - alpha)))
    return ordered[rank - 1]


def rank_correlation(left, right):
    """Pearson correlation of average ranks; returns 0 for a constant vector."""
    def ranks(values):
        order = sorted(range(len(values)), key=lambda index: (values[index], index))
        result = [0.0] * len(values)
        position = 0
        while position < len(order):
            end = position + 1
            while end < len(order) and values[order[end]] == values[order[position]]:
                end += 1
            average = (position + end - 1) / 2
            for index in order[position:end]:
                result[index] = average
            position = end
        return np.asarray(result)
    a, b = ranks(left), ranks(right)
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def route_features(route, packages, costs, shifted):
    stops = route["stops"]
    depot = next(stop for stop, value in stops.items() if value["type"] == "Station")
    zones = normalized_zones(route)
    non_depot = [stop for stop in stops if stop != depot]
    zone_counts = Counter(zones[stop] for stop in non_depot)
    blocks = compressed_zones(shifted, zones)
    nn = nearest_neighbor(costs, depot)
    nn2 = two_opt(nn, costs)
    values = [float(costs[a][b]) for a in costs for b in costs if a != b]
    asymmetric = [abs(float(costs[a][b]) - float(costs[b][a])) for a in costs for b in costs if a < b]
    mean_cost = statistics.fmean(values)
    shifted_travel = travel_time(shifted, costs)
    package_rows = [package for stop in packages.values() for package in stop.values()]
    windowed = sum(
        package["time_window"].get("start_time_utc") is not None
        or package["time_window"].get("end_time_utc") is not None
        for package in package_rows
    )
    coordinates = [(float(value["lat"]), float(value["lng"])) for value in stops.values()]
    features = {
        "stop_count": float(len(stops)),
        "zone_count": float(len(zone_counts)),
        "missing_zone_fraction": sum(not stops[stop].get("zone_id") for stop in non_depot) / max(1, len(non_depot)),
        "mean_stops_per_zone": statistics.fmean(zone_counts.values()),
        "max_stops_per_zone": float(max(zone_counts.values())),
        "zone_blocks_per_stop": len(blocks) / len(stops),
        "repeated_zone_blocks": float(len(blocks) - len(set(blocks))),
        "matrix_mean": mean_cost,
        "matrix_cv": statistics.pstdev(values) / max(mean_cost, 1e-12),
        "matrix_asymmetry": statistics.fmean(asymmetric) / max(mean_cost, 1e-12),
        "shifted_to_nn_travel": shifted_travel / max(travel_time(nn, costs), 1e-12),
        "shifted_to_nn2opt_travel": shifted_travel / max(travel_time(nn2, costs), 1e-12),
        "edge_overlap_nn": overlap(directed_edges(shifted), directed_edges(nn)),
        "edge_overlap_nn2opt": overlap(directed_edges(shifted), directed_edges(nn2)),
        "lat_span": max(value[0] for value in coordinates) - min(value[0] for value in coordinates),
        "lng_span": max(value[1] for value in coordinates) - min(value[1] for value in coordinates),
        "package_count_per_stop": len(package_rows) / len(stops),
        "windowed_package_fraction": windowed / max(1, len(package_rows)),
        "service_seconds_per_stop": sum(float(p["planned_service_time_seconds"]) for p in package_rows) / len(stops),
    }
    if set(features) != set(FEATURE_NAMES) or not all(math.isfinite(value) for value in features.values()):
        raise RuntimeError("invalid feature vector")
    return features


def prepare(out: Path, expanded=False):
    if out.exists():
        raise FileExistsError(out)
    sources = [PILOT_SOURCE, EXPANSION_SOURCE] if expanded else [PILOT_SOURCE]
    groups = []
    examples = []
    for source in sources:
        phase_a = json.loads((source / "protocol.json").read_text())
        predictions = json.loads((source / "predictions.json").read_text())["station"]
        source_groups = phase_a["groups"]["station"]
        groups.extend(source_groups)
        for index, group in enumerate(source_groups):
            folder = source / f"station_{index}"
            routes = json.loads((folder / "target_routes.json").read_text())
            packages = json.loads((folder / "target_packages.json").read_text())
            times = json.loads((folder / "target_times.json").read_text())
            fold = json.loads((folder / "fold.json").read_text())
            if fold["group"] != group or set(routes) != set(fold["targets"]):
                raise RuntimeError("station-shift fold mismatch")
            for route_id in sorted(routes):
                examples.append({
                    "route_id": route_id, "station": group,
                    "features": route_features(
                        routes[route_id], packages[route_id], times[route_id],
                        predictions[route_id]["sequence"],
                    ),
                })
    expected = 204 if expanded else 60
    if len(examples) != expected or len({row["route_id"] for row in examples}) != expected:
        raise RuntimeError(f"expected exactly {expected} unique station-shift routes")
    if len(groups) != len(set(groups)):
        raise RuntimeError("station groups overlap across sources")
    calibration_groups = 4 if expanded else 1
    out.mkdir(parents=True)
    protocol = {
        "version": 1,
        "scope": "fit-only bounded SARP Phase-B pilot; no outer confirmation/evaluation access",
        "task": "predict station-cold-start JPT degradation and abstain on unsafe routes",
        "feature_policy": "operational inputs and frozen shifted tour only; no actual sequence, route quality, or random-OOF score",
        "validation": f"leave one station out; {calibration_groups} whole station group(s) calibrate the conformal bound",
        "calibration_rule": f"for test station i, the next {calibration_groups} station(s) cyclically calibrate; all other stations train",
        "model": {"type": "ExtraTreesRegressor", "trees": 400, "min_samples_leaf": 4, "seed": SEED},
        "risk": {"alpha": ALPHA, "tolerance_delta": RISK_TOLERANCE, "harmful": "shifted_score - random_oof_score > 0"},
        "predeclared_gate": {
            "upper_bound_coverage_min": 0.90,
            "accepted_routes_min": math.ceil(0.10 * expected),
            "accepted_harmful_rate_max": 0.10,
            "harmful_recall_min": 0.80,
            "roc_auc_min": 0.65,
        },
        "feature_names": FEATURE_NAMES,
        "groups": groups, "source_roots": [str(source) for source in sources],
        "source_hashes": {str(path): fingerprint(path) for path in [
            *(source / name for source in sources for name in ("protocol.json", "predictions.json", "results.jsonl")),
        ]},
    }
    atomic_dump(out / "protocol.json", protocol)
    with (out / "features.jsonl").open("x") as handle:
        for row in examples:
            handle.write(json.dumps(row) + "\n")
    atomic_dump(out / "feature_freeze.json", {
        "routes": len(examples), "stations": groups,
        "features_sha256": fingerprint(out / "features.jsonl"),
        "labels_attached": False,
    })
    print(f"Frozen {len(examples)} inference-only feature vectors before labels", flush=True)


def fit_model(rows):
    model = ExtraTreesRegressor(
        n_estimators=400, min_samples_leaf=4, max_features=0.8,
        random_state=SEED, n_jobs=1,
    )
    model.fit([[row["features"][name] for name in FEATURE_NAMES] for row in rows],
              [row["delta"] for row in rows])
    return model


def predict(model, rows):
    return model.predict([[row["features"][name] for name in FEATURE_NAMES] for row in rows])


def evaluate(out: Path):
    protocol = json.loads((out / "protocol.json").read_text())
    freeze = json.loads((out / "feature_freeze.json").read_text())
    if fingerprint(out / "features.jsonl") != freeze["features_sha256"]:
        raise RuntimeError("features changed after freeze")
    if not all(fingerprint(Path(path)) == value for path, value in protocol["source_hashes"].items()):
        raise RuntimeError("Phase-A source changed after protocol lock")
    rows = [json.loads(line) for line in (out / "features.jsonl").read_text().splitlines()]

    scores = {}
    for source in map(Path, protocol["source_roots"]):
        for line in (source / "results.jsonl").read_text().splitlines():
            item = json.loads(line)
            if item["mode"] == "station":
                scores[item["route_id"], item["method"]] = float(item["official_score"])
    for row in rows:
        row["random_oof_score"] = scores[row["route_id"], "random_oof_jpt"]
        row["shifted_score"] = scores[row["route_id"], "shifted_jpt"]
        row["delta"] = row["shifted_score"] - row["random_oof_score"]
        row["harmful"] = row["delta"] > 1e-12

    groups = protocol["groups"]
    outputs = []
    folds = []
    calibration_count = 4 if len(groups) > 5 else 1
    for test_index, test_group in enumerate(groups):
        calibration_groups = [groups[(test_index + offset) % len(groups)] for offset in range(1, calibration_count + 1)]
        excluded = {test_group, *calibration_groups}
        training = [row for row in rows if row["station"] not in excluded]
        calibration = [row for row in rows if row["station"] in calibration_groups]
        testing = [row for row in rows if row["station"] == test_group]
        model = fit_model(training)
        calibration_prediction = predict(model, calibration)
        quantile = finite_sample_upper_quantile(
            [row["delta"] - estimate for row, estimate in zip(calibration, calibration_prediction)]
        )
        test_prediction = predict(model, testing)
        for row, estimate in zip(testing, test_prediction):
            upper = float(estimate + quantile)
            outputs.append({
                "route_id": row["route_id"], "station": test_group,
                "calibration_stations": calibration_groups,
                "predicted_delta": float(estimate), "upper_delta": upper,
                "accepted": upper <= RISK_TOLERANCE,
                "delta": row["delta"], "harmful": row["harmful"],
                "random_oof_score": row["random_oof_score"], "shifted_score": row["shifted_score"],
            })
        folds.append({"test_station": test_group, "calibration_stations": calibration_groups,
                      "training_stations": sorted({row["station"] for row in training}),
                      "training_routes": len(training), "calibration_routes": len(calibration),
                      "test_routes": len(testing), "conformal_quantile": quantile})

    actual = [row["delta"] for row in outputs]
    predicted = [row["predicted_delta"] for row in outputs]
    harmful = [row["harmful"] for row in outputs]
    accepted = [row for row in outputs if row["accepted"]]
    abstained = [row for row in outputs if not row["accepted"]]
    harmful_total = sum(harmful)
    summary = {
        "routes": len(outputs), "stations": len(groups),
        "accept_all_baseline": {
            "harmful_routes": harmful_total, "harmful_rate": harmful_total / len(outputs),
            "mean_delta": statistics.fmean(actual),
        },
        "prediction": {
            "mae": statistics.fmean(abs(a - p) for a, p in zip(actual, predicted)),
            "spearman": rank_correlation(actual, predicted),
            "roc_auc": float(roc_auc_score(harmful, predicted)),
        },
        "risk_control": {
            "upper_bound_coverage": sum(row["delta"] <= row["upper_delta"] + 1e-12 for row in outputs) / len(outputs),
            "accepted_routes": len(accepted), "accepted_fraction": len(accepted) / len(outputs),
            "accepted_harmful_routes": sum(row["harmful"] for row in accepted),
            "accepted_harmful_rate": (sum(row["harmful"] for row in accepted) / len(accepted)) if accepted else None,
            "accepted_mean_delta": statistics.fmean(row["delta"] for row in accepted) if accepted else None,
            "abstained_routes": len(abstained),
            "harmful_recall": sum(row["harmful"] for row in abstained) / max(1, harmful_total),
        },
        "folds": folds,
    }
    # Descriptive only: this curve is computed after labels are opened and must
    # not be used to select a deployment threshold on this same pilot.
    ordered = sorted(outputs, key=lambda row: row["predicted_delta"])
    summary["posthoc_selective_diagnostic"] = [
        {
            "accepted_routes": count,
            "accepted_fraction": count / len(outputs),
            "harmful_rate": sum(row["harmful"] for row in ordered[:count]) / count,
            "mean_delta": statistics.fmean(row["delta"] for row in ordered[:count]),
        }
        for count in range(6, len(outputs) + 1, 6)
    ]
    gate = protocol["predeclared_gate"]
    risk = summary["risk_control"]
    checks = {
        "upper_bound_coverage": risk["upper_bound_coverage"] >= gate["upper_bound_coverage_min"],
        "accepted_routes": risk["accepted_routes"] >= gate["accepted_routes_min"],
        "accepted_harmful_rate": risk["accepted_harmful_rate"] is not None and risk["accepted_harmful_rate"] <= gate["accepted_harmful_rate_max"],
        "harmful_recall": risk["harmful_recall"] >= gate["harmful_recall_min"],
        "roc_auc": summary["prediction"]["roc_auc"] >= gate["roc_auc_min"],
    }
    summary["gate_checks"] = checks
    summary["decision"] = "PASS: EXPAND SARP" if all(checks.values()) else "FAIL PILOT GATE: DO NOT DEPLOY"
    summary["recommended_next_step"] = (
        "Run a separately frozen outer confirmation cohort before any deployment."
        if all(checks.values()) else
        "Stop the per-route abstention claim and reposition the contribution as a station-level shift benchmark/audit; do not tune a threshold on these labels."
    )
    atomic_dump(out / "oof_predictions.json", outputs)
    atomic_dump(out / "summary.json", summary)

    rate = risk["accepted_harmful_rate"]
    report = [
        "# SARP Phase-B cold-start risk pilot", "", f"**{summary['decision']}**", "",
        "## Locked protocol", "",
        f"- {len(outputs)} routes; {len(groups)} held-out stations.",
        f"- Each fold trains on {len(groups) - calibration_count - 1} stations, calibrates on {calibration_count} whole station(s), and tests on one unseen station.",
        "- The random-OOF score is used only as the post-freeze label.",
        "- Action is accept cold-start JPT or abstain; NN/2-opt is not an automatic fallback.", "",
        "## Results", "",
        f"- Accept-all harmful rate: {100 * summary['accept_all_baseline']['harmful_rate']:.2f}% ({harmful_total}/{len(outputs)})",
        f"- Regression MAE: {summary['prediction']['mae']:.8f}",
        f"- Spearman / harmful AUROC: {summary['prediction']['spearman']:.3f} / {summary['prediction']['roc_auc']:.3f}",
        f"- Conformal upper-bound coverage: {100 * risk['upper_bound_coverage']:.2f}%",
        f"- Accepted: {risk['accepted_routes']}/{len(outputs)} ({100 * risk['accepted_fraction']:.2f}%)",
        f"- Accepted harmful rate: {'N/A' if rate is None else f'{100 * rate:.2f}%'}",
        f"- Recall of harmful routes among abstentions: {100 * risk['harmful_recall']:.2f}%", "",
        "## Interpretation", "",
        "Passing requires calibration coverage, non-trivial acceptance, low accepted risk, harmful-route recall, and discrimination simultaneously. A zero-coverage policy therefore cannot pass.",
        "The post-hoc risk-coverage curve is diagnostic only and cannot justify tuning a deployment threshold on these labels.",
        summary["recommended_next_step"],
    ]
    (out / "report.md").write_text("\n".join(report) + "\n")
    (ROOT / f"reports/{out.name}.md").write_text("\n".join(report) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/sarp_phase_b_pilot_v1"))
    parser.add_argument("--stage", choices=["prepare", "evaluate", "all"], default="all")
    parser.add_argument("--expanded", action="store_true", help="combine pilot and 12-station expansion")
    args = parser.parse_args()
    out = args.output.resolve()
    if args.stage in ("prepare", "all"):
        prepare(out, expanded=args.expanded)
    if args.stage in ("evaluate", "all"):
        evaluate(out)


if __name__ == "__main__":
    main()
