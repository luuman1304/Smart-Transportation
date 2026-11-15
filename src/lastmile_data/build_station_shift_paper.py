"""Build station-cluster statistics, exact baselines, tables, and paper figures."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

from .baselines import nearest_neighbor, two_opt
from .research_utils import atomic_dump
from .reproduce_jpt import ROOT, official_score


PILOT = ROOT / "outputs/shift_phase_a_v1"
CONFIRMATION = ROOT / "outputs/station_shift_expansion_v1"
SHARED = ROOT / "outputs/shared_zone_grammar_v1"
SOFT = ROOT / "outputs/soft_zone_prior_v1"
LADE_EXTERNAL = ROOT / "outputs/lade_external_v1"
LADE_PAIRWISE = ROOT / "outputs/lade_pairwise_v1"


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def station_shift_rows():
    rows = []
    for phase_name, root in (("pilot", PILOT), ("confirmation", CONFIRMATION)):
        scores = defaultdict(dict)
        groups = {}
        for row in jsonl(root / "results.jsonl"):
            if row["mode"] == "station":
                scores[row["route_id"]][row["method"]] = float(row["official_score"])
                groups[row["route_id"]] = row["group"]
        for route_id in sorted(scores):
            rows.append({
                "route_id": route_id, "station": groups[route_id], "phase": phase_name,
                "random_oof_jpt": scores[route_id]["random_oof_jpt"],
                "cold_start_jpt": scores[route_id]["shifted_jpt"],
                "delta": scores[route_id]["shifted_jpt"] - scores[route_id]["random_oof_jpt"],
            })
    return rows


def station_means(rows, field="delta"):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["station"]].append(float(row[field]))
    return {station: statistics.fmean(values) for station, values in sorted(grouped.items())}


def cluster_bootstrap(values, seed=42, repetitions=100_000):
    rng = random.Random(seed)
    values = list(values)
    samples = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(repetitions))
    return [samples[int(.025 * repetitions)], samples[int(.975 * repetitions)]]


def exact_sign_test(values):
    nonzero = [value for value in values if value != 0]
    n, positives = len(nonzero), sum(value > 0 for value in nonzero)
    tail = min(positives, n - positives)
    probability = 2 * sum(math.comb(n, k) for k in range(tail + 1)) / (2 ** n)
    return {"nonzero_stations": n, "positive_stations": positives, "two_sided_p": min(1.0, probability)}


def exact_sign_flip_test(values):
    values = list(values)
    observed = abs(statistics.fmean(values))
    exceed = 0
    for signs in itertools.product((-1, 1), repeat=len(values)):
        statistic = abs(statistics.fmean(sign * value for sign, value in zip(signs, values)))
        exceed += statistic >= observed - 1e-15
    return {"permutations": 2 ** len(values), "two_sided_p": exceed / (2 ** len(values))}


def effect_summary(rows):
    station = station_means(rows)
    values = list(station.values())
    route_deltas = [row["delta"] for row in rows]
    random_mean = statistics.fmean(row["random_oof_jpt"] for row in rows)
    cold_mean = statistics.fmean(row["cold_start_jpt"] for row in rows)
    return {
        "routes": len(rows), "stations": len(station),
        "random_oof_jpt_mean": random_mean, "cold_start_jpt_mean": cold_mean,
        "relative_degradation": cold_mean / random_mean - 1,
        "route_weighted_mean_delta": statistics.fmean(route_deltas),
        "station_weighted_mean_delta": statistics.fmean(values),
        "station_cluster_bootstrap_ci95": cluster_bootstrap(values),
        "station_effect_sd": statistics.stdev(values),
        "station_standardized_effect_dz": statistics.fmean(values) / statistics.stdev(values),
        "stations_worse_tied_better": [sum(v > 0 for v in values), sum(v == 0 for v in values), sum(v < 0 for v in values)],
        "exact_sign_test": exact_sign_test(values),
        "exact_sign_flip_test": exact_sign_flip_test(values),
    }


def load_method_scores(root):
    return {
        row["route_id"]: float(row["official_score"])
        for row in jsonl(root / "results.jsonl") if row["method"] == "shifted_jpt"
    }


def compute_baselines(rows):
    by_id = {row["route_id"]: row for row in rows}
    result = {}
    actual = json.loads((ROOT / "outputs/jpt_reproduction_v1/fit_sequences.json").read_text())
    for root in (PILOT, CONFIRMATION):
        protocol = json.loads((root / "protocol.json").read_text())
        for index, station in enumerate(protocol["groups"]["station"]):
            folder = root / f"station_{index}"
            routes = json.loads((folder / "target_routes.json").read_text())
            times = json.loads((folder / "target_times.json").read_text())
            for route_id, route in routes.items():
                reference = sorted(actual[route_id]["actual"], key=actual[route_id]["actual"].get)
                depot = reference[0]
                nn = nearest_neighbor(times[route_id], depot)
                nn2 = two_opt(nn, times[route_id])
                result[route_id] = {
                    "route_id": route_id, "station": station,
                    "nearest_neighbor": official_score(reference, nn, times[route_id]),
                    "nearest_neighbor_2opt": official_score(reference, nn2, times[route_id]),
                    "random_oof_jpt": by_id[route_id]["random_oof_jpt"],
                    "cold_start_jpt": by_id[route_id]["cold_start_jpt"],
                }
    shared, soft = load_method_scores(SHARED), load_method_scores(SOFT)
    for route_id in result:
        result[route_id]["shared_hard_grammar"] = shared[route_id]
        result[route_id]["station_balanced_soft_prior"] = soft[route_id]
    return [result[route_id] for route_id in sorted(result)]


def method_table(baselines):
    names = [key for key in baselines[0] if key not in ("route_id", "station")]
    cold = {row["route_id"]: row["cold_start_jpt"] for row in baselines}
    table = []
    for name in names:
        values = [row[name] for row in baselines]
        deltas = [row[name] - cold[row["route_id"]] for row in baselines]
        grouped = defaultdict(list)
        for row, delta in zip(baselines, deltas):
            grouped[row["station"]].append(delta)
        station_delta = [statistics.fmean(value) for value in grouped.values()]
        table.append({
            "method": name, "mean_official_score": statistics.fmean(values),
            "relative_change_vs_cold_jpt": statistics.fmean(values) / statistics.fmean(cold.values()) - 1,
            "route_wins_ties_losses_vs_cold": [sum(v < 0 for v in deltas), sum(v == 0 for v in deltas), sum(v > 0 for v in deltas)],
            "station_mean_wins_vs_cold": sum(v < 0 for v in station_delta),
        })
    return table


def svg_station_effects(path, station_rows):
    width, height, margin = 980, 520, 80
    values = [row["mean_delta"] for row in station_rows]
    bound = max(abs(min(values)), abs(max(values))) * 1.12
    zero = margin + (height - 2 * margin) * (bound / (2 * bound))
    bar_width = (width - 2 * margin) / len(values) * .72
    step = (width - 2 * margin) / len(values)
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="490" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">Station-level cold-start effect (positive is worse)</text>',
             f'<line x1="{margin}" x2="{width-margin}" y1="{zero:.1f}" y2="{zero:.1f}" stroke="#333"/>']
    for index, row in enumerate(station_rows):
        x = margin + index * step + (step - bar_width) / 2
        y_value = zero - row["mean_delta"] / (2 * bound) * (height - 2 * margin)
        y, h = min(zero, y_value), abs(zero - y_value)
        color = "#c23b22" if row["mean_delta"] > 0 else "#2b6cb0"
        lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{h:.1f}" fill="{color}"/>')
        lines.append(f'<text x="{x+bar_width/2:.1f}" y="{height-margin+18}" text-anchor="end" transform="rotate(-55 {x+bar_width/2:.1f},{height-margin+18})" font-family="sans-serif" font-size="11">{row["station"]}</text>')
    lines += [f'<text x="18" y="{height/2}" transform="rotate(-90 18,{height/2})" text-anchor="middle" font-family="sans-serif" font-size="13">Mean official-score delta</text>', '</svg>']
    path.write_text("\n".join(lines))


def svg_methods(path, table):
    ordered = sorted(table, key=lambda row: row["mean_official_score"])
    width, height, margin = 900, 410, 210
    max_score = max(row["mean_official_score"] for row in ordered) * 1.05
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', '<text x="450" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">Official score on the 204-route station-shift cohort (lower is better)</text>']
    for index, row in enumerate(ordered):
        y = 55 + index * 50
        bar = row["mean_official_score"] / max_score * (width - margin - 50)
        color = "#2b6cb0" if row["method"] == "cold_start_jpt" else "#718096"
        lines.append(f'<text x="{margin-8}" y="{y+18}" text-anchor="end" font-family="sans-serif" font-size="12">{row["method"]}</text>')
        lines.append(f'<rect x="{margin}" y="{y}" width="{bar:.1f}" height="24" fill="{color}"/>')
        lines.append(f'<text x="{margin+bar+6:.1f}" y="{y+17}" font-family="monospace" font-size="12">{row["mean_official_score"]:.5f}</text>')
    lines.append('</svg>')
    path.write_text("\n".join(lines))


def build(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    rows = station_shift_rows()
    pilot = [row for row in rows if row["phase"] == "pilot"]
    confirmation = [row for row in rows if row["phase"] == "confirmation"]
    station = station_means(rows)
    station_rows = []
    for name, delta in station.items():
        members = [row for row in rows if row["station"] == name]
        station_rows.append({
            "station": name, "phase": members[0]["phase"], "routes": len(members),
            "random_oof_jpt": statistics.fmean(row["random_oof_jpt"] for row in members),
            "cold_start_jpt": statistics.fmean(row["cold_start_jpt"] for row in members),
            "mean_delta": delta,
        })
    baselines = compute_baselines(rows)
    methods = method_table(baselines)
    lade_external = json.loads((LADE_EXTERNAL / "analysis.json").read_text())
    lade_pairwise = json.loads((LADE_PAIRWISE / "analysis.json").read_text())
    if not lade_external["audit"]["passed"] or not lade_pairwise["audit"]["passed"]:
        raise RuntimeError("refusing to build paper evidence from a failed LaDe audit")
    analysis = {
        "confirmatory_primary": effect_summary(confirmation),
        "pilot_hypothesis_generating": effect_summary(pilot),
        "combined_supportive": effect_summary(rows),
        "methods": methods,
        "external_lade_lightweight": lade_external,
        "external_lade_pairwise_sensitivity": lade_pairwise,
        "integrity": {
            "unique_routes": len({row["route_id"] for row in rows}),
            "stations": len(station), "routes_per_station": sorted({sum(row["station"] == name for row in rows) for name in station}),
            "pilot_audit_passed": json.loads((PILOT / "audit.json").read_text())["passed"],
            "confirmation_audit_passed": json.loads((CONFIRMATION / "audit.json").read_text())["passed"],
            "lade_lightweight_audit_passed": lade_external["audit"]["passed"],
            "lade_pairwise_audit_passed": lade_pairwise["audit"]["passed"],
        },
    }
    atomic_dump(out / "analysis.json", analysis)
    for name, data in (("route_effects.csv", rows), ("station_effects.csv", station_rows), ("method_scores.csv", baselines), ("method_summary.csv", methods)):
        with (out / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader(); writer.writerows(data)
    svg_station_effects(out / "figures/station_effects.svg", station_rows)
    svg_methods(out / "figures/method_scores.svg", methods)
    print(json.dumps(analysis, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("paper/evidence"))
    args = parser.parse_args()
    build(args.output.resolve())


if __name__ == "__main__":
    main()
