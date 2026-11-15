"""Evaluate a bounded station-balanced soft zone-transition prior for cold-start JPT."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import evaluate_shift_phase_a as phase
from .research_utils import atomic_dump, normalized_zones
from .evaluate_shared_zone_grammar import REFERENCE, SOURCES, source_folds
from .reproduce_jpt import JPT, ROOT, command, dump, fingerprint


ALPHA = 0.05
SMOOTHING = 0.5
LOG_CAP = 4.0
METHOD = "station_balanced_soft_zone_prior"


def station_transition_counts(model):
    result = {}
    for station, histories in model["station_infos"].items():
        edges, outgoing, vocabulary = Counter(), Counter(), set()
        for history in histories.values():
            sequence = [zone for zone in history["zone_list"] if isinstance(zone, str) and zone]
            vocabulary.update(sequence)
            for left, right in zip(sequence, sequence[1:]):
                if left != right:
                    edges[left, right] += 1
                    outgoing[left] += 1
        result[station] = {"edges": edges, "outgoing": outgoing, "vocabulary": vocabulary}
    return result


def balanced_probabilities(model, target_zones, smoothing=SMOOTHING, station_counts=None):
    """Average per-station smoothed probabilities; absent source zones are neutral."""
    station_counts = station_counts or station_transition_counts(model)
    probabilities = {}
    target_zones = sorted(set(target_zones))
    for left in target_zones:
        supporting = [value for value in station_counts.values() if value["outgoing"][left] > 0]
        if not supporting:
            for right in target_zones:
                probabilities[left, right] = 1.0
            continue
        destinations = set(target_zones)
        for value in supporting:
            destinations.update(value["vocabulary"])
        cardinality = max(1, len(destinations))
        for right in target_zones:
            probabilities[left, right] = statistics.fmean(
                (value["edges"][left, right] + smoothing)
                / (value["outgoing"][left] + smoothing * cardinality)
                for value in supporting
            )
    return probabilities


def route_penalties(route, model, station_counts=None):
    zones = normalized_zones(route)
    unique = sorted(set(zones.values()))
    probabilities = balanced_probabilities(model, unique, station_counts=station_counts)
    penalties = {}
    for left in unique:
        best = max(probabilities[left, right] for right in unique)
        for right in unique:
            if left == right or probabilities[left, right] == 1.0:
                penalty = 0.0
            else:
                penalty = min(LOG_CAP, math.log(best / probabilities[left, right])) / LOG_CAP
            penalties[left, right] = penalty
    return zones, penalties


def soften_times(routes, times, model, alpha=ALPHA):
    softened, diagnostics = {}, []
    counts = station_transition_counts(model)
    for route_id, costs in times.items():
        zones, penalties = route_penalties(routes[route_id], model, counts)
        off_diagonal = [float(costs[a][b]) for a in costs for b in costs if a != b]
        scale = statistics.median(off_diagonal)
        rows = {}
        changes = []
        for left, row in costs.items():
            rows[left] = {}
            for right, value in row.items():
                addition = alpha * scale * penalties[zones[left], zones[right]] if left != right else 0.0
                rows[left][right] = float(value) + addition
                changes.append(addition)
        softened[route_id] = rows
        diagnostics.append({
            "route_id": route_id, "median_edge_seconds": scale,
            "mean_added_seconds": statistics.fmean(changes),
            "max_added_seconds": max(changes),
            "max_allowed_seconds": alpha * scale,
        })
    return softened, diagnostics


def initialize(out: Path):
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    groups, selected, diagnostics, hashes = [], {}, [], {}
    for source, source_index, group, targets in source_folds():
        index = len(groups)
        groups.append(group)
        selected[group] = targets
        source_folder = source / f"station_{source_index}"
        folder = out / f"station_{index}"
        folder.mkdir()
        for name in ("target_routes.json", "target_packages.json", "target_times.json", "model.json"):
            shutil.copyfile(source_folder / name, folder / name)
            hashes[str(source_folder / name)] = fingerprint(source_folder / name)
        fold = json.loads((source_folder / "fold.json").read_text())
        dump(folder / "fold.json", {**fold, "index": index, "source": str(source), "source_index": source_index})
        routes = json.loads((folder / "target_routes.json").read_text())
        times = json.loads((folder / "target_times.json").read_text())
        model = json.loads((folder / "model.json").read_text())
        if group in model["station_infos"]:
            raise RuntimeError("target station present in transition model")
        soft, rows = soften_times(routes, times, model)
        dump(folder / "soft_times.json", soft)
        diagnostics.extend({"station": group, **row} for row in rows)
    if len(groups) != 17 or len(diagnostics) != 204:
        raise RuntimeError("expected 17 stations and 204 routes")
    diagnostic_summary = {
        "routes": len(diagnostics),
        "mean_added_seconds": statistics.fmean(row["mean_added_seconds"] for row in diagnostics),
        "maximum_added_seconds": max(row["max_added_seconds"] for row in diagnostics),
        "all_penalties_bounded": all(row["max_added_seconds"] <= row["max_allowed_seconds"] + 1e-12 for row in diagnostics),
    }
    atomic_dump(out / "penalty_diagnostic.json", {"summary": diagnostic_summary, "routes": diagnostics})
    protocol = {
        "version": 1,
        "scope": "reused fit-only development evidence; no outer confirmation/evaluation access",
        "method": METHOD,
        "formula": "c'(i,j)=c(i,j)+alpha*median_route_edge*min(log(max_b p(b|z_i)/p(z_j|z_i)),cap)/cap",
        "parameters_locked_pre_score": {"alpha": ALPHA, "smoothing": SMOOTHING, "log_cap": LOG_CAP},
        "transition_estimator": "per-station smoothed transition probabilities averaged equally across supporting training stations",
        "neutral_rules": "same-zone edges and source zones absent from all training stations receive zero penalty",
        "target_history_used": False,
        "groups": {"station": groups}, "selected": {"station": selected},
        "penalty_diagnostic": diagnostic_summary,
        "solver": {"variants": 2, "seconds_per_variant": 1, "seed": 1, "workers": 4},
        "success_gate": "paired improvement over station-cold-start JPT >=1% and paired bootstrap delta CI95 upper bound <0",
        "source_hashes": hashes,
    }
    dump(out / "protocol.json", protocol)
    print(json.dumps(diagnostic_summary, indent=2), flush=True)


def build(out: Path, workers=4):
    protocol = json.loads((out / "protocol.json").read_text())
    if not protocol["penalty_diagnostic"]["all_penalties_bounded"]:
        raise RuntimeError("penalty bound failed")
    if not all(fingerprint(Path(path)) == value for path, value in protocol["source_hashes"].items()):
        raise RuntimeError("source changed after protocol lock")
    fit_routes = json.loads((REFERENCE / "fit_routes.json").read_text())

    def one(index):
        folder = out / f"station_{index}"
        fold = json.loads((folder / "fold.json").read_text())
        with tempfile.TemporaryDirectory(prefix=f"soft-prior-{index}-") as temporary:
            train_path = Path(temporary) / "train_routes.json"
            dump(train_path, {route_id: fit_routes[route_id] for route_id in fold["train"]})
            for variant in (1, 2):
                arguments = [
                    Path(__import__("sys").executable), JPT / "scripts/build_TSPLIB.py",
                    "--r_file", "target_routes.json", "--t_file", "soft_times.json",
                    "--p_file", "target_packages.json", "--z_file", "model.json",
                    "--br_file", train_path, "-noPrune", "-noPruneFailed",
                    "-zoneNeighborTrans", "-superNeighborTrans",
                ]
                if variant == 2:
                    arguments += ["-superPred", "-trans"]
                arguments += [f"TSPLIB_{variant}"]
                command(arguments, folder, folder / f"build_variant_{variant}.log")
                if len(list((folder / f"TSPLIB_{variant}").glob("*.ctsptw"))) != 12:
                    raise RuntimeError(f"station fold {index} has incomplete instances")
        return index

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for completed, index in enumerate(pool.map(one, range(17)), 1):
            print(f"Built soft-prior fold {completed}/17 (station_{index})", flush=True)


def configure_phase():
    phase.MODES = ("station",)
    phase.GROUPS_PER_MODE = 17


def score_and_compare(out: Path):
    configure_phase()
    phase.score(out)  # target_times.json remains the unmodified official scoring matrix
    rows = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
    proposed = {row["route_id"]: row["official_score"] for row in rows if row["method"] == "shifted_jpt"}
    cold = {}
    for source in SOURCES:
        for line in (source / "results.jsonl").read_text().splitlines():
            row = json.loads(line)
            if row["mode"] == "station" and row["method"] == "shifted_jpt":
                cold[row["route_id"]] = row["official_score"]
    if set(proposed) != set(cold) or len(proposed) != 204:
        raise RuntimeError("comparison cohort mismatch")
    route_ids = sorted(proposed)
    deltas = [proposed[route_id] - cold[route_id] for route_id in route_ids]
    cold_mean, proposed_mean = statistics.fmean(cold.values()), statistics.fmean(proposed.values())
    improvement, ci = 1 - proposed_mean / cold_mean, phase.bootstrap_ci(deltas)
    passed = improvement >= .01 and ci[1] < 0
    comparison = {
        "routes": 204, "cold_start_jpt_mean_score": cold_mean,
        "soft_prior_mean_score": proposed_mean,
        "relative_improvement_over_cold_jpt": improvement,
        "mean_delta": statistics.fmean(deltas), "paired_delta_ci95": ci,
        "improved_routes": sum(delta < 0 for delta in deltas),
        "tied_routes": sum(delta == 0 for delta in deltas),
        "worsened_routes": sum(delta > 0 for delta in deltas),
        "gate_passed": passed,
        "decision": "PROMOTE TO FROZEN CONFIRMATION" if passed else "STOP SOFT-PRIOR BRANCH",
        "evidence_status": "reused development cohort; not confirmation evidence",
    }
    atomic_dump(out / "comparison.json", comparison)
    report = [
        "# Station-balanced soft zone prior", "", f"**{comparison['decision']}**", "",
        f"- Cold-start JPT: {cold_mean:.8f}", f"- Soft-prior JPT: {proposed_mean:.8f}",
        f"- Relative improvement: {100 * improvement:.2f}%",
        f"- Paired delta CI95: [{ci[0]:.8f}, {ci[1]:.8f}]",
        f"- Better/tied/worse: {comparison['improved_routes']}/{comparison['tied_routes']}/{comparison['worsened_routes']}", "",
        "The fixed gate requires at least 1% improvement and a paired CI entirely below zero. Original travel times, not modified solver costs, are used for official scoring.",
    ]
    (out / "comparison_report.md").write_text("\n".join(report) + "\n")
    (ROOT / "reports/soft_zone_prior_v1.md").write_text("\n".join(report) + "\n")
    print(json.dumps(comparison, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/soft_zone_prior_v1"))
    parser.add_argument("--stage", choices=["init", "build", "solve", "freeze", "score", "all"], default="all")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    out = args.output.resolve()
    configure_phase()
    stages = [
        ("init", initialize), ("build", lambda path: build(path, args.workers)),
        ("solve", lambda path: phase.solve(path, args.workers)),
        ("freeze", phase.freeze), ("score", score_and_compare),
    ]
    for name, function in stages:
        if args.stage in (name, "all"):
            print("Stage: " + name, flush=True)
            function(out)


if __name__ == "__main__":
    main()
