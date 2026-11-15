"""Evaluate a station-invariant shared-zone grammar inside the JPT solver.

For an unseen station, pool historical zone paths and super-zone paths from all
training stations under the target station key. JPT's original nearest-overlap
matching then selects the relevant cross-station history without target labels.
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import evaluate_shift_phase_a as phase
from .research_utils import atomic_dump
from .reproduce_jpt import JPT, ROOT, command, dump, fingerprint


SOURCES = [ROOT / "outputs/shift_phase_a_v1", ROOT / "outputs/station_shift_expansion_v1"]
REFERENCE = ROOT / "outputs/jpt_reproduction_v1"
METHOD = "shared_zone_grammar_jpt"


def source_folds():
    result = []
    for source in SOURCES:
        protocol = json.loads((source / "protocol.json").read_text())
        for index, group in enumerate(protocol["groups"]["station"]):
            result.append((source, index, group, protocol["selected"]["station"][group]))
    return result


def pool_target_history(model, target_station):
    """Copy all non-target histories under target key; inputs already exclude target."""
    pooled_info = {}
    for station, routes in model["station_infos"].items():
        if station == target_station:
            raise RuntimeError("target station leaked into historical model")
        for route_id, value in routes.items():
            pooled_info[f"{station}:{route_id}"] = value
    pooled_super = {}
    for station, routes in model["superSets"].items():
        if station == target_station:
            raise RuntimeError("target station leaked into super-set model")
        for route_id, value in routes.items():
            pooled_super[f"{station}:{route_id}"] = value
    result = json.loads(json.dumps(model))
    result["station_infos"][target_station] = pooled_info
    result["superSets"][target_station] = pooled_super
    return result


def overlap_diagnostic(routes, model):
    histories = [
        set(value["zone_list"])
        for station in model["station_infos"].values() for value in station.values()
    ]
    rows = []
    for route_id, route in routes.items():
        zones = {stop.get("zone_id") for stop in route["stops"].values() if stop.get("zone_id")}
        if not zones:
            rows.append({"route_id": route_id, "best_jaccard": 0.0, "zone_coverage": 0.0})
            continue
        best = max(histories, key=lambda history: len(zones & history) / max(1, len(zones | history)))
        rows.append({
            "route_id": route_id,
            "best_jaccard": len(zones & best) / max(1, len(zones | best)),
            "zone_coverage": len(zones & best) / len(zones),
        })
    return rows


def initialize(out: Path):
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    groups, selected, diagnostics = [], {}, []
    hashes = {}
    for source, source_index, group, targets in source_folds():
        index = len(groups)
        groups.append(group)
        selected[group] = targets
        source_folder = source / f"station_{source_index}"
        folder = out / f"station_{index}"
        folder.mkdir()
        for name in ("target_routes.json", "target_packages.json", "target_times.json"):
            shutil.copyfile(source_folder / name, folder / name)
            hashes[str(source_folder / name)] = fingerprint(source_folder / name)
        source_fold = json.loads((source_folder / "fold.json").read_text())
        dump(folder / "fold.json", {
            **source_fold, "index": index, "source": str(source), "source_index": source_index,
        })
        model = json.loads((source_folder / "model.json").read_text())
        routes = json.loads((folder / "target_routes.json").read_text())
        diagnostics.extend({"station": group, **row} for row in overlap_diagnostic(routes, model))
        dump(folder / "model.json", pool_target_history(model, group))
        hashes[str(source_folder / "model.json")] = fingerprint(source_folder / "model.json")
    if len(groups) != 17 or len(diagnostics) != 204:
        raise RuntimeError("expected 17 stations and 204 routes")
    diagnostic_summary = {
        "routes": len(diagnostics),
        "mean_best_jaccard": statistics.fmean(row["best_jaccard"] for row in diagnostics),
        "median_best_jaccard": statistics.median(row["best_jaccard"] for row in diagnostics),
        "mean_zone_coverage": statistics.fmean(row["zone_coverage"] for row in diagnostics),
        "median_zone_coverage": statistics.median(row["zone_coverage"] for row in diagnostics),
        "routes_jaccard_at_least_0_5": sum(row["best_jaccard"] >= .5 for row in diagnostics),
        "headroom_gate": "PASS" if statistics.median(row["best_jaccard"] for row in diagnostics) >= .5 else "FAIL",
    }
    atomic_dump(out / "overlap_diagnostic.json", {"summary": diagnostic_summary, "routes": diagnostics})
    protocol = {
        "version": 1,
        "scope": "reused fit-only development evidence; no outer confirmation/evaluation access",
        "method": METHOD,
        "mechanism": "pool non-target station_infos and superSets under unseen target station; retain original JPT nearest-overlap matching",
        "target_history_used": False,
        "groups": {"station": groups}, "selected": {"station": selected},
        "headroom_gate_pre_score": "median nearest-history zone-set Jaccard >= 0.5",
        "headroom": diagnostic_summary,
        "solver": {"variants": 2, "seconds_per_variant": 1, "seed": 1, "workers": 4},
        "success_gate": "paired improvement over station-cold-start JPT >=1% and paired bootstrap delta CI95 upper bound <0",
        "source_hashes": hashes,
    }
    dump(out / "protocol.json", protocol)
    print(json.dumps(diagnostic_summary, indent=2), flush=True)


def build(out: Path, workers=4):
    protocol = json.loads((out / "protocol.json").read_text())
    if protocol["headroom"]["headroom_gate"] != "PASS":
        raise RuntimeError("headroom gate failed")
    if not all(fingerprint(Path(path)) == value for path, value in protocol["source_hashes"].items()):
        raise RuntimeError("source changed after protocol lock")
    fit_routes = json.loads((REFERENCE / "fit_routes.json").read_text())

    def one(index):
        folder = out / f"station_{index}"
        fold = json.loads((folder / "fold.json").read_text())
        with tempfile.TemporaryDirectory(prefix=f"shared-grammar-{index}-") as temporary:
            train_path = Path(temporary) / "train_routes.json"
            dump(train_path, {route_id: fit_routes[route_id] for route_id in fold["train"]})
            for variant in (1, 2):
                arguments = [
                    Path(__import__("sys").executable), JPT / "scripts/build_TSPLIB.py",
                    "--r_file", "target_routes.json", "--t_file", "target_times.json",
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
            print(f"Built shared grammar fold {completed}/17 (station_{index})", flush=True)


def configure_phase():
    phase.MODES = ("station",)
    phase.GROUPS_PER_MODE = 17


def score_and_compare(out: Path):
    configure_phase()
    phase.score(out)
    new_rows = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
    new_scores = {row["route_id"]: row["official_score"] for row in new_rows if row["method"] == "shifted_jpt"}
    cold_scores = {}
    for source in SOURCES:
        for line in (source / "results.jsonl").read_text().splitlines():
            row = json.loads(line)
            if row["mode"] == "station" and row["method"] == "shifted_jpt":
                cold_scores[row["route_id"]] = row["official_score"]
    if set(new_scores) != set(cold_scores) or len(new_scores) != 204:
        raise RuntimeError("comparison cohort mismatch")
    route_ids = sorted(new_scores)
    deltas = [new_scores[route_id] - cold_scores[route_id] for route_id in route_ids]
    cold_mean = statistics.fmean(cold_scores.values())
    shared_mean = statistics.fmean(new_scores.values())
    ci = phase.bootstrap_ci(deltas)
    improvement = 1 - shared_mean / cold_mean
    comparison = {
        "routes": 204, "cold_start_jpt_mean_score": cold_mean,
        "shared_zone_grammar_mean_score": shared_mean,
        "relative_improvement_over_cold_jpt": improvement,
        "mean_delta": statistics.fmean(deltas), "paired_delta_ci95": ci,
        "improved_routes": sum(delta < 0 for delta in deltas),
        "tied_routes": sum(delta == 0 for delta in deltas),
        "worsened_routes": sum(delta > 0 for delta in deltas),
        "gate_passed": improvement >= .01 and ci[1] < 0,
        "decision": "PROMOTE TO FROZEN CONFIRMATION" if improvement >= .01 and ci[1] < 0 else "STOP SHARED-GRAMMAR BRANCH",
        "evidence_status": "reused development cohort; not confirmation evidence",
    }
    atomic_dump(out / "comparison.json", comparison)
    report = [
        "# Shared station-invariant zone grammar", "", f"**{comparison['decision']}**", "",
        f"- Routes / stations: 204 / 17",
        f"- Cold-start JPT: {cold_mean:.8f}",
        f"- Shared-zone grammar JPT: {shared_mean:.8f}",
        f"- Relative improvement: {100 * improvement:.2f}%",
        f"- Paired delta CI95: [{ci[0]:.8f}, {ci[1]:.8f}]",
        f"- Better/tied/worse: {comparison['improved_routes']}/{comparison['tied_routes']}/{comparison['worsened_routes']}", "",
        "The gate was fixed before scoring: at least 1% mean improvement and a paired bootstrap CI entirely below zero. This reused cohort is development evidence only.",
    ]
    (out / "comparison_report.md").write_text("\n".join(report) + "\n")
    (ROOT / "reports/shared_zone_grammar_v1.md").write_text("\n".join(report) + "\n")
    print(json.dumps(comparison, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/shared_zone_grammar_v1"))
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
