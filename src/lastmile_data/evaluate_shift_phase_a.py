"""Bounded Phase-A stress test for station and temporal distribution shift."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import statistics
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .baselines import evaluate, validate_sequence
from .benchmark import objects
from .research_utils import atomic_dump
from .reproduce_jpt import JPT, ROOT, command, dump, fingerprint, official_score, read_tour


REFERENCE = ROOT / "outputs/jpt_reproduction_v1"
RANDOM_OOF = ROOT / "outputs/residual_oof_full_v1"
RAW = ROOT / "data/raw/amazon/almrrc2021-data-training/model_build_inputs"
MODES = ("station", "temporal")
GROUPS_PER_MODE = 5
TARGETS_PER_GROUP = 12


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def choose_groups(routes, eligible):
    """Choose well-supported station and late-date groups before scoring."""
    stations = {}
    dates = {}
    for route_id in eligible:
        route = routes[route_id]
        stations.setdefault(route["station_code"], []).append(route_id)
        dates.setdefault(route["date_YYYY_MM_DD"], []).append(route_id)
    station_groups = sorted(
        (group for group, ids in stations.items() if len(ids) >= TARGETS_PER_GROUP),
        key=lambda group: (-len(stations[group]), group),
    )[:GROUPS_PER_MODE]
    date_groups = sorted(
        (group for group, ids in dates.items() if len(ids) >= TARGETS_PER_GROUP), reverse=True,
    )[:GROUPS_PER_MODE]
    if len(station_groups) != GROUPS_PER_MODE or len(date_groups) != GROUPS_PER_MODE:
        raise RuntimeError("insufficient groups for locked shift pilot")
    return {"station": station_groups, "temporal": date_groups}


def select_targets(routes, eligible, groups):
    selected = {mode: {} for mode in MODES}
    for mode in MODES:
        field = "station_code" if mode == "station" else "date_YYYY_MM_DD"
        for group in groups[mode]:
            candidates = [rid for rid in eligible if routes[rid][field] == group]
            selected[mode][group] = sorted(
                candidates, key=lambda rid: digest(f"shift-phase-a:{mode}:{group}:{rid}")
            )[:TARGETS_PER_GROUP]
    return selected


def bootstrap_ci(values, seed=42, repetitions=10_000):
    rng = random.Random(seed)
    samples = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(repetitions))
    return [samples[int(0.025 * repetitions)], samples[int(0.975 * repetitions)]]


def initialize(out: Path):
    if out.exists():
        raise FileExistsError(out)
    routes = json.loads((REFERENCE / "fit_routes.json").read_text())
    actual = json.loads((REFERENCE / "fit_sequences.json").read_text())
    oof_protocol = json.loads((RANDOM_OOF / "protocol.json").read_text())
    eligible = set(oof_protocol["selected"])
    groups = choose_groups(routes, eligible)
    selected = select_targets(routes, eligible, groups)
    all_targets = {rid for mode in MODES for ids in selected[mode].values() for rid in ids}
    if not all_targets <= eligible:
        raise RuntimeError("targets lack random-OOF controls")
    out.mkdir(parents=True)
    protocol = {
        "version": 1,
        "scope": "fit-only Phase-A shift pilot; no outer development, confirmation, or evaluation",
        "population": "High-quality routes with existing leakage-controlled random-OOF JPT controls",
        "groups": groups, "selected": selected,
        "targets_per_group": TARGETS_PER_GROUP,
        "station_training": "all fit routes from every other station; held station completely unseen",
        "temporal_training": "all fit routes with date strictly earlier than target date",
        "target_labels_unavailable_until_all_shifted_tours_are_frozen": True,
        "solver": {"variants": 2, "seconds_per_variant": 1.0, "seed": 1, "workers": 4},
        "phase_b_gate": "relative score degradation >=5% and paired bootstrap delta CI95 entirely >0 in either mode",
        "bootstrap": {"seed": 42, "replicates": 10_000},
        "quality_is_not_an_inference_feature": True,
        "evaluation_access": False,
        "source_hashes": {str(path): fingerprint(path) for path in [
            Path(__file__), REFERENCE / "fit_routes.json", REFERENCE / "fit_sequences.json",
            RANDOM_OOF / "protocol.json", RANDOM_OOF / "anchors.json", RANDOM_OOF / "routes.json",
            JPT / "scripts/analyze_zone_id_order_main.py", JPT / "scripts/build_TSPLIB.py",
            JPT / "LKH-AMZ/LKH",
        ]},
    }
    dump(out / "protocol.json", protocol)

    packages = {}
    for rid, value in objects(RAW / "package_data.json"):
        if rid in all_targets:
            cleaned = copy.deepcopy(value)
            for stop in cleaned.values():
                for package in stop.values():
                    package.pop("scan_status", None)
            packages[rid] = cleaned
    times = {rid: value for rid, value in objects(RAW / "travel_times.json") if rid in all_targets}
    if set(packages) != all_targets or set(times) != all_targets:
        raise RuntimeError("incomplete operational inputs")

    for mode in MODES:
        for index, group in enumerate(groups[mode]):
            folder = out / f"{mode}_{index}"
            folder.mkdir()
            targets = selected[mode][group]
            if mode == "station":
                train = [rid for rid, route in routes.items() if route["station_code"] != group]
                if any(routes[rid]["station_code"] == group for rid in train):
                    raise RuntimeError("station leakage")
            else:
                train = [rid for rid, route in routes.items() if route["date_YYYY_MM_DD"] < group]
                if any(routes[rid]["date_YYYY_MM_DD"] >= group for rid in train):
                    raise RuntimeError("temporal leakage")
            dump(folder / "fold.json", {"mode": mode, "index": index, "group": group,
                                         "train": train, "targets": targets})
            dump(folder / "target_routes.json", {
                rid: {key: value for key, value in routes[rid].items() if key != "route_score"}
                for rid in targets
            })
            dump(folder / "target_packages.json", {rid: packages[rid] for rid in targets})
            dump(folder / "target_times.json", {rid: times[rid] for rid in targets})
    print(f"Prepared {sum(len(v) for mode in selected.values() for v in mode.values())} shifted evaluations", flush=True)


def build(out: Path):
    protocol = json.loads((out / "protocol.json").read_text())
    if not all(fingerprint(Path(path)) == value for path, value in protocol["source_hashes"].items()):
        raise RuntimeError("source changed after protocol lock")
    routes = json.loads((REFERENCE / "fit_routes.json").read_text())
    actual = json.loads((REFERENCE / "fit_sequences.json").read_text())
    for mode in MODES:
        for index, group in enumerate(protocol["groups"][mode]):
            folder = out / f"{mode}_{index}"
            train = json.loads((folder / "fold.json").read_text())["train"]
            with tempfile.TemporaryDirectory(prefix=f"shift-{mode}-{index}-") as temporary:
                temporary = Path(temporary)
                dump(temporary / "train_routes.json", {rid: routes[rid] for rid in train})
                dump(temporary / "train_sequences.json", {rid: actual[rid] for rid in train})
                command([
                    Path(__import__("sys").executable), JPT / "scripts/analyze_zone_id_order_main.py",
                    "--r_file", temporary / "train_routes.json",
                    "--s_file", temporary / "train_sequences.json", "--out_file", "model.json",
                ], folder, folder / "build.log")
                for variant in (1, 2):
                    arguments = [
                        Path(__import__("sys").executable), JPT / "scripts/build_TSPLIB.py",
                        "--r_file", "target_routes.json", "--t_file", "target_times.json",
                        "--p_file", "target_packages.json", "--z_file", "model.json",
                        "--br_file", temporary / "train_routes.json", "-noPrune", "-noPruneFailed",
                        "-zoneNeighborTrans", "-superNeighborTrans",
                    ]
                    if variant == 2:
                        arguments += ["-superPred", "-trans"]
                    arguments += [f"TSPLIB_{variant}"]
                    command(arguments, folder, folder / f"build_variant_{variant}.log")
                    count = len(list((folder / f"TSPLIB_{variant}").glob("*.ctsptw")))
                    if count != TARGETS_PER_GROUP:
                        raise RuntimeError(f"{mode} {group} variant {variant}: {count} instances")
            print(f"Built {mode} group {index + 1}/{GROUPS_PER_MODE}: {group}", flush=True)


def solve(out: Path, workers=4):
    protocol = json.loads((out / "protocol.json").read_text())
    tasks = []
    for mode in MODES:
        for index in range(GROUPS_PER_MODE):
            folder = out / f"{mode}_{index}"
            for variant in (1, 2):
                (folder / f"tours_{variant}").mkdir(exist_ok=True)
                (folder / f"params_{variant}").mkdir(exist_ok=True)
                for instance in sorted((folder / f"TSPLIB_{variant}").glob("*.ctsptw")):
                    tasks.append((mode, index, variant, instance.stem))

    def one(task):
        mode, index, variant, name = task
        folder = out / f"{mode}_{index}"
        tour = folder / f"tours_{variant}/{name}.tour"
        parameter = folder / f"params_{variant}/{name}.par"
        parameter.write_text(
            f"PROBLEM_FILE = TSPLIB_{variant}/{name}.ctsptw\nCTSP_TRANSFORM = YES\n"
            f"SEED = 1\nTIME_LIMIT = 1\nTRACE_LEVEL = 0\nTOUR_FILE = tours_{variant}/{name}.tour\n"
        )
        begin = time.perf_counter()
        command([JPT / "LKH-AMZ/LKH", parameter.relative_to(folder)], folder, parameter.with_suffix(".log"))
        read_tour(tour)
        return {"mode": mode, "group_index": index, "variant": variant,
                "name": name, "seconds": time.perf_counter() - begin}

    timings = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for completed, result in enumerate(pool.map(one, tasks), 1):
            timings.append(result)
            if completed % 40 == 0 or completed == len(tasks):
                print(f"Shift solvers {completed}/{len(tasks)}", flush=True)
    dump(out / "solver.json", {"workers": workers, "timings": timings})


def freeze(out: Path):
    predictions = {}
    for mode in MODES:
        predictions[mode] = {}
        for index in range(GROUPS_PER_MODE):
            folder = out / f"{mode}_{index}"
            mapping = json.loads((folder / "TSPLIB_1/tsp2amz.json").read_text())
            if mapping != json.loads((folder / "TSPLIB_2/tsp2amz.json").read_text()):
                raise RuntimeError("variant mappings differ")
            for name, entry in mapping.items():
                length1, nodes1 = read_tour(folder / f"tours_1/{name}.tour")
                length2, nodes2 = read_tour(folder / f"tours_2/{name}.tour")
                take1 = length1 <= (101 * length2) // 100
                nodes = nodes1 if take1 else nodes2
                rid = entry["route_id"]
                predictions[mode][rid] = {
                    "sequence": [entry["stops"][node] for node in nodes],
                    "variant": 1 if take1 else 2, "group_index": index,
                }
    dump(out / "predictions.json", predictions)
    print("All shifted predictions frozen before labels", flush=True)


def score(out: Path):
    protocol = json.loads((out / "protocol.json").read_text())
    predictions = json.loads((out / "predictions.json").read_text())
    random_anchors = json.loads((RANDOM_OOF / "anchors.json").read_text())
    oracle_rows = {row["route_id"]: row for row in json.loads((RANDOM_OOF / "routes.json").read_text())}
    actual = json.loads((REFERENCE / "fit_sequences.json").read_text())
    rows = []
    for mode in MODES:
        expected = {rid for ids in protocol["selected"][mode].values() for rid in ids}
        if set(predictions[mode]) != expected:
            raise RuntimeError(f"incomplete frozen predictions for {mode}")
        for index in range(GROUPS_PER_MODE):
            folder = out / f"{mode}_{index}"
            routes = json.loads((folder / "target_routes.json").read_text())
            packages = json.loads((folder / "target_packages.json").read_text())
            group = json.loads((folder / "fold.json").read_text())["group"]
            for rid, costs in objects(folder / "target_times.json"):
                reference = sorted(actual[rid]["actual"], key=actual[rid]["actual"].get)
                methods = {
                    "random_oof_jpt": random_anchors[rid]["sequence"],
                    "shifted_jpt": predictions[mode][rid]["sequence"],
                }
                for method, sequence in methods.items():
                    depot = reference[0]
                    validate_sequence(sequence, routes[rid]["stops"], depot)
                    rows.append({"route_id": rid, "mode": mode, "group": group, "method": method,
                                 "official_score": official_score(reference, sequence, costs),
                                 **evaluate(sequence, routes[rid], packages[rid], costs)})
    with (out / "results.jsonl").open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    summaries = {}
    phase_b = False
    for mode in MODES:
        subset = [row for row in rows if row["mode"] == mode]
        lookup = {(row["route_id"], row["method"]): row["official_score"] for row in subset}
        route_ids = sorted({row["route_id"] for row in subset})
        random_scores = [lookup[rid, "random_oof_jpt"] for rid in route_ids]
        shifted_scores = [lookup[rid, "shifted_jpt"] for rid in route_ids]
        deltas = [shifted - control for shifted, control in zip(shifted_scores, random_scores)]
        random_mean = statistics.fmean(random_scores)
        shifted_mean = statistics.fmean(shifted_scores)
        ci = bootstrap_ci(deltas)
        degradation = shifted_mean / random_mean - 1
        qualifies = degradation >= 0.05 and ci[0] > 0
        phase_b |= qualifies
        group_rows = []
        for group in protocol["groups"][mode]:
            ids = sorted({row["route_id"] for row in subset if row["group"] == group})
            group_rows.append({
                "group": group, "routes": len(ids),
                "random_oof_score": statistics.fmean(lookup[rid, "random_oof_jpt"] for rid in ids),
                "shifted_score": statistics.fmean(lookup[rid, "shifted_jpt"] for rid in ids),
            })
        summaries[mode] = {
            "routes": len(route_ids), "random_oof_mean_score": random_mean,
            "shifted_mean_score": shifted_mean, "relative_degradation": degradation,
            "mean_delta": statistics.fmean(deltas), "delta_ci95": ci,
            "routes_shifted_better": sum(delta < 0 for delta in deltas),
            "routes_shifted_tied": sum(delta == 0 for delta in deltas),
            "routes_shifted_worse": sum(delta > 0 for delta in deltas),
            "oracle_residual_mean_score": statistics.fmean(oracle_rows[rid]["oracle_score"] for rid in route_ids),
            "qualifies_for_phase_b": qualifies, "groups": group_rows,
        }
    summary = {"modes": summaries, "decision": "PROCEED TO SARP PHASE B" if phase_b else "STOP BEFORE SARP",
               "phase_b_gate_passed": phase_b}
    atomic_dump(out / "summary.json", summary)
    report_lines = ["# Distribution-shift Phase A", "", f"**{summary['decision']}**", ""]
    for mode in MODES:
        item = summaries[mode]
        report_lines += [
            f"## {mode.title()} shift", "",
            f"- Routes: {item['routes']}",
            f"- Random-OOF JPT: {item['random_oof_mean_score']:.8f}",
            f"- Shifted-history JPT: {item['shifted_mean_score']:.8f}",
            f"- Relative degradation: {100 * item['relative_degradation']:.2f}%",
            f"- Paired delta CI95: [{item['delta_ci95'][0]:.8f}, {item['delta_ci95'][1]:.8f}]",
            f"- Better/tied/worse: {item['routes_shifted_better']}/{item['routes_shifted_tied']}/{item['routes_shifted_worse']}",
            f"- Existing residual oracle: {item['oracle_residual_mean_score']:.8f}", "",
        ]
    (ROOT / f"reports/{out.name}.md").write_text("\n".join(report_lines))
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/shift_phase_a_v1"))
    parser.add_argument("--stage", choices=["init", "build", "solve", "freeze", "score", "all"], default="all")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    out = args.output.resolve()
    stages = [("init", initialize), ("build", build),
              ("solve", lambda path: solve(path, args.workers)), ("freeze", freeze), ("score", score)]
    for name, function in stages:
        if args.stage in (name, "all"):
            print("Stage: " + name, flush=True)
            function(out)


if __name__ == "__main__":
    main()
