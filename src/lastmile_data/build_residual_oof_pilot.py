"""Leakage-controlled pilot for a large JPT-residual training corpus."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .baselines import evaluate, travel_time, validate_sequence
from .benchmark import objects
from .research_utils import atomic_dump, bootstrap_ci, normalized_zones
from .reproduce_jpt import JPT, ROOT, command, dump, fingerprint, official_score, read_tour
from .research_utils import stratified_hash_folds
from .zone_repair import moves


REFERENCE = ROOT / "outputs/jpt_reproduction_v1"
RAW = ROOT / "data/raw/amazon/almrrc2021-data-training/model_build_inputs"
FOLDS = 5
SAMPLE_PER_FOLD = 25
POOL_SIZE = 32
TRAVEL_BUDGET = 0.10


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def route_with_filled_zones(route):
    result = copy.deepcopy(route)
    for stop, zone in normalized_zones(route).items():
        if result["stops"][stop]["type"] != "Station":
            result["stops"][stop]["zone_id"] = zone
    return result


def residual_candidates(anchor, route, costs, limit=POOL_SIZE, budget=TRAVEL_BUDGET):
    """Hash-sample one/two-move candidates without using outcome labels."""
    route = route_with_filled_zones(route)
    depot = anchor[0]
    validate_sequence(anchor, route["stops"], depot)
    ceiling = travel_time(anchor, costs) * (1 + budget) + 1e-8

    def valid(sequence):
        return sequence[0] == depot and travel_time(sequence, costs) <= ceiling

    one = {tuple(sequence) for sequence in moves(anchor, route, True) if valid(sequence)}
    ordered_one = sorted(one, key=lambda sequence: digest("residual-one-v1:" + "|".join(sequence)))
    chosen = [tuple(anchor), *ordered_one[: min(24, limit - 1)]]
    # A small deterministic two-hop tail broadens the pool while staying near JPT.
    two = set()
    for seed in ordered_one[:8]:
        for sequence in moves(list(seed), route, True):
            value = tuple(sequence)
            if value not in one and value != tuple(anchor) and valid(sequence):
                two.add(value)
    for sequence in sorted(two, key=lambda value: digest("residual-two-v1:" + "|".join(value))):
        if len(chosen) >= limit:
            break
        chosen.append(sequence)
    result = [list(sequence) for sequence in dict.fromkeys(chosen)]
    for sequence in result:
        validate_sequence(sequence, route["stops"], depot)
    return result


def initialize(out: Path):
    if out.exists():
        raise FileExistsError(out)
    routes = json.loads((REFERENCE / "fit_routes.json").read_text())
    actual = json.loads((REFERENCE / "fit_sequences.json").read_text())
    assert set(routes) == set(actual) and len(routes) == 5005
    stations = {rid: route["station_code"] for rid, route in routes.items()}
    folds = stratified_hash_folds(routes, stations, FOLDS, "residual-oof-v1:")
    selected = []
    for fold in range(FOLDS):
        eligible = [rid for rid in routes if folds[rid] == fold and routes[rid]["route_score"] == "High"]
        chosen = sorted(eligible, key=lambda rid: digest("residual-pilot-v1:" + rid))[:SAMPLE_PER_FOLD]
        if len(chosen) != SAMPLE_PER_FOLD:
            raise RuntimeError(f"insufficient High routes in fold {fold}")
        selected.extend(chosen)
    out.mkdir(parents=True)
    protocol = {
        "version": 1,
        "scope": "fit-only OOF pilot; no outer development, confirmation or evaluation access",
        "folds": folds,
        "selected": selected,
        "sample_per_fold": SAMPLE_PER_FOLD,
        "selection_population": "High-quality fit routes; quality is never an inference feature",
        "jpt_history_excludes_entire_target_fold": True,
        "pool_size": POOL_SIZE,
        "travel_budget": TRAVEL_BUDGET,
        "candidate_sampling": "24 deterministic one-move plus up to 7 deterministic two-move residuals",
        "labels_loaded_after_candidate_generation": True,
        "source_hashes": {
            str(path): fingerprint(path)
            for path in [
                Path(__file__), REFERENCE / "fit_routes.json", REFERENCE / "fit_sequences.json",
                REFERENCE / "provenance.json", JPT / "scripts/analyze_zone_id_order_main.py",
                JPT / "scripts/build_TSPLIB.py", JPT / "LKH-AMZ/LKH",
            ]
        },
        "evaluation_access": False,
    }
    atomic_dump(out / "protocol.json", protocol)

    selected_set = set(selected)
    packages = {}
    for rid, value in objects(RAW / "package_data.json"):
        if rid in selected_set:
            cleaned = copy.deepcopy(value)
            for stop in cleaned.values():
                for package in stop.values():
                    package.pop("scan_status", None)
            packages[rid] = cleaned
    times = {rid: value for rid, value in objects(RAW / "travel_times.json") if rid in selected_set}
    if set(packages) != selected_set or set(times) != selected_set:
        raise RuntimeError("incomplete selected operational inputs")

    for fold in range(FOLDS):
        folder = out / f"fold_{fold}"
        folder.mkdir()
        targets = [rid for rid in selected if folds[rid] == fold]
        train = [rid for rid in routes if folds[rid] != fold]
        assert not set(targets) & set(train)
        dump(folder / "train_routes.json", {rid: routes[rid] for rid in train})
        dump(folder / "train_sequences.json", {rid: actual[rid] for rid in train})
        dump(folder / "target_routes.json", {
            rid: {key: value for key, value in routes[rid].items() if key != "route_score"} for rid in targets
        })
        dump(folder / "target_packages.json", {rid: packages[rid] for rid in targets})
        dump(folder / "target_times.json", {rid: times[rid] for rid in targets})
        dump(folder / "fold.json", {"fold": fold, "train": train, "targets": targets})
    print(f"Prepared {len(selected)} OOF pilot targets", flush=True)


def build(out: Path):
    protocol = json.loads((out / "protocol.json").read_text())
    for fold in range(FOLDS):
        folder = out / f"fold_{fold}"
        command([
            Path(__import__("sys").executable), JPT / "scripts/analyze_zone_id_order_main.py",
            "--r_file", "train_routes.json", "--s_file", "train_sequences.json", "--out_file", "model.json",
        ], folder, folder / "build.log")
        for variant in (1, 2):
            arguments = [
                Path(__import__("sys").executable), JPT / "scripts/build_TSPLIB.py",
                "--r_file", "target_routes.json", "--t_file", "target_times.json",
                "--p_file", "target_packages.json", "--z_file", "model.json",
                "--br_file", "train_routes.json", "-noPrune", "-noPruneFailed",
                "-zoneNeighborTrans", "-superNeighborTrans",
            ]
            if variant == 2:
                arguments += ["-superPred", "-trans"]
            arguments += [f"TSPLIB_{variant}"]
            command(arguments, folder, folder / f"build_variant_{variant}.log")
            count = len(list((folder / f"TSPLIB_{variant}").glob("*.ctsptw")))
            if count != SAMPLE_PER_FOLD:
                raise RuntimeError(f"fold {fold} variant {variant}: {count} instances")
        print(f"Built JPT OOF fold {fold + 1}/{FOLDS}", flush=True)
    assert len(protocol["selected"]) == FOLDS * SAMPLE_PER_FOLD


def solve(out: Path, workers: int, seconds: float):
    tasks = []
    for fold in range(FOLDS):
        folder = out / f"fold_{fold}"
        for variant in (1, 2):
            (folder / f"tours_{variant}").mkdir(exist_ok=True)
            (folder / f"params_{variant}").mkdir(exist_ok=True)
            for instance in sorted((folder / f"TSPLIB_{variant}").glob("*.ctsptw")):
                tasks.append((fold, variant, instance.stem))

    def one(task):
        fold, variant, name = task
        folder = out / f"fold_{fold}"
        parameter = folder / f"params_{variant}/{name}.par"
        parameter.write_text(
            f"PROBLEM_FILE = TSPLIB_{variant}/{name}.ctsptw\nCTSP_TRANSFORM = YES\n"
            f"SEED = 1\nTIME_LIMIT = {seconds}\nTRACE_LEVEL = 0\nTOUR_FILE = tours_{variant}/{name}.tour\n"
        )
        begin = time.perf_counter()
        command([JPT / "LKH-AMZ/LKH", parameter.relative_to(folder)], folder, parameter.with_suffix(".log"))
        read_tour(folder / f"tours_{variant}/{name}.tour")
        return {"fold": fold, "variant": variant, "name": name, "seconds": time.perf_counter() - begin}

    timings = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, result in enumerate(pool.map(one, tasks), 1):
            timings.append(result)
            if index % 50 == 0 or index == len(tasks):
                print(f"JPT OOF solvers {index}/{len(tasks)}", flush=True)
    atomic_dump(out / "solver.json", {"seconds_per_variant": seconds, "workers": workers, "timings": timings})


def anchors(out: Path):
    predictions = {}
    for fold in range(FOLDS):
        folder = out / f"fold_{fold}"
        mapping = json.loads((folder / "TSPLIB_1/tsp2amz.json").read_text())
        mapping2 = json.loads((folder / "TSPLIB_2/tsp2amz.json").read_text())
        if mapping != mapping2:
            raise RuntimeError("variant mappings differ")
        for name, entry in mapping.items():
            length1, nodes1 = read_tour(folder / f"tours_1/{name}.tour")
            length2, nodes2 = read_tour(folder / f"tours_2/{name}.tour")
            take1 = length1 <= (101 * length2) // 100
            nodes = nodes1 if take1 else nodes2
            rid = entry["route_id"]
            if rid in predictions:
                raise RuntimeError("duplicate target")
            predictions[rid] = {
                "sequence": [entry["stops"][node] for node in nodes],
                "variant": 1 if take1 else 2,
                "fold": fold,
            }
    atomic_dump(out / "anchors.json", predictions)


def generate(out: Path):
    protocol = json.loads((out / "protocol.json").read_text())
    anchors_data = json.loads((out / "anchors.json").read_text())
    path = out / "candidates.jsonl"
    if path.exists():
        raise FileExistsError(path)
    selected = set(protocol["selected"])
    with path.open("x") as handle:
        completed = 0
        for fold in range(FOLDS):
            folder = out / f"fold_{fold}"
            routes = json.loads((folder / "target_routes.json").read_text())
            for rid, costs in objects(folder / "target_times.json"):
                anchor = anchors_data[rid]["sequence"]
                sequences = residual_candidates(anchor, routes[rid], costs)
                handle.write(json.dumps({
                    "route_id": rid, "fold": fold, "anchor_variant": anchors_data[rid]["variant"],
                    "sequences": sequences,
                }) + "\n")
                handle.flush()
                completed += 1
                if completed % 25 == 0:
                    print(f"Residual candidates {completed}/{len(selected)}", flush=True)
    if completed != len(selected):
        raise RuntimeError("incomplete residual generation")


def score(out: Path):
    protocol = json.loads((out / "protocol.json").read_text())
    records = {}
    with (out / "candidates.jsonl").open() as handle:
        for line in handle:
            item = json.loads(line)
            records[item["route_id"]] = item
    selected = set(protocol["selected"])
    if set(records) != selected:
        raise RuntimeError("freeze all candidates before loading labels")
    labels = json.loads((REFERENCE / "fit_sequences.json").read_text())
    rows = []
    route_summaries = []
    for fold in range(FOLDS):
        folder = out / f"fold_{fold}"
        routes = json.loads((folder / "target_routes.json").read_text())
        packages = json.loads((folder / "target_packages.json").read_text())
        for rid, costs in objects(folder / "target_times.json"):
            reference = sorted(labels[rid]["actual"], key=labels[rid]["actual"].get)
            values = []
            for index, sequence in enumerate(records[rid]["sequences"]):
                value = official_score(reference, sequence, costs)
                metrics = evaluate(sequence, routes[rid], packages[rid], costs)
                rows.append({
                    "route_id": rid, "fold": fold, "candidate": index,
                    "official_score": value, "sequence": sequence, **metrics,
                })
                values.append(value)
            base, best = values[0], min(values)
            route_summaries.append({
                "route_id": rid, "fold": fold, "candidates": len(values),
                "anchor_score": base, "oracle_score": best, "delta": best - base,
                "improving_candidates": sum(value < base - 1e-14 for value in values[1:]),
            })
            if len(route_summaries) % 25 == 0:
                print(f"Residual scored {len(route_summaries)}/{len(selected)}", flush=True)
    deltas = np.array([row["delta"] for row in route_summaries])
    anchor = np.array([row["anchor_score"] for row in route_summaries])
    oracle = np.array([row["oracle_score"] for row in route_summaries])
    total_nonanchor = sum(row["candidates"] - 1 for row in route_summaries)
    positives = sum(row["improving_candidates"] for row in route_summaries)
    summary = {
        "routes": len(route_summaries),
        "candidate_records": len(rows),
        "mean_candidates_per_route": float(np.mean([row["candidates"] for row in route_summaries])),
        "anchor_mean_score": float(np.mean(anchor)),
        "oracle_mean_score": float(np.mean(oracle)),
        "oracle_relative_improvement": float(1 - np.mean(oracle) / np.mean(anchor)),
        "oracle_mean_delta": float(np.mean(deltas)),
        "oracle_delta_ci95": bootstrap_ci(deltas),
        "routes_with_improvement": int(np.sum(deltas < 0)),
        "positive_candidate_records": positives,
        "positive_candidate_rate": positives / max(1, total_nonanchor),
        "source_hashes_match": all(
            fingerprint(Path(path)) == value for path, value in protocol["source_hashes"].items()
        ),
        "decision": "EXPAND FULL OOF" if float(1 - np.mean(oracle) / np.mean(anchor)) >= 0.03 and positives / max(1, total_nonanchor) >= 0.10 else "REVISE GENERATOR BEFORE FULL OOF",
        "caveat": "Label-selected oracle on fit-only OOF pilot; not deployable or confirmation evidence",
    }
    atomic_dump(out / "results.json", rows)
    atomic_dump(out / "routes.json", route_summaries)
    atomic_dump(out / "summary.json", summary)
    (out / "report.md").write_text(f"""# Fit-only OOF JPT-residual pilot

**{summary['decision']}**

- Routes: {summary['routes']}
- Candidate records: {summary['candidate_records']}
- Mean candidates per route: {summary['mean_candidates_per_route']:.2f}
- JPT OOF anchor score: {summary['anchor_mean_score']:.8f}
- Oracle residual score: {summary['oracle_mean_score']:.8f}
- Oracle relative improvement: {100 * summary['oracle_relative_improvement']:.3f}%
- Oracle delta CI95: [{summary['oracle_delta_ci95'][0]:.8f}, {summary['oracle_delta_ci95'][1]:.8f}]
- Routes with an improving candidate: {summary['routes_with_improvement']}
- Positive non-anchor candidate rate: {100 * summary['positive_candidate_rate']:.2f}%

This pilot uses only High-quality routes from the 5,005 fit population. Every
target route is excluded from the full history fold used to build its JPT anchor.
Actual sequences are loaded only after every candidate is frozen.
""")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/residual_oof_pilot_v1"))
    parser.add_argument("--stage", choices=["init", "build", "solve", "anchors", "generate", "score", "all"], default="all")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=1.0)
    args = parser.parse_args()
    out = args.output.resolve()
    stages = [
        ("init", initialize), ("build", build),
        ("solve", lambda path: solve(path, args.workers, args.seconds)),
        ("anchors", anchors), ("generate", generate), ("score", score),
    ]
    for name, function in stages:
        if args.stage in (name, "all"):
            print(f"Stage: {name}", flush=True)
            function(out)


if __name__ == "__main__":
    main()
