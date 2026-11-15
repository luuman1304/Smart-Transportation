"""Build the full High-route, leakage-controlled JPT-residual OOF corpus.

The expensive scoring stage is route-parallel and append-checkpointed.  Actual
sequences are not read until every candidate sequence has been frozen.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .baselines import evaluate
from .benchmark import objects
from .build_residual_oof_pilot import (
    FOLDS, JPT, RAW, REFERENCE, TRAVEL_BUDGET, digest,
)
from .build_residual_oof_pilot_v2 import residual_candidates_v2
from .research_utils import atomic_dump, bootstrap_ci
from .reproduce_jpt import ROOT, command, dump, fingerprint, load_scorer, read_tour
from .research_utils import stratified_hash_folds


POOL_SIZE = 32
QUALITY = "High"


def initialize(out: Path) -> None:
    if out.exists():
        raise FileExistsError(out)
    routes = json.loads((REFERENCE / "fit_routes.json").read_text())
    actual = json.loads((REFERENCE / "fit_sequences.json").read_text())
    if set(routes) != set(actual) or len(routes) != 5005:
        raise RuntimeError("unexpected JPT fit population")
    stations = {rid: route["station_code"] for rid, route in routes.items()}
    folds = stratified_hash_folds(routes, stations, FOLDS, "residual-oof-v1:")
    selected = sorted(rid for rid, route in routes.items() if route["route_score"] == QUALITY)
    out.mkdir(parents=True)
    protocol = {
        "version": 1,
        "scope": "full fit-only OOF corpus; no holdout development, confirmation or evaluation access",
        "folds": folds,
        "selected": selected,
        "selection_population": "all High-quality fit routes; quality is never an inference feature",
        "jpt_history_excludes_entire_target_fold": True,
        "pool_size": POOL_SIZE,
        "travel_budget": TRAVEL_BUDGET,
        "candidate_sampling": "31 deterministic one-hop residuals, validated on two disjoint pilots",
        "labels_loaded_after_candidate_generation": True,
        "scoring_checkpointed_by_route": True,
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
        if set(targets) & set(train):
            raise RuntimeError("OOF target leaked into JPT history")
        dump(folder / "train_routes.json", {rid: routes[rid] for rid in train})
        dump(folder / "train_sequences.json", {rid: actual[rid] for rid in train})
        dump(folder / "target_routes.json", {
            rid: {key: value for key, value in routes[rid].items() if key != "route_score"}
            for rid in targets
        })
        dump(folder / "target_packages.json", {rid: packages[rid] for rid in targets})
        dump(folder / "target_times.json", {rid: times[rid] for rid in targets})
        dump(folder / "fold.json", {"fold": fold, "train": train, "targets": targets})
    print(f"Prepared {len(selected)} full OOF High targets", flush=True)


def build(out: Path) -> None:
    for fold in range(FOLDS):
        folder = out / f"fold_{fold}"
        target_count = len(json.loads((folder / "fold.json").read_text())["targets"])
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
            if count != target_count:
                raise RuntimeError(f"fold {fold} variant {variant}: {count}/{target_count} instances")
        print(f"Built full JPT OOF fold {fold + 1}/{FOLDS}", flush=True)


def solve(out: Path, workers: int, seconds: float) -> None:
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
        tour = folder / f"tours_{variant}/{name}.tour"
        if tour.exists():
            read_tour(tour)
            return {"fold": fold, "variant": variant, "name": name, "seconds": 0.0, "reused": True}
        parameter = folder / f"params_{variant}/{name}.par"
        parameter.write_text(
            f"PROBLEM_FILE = TSPLIB_{variant}/{name}.ctsptw\nCTSP_TRANSFORM = YES\n"
            f"SEED = 1\nTIME_LIMIT = {seconds}\nTRACE_LEVEL = 0\nTOUR_FILE = tours_{variant}/{name}.tour\n"
        )
        begin = time.perf_counter()
        command([JPT / "LKH-AMZ/LKH", parameter.relative_to(folder)], folder, parameter.with_suffix(".log"))
        read_tour(tour)
        return {"fold": fold, "variant": variant, "name": name, "seconds": time.perf_counter() - begin, "reused": False}

    timings = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, result in enumerate(pool.map(one, tasks), 1):
            timings.append(result)
            if index % 100 == 0 or index == len(tasks):
                print(f"Full JPT OOF solvers {index}/{len(tasks)}", flush=True)
    atomic_dump(out / "solver.json", {"seconds_per_variant": seconds, "workers": workers, "timings": timings})


def anchors(out: Path) -> None:
    predictions = {}
    for fold in range(FOLDS):
        folder = out / f"fold_{fold}"
        mapping = json.loads((folder / "TSPLIB_1/tsp2amz.json").read_text())
        if mapping != json.loads((folder / "TSPLIB_2/tsp2amz.json").read_text()):
            raise RuntimeError("variant mappings differ")
        for name, entry in mapping.items():
            length1, nodes1 = read_tour(folder / f"tours_1/{name}.tour")
            length2, nodes2 = read_tour(folder / f"tours_2/{name}.tour")
            take1 = length1 <= (101 * length2) // 100
            rid = entry["route_id"]
            predictions[rid] = {
                "sequence": [entry["stops"][node] for node in (nodes1 if take1 else nodes2)],
                "variant": 1 if take1 else 2,
                "fold": fold,
            }
    selected = set(json.loads((out / "protocol.json").read_text())["selected"])
    if set(predictions) != selected:
        raise RuntimeError("incomplete anchors")
    atomic_dump(out / "anchors.json", predictions)


def generate(out: Path) -> None:
    protocol = json.loads((out / "protocol.json").read_text())
    anchor_data = json.loads((out / "anchors.json").read_text())
    path = out / "candidates.jsonl"
    if path.exists():
        raise FileExistsError(path)
    completed = 0
    with path.open("x") as handle:
        for fold in range(FOLDS):
            folder = out / f"fold_{fold}"
            routes = json.loads((folder / "target_routes.json").read_text())
            for rid, costs in objects(folder / "target_times.json"):
                sequences = residual_candidates_v2(
                    anchor_data[rid]["sequence"], routes[rid], costs, limit=POOL_SIZE,
                )
                handle.write(json.dumps({
                    "route_id": rid, "fold": fold, "anchor_variant": anchor_data[rid]["variant"],
                    "sequences": sequences,
                }) + "\n")
                completed += 1
                if completed % 100 == 0 or completed == len(protocol["selected"]):
                    print(f"Full residual candidates {completed}/{len(protocol['selected'])}", flush=True)
    if completed != len(protocol["selected"]):
        raise RuntimeError("incomplete candidate generation")


def _score_route(task):
    rid, fold, reference, sequences, costs, route, packages = task
    scorer = load_scorer()
    # normalize_matrix mutates nested rows, so isolate once per route.  This is
    # algebraically identical to scorer.score for every candidate.
    normalized = scorer.normalize_matrix(copy.deepcopy(costs))
    actual = reference + reference[:1]
    rows, values = [], []
    for index, sequence in enumerate(sequences):
        proposed = sequence + sequence[:1]
        value = float(scorer.seq_dev(actual, proposed) * scorer.erp_per_edit(actual, proposed, normalized))
        rows.append({
            "route_id": rid, "fold": fold, "candidate": index,
            "official_score": value, "sequence": sequence,
            **evaluate(sequence, route, packages, costs),
        })
        values.append(value)
    base, best = values[0], min(values)
    summary = {
        "route_id": rid, "fold": fold, "candidates": len(values),
        "anchor_score": base, "oracle_score": best, "delta": best - base,
        "improving_candidates": sum(value < base - 1e-14 for value in values[1:]),
    }
    return {"route_id": rid, "rows": rows, "summary": summary}


def _checkpoint_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rid = json.loads(line)["route_id"]
                if rid in ids:
                    raise RuntimeError(f"duplicate scoring checkpoint: {rid}")
                ids.add(rid)
    return ids


def score(out: Path, workers: int) -> None:
    protocol = json.loads((out / "protocol.json").read_text())
    records = {}
    with (out / "candidates.jsonl").open() as handle:
        for line in handle:
            item = json.loads(line)
            records[item["route_id"]] = item
    selected = set(protocol["selected"])
    if set(records) != selected:
        raise RuntimeError("freeze all candidates before loading labels")

    checkpoint = out / "scored_routes.jsonl"
    completed = _checkpoint_ids(checkpoint)
    labels = json.loads((REFERENCE / "fit_sequences.json").read_text())
    tasks = []
    for fold in range(FOLDS):
        folder = out / f"fold_{fold}"
        routes = json.loads((folder / "target_routes.json").read_text())
        packages = json.loads((folder / "target_packages.json").read_text())
        for rid, costs in objects(folder / "target_times.json"):
            if rid not in completed:
                reference = sorted(labels[rid]["actual"], key=labels[rid]["actual"].get)
                tasks.append((rid, fold, reference, records[rid]["sequences"], costs, routes[rid], packages[rid]))
    print(f"Full residual scoring: reuse {len(completed)}, pending {len(tasks)}", flush=True)
    with checkpoint.open("a") as handle, ProcessPoolExecutor(max_workers=workers) as pool:
        for item in pool.map(_score_route, tasks, chunksize=1):
            handle.write(json.dumps(item) + "\n")
            handle.flush()
            completed.add(item["route_id"])
            if len(completed) % 50 == 0 or len(completed) == len(selected):
                print(f"Full residual scored {len(completed)}/{len(selected)}", flush=True)
    if completed != selected:
        raise RuntimeError("incomplete scoring checkpoint")

    rows, route_summaries = [], []
    with checkpoint.open() as handle:
        for line in handle:
            item = json.loads(line)
            rows.extend(item["rows"])
            route_summaries.append(item["summary"])
    route_summaries.sort(key=lambda row: row["route_id"])
    rows.sort(key=lambda row: (row["route_id"], row["candidate"]))
    deltas = np.array([row["delta"] for row in route_summaries])
    anchor_scores = np.array([row["anchor_score"] for row in route_summaries])
    oracle_scores = np.array([row["oracle_score"] for row in route_summaries])
    nonanchors = sum(row["candidates"] - 1 for row in route_summaries)
    positives = sum(row["improving_candidates"] for row in route_summaries)
    summary = {
        "routes": len(route_summaries),
        "candidate_records": len(rows),
        "mean_candidates_per_route": float(np.mean([row["candidates"] for row in route_summaries])),
        "anchor_mean_score": float(np.mean(anchor_scores)),
        "oracle_mean_score": float(np.mean(oracle_scores)),
        "oracle_relative_improvement": float(1 - np.mean(oracle_scores) / np.mean(anchor_scores)),
        "oracle_mean_delta": float(np.mean(deltas)),
        "oracle_delta_ci95": bootstrap_ci(deltas),
        "routes_with_improvement": int(np.sum(deltas < 0)),
        "positive_candidate_records": positives,
        "positive_candidate_rate": positives / max(1, nonanchors),
        "source_hashes_match": all(
            fingerprint(Path(path)) == value for path, value in protocol["source_hashes"].items()
        ),
        "decision": "TRAIN NESTED OOF RANKER",
        "caveat": "Label-selected oracle on fit-only OOF data; not deployable evidence",
    }
    atomic_dump(out / "results.json", rows)
    atomic_dump(out / "routes.json", route_summaries)
    atomic_dump(out / "summary.json", summary)
    (out / "report.md").write_text(
        "# Full fit-only OOF JPT-residual corpus\n\n"
        f"**{summary['decision']}**\n\n"
        f"- Routes: {summary['routes']}\n"
        f"- Candidate records: {summary['candidate_records']}\n"
        f"- JPT OOF anchor score: {summary['anchor_mean_score']:.8f}\n"
        f"- Oracle residual score: {summary['oracle_mean_score']:.8f}\n"
        f"- Oracle relative improvement: {100 * summary['oracle_relative_improvement']:.3f}%\n"
        f"- Oracle delta CI95: [{summary['oracle_delta_ci95'][0]:.8f}, {summary['oracle_delta_ci95'][1]:.8f}]\n"
        f"- Routes with an improving candidate: {summary['routes_with_improvement']}\n"
        f"- Positive non-anchor candidate rate: {100 * summary['positive_candidate_rate']:.2f}%\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/residual_oof_full_v1"))
    parser.add_argument(
        "--stage", choices=["init", "build", "solve", "anchors", "generate", "score", "all"], default="all",
    )
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--seconds", type=float, default=1.0)
    args = parser.parse_args()
    out = args.output.resolve()
    stages = [
        ("init", initialize), ("build", build),
        ("solve", lambda path: solve(path, args.workers, args.seconds)),
        ("anchors", anchors), ("generate", generate),
        ("score", lambda path: score(path, args.workers)),
    ]
    for name, function in stages:
        if args.stage in (name, "all"):
            print(f"Stage: {name}", flush=True)
            function(out)


if __name__ == "__main__":
    main()
