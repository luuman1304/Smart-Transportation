"""Disjoint validation pilot using only one-hop residual candidates."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from .benchmark import objects
from .build_residual_oof_pilot import (
    FOLDS, JPT, POOL_SIZE, RAW, REFERENCE, SAMPLE_PER_FOLD, TRAVEL_BUDGET,
    anchors, build, digest, route_with_filled_zones, score, solve,
)
from .research_utils import atomic_dump
from .reproduce_jpt import ROOT, dump, fingerprint
from .research_utils import stratified_hash_folds
from .baselines import travel_time, validate_sequence
from .zone_repair import moves


PILOT_V1 = ROOT / "outputs/residual_oof_pilot_v1/protocol.json"


def residual_candidates_v2(anchor, route, costs, limit=POOL_SIZE, budget=TRAVEL_BUDGET):
    route = route_with_filled_zones(route)
    validate_sequence(anchor, route["stops"], anchor[0])
    ceiling = travel_time(anchor, costs) * (1 + budget) + 1e-8
    candidates = {
        tuple(sequence) for sequence in moves(anchor, route, True)
        if sequence[0] == anchor[0] and travel_time(sequence, costs) <= ceiling
    }
    ordered = sorted(candidates, key=lambda sequence: digest("residual-one-v2:" + "|".join(sequence)))
    result = [list(sequence) for sequence in [tuple(anchor), *ordered[: limit - 1]]]
    for sequence in result:
        validate_sequence(sequence, route["stops"], anchor[0])
    return result


def initialize(out: Path):
    if out.exists():
        raise FileExistsError(out)
    prior = json.loads(PILOT_V1.read_text())
    excluded = set(prior["selected"])
    routes = json.loads((REFERENCE / "fit_routes.json").read_text())
    actual = json.loads((REFERENCE / "fit_sequences.json").read_text())
    stations = {rid: route["station_code"] for rid, route in routes.items()}
    folds = stratified_hash_folds(routes, stations, FOLDS, "residual-oof-v1:")
    selected = []
    for fold in range(FOLDS):
        eligible = [
            rid for rid in routes
            if folds[rid] == fold and routes[rid]["route_score"] == "High" and rid not in excluded
        ]
        chosen = sorted(eligible, key=lambda rid: digest("residual-pilot-v2:" + rid))[:SAMPLE_PER_FOLD]
        if len(chosen) != SAMPLE_PER_FOLD:
            raise RuntimeError(f"insufficient disjoint High routes in fold {fold}")
        selected.extend(chosen)
    out.mkdir(parents=True)
    protocol = {
        "version": 2,
        "scope": "disjoint fit-only OOF validation pilot; no outer development/confirmation/evaluation",
        "folds": folds,
        "selected": selected,
        "excluded_pilot_v1": sorted(excluded),
        "sample_per_fold": SAMPLE_PER_FOLD,
        "selection_population": "High-quality fit routes; quality is never an inference feature",
        "jpt_history_excludes_entire_target_fold": True,
        "pool_size": POOL_SIZE,
        "travel_budget": TRAVEL_BUDGET,
        "candidate_sampling": "31 deterministic one-hop residuals; two-hop removed before selecting this disjoint sample",
        "labels_loaded_after_candidate_generation": True,
        "source_hashes": {
            str(path): fingerprint(path)
            for path in [
                Path(__file__), PILOT_V1, REFERENCE / "fit_routes.json", REFERENCE / "fit_sequences.json",
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
        dump(folder / "train_routes.json", {rid: routes[rid] for rid in train})
        dump(folder / "train_sequences.json", {rid: actual[rid] for rid in train})
        dump(folder / "target_routes.json", {
            rid: {key: value for key, value in routes[rid].items() if key != "route_score"} for rid in targets
        })
        dump(folder / "target_packages.json", {rid: packages[rid] for rid in targets})
        dump(folder / "target_times.json", {rid: times[rid] for rid in targets})
        dump(folder / "fold.json", {"fold": fold, "train": train, "targets": targets})
    print(f"Prepared {len(selected)} disjoint v2 targets", flush=True)


def generate(out: Path):
    protocol = json.loads((out / "protocol.json").read_text())
    anchor_data = json.loads((out / "anchors.json").read_text())
    path = out / "candidates.jsonl"
    if path.exists():
        raise FileExistsError(path)
    with path.open("x") as handle:
        completed = 0
        for fold in range(FOLDS):
            folder = out / f"fold_{fold}"
            routes = json.loads((folder / "target_routes.json").read_text())
            for rid, costs in objects(folder / "target_times.json"):
                anchor = anchor_data[rid]["sequence"]
                sequences = residual_candidates_v2(anchor, routes[rid], costs)
                handle.write(json.dumps({
                    "route_id": rid, "fold": fold, "anchor_variant": anchor_data[rid]["variant"],
                    "sequences": sequences,
                }) + "\n")
                handle.flush()
                completed += 1
                if completed % 25 == 0:
                    print(f"Residual v2 candidates {completed}/{len(protocol['selected'])}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/residual_oof_pilot_v2"))
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
