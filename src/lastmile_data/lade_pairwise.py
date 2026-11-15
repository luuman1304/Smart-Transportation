"""Post-hoc strong-model sensitivity for the frozen LaDe route reconstruction.

This analysis is explicitly exploratory.  It uses a common temporal boundary,
five service-region folds, matched training size, five fixed seeds, and a
pairwise histogram-gradient-boosting ranker.  The model and hyperparameters are
fixed in this file before its first outcome run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .lade_external import (
    ROOT,
    ROUTES,
    _cluster_bootstrap,
    _metric_row,
    _round_robin_sample,
    _sign_flip,
    _sign_test,
    atomic_json,
    predict as transition_predict,
)


OUT = ROOT / "outputs/lade_pairwise_v1"
TRAIN_BEFORE = "10-01"
TEST_FROM = "10-15"
TARGETS_PER_GROUP = 10
MIN_TRAIN_PER_GROUP = 20
TRAIN_ROUTES = 4_000
FOLDS = 5
SEEDS = [17, 42, 91, 123, 2026]
NEGATIVES = 3
FEATURE_NAMES = [
    "log_current_distance", "distance_rank", "log_candidate_to_remaining_centroid",
    "log_candidate_radius", "log_current_radius", "direction_x", "direction_y",
    "turn_cos", "turn_sin", "progress", "remaining_fraction", "same_type",
    "current_type", "candidate_type", "global_type_log_probability",
    "regional_edge_log_count", "regional_destination_log_count", "radial_change",
]

PROTOCOL = {
    "status": "post-hoc strong-model sensitivity; not confirmatory",
    "route_artifact": str(ROUTES),
    "train_date_rule": f"date < {TRAIN_BEFORE}",
    "test_date_rule": f"date >= {TEST_FROM}",
    "targets_per_group": TARGETS_PER_GROUP,
    "minimum_train_routes_per_group": MIN_TRAIN_PER_GROUP,
    "training_routes_per_model": TRAIN_ROUTES,
    "group_folds": FOLDS,
    "seeds": SEEDS,
    "negative_candidates_per_positive": NEGATIVES,
    "model": {
        "class": "sklearn.ensemble.HistGradientBoostingClassifier",
        "learning_rate": 0.08,
        "max_iter": 80,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 40,
        "l2_regularization": 0.1,
        "categorical_features": ["current_type", "candidate_type"],
    },
    "primary_metric": "macro-region normalized Kendall loss",
    "promotion_gate": "ranker must beat nearest-neighbor and unseen-minus-seen CI must be reported regardless of sign",
}


def stable_fold(group: str) -> int:
    return int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % FOLDS


def eligible_split(routes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = routes[routes["date"] < TRAIN_BEFORE]
    test = routes[routes["date"] >= TEST_FROM]
    train_counts, test_counts = train.groupby("group").size(), test.groupby("group").size()
    groups = sorted(set(train_counts[train_counts >= MIN_TRAIN_PER_GROUP].index) &
                    set(test_counts[test_counts >= TARGETS_PER_GROUP].index))
    train = train[train["group"].isin(groups)].copy()
    targets = (test[test["group"].isin(groups)]
               .sort_values(["group", "date", "route_id"], kind="mergesort")
               .groupby("group", sort=True).tail(TARGETS_PER_GROUP).copy())
    return train, targets


@dataclass
class History:
    type_edges: Counter
    type_totals: Counter
    region_edges: Counter
    region_destinations: Counter
    type_vocab: int

    @classmethod
    def fit(cls, routes: pd.DataFrame) -> "History":
        type_edges, type_totals, region_edges, destinations, types = Counter(), Counter(), Counter(), Counter(), set()
        for row in routes.itertuples(index=False):
            ids, kinds = [int(x) for x in row.aoi_ids], [int(x) for x in row.aoi_types]
            types.update(kinds)
            for left, right, left_type, right_type in zip(ids, ids[1:], kinds, kinds[1:]):
                type_edges[(left_type, right_type)] += 1
                type_totals[left_type] += 1
                region_edges[(row.group, left, right)] += 1
                destinations[(row.group, right)] += 1
        return cls(type_edges, type_totals, region_edges, destinations, max(1, len(types)))


def _xy(lng: float, lat: float, reference_lat: float) -> tuple[float, float]:
    return lng * 111.320 * math.cos(math.radians(reference_lat)), lat * 110.574


@dataclass
class Geometry:
    ids: list[int]
    kinds: list[int]
    index: dict[int, int]
    points: dict[int, tuple[float, float]]
    cx: float
    cy: float

    @classmethod
    def from_row(cls, row) -> "Geometry":
        ids = [int(x) for x in row.aoi_ids]
        kinds = [int(x) for x in row.aoi_types]
        index = {identifier: i for i, identifier in enumerate(ids)}
        reference_lat = statistics.fmean(float(x) for x in row.lats)
        points = {identifier: _xy(float(row.lngs[i]), float(row.lats[i]), reference_lat)
                  for i, identifier in enumerate(ids)}
        return cls(ids, kinds, index, points,
                   statistics.fmean(points[value][0] for value in ids),
                   statistics.fmean(points[value][1] for value in ids))


@dataclass
class State:
    current: int
    remaining: list[int]
    previous: int | None
    rx: float
    ry: float
    x0: float
    y0: float
    current_radius: float
    distance_rank: dict[int, float]
    previous_vector: tuple[float, float, float] | None

    @classmethod
    def build(cls, geometry: Geometry, current: int, remaining: list[int], previous: int | None) -> "State":
        rx = statistics.fmean(geometry.points[value][0] for value in remaining)
        ry = statistics.fmean(geometry.points[value][1] for value in remaining)
        x0, y0 = geometry.points[current]
        ordered = sorted(remaining, key=lambda value: (math.hypot(
            geometry.points[value][0] - x0, geometry.points[value][1] - y0), value))
        denominator = max(1, len(ordered) - 1)
        ranks = {value: rank / denominator for rank, value in enumerate(ordered)}
        if previous is None:
            previous_vector = None
        else:
            xp, yp = geometry.points[previous]
            pdx, pdy = x0 - xp, y0 - yp
            previous_vector = (pdx, pdy, max(1e-9, math.hypot(pdx, pdy)))
        return cls(current, remaining, previous, rx, ry, x0, y0,
                   math.hypot(x0 - geometry.cx, y0 - geometry.cy), ranks, previous_vector)


def _features(row, geometry: Geometry, state: State, candidate: int, history: History) -> list[float]:
    x0, y0 = state.x0, state.y0
    x1, y1 = geometry.points[candidate]
    dx, dy = x1 - x0, y1 - y0
    distance = max(1e-9, math.hypot(dx, dy))
    candidate_radius = math.hypot(x1 - geometry.cx, y1 - geometry.cy)
    if state.previous_vector is None:
        turn_cos, turn_sin = 0.0, 0.0
    else:
        pdx, pdy, previous_length = state.previous_vector
        turn_cos = (pdx * dx + pdy * dy) / (previous_length * distance)
        turn_sin = (pdx * dy - pdy * dx) / (previous_length * distance)
    current_type = geometry.kinds[geometry.index[state.current]]
    candidate_type = geometry.kinds[geometry.index[candidate]]
    type_probability = (history.type_edges[(current_type, candidate_type)] + 1) / (
        history.type_totals[current_type] + history.type_vocab
    )
    return [
        math.log1p(distance), state.distance_rank[candidate], math.log1p(math.hypot(x1 - state.rx, y1 - state.ry)),
        math.log1p(candidate_radius), math.log1p(state.current_radius), dx / distance, dy / distance,
        turn_cos, turn_sin, (len(geometry.ids) - len(state.remaining)) / len(geometry.ids), len(state.remaining) / len(geometry.ids),
        float(current_type == candidate_type), float(current_type), float(candidate_type),
        math.log(type_probability), math.log1p(history.region_edges[(row.group, state.current, candidate)]),
        math.log1p(history.region_destinations[(row.group, candidate)]), candidate_radius - state.current_radius,
    ]


def training_matrix(routes: pd.DataFrame, history: History, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng, features, labels = random.Random(seed), [], []
    for row in routes.itertuples(index=False):
        geometry = Geometry.from_row(row)
        ids = geometry.ids
        for position in range(len(ids) - 1):
            current, positive = ids[position], ids[position + 1]
            remaining = ids[position + 1:]
            previous = ids[position - 1] if position else None
            state = State.build(geometry, current, remaining, previous)
            features.append(_features(row, geometry, state, positive, history)); labels.append(1)
            negatives = [value for value in remaining if value != positive]
            for negative in rng.sample(negatives, min(NEGATIVES, len(negatives))):
                features.append(_features(row, geometry, state, negative, history)); labels.append(0)
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int8)


def fit_ranker(routes: pd.DataFrame, seed: int) -> tuple[HistGradientBoostingClassifier, History, dict]:
    history = History.fit(routes)
    x, y = training_matrix(routes, history, seed)
    model = HistGradientBoostingClassifier(
        learning_rate=0.08, max_iter=80, max_leaf_nodes=31, min_samples_leaf=40,
        l2_regularization=0.1, categorical_features=[12, 13], random_state=seed,
    )
    model.fit(x, y)
    return model, history, {"examples": len(y), "positive_fraction": float(y.mean())}


def predict(row, model: HistGradientBoostingClassifier, history: History) -> list[int]:
    geometry = Geometry.from_row(row)
    ids = geometry.ids
    order, remaining = [ids[0]], ids[1:]
    while remaining:
        previous = order[-2] if len(order) > 1 else None
        state = State.build(geometry, order[-1], remaining, previous)
        x = np.asarray([_features(row, geometry, state, candidate, history)
                        for candidate in remaining], dtype=np.float32)
        probabilities = model.predict_proba(x)[:, 1]
        chosen_index = max(range(len(remaining)), key=lambda i: (probabilities[i], -remaining[i]))
        order.append(remaining.pop(chosen_index))
    return order


def evaluate(out: Path) -> None:
    routes = pd.read_parquet(ROUTES)
    train_pool, targets = eligible_split(routes)
    out.mkdir(parents=True, exist_ok=True)
    protocol = dict(PROTOCOL)
    protocol.update({"eligible_groups": int(targets["group"].nunique()), "targets": len(targets)})
    atomic_json(out / "protocol.json", protocol)
    rows, fits = [], []
    nearest = {}
    for target in targets.itertuples(index=False):
        nearest[target.route_id] = transition_predict(target, None)
    for seed in SEEDS:
        seen_train = _round_robin_sample(train_pool, TRAIN_ROUTES, seed)
        print(f"seed={seed}: fit seen", flush=True)
        seen_model, seen_history, fit_info = fit_ranker(seen_train, seed)
        fits.append({"seed": seed, "regime": "seen", "fold": -1, "groups": seen_train.group.nunique(), **fit_info})
        for target in targets.itertuples(index=False):
            _append(rows, target, seed, "seen", "pairwise_hgb", predict(target, seen_model, seen_history))
        for fold in range(FOLDS):
            fold_targets = targets[targets["group"].map(stable_fold) == fold]
            unseen_pool = train_pool[train_pool["group"].map(stable_fold) != fold]
            unseen_train = _round_robin_sample(unseen_pool, TRAIN_ROUTES, seed)
            print(f"seed={seed}: fit unseen fold={fold}, groups={len(set(fold_targets.group))}", flush=True)
            model, history, fit_info = fit_ranker(unseen_train, seed ^ (fold + 1) * 1009)
            fits.append({"seed": seed, "regime": "unseen", "fold": fold, "groups": unseen_train.group.nunique(), **fit_info})
            for target in fold_targets.itertuples(index=False):
                _append(rows, target, seed, "unseen", "pairwise_hgb", predict(target, model, history))
        for target in targets.itertuples(index=False):
            _append(rows, target, seed, "none", "nearest_neighbor", nearest[target.route_id])
    pd.DataFrame(rows).to_parquet(out / "route_metrics.parquet", index=False)
    atomic_json(out / "fits.json", fits)
    print(json.dumps({"groups": targets.group.nunique(), "targets": len(targets), "metric_rows": len(rows)}, indent=2))


def _append(rows: list, target, seed: int, regime: str, method: str, prediction: list[int]) -> None:
    value = {"route_id": target.route_id, "group": target.group, "city": target.city,
             "seed": seed, "regime": regime, "method": method, "n_stops": int(target.n_stops)}
    value.update(_metric_row(target, prediction)); rows.append(value)


def report(out: Path) -> None:
    data = pd.read_parquet(out / "route_metrics.parquet")
    ranker = data[data.method == "pairwise_hgb"]
    route = ranker.groupby(["group", "city", "route_id", "regime"], as_index=False).normalized_kendall_loss.mean()
    region = route.groupby(["group", "city", "regime"], as_index=False).normalized_kendall_loss.mean()
    pivot = region.pivot(index=["group", "city"], columns="regime", values="normalized_kendall_loss").reset_index()
    pivot["delta"] = pivot.unseen - pivot.seen
    values = pivot.delta.tolist()
    nn = (data[data.method == "nearest_neighbor"].groupby(["group", "route_id"], as_index=False)
          .normalized_kendall_loss.mean().groupby("group").normalized_kendall_loss.mean().mean())
    summary = {
        "groups": len(pivot), "cities": int(pivot.city.nunique()),
        "seen": float(pivot.seen.mean()), "unseen": float(pivot.unseen.mean()),
        "relative_degradation": float(pivot.unseen.mean() / pivot.seen.mean() - 1),
        "absolute_delta": float(pivot.delta.mean()), "cluster_ci95": _cluster_bootstrap(values),
        "groups_worse_tied_better": [sum(x > 0 for x in values), sum(x == 0 for x in values), sum(x < 0 for x in values)],
        "sign_test": _sign_test(values), "sign_flip": _sign_flip(values),
        "nearest_neighbor": float(nn),
        "ranker_beats_nearest_neighbor_seen": bool(pivot.seen.mean() < nn),
        "ranker_beats_nearest_neighbor_unseen": bool(pivot.unseen.mean() < nn),
    }
    seed_rows = []
    for seed, members in ranker.groupby("seed"):
        per_route = members.groupby(["group", "route_id", "regime"], as_index=False).normalized_kendall_loss.mean()
        per_region = per_route.groupby(["group", "regime"], as_index=False).normalized_kendall_loss.mean()
        p = per_region.pivot(index="group", columns="regime", values="normalized_kendall_loss")
        seed_rows.append({"seed": int(seed), "seen": float(p.seen.mean()), "unseen": float(p.unseen.mean()),
                          "delta": float((p.unseen - p.seen).mean())})
    split_audit = audit_splits(out)
    audit = {
        "five_seeds": data[data.method == "pairwise_hgb"].groupby(["group", "regime"]).seed.nunique().eq(5).all(),
        "same_target_count": data[data.method == "pairwise_hgb"].groupby(["regime", "seed"]).route_id.nunique().nunique() == 1,
        "finite": np.isfinite(data[["normalized_kendall_loss", "normalized_edit_distance", "normalized_footrule", "route_length_ratio"]]).all().all(),
        "bounded_rank_metrics": data[["normalized_kendall_loss", "normalized_edit_distance", "normalized_footrule"]].apply(lambda x: x.between(0, 1)).all().all(),
        **split_audit["checks"],
    }
    audit = {key: bool(value) for key, value in audit.items()}
    result = {"summary": summary, "seed_summary": seed_rows, "audit": {"passed": all(audit.values()), "checks": audit}}
    atomic_json(out / "analysis.json", result)
    pivot.to_csv(out / "region_effects.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(out / "seed_summary.csv", index=False)
    if not all(audit.values()):
        raise RuntimeError("pairwise LaDe audit failed")
    print(json.dumps(result, indent=2))


def audit_splits(out: Path) -> dict:
    routes = pd.read_parquet(ROUTES)
    train_pool, targets = eligible_split(routes)
    target_groups = set(targets.group)
    records, checks = [], []
    for seed in SEEDS:
        seen = _round_robin_sample(train_pool, TRAIN_ROUTES, seed)
        seen_counts = seen.groupby("group").size()
        checks.append({
            "seed": seed, "regime": "seen", "fold": -1, "routes": len(seen),
            "groups": seen.group.nunique(), "count_min": int(seen_counts.min()), "count_max": int(seen_counts.max()),
            "max_date": seen.date.max(), "target_group_overlap": len(set(seen.group) & target_groups),
            "target_route_overlap": len(set(seen.route_id) & set(targets.route_id)),
        })
        for fold in range(FOLDS):
            held_out = {group for group in target_groups if stable_fold(group) == fold}
            pool = train_pool[train_pool.group.map(stable_fold) != fold]
            sample = _round_robin_sample(pool, TRAIN_ROUTES, seed)
            counts = sample.groupby("group").size()
            checks.append({
                "seed": seed, "regime": "unseen", "fold": fold, "routes": len(sample),
                "groups": sample.group.nunique(), "count_min": int(counts.min()), "count_max": int(counts.max()),
                "max_date": sample.date.max(), "target_group_overlap": len(set(sample.group) & held_out),
                "target_route_overlap": len(set(sample.route_id) & set(targets.route_id)),
            })
    result = {
        "eligible_groups": targets.group.nunique(), "target_routes": len(targets), "records": checks,
        "checks": {
            "fixed_common_temporal_boundary": all(row["max_date"] < TRAIN_BEFORE for row in checks) and targets.date.min() >= TEST_FROM,
            "matched_4000_route_training": all(row["routes"] == TRAIN_ROUTES for row in checks),
            "balanced_source_region_counts": all(row["count_max"] - row["count_min"] <= 1 for row in checks),
            "no_target_route_overlap": all(row["target_route_overlap"] == 0 for row in checks),
            "unseen_target_groups_absent": all(row["regime"] != "unseen" or row["target_group_overlap"] == 0 for row in checks),
            "seen_contains_all_target_groups": all(row["regime"] != "seen" or row["target_group_overlap"] == len(target_groups) for row in checks),
            "ten_targets_per_group": targets.groupby("group").size().eq(TARGETS_PER_GROUP).all(),
        },
    }
    result["checks"] = {key: bool(value) for key, value in result["checks"].items()}
    atomic_json(out / "split_audit.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--stage", choices=["evaluate", "report", "all"], default="all")
    args = parser.parse_args()
    if args.stage in ("evaluate", "all"):
        evaluate(args.output.resolve())
    if args.stage in ("report", "all"):
        report(args.output.resolve())


if __name__ == "__main__":
    main()
