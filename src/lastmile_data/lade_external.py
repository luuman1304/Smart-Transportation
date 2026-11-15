"""Leakage-resistant external replication on the Cainiao LaDe-D dataset.

The experiment asks whether a route model evaluated with history from the
target service region looks better than the same model deployed in an unseen
region.  It deliberately does not call a LaDe region an Amazon station.

Stages are explicit so the protocol can be frozen before outcomes are built:

    init -> prepare -> evaluate -> audit -> report

The primary estimand is the macro-region paired increase in normalized Kendall
loss for a region-aware transition model under leave-one-region-out training.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/lade_d"
ROUTES = ROOT / "data/processed/lade_d_routes.parquet"
DEFAULT_OUT = ROOT / "outputs/lade_external_v1"
REVISION = "e1f85a3339848b9eb606ce543fcaaba3f4be1893"
EXPECTED_ROWS = 4_514_661
EXPECTED_FILES = {
    "delivery_cq-00000-of-00001-465887add76aeabc.parquet": "ea503be5d475aa9a3ceb27f09ce6f910f48a02e78122428fbb4f9c724820323b",
    "delivery_hz-00000-of-00001-8090c86f64781f71.parquet": "00b6b4dca471cadee796a3fc7f07dcd919499ec58459ea67db784dcb06089e75",
    "delivery_jl-00000-of-00001-a4fbefe3c368583c.parquet": "ed0296b83d84f739bb91ab3d3cf1a430cf71e3793e25ddf50f88aa3ef34b51fc",
    "delivery_sh-00000-of-00001-ad9a4b1d79823540.parquet": "a0561b186b924ffb6ac2612acbd8fd94ab75a752b3bd70276e5fa7b335a778f7",
    "delivery_yt-00000-of-00001-cc85c1fcb1d10955.parquet": "2c5e5566ca8e4976e215822d8d5f3d5db157aba0dfe85a952b0cb357dee0ec5d",
}

PROTOCOL = {
    "version": 1,
    "dataset": "Cainiao-AI/LaDe-D",
    "dataset_revision": REVISION,
    "estimand": "macro-region paired unseen-minus-seen normalized Kendall loss",
    "unit": "courier-delivery-day, conditioned on the observed first AOI",
    "group": "city:region_id (service region; not claimed to be a depot/station)",
    "route_construction": {
        "date_source": "MM-DD prefix of delivery_time; ds is not used as route date",
        "stop": "first completion event per AOI after chronological sorting",
        "region_assignment": "modal region over collapsed AOI stops",
        "minimum_region_purity": 0.90,
        "minimum_stops": 5,
        "maximum_stops": 25,
    },
    "folds": {
        "targets_per_region": 10,
        "minimum_prior_routes": 20,
        "training_routes": 1500,
        "strict_temporal_cutoff": True,
        "balanced_source_region_sampling": True,
        "matched_training_size": True,
        "seeds": [17, 42, 91, 123, 2026],
    },
    "models": {
        "nearest_neighbor": "non-learning negative control",
        "global_transition": "geometry plus globally pooled AOI-type transition prior",
        "regional_transition": "global_transition plus shrunk within-region AOI-edge and destination history",
        "fixed_weights": {"type": 0.25, "edge": 0.35, "destination": 0.10},
    },
    "primary_metric": "normalized_kendall_loss",
    "secondary_metrics": ["normalized_edit_distance", "normalized_footrule", "route_length_ratio"],
    "inference": "region-cluster bootstrap CI and two-sided sign/sign-flip tests",
    "imbalance_controls": [
        "equal target count per eligible region",
        "round-robin source-region training sampler",
        "macro-region and macro-city reporting",
        "route-weighted results reported only as secondary",
    ],
    "overfitting_controls": [
        "all training dates precede the earliest target date in each fold",
        "identical target routes for seen and unseen regimes",
        "identical training-route count for seen and unseen regimes",
        "target region is absent from unseen training",
        "five pre-specified sampling seeds",
        "fixed model weights and frozen protocol before evaluation",
        "global-only and non-learning negative controls",
    ],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def init(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    total = 0
    for name, expected_hash in EXPECTED_FILES.items():
        path = RAW / name
        if not path.exists():
            raise FileNotFoundError(f"missing official LaDe-D file: {path}")
        observed_hash = sha256(path)
        if observed_hash != expected_hash:
            raise RuntimeError(f"hash mismatch for {name}: {observed_hash}")
        rows = len(pd.read_parquet(path, columns=["order_id"]))
        total += rows
        manifest.append({"file": name, "sha256": observed_hash, "rows": rows, "bytes": path.stat().st_size})
    if total != EXPECTED_ROWS:
        raise RuntimeError(f"expected {EXPECTED_ROWS:,} package rows, found {total:,}")
    frozen = dict(PROTOCOL)
    frozen["raw_manifest"] = manifest
    frozen["raw_total_rows"] = total
    frozen["protocol_sha256"] = hashlib.sha256(
        json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    atomic_json(out / "protocol.json", frozen)
    print(json.dumps({"protocol": str(out / "protocol.json"), "raw_rows": total}, indent=2))


def _valid_time(series: pd.Series) -> pd.Series:
    return series.astype("string").str.match(r"^\d{2}-\d{2} \d{2}:\d{2}:\d{2}$", na=False)


def _prepare_city(path: Path) -> tuple[pd.DataFrame, dict]:
    columns = ["city", "region_id", "courier_id", "aoi_id", "aoi_type", "lng", "lat", "delivery_time", "ds"]
    frame = pd.read_parquet(path, columns=columns)
    raw_rows = len(frame)
    valid = (
        _valid_time(frame["delivery_time"])
        & frame[["region_id", "courier_id", "aoi_id", "lng", "lat"]].notna().all(axis=1)
        & frame["lat"].between(-90, 90)
        & frame["lng"].between(-180, 180)
    )
    frame = frame.loc[valid].copy()
    frame["date"] = frame["delivery_time"].str.slice(0, 5)
    frame["clock"] = frame["delivery_time"].str.slice(6)
    keys = ["city", "courier_id", "date"]
    frame.sort_values(keys + ["clock", "aoi_id"], inplace=True, kind="mergesort")
    # Multiple packages at one AOI are one physical stop.  The first completion
    # event fixes that stop's position, matching route-prediction semantics.
    stops = frame.drop_duplicates(keys + ["aoi_id"], keep="first")
    region_counts = stops.groupby(keys + ["region_id"], sort=False).size().rename("region_stops").reset_index()
    region_counts.sort_values(keys + ["region_stops", "region_id"], ascending=[True, True, True, False, True], inplace=True)
    modal = region_counts.drop_duplicates(keys, keep="first").rename(columns={"region_id": "modal_region"})
    grouped = stops.groupby(keys, sort=False).agg(
        aoi_ids=("aoi_id", list),
        aoi_types=("aoi_type", list),
        lngs=("lng", list),
        lats=("lat", list),
        n_stops=("aoi_id", "size"),
    ).reset_index()
    grouped = grouped.merge(modal[keys + ["modal_region", "region_stops"]], on=keys, how="inner", validate="one_to_one")
    grouped["region_purity"] = grouped["region_stops"] / grouped["n_stops"]
    grouped["group"] = grouped["city"].astype(str) + ":" + grouped["modal_region"].astype(str)
    grouped["route_id"] = grouped["city"].astype(str) + ":" + grouped["courier_id"].astype(str) + ":" + grouped["date"]
    eligible = grouped[
        grouped["n_stops"].between(PROTOCOL["route_construction"]["minimum_stops"], PROTOCOL["route_construction"]["maximum_stops"])
        & grouped["region_purity"].ge(PROTOCOL["route_construction"]["minimum_region_purity"])
    ].copy()
    audit = {
        "file": path.name,
        "city": str(frame["city"].iloc[0]) if len(frame) else None,
        "raw_package_rows": raw_rows,
        "valid_package_rows": len(frame),
        "invalid_package_rows": raw_rows - len(frame),
        "courier_days_before_filter": len(grouped),
        "eligible_routes": len(eligible),
        "regions": int(eligible["group"].nunique()),
        "median_stops": float(eligible["n_stops"].median()) if len(eligible) else None,
    }
    return eligible, audit


def prepare(out: Path) -> None:
    if not (out / "protocol.json").exists():
        raise RuntimeError("run --stage init before prepare")
    frames, audits = [], []
    for name in EXPECTED_FILES:
        print(f"Preparing {name}", flush=True)
        frame, audit = _prepare_city(RAW / name)
        frames.append(frame)
        audits.append(audit)
    routes = pd.concat(frames, ignore_index=True)
    if routes["route_id"].duplicated().any():
        duplicates = routes.loc[routes["route_id"].duplicated(), "route_id"].head().tolist()
        raise RuntimeError(f"route IDs are not unique: {duplicates}")
    ROUTES.parent.mkdir(parents=True, exist_ok=True)
    routes.to_parquet(ROUTES, index=False)
    summary = {
        "route_file": str(ROUTES),
        "route_file_sha256": sha256(ROUTES),
        "eligible_routes": len(routes),
        "cities": int(routes["city"].nunique()),
        "regions": int(routes["group"].nunique()),
        "city_audits": audits,
        "purity_quantiles": {str(q): float(routes["region_purity"].quantile(q)) for q in (0, .25, .5, .75, 1)},
        "stop_quantiles": {str(q): float(routes["n_stops"].quantile(q)) for q in (0, .25, .5, .75, 1)},
    }
    atomic_json(out / "preparation_audit.json", summary)
    print(json.dumps(summary, indent=2))


def _round_robin_sample(frame: pd.DataFrame, total: int, seed: int) -> pd.DataFrame:
    """Sample deterministically while keeping source-region counts within one."""
    if total > len(frame):
        raise ValueError("sample exceeds pool")
    # A naïve round robin leaves a sparse source group at one route while dense
    # groups continue to ten or more. Remove groups that cannot meet the common
    # floor quota, recomputing that quota until the retained set is feasible.
    counts = frame.groupby("group").size()
    retained = set(counts.index)
    while retained:
        floor_quota = total // len(retained)
        next_retained = {group for group in retained if counts[group] >= floor_quota}
        if next_retained == retained:
            break
        retained = next_retained
    if not retained or sum(counts[group] for group in retained) < total:
        raise ValueError("insufficient balanced source-region pool")
    frame = frame[frame["group"].isin(retained)]
    buckets = {}
    for group, members in frame.groupby("group", sort=True):
        buckets[group] = list(members.sample(frac=1, random_state=seed ^ _stable_int(group)).index)
    selected = []
    groups = sorted(buckets)
    cursor = {group: 0 for group in groups}
    while len(selected) < total:
        progressed = False
        for group in groups:
            position = cursor[group]
            if position < len(buckets[group]):
                selected.append(buckets[group][position])
                cursor[group] += 1
                progressed = True
                if len(selected) == total:
                    break
        if not progressed:
            raise RuntimeError("balanced sampler exhausted unexpectedly")
    return frame.loc[selected].copy()


def _stable_int(value: str) -> int:
    return int(hashlib.sha256(value.encode()).hexdigest()[:8], 16)


def _haversine(lng1: float, lat1: float, lng2: float, lat2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371.0088 * 2 * math.asin(min(1.0, math.sqrt(a)))


@dataclass
class TransitionModel:
    type_edges: Counter
    type_totals: Counter
    type_vocab: int
    region_edges: Counter
    region_destinations: Counter
    use_regional_history: bool

    @classmethod
    def fit(cls, routes: pd.DataFrame, use_regional_history: bool) -> "TransitionModel":
        type_edges, type_totals = Counter(), Counter()
        region_edges, region_destinations = Counter(), Counter()
        types = set()
        for row in routes.itertuples(index=False):
            ids, kinds, group = list(row.aoi_ids), list(row.aoi_types), row.group
            types.update(kinds)
            for index in range(len(ids) - 1):
                type_edges[(int(kinds[index]), int(kinds[index + 1]))] += 1
                type_totals[int(kinds[index])] += 1
                region_edges[(group, int(ids[index]), int(ids[index + 1]))] += 1
                region_destinations[(group, int(ids[index + 1]))] += 1
        return cls(type_edges, type_totals, max(1, len(types)), region_edges, region_destinations, use_regional_history)

    def score(self, group: str, current: int, candidate: int, current_type: int,
              candidate_type: int, distance_km: float) -> float:
        type_probability = (
            self.type_edges[(current_type, candidate_type)] + 1
        ) / (self.type_totals[current_type] + self.type_vocab)
        value = -math.log1p(distance_km) + 0.25 * math.log(type_probability)
        if self.use_regional_history:
            # log1p and small fixed coefficients shrink sparse exact-AOI history
            # rather than allowing a single observed transition to dominate geometry.
            value += 0.35 * math.log1p(self.region_edges[(group, current, candidate)])
            value += 0.10 * math.log1p(self.region_destinations[(group, candidate)])
        return value


def predict(row, model: TransitionModel | None) -> list[int]:
    ids, kinds = [int(x) for x in row.aoi_ids], [int(x) for x in row.aoi_types]
    lngs, lats = [float(x) for x in row.lngs], [float(x) for x in row.lats]
    attributes = {identifier: (kinds[i], lngs[i], lats[i]) for i, identifier in enumerate(ids)}
    order, remaining = [ids[0]], set(ids[1:])
    while remaining:
        current = order[-1]
        current_type, current_lng, current_lat = attributes[current]
        scored = []
        for candidate in remaining:
            candidate_type, candidate_lng, candidate_lat = attributes[candidate]
            distance = _haversine(current_lng, current_lat, candidate_lng, candidate_lat)
            score = -distance if model is None else model.score(
                row.group, current, candidate, current_type, candidate_type, distance
            )
            scored.append((score, -candidate, candidate))
        chosen = max(scored)[2]
        order.append(chosen)
        remaining.remove(chosen)
    return order


def normalized_kendall_loss(actual: Sequence[int], predicted: Sequence[int]) -> float:
    position = {value: index for index, value in enumerate(actual)}
    ranks = [position[value] for value in predicted]
    inversions = sum(ranks[i] > ranks[j] for i in range(len(ranks)) for j in range(i + 1, len(ranks)))
    return inversions / max(1, math.comb(len(ranks), 2))


def normalized_edit_distance(actual: Sequence[int], predicted: Sequence[int]) -> float:
    previous = list(range(len(predicted) + 1))
    for i, left in enumerate(actual, 1):
        current = [i]
        for j, right in enumerate(predicted, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1] / max(1, len(actual))


def normalized_footrule(actual: Sequence[int], predicted: Sequence[int]) -> float:
    n = len(actual)
    position = {value: index for index, value in enumerate(actual)}
    distance = sum(abs(index - position[value]) for index, value in enumerate(predicted))
    maximum = (n * n) // 2
    return distance / max(1, maximum)


def route_length(order: Sequence[int], row) -> float:
    coordinates = {int(identifier): (float(row.lngs[i]), float(row.lats[i])) for i, identifier in enumerate(row.aoi_ids)}
    return sum(_haversine(*coordinates[left], *coordinates[right]) for left, right in zip(order, order[1:]))


def _metric_row(row, prediction: Sequence[int]) -> dict:
    actual = [int(value) for value in row.aoi_ids]
    actual_length = route_length(actual, row)
    return {
        "normalized_kendall_loss": normalized_kendall_loss(actual, prediction),
        "normalized_edit_distance": normalized_edit_distance(actual, prediction),
        "normalized_footrule": normalized_footrule(actual, prediction),
        "route_length_ratio": route_length(prediction, row) / actual_length if actual_length > 0 else 1.0,
    }


def _eligible_targets(routes: pd.DataFrame) -> dict[str, pd.DataFrame]:
    targets, config = {}, PROTOCOL["folds"]
    for group, members in routes.groupby("group", sort=True):
        members = members.sort_values(["date", "route_id"], kind="mergesort")
        if len(members) < config["minimum_prior_routes"] + config["targets_per_region"]:
            continue
        candidate = members.tail(config["targets_per_region"])
        cutoff = candidate["date"].min()
        if sum(members["date"] < cutoff) < config["minimum_prior_routes"]:
            continue
        targets[group] = candidate
    return targets


def evaluate(out: Path) -> None:
    protocol_path = out / "protocol.json"
    if not protocol_path.exists() or not ROUTES.exists():
        raise RuntimeError("run init and prepare before evaluate")
    routes = pd.read_parquet(ROUTES)
    targets = _eligible_targets(routes)
    if len(targets) < 10:
        raise RuntimeError(f"too few eligible service regions: {len(targets)}")
    rows, folds = [], []
    config = PROTOCOL["folds"]
    for fold_index, (target_group, target_frame) in enumerate(sorted(targets.items()), 1):
        cutoff = target_frame["date"].min()
        historical = routes[routes["date"] < cutoff]
        unseen_pool = historical[historical["group"] != target_group]
        seen_pool = historical
        training_size = min(config["training_routes"], len(unseen_pool), len(seen_pool))
        if training_size < 100:
            continue
        print(f"Fold {fold_index}/{len(targets)} {target_group}: cutoff={cutoff}, train={training_size}", flush=True)
        for seed in config["seeds"]:
            training = {
                "seen": _round_robin_sample(seen_pool, training_size, seed),
                "unseen": _round_robin_sample(unseen_pool, training_size, seed),
            }
            fold_record = {
                "target_group": target_group,
                "city": target_frame["city"].iloc[0],
                "cutoff": cutoff,
                "seed": seed,
                "targets": target_frame["route_id"].tolist(),
                "train_size_seen": len(training["seen"]),
                "train_size_unseen": len(training["unseen"]),
                "target_in_seen_groups": target_group in set(training["seen"]["group"]),
                "target_in_unseen_groups": target_group in set(training["unseen"]["group"]),
                "max_seen_date": training["seen"]["date"].max(),
                "max_unseen_date": training["unseen"]["date"].max(),
                "seen_group_count_range": _count_range(training["seen"]),
                "unseen_group_count_range": _count_range(training["unseen"]),
            }
            folds.append(fold_record)
            predictions = {}
            # NN is independent of training regime and is computed once per target.
            for target in target_frame.itertuples(index=False):
                predictions[("none", "nearest_neighbor", target.route_id)] = predict(target, None)
            for regime, train in training.items():
                global_model = TransitionModel.fit(train, use_regional_history=False)
                regional_model = TransitionModel.fit(train, use_regional_history=True)
                for target in target_frame.itertuples(index=False):
                    predictions[(regime, "global_transition", target.route_id)] = predict(target, global_model)
                    predictions[(regime, "regional_transition", target.route_id)] = predict(target, regional_model)
            for target in target_frame.itertuples(index=False):
                for (regime, method, route_id), prediction in predictions.items():
                    if route_id != target.route_id:
                        continue
                    result = {
                        "route_id": route_id,
                        "group": target_group,
                        "city": target.city,
                        "date": target.date,
                        "n_stops": int(target.n_stops),
                        "seed": seed,
                        "regime": regime,
                        "method": method,
                    }
                    result.update(_metric_row(target, prediction))
                    rows.append(result)
    pd.DataFrame(rows).to_parquet(out / "route_metrics.parquet", index=False)
    atomic_json(out / "folds.json", folds)
    print(json.dumps({"regions": len({row['target_group'] for row in folds}), "fold_records": len(folds), "metric_rows": len(rows)}, indent=2))


def _count_range(frame: pd.DataFrame) -> list[int]:
    values = frame.groupby("group").size()
    return [int(values.min()), int(values.max())]


def _cluster_bootstrap(values: Sequence[float], seed: int = 42, repetitions: int = 100_000) -> list[float]:
    rng, values = random.Random(seed), list(values)
    samples = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(repetitions))
    return [samples[int(.025 * repetitions)], samples[int(.975 * repetitions)]]


def _sign_test(values: Sequence[float]) -> dict:
    nonzero = [value for value in values if value != 0]
    positives = sum(value > 0 for value in nonzero)
    tail = min(positives, len(nonzero) - positives)
    probability = 2 * sum(math.comb(len(nonzero), k) for k in range(tail + 1)) / (2 ** len(nonzero))
    return {"nonzero": len(nonzero), "positive": positives, "two_sided_p": min(1.0, probability)}


def _sign_flip(values: Sequence[float], seed: int = 42, repetitions: int = 100_000) -> dict:
    values, observed = list(values), abs(statistics.fmean(values))
    if len(values) <= 20:
        statistics_ = [abs(statistics.fmean(sign * value for sign, value in zip(signs, values)))
                       for signs in itertools.product((-1, 1), repeat=len(values))]
        return {"mode": "exact", "permutations": len(statistics_), "two_sided_p": sum(x >= observed - 1e-15 for x in statistics_) / len(statistics_)}
    rng, exceed = random.Random(seed), 0
    for _ in range(repetitions):
        statistic = abs(statistics.fmean((1 if rng.random() < .5 else -1) * value for value in values))
        exceed += statistic >= observed - 1e-15
    return {"mode": "monte_carlo", "permutations": repetitions, "two_sided_p": (exceed + 1) / (repetitions + 1)}


def audit(out: Path) -> None:
    folds = json.loads((out / "folds.json").read_text())
    metrics = pd.read_parquet(out / "route_metrics.parquet")
    checks = {
        "matched_training_size": all(row["train_size_seen"] == row["train_size_unseen"] for row in folds),
        "target_present_in_seen_training": all(row["target_in_seen_groups"] for row in folds),
        "target_absent_from_unseen_training": all(not row["target_in_unseen_groups"] for row in folds),
        "strict_seen_temporal_cutoff": all(row["max_seen_date"] < row["cutoff"] for row in folds),
        "strict_unseen_temporal_cutoff": all(row["max_unseen_date"] < row["cutoff"] for row in folds),
        "ten_targets_per_fold": all(len(row["targets"]) == PROTOCOL["folds"]["targets_per_region"] for row in folds),
        "five_seeds_per_region": metrics[metrics["method"] == "regional_transition"].groupby(["group", "regime"])["seed"].nunique().eq(5).all(),
        "balanced_seen_source_regions": all(row["seen_group_count_range"][1] - row["seen_group_count_range"][0] <= 1 for row in folds),
        "balanced_unseen_source_regions": all(row["unseen_group_count_range"][1] - row["unseen_group_count_range"][0] <= 1 for row in folds),
        "finite_metrics": metrics[["normalized_kendall_loss", "normalized_edit_distance", "normalized_footrule", "route_length_ratio"]].notna().all().all(),
        "bounded_rank_metrics": metrics[["normalized_kendall_loss", "normalized_edit_distance", "normalized_footrule"]].apply(lambda x: x.between(0, 1)).all().all(),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    result = {"passed": all(checks.values()), "checks": checks, "fold_records": len(folds), "metric_rows": len(metrics)}
    atomic_json(out / "audit.json", result)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise RuntimeError("LaDe external-replication audit failed")


def report(out: Path) -> None:
    audit_result = json.loads((out / "audit.json").read_text())
    if not audit_result["passed"]:
        raise RuntimeError("refusing to report a failed audit")
    data = pd.read_parquet(out / "route_metrics.parquet")
    rank = data[data["method"] == "regional_transition"].copy()
    # First average stochastic training samples within route/regime, then routes
    # within region.  Regions, rather than packages or routes, are the inferential units.
    route = rank.groupby(["group", "city", "route_id", "regime"], as_index=False)["normalized_kendall_loss"].mean()
    region = route.groupby(["group", "city", "regime"], as_index=False)["normalized_kendall_loss"].mean()
    pivot = region.pivot(index=["group", "city"], columns="regime", values="normalized_kendall_loss").reset_index()
    pivot["delta_unseen_minus_seen"] = pivot["unseen"] - pivot["seen"]
    values = pivot["delta_unseen_minus_seen"].tolist()
    primary = {
        "regions": len(pivot),
        "cities": int(pivot["city"].nunique()),
        "seen_macro_region_mean": float(pivot["seen"].mean()),
        "unseen_macro_region_mean": float(pivot["unseen"].mean()),
        "absolute_degradation": float(pivot["delta_unseen_minus_seen"].mean()),
        "relative_degradation": float(pivot["unseen"].mean() / pivot["seen"].mean() - 1),
        "region_cluster_bootstrap_ci95": _cluster_bootstrap(values),
        "regions_worse_tied_better": [sum(x > 0 for x in values), sum(x == 0 for x in values), sum(x < 0 for x in values)],
        "exact_sign_test": _sign_test(values),
        "sign_flip_test": _sign_flip(values),
    }
    methods = []
    for (method, regime), members in data.groupby(["method", "regime"], sort=True):
        route_mean = members.groupby(["group", "city", "route_id"], as_index=False)[
            ["normalized_kendall_loss", "normalized_edit_distance", "normalized_footrule", "route_length_ratio"]
        ].mean()
        region_mean = route_mean.groupby(["group", "city"], as_index=False)[
            ["normalized_kendall_loss", "normalized_edit_distance", "normalized_footrule", "route_length_ratio"]
        ].mean()
        methods.append({
            "method": method,
            "regime": regime,
            "regions": len(region_mean),
            **{column: float(region_mean[column].mean()) for column in ["normalized_kendall_loss", "normalized_edit_distance", "normalized_footrule", "route_length_ratio"]},
        })
    city_rows = []
    for city, members in pivot.groupby("city", sort=True):
        city_rows.append({
            "city": city,
            "regions": len(members),
            "seen": float(members["seen"].mean()),
            "unseen": float(members["unseen"].mean()),
            "delta": float(members["delta_unseen_minus_seen"].mean()),
            "regions_worse": int((members["delta_unseen_minus_seen"] > 0).sum()),
        })
    seed_rows = []
    for seed, members in rank.groupby("seed", sort=True):
        seed_route = members.groupby(["group", "route_id", "regime"], as_index=False)["normalized_kendall_loss"].mean()
        seed_region = seed_route.groupby(["group", "regime"], as_index=False)["normalized_kendall_loss"].mean()
        seed_pivot = seed_region.pivot(index="group", columns="regime", values="normalized_kendall_loss")
        seed_rows.append({
            "seed": int(seed),
            "seen": float(seed_pivot["seen"].mean()),
            "unseen": float(seed_pivot["unseen"].mean()),
            "delta": float((seed_pivot["unseen"] - seed_pivot["seen"]).mean()),
        })
    analysis = {"primary": primary, "method_summary": methods, "city_summary": city_rows, "seed_summary": seed_rows, "audit": audit_result}
    atomic_json(out / "analysis.json", analysis)
    pivot.to_csv(out / "region_effects.csv", index=False)
    pd.DataFrame(methods).to_csv(out / "method_summary.csv", index=False)
    pd.DataFrame(city_rows).to_csv(out / "city_summary.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(out / "seed_summary.csv", index=False)
    lines = [
        "# LaDe-D external replication v1",
        "",
        "This is an unseen service-region replication, not a claim that LaDe regions are Amazon stations.",
        "",
        "## Primary result",
        "",
        f"- Eligible regions: {primary['regions']} across {primary['cities']} cities.",
        f"- Seen-region macro Kendall loss: {primary['seen_macro_region_mean']:.6f}.",
        f"- Unseen-region macro Kendall loss: {primary['unseen_macro_region_mean']:.6f}.",
        f"- Relative degradation: {100 * primary['relative_degradation']:.2f}%.",
        f"- Region-cluster 95% bootstrap CI for absolute delta: [{primary['region_cluster_bootstrap_ci95'][0]:.6f}, {primary['region_cluster_bootstrap_ci95'][1]:.6f}].",
        f"- Regions worse/tied/better: {primary['regions_worse_tied_better']}.",
        f"- Exact sign-test p: {primary['exact_sign_test']['two_sided_p']:.6g}.",
        f"- Sign-flip p: {primary['sign_flip_test']['two_sided_p']:.6g}.",
        "",
        "## Validity controls",
        "",
    ] + [f"- {name}: PASS" for name, passed in audit_result["checks"].items() if passed]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(analysis, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stage", choices=["init", "prepare", "evaluate", "audit", "report", "all"], default="all")
    args = parser.parse_args()
    out = args.output.resolve()
    stages = [("init", init), ("prepare", prepare), ("evaluate", evaluate), ("audit", audit), ("report", report)]
    for name, function in stages:
        if args.stage in (name, "all"):
            print(f"Stage: {name}", flush=True)
            function(out)


if __name__ == "__main__":
    main()
