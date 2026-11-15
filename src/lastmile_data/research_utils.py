"""Shared utilities used by the frozen station cold-start experiments."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def atomic_dump(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def bootstrap_ci(values, seed=42, replicates=10_000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    sampled = [
        float(np.mean(rng.choice(values, len(values), replace=True)))
        for _ in range(replicates)
    ]
    return np.quantile(sampled, [0.025, 0.975]).tolist()


def haversine_radians(a, b):
    lat1, lon1 = np.radians(a)
    lat2, lon2 = np.radians(b)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    value = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )
    return 2 * np.arcsin(np.sqrt(value))


def normalized_zones(route):
    """Fill missing zones from the nearest known stop and mark the station."""
    stops = route["stops"]
    result = {}
    known = [
        (stop, data["zone_id"], (data["lat"], data["lng"]))
        for stop, data in stops.items()
        if data["type"] != "Station"
        and isinstance(data.get("zone_id"), str)
        and data["zone_id"]
    ]
    if not known:
        raise ValueError("route has no known non-station zone")
    for stop, data in stops.items():
        if data["type"] == "Station":
            result[stop] = "stz"
        elif isinstance(data.get("zone_id"), str) and data["zone_id"]:
            result[stop] = data["zone_id"]
        else:
            point = (data["lat"], data["lng"])
            result[stop] = min(
                known,
                key=lambda item: (haversine_radians(point, item[2]), item[0]),
            )[1]
    return result


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def stratified_hash_folds(route_ids, stations, folds, salt):
    grouped = defaultdict(list)
    for route_id in route_ids:
        grouped[stations[route_id]].append(route_id)
    result = {}
    for route_ids_in_station in grouped.values():
        ordered = sorted(
            route_ids_in_station,
            key=lambda value: digest(salt + value),
        )
        for index, route_id in enumerate(ordered):
            result[route_id] = index % folds
    return result


def directed_edges(sequence):
    return set(zip(sequence, sequence[1:] + sequence[:1]))


def compressed_zones(sequence, zones):
    values = [zones[stop] for stop in sequence]
    return [
        zone
        for index, zone in enumerate(values)
        if index == 0 or zone != values[index - 1]
    ]


def overlap(left, right):
    return len(left & right) / max(1, len(left | right))
