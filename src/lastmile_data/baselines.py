"""Deterministic travel-time baselines; directed costs, fixed depot, closed tours."""
from __future__ import annotations

import math
from datetime import datetime


def validate_sequence(sequence, stops, depot):
    if (not sequence or sequence[0] != depot or len(sequence) != len(stops)
            or len(set(sequence)) != len(sequence) or set(sequence) != set(stops)):
        raise ValueError("sequence must start at depot and visit every stop exactly once")


def travel_time(sequence, costs):
    return sum(costs[a][b] for a, b in zip(sequence, sequence[1:] + sequence[:1]))


def nearest_neighbor(costs, depot):
    remaining = set(costs) - {depot}
    sequence = [depot]
    while remaining:
        stop = min(remaining, key=lambda s: (costs[sequence[-1]][s], s))
        sequence.append(stop)
        remaining.remove(stop)
    return sequence


def two_opt(sequence, costs, max_passes=5):
    """Best-improvement directed 2-opt, including all reversed internal arcs.

    Bounded passes are part of the algorithm, not a claim of local optimality.
    """
    result = list(sequence)
    for _ in range(max_passes):
        n = len(result)
        prefix = [0.0]
        for a, b in zip(result, result[1:]):
            prefix.append(prefix[-1] + costs[b][a] - costs[a][b])
        best_delta, best_pair = -1e-9, None
        for i in range(1, n - 1):
            a, b = result[i - 1], result[i]
            for j in range(i + 1, n):
                c, d = result[j], result[(j + 1) % n]
                delta = (costs[a][c] + costs[b][d] - costs[a][b] - costs[c][d]
                         + prefix[j] - prefix[i])
                if delta < best_delta:
                    best_delta, best_pair = delta, (i, j)
        if best_pair is None:
            break
        i, j = best_pair
        result[i:j + 1] = reversed(result[i:j + 1])
    return result


def evaluate(sequence, route, packages, costs):
    """Simulation: wait for latest stop-level release, then serve packages.

    Windows measured at service START, summed service seconds per stop, depot
    return included. These are declared simulation metrics, not actual driver
    timestamps or the official Amazon sequence score. Scan status is not used.
    """
    stops = route['stops']
    depots = [s for s, v in stops.items() if v['type'] == 'Station']
    if len(depots) != 1:
        raise ValueError('expected exactly one depot')
    depot = depots[0]
    validate_sequence(sequence, stops, depot)
    if set(packages) != set(stops) or set(costs) != set(stops):
        raise ValueError('stop keys differ across inputs')
    for row in costs.values():
        if set(row) != set(stops) or any(not math.isfinite(v) or v < 0 for v in row.values()):
            raise ValueError('invalid travel-time matrix')
    start = datetime.fromisoformat(route['date_YYYY_MM_DD'] + ' ' + route['departure_time_utc'])
    def offset(value):
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return (datetime.fromisoformat(value) - start).total_seconds()
    elapsed = service = waiting = lateness = 0.0
    late_count = windows = 0
    previous = depot
    for stop in sequence[1:]:
        elapsed += costs[previous][stop]
        items = list(packages[stop].values())
        bounds = [(offset(p['time_window']['start_time_utc']),
                   offset(p['time_window']['end_time_utc'])) for p in items]
        if any(a is not None and b is not None and a > b for a, b in bounds):
            raise ValueError('reversed time window')
        release = max([elapsed] + [a for a, _ in bounds if a is not None])
        waiting += release - elapsed
        elapsed = release
        for a, b in bounds:
            windows += int(a is not None or b is not None)
            if b is not None:
                delay = max(0.0, elapsed - b)
                lateness += delay
                late_count += int(delay > 0)
        seconds = [p['planned_service_time_seconds'] for p in items]
        if any(not math.isfinite(v) or v < 0 for v in seconds):
            raise ValueError('invalid service time')
        duration = sum(seconds)
        service += duration
        elapsed += duration
        previous = stop
    elapsed += costs[previous][depot]
    return dict(travel_seconds=travel_time(sequence, costs), service_seconds=service,
                waiting_seconds=waiting, duration_seconds=elapsed,
                package_lateness_seconds=lateness, late_packages=late_count,
                windowed_packages=windows, sequence_valid=True,
                time_windows_satisfied=late_count == 0)
