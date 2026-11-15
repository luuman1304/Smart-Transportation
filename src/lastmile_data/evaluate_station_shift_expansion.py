"""Evaluate the 12 station-cold-start groups omitted from the Phase-A pilot.

The implementation deliberately reuses the locked Phase-A build/solve/freeze/
score machinery.  Only deterministic group selection changes: the original
five pilot stations are excluded and every other eligible station is included.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from . import evaluate_shift_phase_a as phase
from .reproduce_jpt import ROOT, dump, fingerprint


PILOT = ROOT / "outputs/shift_phase_a_v1/protocol.json"
EXPECTED_GROUPS = 12


def expansion_groups(routes, eligible):
    pilot = json.loads(PILOT.read_text())
    excluded = set(pilot["groups"]["station"])
    counts = Counter(routes[route_id]["station_code"] for route_id in eligible)
    groups = sorted(
        (station for station, count in counts.items() if count >= phase.TARGETS_PER_GROUP and station not in excluded),
        key=lambda station: (-counts[station], station),
    )
    if len(groups) != EXPECTED_GROUPS:
        raise RuntimeError(f"expected {EXPECTED_GROUPS} remaining eligible stations, found {len(groups)}")
    return {"station": groups}


def configure():
    phase.MODES = ("station",)
    phase.GROUPS_PER_MODE = EXPECTED_GROUPS
    phase.choose_groups = expansion_groups


def initialize(out: Path):
    configure()
    phase.initialize(out)
    protocol_path = out / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol.update({
        "version": 2,
        "scope": "fit-only station-shift expansion; original five Phase-A stations excluded",
        "pilot_protocol": str(PILOT),
        "pilot_protocol_hash": fingerprint(PILOT),
        "expansion_script_hash": fingerprint(Path(__file__)),
        "purpose": "increase independent station groups from 5 to 17 before repeating SARP risk evaluation",
    })
    protocol["source_hashes"][str(Path(__file__))] = protocol["expansion_script_hash"]
    protocol["source_hashes"][str(PILOT)] = protocol["pilot_protocol_hash"]
    dump(protocol_path, protocol)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/station_shift_expansion_v1"))
    parser.add_argument("--stage", choices=["init", "build", "solve", "freeze", "score", "all"], default="all")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    out = args.output.resolve()
    configure()
    stages = [
        ("init", initialize), ("build", phase.build),
        ("solve", lambda path: phase.solve(path, args.workers)),
        ("freeze", phase.freeze), ("score", phase.score),
    ]
    for name, function in stages:
        if args.stage in (name, "all"):
            print("Stage: " + name, flush=True)
            function(out)


if __name__ == "__main__":
    main()
