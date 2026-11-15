"""Validation, imbalance reporting, and leakage checks.

These utilities operate on iterable dictionaries so the raw-data adapter can be
implemented after the exact Amazon schema is inspected. They deliberately fail
closed on missing routing identity or split metadata.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

from .contracts import (
    ALLOWED_ROUTE_QUALITIES,
    ALLOWED_SOURCE_SPLITS,
    REQUIRED_FIELDS,
)


class SplitLeakageError(ValueError):
    """Raised when a route appears in more than one split."""


@dataclass(frozen=True)
class ImbalanceReport:
    counts: dict[str, int]
    proportions: dict[str, float]
    minority_class: Optional[str]
    imbalance_ratio: Optional[float]


def validate_canonical_records(records: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    """Validate and materialize canonical route-level records.

    Required identifiers must be non-empty. Unknown labels are rejected instead
    of silently being mapped, because label mistakes can create leakage or hide
    class imbalance.
    """

    materialized = [dict(record) for record in records]
    seen_ids: set[str] = set()
    for index, record in enumerate(materialized):
        missing = REQUIRED_FIELDS - record.keys()
        if missing:
            raise ValueError(f"record {index} is missing required fields: {sorted(missing)}")
        route_id = str(record["route_id"]).strip()
        city = str(record["city"]).strip()
        source_split = str(record["source_split"]).strip().lower()
        if not route_id or not city:
            raise ValueError(f"record {index} has an empty route_id or city")
        if source_split not in ALLOWED_SOURCE_SPLITS:
            raise ValueError(f"record {index} has invalid source_split={source_split!r}")
        if route_id in seen_ids:
            raise ValueError(f"duplicate route_id={route_id!r}")
        seen_ids.add(route_id)
        record["route_id"] = route_id
        record["city"] = city
        record["source_split"] = source_split

        if record.get("route_quality") is not None:
            quality = str(record["route_quality"]).strip().lower()
            if quality not in ALLOWED_ROUTE_QUALITIES:
                raise ValueError(f"record {index} has invalid route_quality={quality!r}")
            record["route_quality"] = quality
    return materialized


def find_route_split_leakage(records: Iterable[Mapping[str, object]]) -> dict[str, set[str]]:
    """Return route IDs assigned to more than one split."""

    route_splits: dict[str, set[str]] = {}
    for record in records:
        route_id = str(record["route_id"])
        route_splits.setdefault(route_id, set()).add(str(record["source_split"]))
    return {route_id: splits for route_id, splits in route_splits.items() if len(splits) > 1}


def assert_no_route_split_leakage(records: Iterable[Mapping[str, object]]) -> None:
    leakage = find_route_split_leakage(records)
    if leakage:
        raise SplitLeakageError(f"route IDs cross splits: {sorted(leakage)}")


def summarize_imbalance(records: Iterable[Mapping[str, object]], label_field: str = "route_quality") -> ImbalanceReport:
    labels = [str(record[label_field]).lower() for record in records if record.get(label_field) is not None]
    counts = Counter(labels)
    if not counts:
        return ImbalanceReport({}, {}, None, None)
    total = sum(counts.values())
    proportions = {label: count / total for label, count in sorted(counts.items())}
    minority_count = min(counts.values())
    majority_count = max(counts.values())
    minority_class = min(counts, key=counts.get)
    return ImbalanceReport(dict(counts), proportions, minority_class, majority_count / minority_count)


def build_class_weights(report: ImbalanceReport) -> dict[str, float]:
    """Return balanced weights; use only on training folds.

    The weighting formula is total/(number_of_classes*class_count), which keeps
    the mean training weight close to one. Validation and test data must not be
    resampled or reweighted when reporting metrics.
    """

    if not report.counts:
        return {}
    total = sum(report.counts.values())
    number_of_classes = len(report.counts)
    return {
        label: total / (number_of_classes * count)
        for label, count in report.counts.items()
    }
