"""Data contracts and validation helpers for the last-mile route study."""

from .contracts import CANONICAL_FIELDS, REQUIRED_FIELDS
from .validation import (
    ImbalanceReport,
    SplitLeakageError,
    build_class_weights,
    assert_no_route_split_leakage,
    find_route_split_leakage,
    summarize_imbalance,
    validate_canonical_records,
)

__all__ = [
    "CANONICAL_FIELDS",
    "REQUIRED_FIELDS",
    "ImbalanceReport",
    "SplitLeakageError",
    "build_class_weights",
    "assert_no_route_split_leakage",
    "find_route_split_leakage",
    "summarize_imbalance",
    "validate_canonical_records",
]
