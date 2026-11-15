"""Small, dependency-free data contracts used before model training."""

REQUIRED_FIELDS = frozenset({"route_id", "city", "source_split"})
CANONICAL_FIELDS = (
    "route_id",
    "city",
    "source_split",
    "route_quality",
    "stop_sequence",
    "route_time",
    "transit_time",
    "service_time",
    "source_file",
)

ALLOWED_SOURCE_SPLITS = frozenset({"training", "evaluation"})
ALLOWED_ROUTE_QUALITIES = frozenset({"low", "medium", "high"})
