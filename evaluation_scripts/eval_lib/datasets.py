"""Dataset ↔ problem-statement registry.

Evaluation is dataset-specific and problem-statement-specific: the same CSV
loading and metrics engine is reused, but each dataset pins down its problem
type, its class labels, and how raw CSV values map onto them. Adding a new
dataset means adding one DatasetConfig here — nothing else in eval_lib needs
to change.

label_map keys are matched case-insensitively after stripping whitespace, so
"Positive", "positive", " Positive " all resolve the same way. Values not
found in label_map fall back to being parsed as int/float directly, which is
what lets plain 0/1 or 1..4 CSVs work with an empty-ish label_map.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    problem_type: str  # "binary" or "multiclass"
    label_map: dict = field(default_factory=dict)
    class_values: Optional[list] = None  # ordered label space; None => infer from data
    description: str = ""

    def __post_init__(self):
        if self.problem_type not in ("binary", "multiclass"):
            raise ValueError(f"Unknown problem_type {self.problem_type!r} for dataset {self.name!r}")


# dataset_id 1 in TANUH-processingtee/config.yml. Ground truth is BI-RADS
# breast tissue density (four ordinal categories, "almost entirely fatty"
# through "extremely dense"), stored as ints 1-4 in the source dataset.csv
# (see TANUH-processingtee/tools/prepare_and_upload_datasets.py). Ordinal ->
# QWK is meaningful here, which is why this is a multiclass config rather
# than 4 independent one-hot classes.
BREAST_CANCER = DatasetConfig(
    name="breast_cancer",
    problem_type="multiclass",
    class_values=[1, 2, 3, 4],
    label_map={
        "a": 1, "b": 2, "c": 3, "d": 4,  # BI-RADS density letter grades, if ever submitted that way
    },
    description="Breast tissue density grading (BI-RADS a-d / 1-4), multiclass + ordinal.",
)

# dataset_id 2 in TANUH-processingtee/config.yml. Ground truth folder names
# are "Non-Suspicious" / "Suspicious" (see prepare_and_upload_datasets.py),
# mapped here to the canonical 0 (negative) / 1 (positive) used throughout
# eval_lib.binary.
ORAL_CANCER = DatasetConfig(
    name="oral_cancer",
    problem_type="binary",
    label_map={
        "non-suspicious": 0, "suspicious": 1,
        "negative": 0, "positive": 1,
        "benign": 0, "malignant": 1,
        "false": 0, "true": 1,
        "no": 0, "yes": 1,
    },
    description="Oral cancer screening (OCS): Suspicious vs Non-Suspicious, binary.",
)

DATASET_REGISTRY: dict[str, DatasetConfig] = {
    BREAST_CANCER.name: BREAST_CANCER,
    ORAL_CANCER.name: ORAL_CANCER,
}


def get_dataset_config(name: str) -> DatasetConfig:
    try:
        return DATASET_REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(DATASET_REGISTRY))
        raise ValueError(f"Unknown dataset {name!r}. Known datasets: {known}") from None


def coerce_label(raw_value, cfg: DatasetConfig):
    """Map one raw CSV value onto this dataset's canonical label space.

    Lookup order: dataset label_map (case-insensitive, whitespace-stripped)
    first, then a direct int parse, then a direct float parse (for ratings
    that arrive as "1.0"). Raises ValueError with the offending value if none
    of those work, rather than silently producing NaN/None.
    """
    text = str(raw_value).strip()
    mapped = cfg.label_map.get(text.lower())
    if mapped is not None:
        return mapped
    try:
        return int(text)
    except ValueError:
        pass
    try:
        as_float = float(text)
        if as_float.is_integer():
            return int(as_float)
        return as_float
    except ValueError:
        raise ValueError(
            f"Could not coerce label {raw_value!r} for dataset {cfg.name!r}. "
            f"Known string labels: {sorted(cfg.label_map)}"
        ) from None


def coerce_labels(raw_values, cfg: DatasetConfig) -> list:
    return [coerce_label(v, cfg) for v in raw_values]
