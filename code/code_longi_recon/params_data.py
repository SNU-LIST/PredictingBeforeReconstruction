import os
from typing import Literal

default_root: str = ""
dataset_mode: Literal[
    "base",
    "pred",
] = "base"

DATA_ROOT: str = os.environ.get("DATA_ROOT", default_root)
DATASET_MODE: Literal["base", "pred"] = os.environ.get("DATASET_MODE", dataset_mode)


if DATASET_MODE == "base":
    TRAIN_DATASET: list[str] = [
        f"{DATA_ROOT}/train",
    ]
    VALID_DATASET: list[str] = [
        f"{DATA_ROOT}/val",
    ]

elif DATASET_MODE == "pred":
    TRAIN_DATASET: list[str] = [
        f"{DATA_ROOT}/train",
    ]
    VALID_DATASET: list[str] = [
        f"{DATA_ROOT}/val",
    ]
