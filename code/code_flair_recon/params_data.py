import os
from typing import Literal

default_root: str = ""
dataset_mode: Literal[
    "base",
    "t1prior",
    "t2prior",
    "t1pred",
    "t2pred",
    "t1t2pred",
] = "t1t2pred"

DATA_ROOT: str = os.environ.get("DATA_ROOT", default_root)
DATASET_MODE: Literal["base", "t1prior", "t2prior", "t1pred", "t2pred", "t1t2pred"] = os.environ.get("DATASET_MODE", dataset_mode)


if DATASET_MODE in {"base", "t1prior"}:
    TRAIN_DATASET: list[str] = [
        f"{DATA_ROOT}/train",
    ]
    VALID_DATASET: list[str] = [
        f"{DATA_ROOT}/val",
    ]

elif DATASET_MODE == "t2prior":
    TRAIN_DATASET: list[str] = [
        f"{DATA_ROOT}/train",
    ]
    VALID_DATASET: list[str] = [
        f"{DATA_ROOT}/val",
    ]

elif DATASET_MODE in {"t1pred"}:
    TRAIN_DATASET: list[str] = [
        f"{DATA_ROOT}/train",
    ]
    VALID_DATASET: list[str] = [
        f"{DATA_ROOT}/val",
    ]

elif DATASET_MODE in {"t2pred"}:
    TRAIN_DATASET: list[str] = [
        f"{DATA_ROOT}/train",
    ]
    VALID_DATASET: list[str] = [
        f"{DATA_ROOT}/val",
    ]

elif DATASET_MODE in {"t1t2pred"}:
    TRAIN_DATASET: list[str] = [
        f"{DATA_ROOT}/train",
    ]
    VALID_DATASET: list[str] = [
        f"{DATA_ROOT}/val",
    ]
