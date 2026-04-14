import os

default_root: str = ""


DATA_ROOT: str = os.environ.get("DATA_ROOT", default_root)

TRAIN_DATASET: list[str] = [DATA_ROOT + "/train"]
VALID_DATASET: list[str] = [DATA_ROOT + "/val"]
