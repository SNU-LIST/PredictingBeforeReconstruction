import argparse
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import torch
from params_data import TRAIN_DATASET, VALID_DATASET

default_run_dir: str = ""
RUN_DIR: str = os.environ.get("RUN_DIR", default_run_dir)


@dataclass
class GeneralConfig:
    # Dataset
    train_dataset: list[str] = field(default_factory=lambda: TRAIN_DATASET)
    valid_dataset: list[str] = field(default_factory=lambda: VALID_DATASET)
    data_type: str = "*.mat"
    debugmode: bool = False

    # Logging
    log_lv: str = "INFO"
    run_dir: Path = Path(RUN_DIR)
    init_time: float = 0.0

    # Model experiment
    model_type: Literal["prediction"] = "prediction"

    # Optimizer
    optimizer: Literal["adam", "adamw"] = "adam"
    loss_model: Literal["l1", "l2"] = "l2"
    lr: float = 1e-4
    lr_decay: float = 0.93
    lr_tol: int = 1

    # Train params
    gpu: str = "0,1,2,3,4,5,6,7"
    train_batch: int = 32
    valid_batch: int = 32
    train_epoch: int = 70
    logging_density: int = 2
    valid_interval: int = 1
    valid_tol: int = 1
    # valid_tol: int = 50
    num_workers: int = 32
    save_val: bool = True
    parallel: bool = True
    device: torch.device | None = None
    save_max_idx: int = 20

    # hyper
    source_contrast: str = "prev"

    tag: str = ""


@dataclass
class PredNetConfig:
    # Model architecture
    recon_net_chan: int = 32
    recon_net_pool: int = 5
    meta_dim: int = 7


@dataclass
class TestConfig:
    # Dataset
    trained_checkpoints: str = ""


# Argparser
parser = argparse.ArgumentParser(description="Training Configuration")
general_config_dict = asdict(GeneralConfig())
prednet_config_dict = asdict(PredNetConfig())
test_config_dict = asdict(TestConfig())

for key, default_value in general_config_dict.items():
    if isinstance(default_value, bool):
        parser.add_argument(
            f"--{key}",
            type=lambda x: x.lower() in ("true", "t", "yes", "y", "1"),
            default=None,
            help=f"Set {key} (true/false, default: {default_value})",
        )
    else:
        parser.add_argument(
            f"--{key}",
            type=type(default_value),
            default=None,
            help=f"Override for {key}",
        )

for key, default_value in prednet_config_dict.items():
    if isinstance(default_value, bool):
        parser.add_argument(
            f"--{key}",
            type=lambda x: x.lower() in ("true", "t", "yes", "y", "1"),
            default=None,
            help=f"Set {key} (true/false, default: {default_value})",
        )
    else:
        parser.add_argument(
            f"--{key}",
            type=type(default_value),
            default=None,
            help=f"Override for {key}",
        )

for key, default_value in test_config_dict.items():
    if isinstance(default_value, bool):
        parser.add_argument(
            f"--{key}",
            type=lambda x: x.lower() in ("true", "t", "yes", "y", "1"),
            default=None,
            help=f"Set {key} (true/false, default: {default_value})",
        )
    else:
        parser.add_argument(
            f"--{key}",
            type=type(default_value),
            default=None,
            help=f"Override for {key}",
        )

# Apply argparser
config = GeneralConfig()
prednetconfig = PredNetConfig()
args = parser.parse_args()

for key, value in vars(args).items():
    if value is not None:
        if hasattr(config, key):
            if isinstance(getattr(config, key), bool):
                setattr(config, key, bool(value))
            else:
                setattr(config, key, value)

        if hasattr(prednetconfig, key):
            if isinstance(getattr(prednetconfig, key), bool):
                setattr(prednetconfig, key, bool(value))
            else:
                setattr(prednetconfig, key, value)
