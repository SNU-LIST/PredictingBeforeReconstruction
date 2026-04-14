import argparse
import os
import random
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from scipy.io import savemat
from torch import Tensor
from torch.utils.data import DataLoader

from common.logger import logger, logger_add_handler
from common.metric import calculate_mse, calculate_ssim
from common.utils import call_next_id, separator, validate_tensor_dimensions, validate_tensors
from common.wrapper import error_wrap
from components.metriccontroller import MetricController
from core_funcs import ModelType, get_loss_func, log_summary
from datawrapper.datawrapper_wholebrain import DataKey, LoaderConfig, get_data_wrapper_loader
from model.prednet import PredNet
from params import PredNetConfig

warnings.filterwarnings("ignore")


default_root: str = ""
default_run_dir: str = ""

DATA_ROOT: str = os.environ.get("DATA_ROOT", default_root)
RUN_DIR: str = os.environ.get("RUN_DIR", default_run_dir)

TEST_DATASET: list[str] = [
    DATA_ROOT,
]

os.environ["PYTHONHASHSEED"] = "0"
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.random.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


@dataclass
class TestConfig:
    # Dataset
    trained_checkpoints: str = ""
    test_dataset: list[str] = field(default_factory=lambda: TEST_DATASET)
    data_type: str = "*.mat"
    debugmode: bool = False

    # Logging
    log_lv: str = "INFO"
    run_dir: Path = Path(RUN_DIR)
    init_time: float = 0.0

    # Model experiment
    model_type: Literal["prediction"] = "prediction"

    # Test params
    gpu: str = "0"
    valid_batch: int = 1
    num_workers: int = 4
    device: torch.device | None = None
    loss_model: Literal["l1", "l2"] = "l2"

    # hyper
    source_contrast: str = "prev"


parser = argparse.ArgumentParser(description="Test Configuration")
test_dict = asdict(TestConfig())
for key, default_value in test_dict.items():
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
args = parser.parse_args()

NET_PREDICTION = PredNet | torch.nn.DataParallel[PredNet]


def test_part_prediction(
    _data: dict[DataKey, Tensor | str],
    test_dir: Path,
    network_prediction: NET_PREDICTION,
    save_val: bool,
    test_state: MetricController,
    config: TestConfig,
) -> None:
    loss_func = get_loss_func(loss_model=config.loss_model)

    img: Tensor = _data[DataKey.IMG].to(config.device)
    img_prev: Tensor = _data[DataKey.PREV].to(config.device)
    mask: Tensor = _data[DataKey.IMG_Mask].to(config.device)
    meta: Tensor = _data[DataKey.Meta].to(config.device)
    name: str = _data[DataKey.Name][0]

    logger.info(f"Testing {name} started...")

    data_range = torch.amax(img.view(img.shape[0], -1), dim=1)

    validate_tensors([img, img_prev, mask, meta])
    validate_tensor_dimensions([img, img_prev, mask], 4)

    batch_cnt = img.shape[0]
    if batch_cnt != 1:
        raise ValueError("Batch size must be 1 for whole brain reconstruction.")

    # output = prediction_wholebrain(
    #     prednet=network_prediction,
    #     img=img,
    #     img_prev=img_prev,
    #     mask=mask,
    #     meta=meta,
    #     batch_size=config.valid_batch,
    # )

    output = img_prev.clone()

    validate_tensors([output])
    validate_tensor_dimensions([output], 4)

    test_state_item = MetricController()

    for slice in range(output.shape[1]):
        mask_slice = mask[:, slice : slice + 1, :, :]
        img_slice = img[:, slice : slice + 1, :, :]
        output_slice = output[:, slice : slice + 1, :, :]

        loss = loss_func(output_slice, img_slice)
        loss = torch.mean(loss, dim=(1, 2, 3), keepdim=True)
        test_state.add("loss", loss)

        if torch.any(mask_slice):
            ssim = calculate_ssim(output_slice, img_slice, mask_slice, data_range)
            mse = calculate_mse(output_slice, img_slice, mask_slice)
            test_state_item.add("ssim", ssim)
            test_state_item.add("mse", mse)

            psnr = 10 * torch.log10((torch.max(img_slice) ** 2) / (mse.mean() + 1e-12)).view(-1, 1, 1, 1)
            test_state_item.add("psnr", psnr)

    for key in test_state_item.state_dict:
        test_state.add(key, torch.tensor(test_state_item.mean(key)).view(-1, 1, 1, 1))

    # logger.info(
    #     f"psnr: {test_state_item.mean('psnr').item():.4f}, ssim: {test_state_item.mean('ssim').item():.4f}, mse: {test_state_item.mean('mse').item():.6f}"  # noqa: E501
    # )

    if not save_val:
        return

    train_div = "results"
    os.makedirs(test_dir / train_div, exist_ok=True)
    save_dict = {
        "img": img.cpu().numpy()[0, ...],
        "out": output.cpu().numpy()[0, ...],
        "mask": mask.cpu().numpy()[0, ...],
        "prev": img_prev.cpu().numpy()[0, ...],
        "meta": meta.cpu().numpy()[0, ...],
    }
    savemat(f"{test_dir}/{train_div}/{name}_res.mat", save_dict)


def test_part(
    valid_state: MetricController,
    valid_loader: DataLoader,
    network_prediction: NET_PREDICTION,
    run_dir: Path,
    save_val: bool,
    config: TestConfig,
) -> float:
    if config.device is None:
        raise TypeError("device is not to be None")

    network_prediction.eval()

    for _data in valid_loader:
        if ModelType.from_string(config.model_type) == ModelType.Prediction:
            test_part_prediction(
                _data=_data,
                test_dir=run_dir,
                network_prediction=network_prediction,
                save_val=save_val,
                test_state=valid_state,
                config=config,
            )
        else:
            raise KeyError("model type not matched")

    log_summary(state=valid_state, log_std=True, init_time=config.init_time)

    primary_metric = valid_state.mean("loss")
    return primary_metric


class Tester:
    run_dir: Path
    network_prediction: NET_PREDICTION
    test_loader: DataLoader
    config: TestConfig
    modelconfig: PredNetConfig

    def __init__(
        self,
    ) -> None:
        self.config = TestConfig()
        for key, value in vars(args).items():
            if value is not None and hasattr(self.config, key):
                if isinstance(getattr(self.config, key), bool):
                    setattr(self.config, key, bool(value))
                else:
                    setattr(self.config, key, value)

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = self.config.gpu

        self.config.init_time = time.time()
        self.config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # dir setting
        self.run_dir = self.config.run_dir / f"{call_next_id(self.config.run_dir):05d}_test"
        logger_add_handler(logger, f"{self.run_dir/'log.log'}", self.config.log_lv)
        logger.info(separator())
        logger.info(f"Run dir: {self.run_dir}")
        os.makedirs(self.run_dir, exist_ok=True)

        # log config
        logger.info(separator())
        logger.info("Text Config")
        config_dict = asdict(self.config)
        for k in config_dict:
            logger.info(f"{k}:{config_dict[k]}")

    def __call__(
        self,
    ) -> None:
        self._set_data()
        self._set_network()
        self._test()

    @error_wrap
    def _set_data(
        self,
    ) -> None:
        logger.info(separator())
        test_loader_cfg = LoaderConfig(
            data_type=self.config.data_type,
            batch=self.config.valid_batch,
            num_workers=self.config.num_workers,
            shuffle=False,
            debug_mode=self.config.debugmode,
            source_contrast=self.config.source_contrast,
        )
        self.test_loader, _, test_len = get_data_wrapper_loader(
            file_path=self.config.test_dataset,
            training_mode=False,
            loader_cfg=test_loader_cfg,
        )
        logger.info(f"Test dataset length : {test_len}")

    @error_wrap
    def _set_network(
        self,
    ) -> None:
        longitudinal_checkpoint_data = torch.load(
            self.config.trained_checkpoints,
            map_location="cpu",
            weights_only=True,
        )

        if not (("model_state_dict" in longitudinal_checkpoint_data) and ("model_config" in longitudinal_checkpoint_data)):
            logger.error("Invalid Checkpoint")
            raise KeyError("Invalid Checkpoint")

        self.modelconfig = PredNetConfig(**longitudinal_checkpoint_data["model_config"])
        self.network_prediction = PredNet(device=self.config.device, prednetconfig=self.modelconfig)
        load_state_dict = longitudinal_checkpoint_data["model_state_dict"]

        _state_dict = {}
        for key, value in load_state_dict.items():
            new_key = key.replace("module.", "")
            _state_dict[new_key] = value

        try:
            self.network_prediction.load_state_dict(_state_dict, strict=True)
        except Exception as err:
            logger.warning(f"Strict load failure. Trying to load weights available: {err}")
            self.network_prediction.load_state_dict(_state_dict, strict=False)

        logger.info(separator())
        logger.info("Model Config")
        config_dict = asdict(self.modelconfig)
        for k in config_dict:
            logger.info(f"{k}:{config_dict[k]}")

        self.network_prediction = self.network_prediction.to(self.config.device)

    @error_wrap
    def _test(self) -> None:
        test_state = MetricController()
        test_state.reset()
        logger.info(separator())
        logger.info("Test")
        with torch.no_grad():
            test_part(
                valid_state=test_state,
                valid_loader=self.test_loader,
                network_prediction=self.network_prediction,
                run_dir=self.run_dir,
                save_val=True,
                config=self.config,
            )


if __name__ == "__main__":
    test = Tester()
    test()
