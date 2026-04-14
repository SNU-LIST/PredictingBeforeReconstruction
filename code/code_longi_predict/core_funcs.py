import os
import time
from collections.abc import Callable
from dataclasses import asdict
from enum import StrEnum
from pathlib import Path

import numpy as np
import torch
from scipy.io import savemat
from torch import Tensor
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader

from common.logger import logger
from common.metric import calculate_mse
from common.utils import seconds_to_dhms
from components.metriccontroller import MetricController
from datawrapper.datawrapper import DataKey
from model.prednet import PredNet
from model.prednet_helper import train_part, validate_part
from params import PredNetConfig, config, prednetconfig

NETWORK = PredNet | torch.nn.DataParallel[PredNet]
OPTIM = Adam | AdamW


class ModelType(StrEnum):
    Prediction = "prediction"

    @classmethod
    def from_string(cls, value: str) -> "ModelType":
        try:
            return cls(value)
        except ValueError as err:
            raise ValueError(f"Invalid ModelType value: {value}. Must be one of {list(cls)} : {err}") from err


def get_network(
    device: torch.device | None,
    model_type: str,
    prednetconfig: PredNetConfig,
) -> NETWORK:
    if device is None:
        raise TypeError("device is not to be None")

    if ModelType.from_string(model_type) == ModelType.Prediction:
        return PredNet(device=device, prednetconfig=prednetconfig)
    else:
        raise KeyError("model type not matched")


def get_optim(
    network: NETWORK | None,
    optimizer: str,
) -> OPTIM | None:
    if network is None:
        return None
    if optimizer == "adam":
        return Adam(network.parameters(), betas=(0.9, 0.99), eps=1e-10)
    elif optimizer == "adamw":
        return AdamW(network.parameters(), betas=(0.9, 0.99), weight_decay=0.0)
    else:
        raise KeyError("optimizer not matched")


def get_loss_func(
    loss_model: str,
) -> Callable:
    if loss_model == "l1":
        return torch.nn.L1Loss(reduction="none")
    elif loss_model == "l2":
        return torch.nn.MSELoss(reduction="none")
    else:
        raise KeyError("loss func not matched")


def get_learning_rate(
    epoch: int,
    lr: float,
    lr_decay: float,
    lr_tol: int,
) -> float:
    factor = epoch - lr_tol if lr_tol < epoch else 0
    return lr * (lr_decay**factor)


def set_optimizer_lr(
    optimizer: OPTIM | None,
    learning_rate: float,
) -> OPTIM | None:
    if optimizer is None:
        return None
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate
    return optimizer


def log_summary(
    init_time: float,
    state: MetricController,
    log_std: bool = False,
) -> None:
    spend_time = seconds_to_dhms(time.time() - init_time)
    for key in state.state_dict:
        if log_std:
            summary = f"{spend_time} | {key}: {state.mean(key):0.3e} + {state.std(key):0.3e} "
            logger.info(summary)
        else:
            summary = f"{spend_time} | {key}: {state.mean(key):0.3e}"
            logger.info(summary)


def save_checkpoint(
    network: NETWORK,
    run_dir: Path,
    epoch: str | int | None = None,
) -> None:
    if epoch is None:
        epoch = "best"
    os.makedirs(run_dir / "checkpoints", exist_ok=True)
    torch.save(
        {
            "model_state_dict": network.state_dict(),
            "model_config": asdict(prednetconfig),
        },
        run_dir / f"checkpoints/checkpoint_{epoch}.ckpt",
    )


def zero_optimizers(
    optim_list: list[OPTIM | None],
) -> None:
    for opt in optim_list:
        if opt is not None:
            opt.zero_grad()


def step_optimizers(
    optim_list: list[OPTIM | None],
) -> None:
    for opt in optim_list:
        if opt is not None:
            opt.step()


def save_result_to_mat(
    test_dir: Path,
    batch_cnt: int,
    tesner_dict: dict[str, Tensor | None],
    img_cnt: int,
) -> None:
    os.makedirs(test_dir, exist_ok=True)
    save_dict = {}

    if batch_cnt == 0:
        logger.warning("batch_cnt is 0, no data to save")
        return

    for i in range(batch_cnt):
        for key, value in tesner_dict.items():
            if value is not None:
                save_dict[key] = value.cpu().detach().numpy()[i, ...]

        idx = img_cnt + i + 1
        savemat(f"{test_dir}/{idx}_res.mat", save_dict)


def save_result_to_npy(
    test_dir: Path,
    batch_cnt: int,
    tesner_dict: dict[str, Tensor | None],
    img_cnt: int,
) -> None:
    os.makedirs(test_dir, exist_ok=True)

    if batch_cnt == 0:
        logger.warning("batch_cnt is 0, no data to save")
        return

    for i in range(batch_cnt):
        save_dict = {}
        for key, value in tesner_dict.items():
            if value is not None:
                save_dict[key] = value.cpu().detach().numpy()[i, ...]

        idx = img_cnt + i + 1
        np.savez_compressed(f"{test_dir}/{idx}_res.npz", **save_dict)


def loss_clip(
    loss: Tensor,
    epoch: int,
    max_loss: float = 1e-1,
) -> Tensor:
    if loss.mean() > max_loss:
        epoch_scale = config.lr_decay ** (config.train_epoch - epoch - 25)
        loss = loss * min(max(epoch_scale, 0.1), 1.0)
    return loss


def train_epoch_prediction(
    _data: dict[DataKey, Tensor | str],
    network: NETWORK,
    epoch: int,
    train_state: MetricController,
    optim_list: list[OPTIM | None],
) -> int:
    loss_func = get_loss_func(config.loss_model)

    img: Tensor = _data[DataKey.IMG].to(config.device)
    img_prev: Tensor = _data[DataKey.PREV].to(config.device)
    meta: Tensor = _data[DataKey.Meta].to(config.device)

    img_cnt_minibatch = img.shape[0]

    zero_optimizers(optim_list=optim_list)

    output, label = train_part(
        network=network,
        img=img,
        img_prev=img_prev,
        meta=meta,
    )

    loss = torch.mean(loss_func(output, label), dim=(1, 2, 3), keepdim=True)
    torch.mean(loss).backward()

    torch.nn.utils.clip_grad_norm_(network.parameters(), 5.0)
    step_optimizers(optim_list=optim_list)

    train_state.add("loss", loss)

    return img_cnt_minibatch


def train_epoch(
    train_loader: DataLoader,
    train_len: int,
    network: NETWORK,
    optim_list: list[OPTIM | None],
    epoch: int,
) -> None:
    train_state = MetricController()
    train_state.reset()
    network.train()

    logging_cnt: int = 1
    img_cnt: int = 0
    for _data in train_loader:
        if ModelType.from_string(config.model_type) == ModelType.Prediction:
            img_cnt_minibatch = train_epoch_prediction(
                _data=_data,
                network=network,
                epoch=epoch,
                train_state=train_state,
                optim_list=optim_list,
            )
        else:
            raise KeyError("model type not matched")

        img_cnt += img_cnt_minibatch
        if img_cnt > (train_len / config.logging_density * logging_cnt):
            log_summary(init_time=config.init_time, state=train_state)
            logging_cnt += 1

    log_summary(init_time=config.init_time, state=train_state)


def test_part_prediction(
    _data: dict[DataKey, Tensor | str],
    test_dir: Path,
    model: NETWORK,
    save_val: bool,
    test_state: MetricController,
    img_cnt: int,
) -> float:
    loss_func = get_loss_func(config.loss_model)

    img: Tensor = _data[DataKey.IMG].to(config.device)
    img_prev: Tensor = _data[DataKey.PREV].to(config.device)
    meta: Tensor = _data[DataKey.Meta].to(config.device)
    mask: Tensor = _data[DataKey.IMG_Mask].to(config.device)

    batch_cnt = img.shape[0]

    output = validate_part(
        network=model,
        img_prev=img_prev,
        meta=meta,
        step=20,
    )

    loss = loss_func(output, img)
    loss = torch.mean(loss, dim=(1, 2, 3), keepdim=True)
    test_state.add("loss", loss)

    for i in range(batch_cnt):
        output_i = output[i : i + 1, :, :, :]
        img_i = img[i : i + 1, :, :, :]
        mask_i = mask[i : i + 1, :, :, :]

        mask_sum = torch.sum(mask_i, dim=(1, 2, 3), keepdim=True)
        if not torch.any(mask_sum == 0):
            output_i = output_i * img_i[mask_i == 1].mean() / output_i[mask_i == 1].mean()
            test_state.add("mse", calculate_mse(output_i, img_i, mask_i))

    # logger.info(f"batch_cnt: {batch_cnt}, img_cnt: {img_cnt}")

    if save_val:
        save_result_to_npy(
            test_dir=test_dir,
            batch_cnt=batch_cnt,
            tesner_dict={
                "visit1": img_prev,
                "visit2": img,
                "output": output,
            },
            img_cnt=img_cnt,
        )

    return batch_cnt


def test_part(
    epoch: int,
    data_loader: DataLoader,
    network: NETWORK,
    run_dir: Path,
    save_val: bool,
) -> float:
    test_state = MetricController()
    test_state.reset()
    network.eval()

    img_cnt: int = 0
    for _data in data_loader:
        if ModelType.from_string(config.model_type) == ModelType.Prediction:
            batch_cnt = test_part_prediction(
                _data=_data,
                test_dir=run_dir / f"test/ep_{epoch}",
                model=network,
                save_val=save_val and img_cnt <= config.save_max_idx,
                test_state=test_state,
                img_cnt=img_cnt,
            )
        else:
            raise KeyError("model type not matched")

        img_cnt += batch_cnt

    log_summary(init_time=config.init_time, state=test_state, log_std=True)

    primary_metric = test_state.mean("mse")
    return primary_metric
