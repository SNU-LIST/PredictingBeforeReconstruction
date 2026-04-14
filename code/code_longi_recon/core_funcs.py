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
from common.utils import seconds_to_dhms, validate_tensor_dimensions
from components.metriccontroller import MetricController
from datawrapper.datawrapper import DataKey
from model.predprior_recon import PredPriorRecon
from model.predprior_recon_helper import train_forward, valid_recon
from params import ModelConfig, config, modelconfig

NETWORK = PredPriorRecon | torch.nn.DataParallel[PredPriorRecon]
OPTIM = Adam | AdamW


class ModelType(StrEnum):
    PredRecon = "predrecon"

    @classmethod
    def from_string(cls, value: str) -> "ModelType":
        try:
            return cls(value)
        except ValueError as err:
            raise ValueError(f"Invalid ModelType value: {value}. Must be one of {list(cls)} : {err}") from err


def get_network(
    device: torch.device | None,
    model_type: str,
    modelconfig: ModelConfig,
) -> NETWORK:
    if device is None:
        raise TypeError("device is not to be None")

    if ModelType(model_type) == ModelType.PredRecon:
        return PredPriorRecon(device=device, modelconfig=modelconfig)
    else:
        raise KeyError("model type not matched")


def get_optim(
    network: NETWORK | None,
    optimizer: str,
) -> OPTIM | None:
    if network is None:
        return None
    if optimizer == "adam":
        return Adam(network.parameters(), betas=(0.9, 0.99))
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
            "model_config": asdict(modelconfig),
            "config": asdict(config),
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


def mask_reg(
    mask: torch.Tensor,
) -> torch.Tensor:
    if mask.dim() != 4:
        raise NotImplementedError("mask has to be 3D")

    half_len = mask.size(-2) // 2
    mask_half = mask[:, :, :, :half_len]
    diff = mask_half[:, :, :, 1:] - mask_half[:, :, :, :-1]
    penalized_diff = torch.abs(torch.clamp(diff, max=0))
    return torch.mean(penalized_diff, dim=(1, 2, 3), keepdim=True)


def grad_norm(
    model: NETWORK,
) -> float:
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm**0.5


def loss_clip(
    loss: Tensor,
    max_loss: float = 1e-3,
) -> Tensor:
    if loss.mean() > max_loss:
        loss = loss * 0.25
    return loss


def train_epoch_predrecon(
    _data: dict[DataKey, Tensor | str],
    network: NETWORK,
    epoch: int,
    train_state: MetricController,
) -> int:
    loss_func = get_loss_func(config.loss_model)

    prior: Tensor = _data[DataKey.Prior].to(config.device)

    img_cnt_minibatch = prior.shape[0]

    (
        pred,
        label,
        mask_prob,
    ) = train_forward(
        network=network,
        prior=prior,
        target=_data[DataKey.Target].to(config.device),
        time=_data[DataKey.Time].to(config.device),
    )

    loss_recon = torch.mean(loss_func(pred, label), dim=(1, 2, 3), keepdim=True)
    train_state.add("loss_recon", loss_recon)

    torch.mean(loss_clip(loss_recon)).backward()

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
        zero_optimizers(optim_list=optim_list)
        if ModelType.from_string(config.model_type) == ModelType.PredRecon:
            img_cnt_minibatch = train_epoch_predrecon(
                _data=_data,
                network=network,
                epoch=epoch,
                train_state=train_state,
            )
        else:
            raise KeyError("model type not matched")

        step_optimizers(optim_list=optim_list)
        img_cnt += img_cnt_minibatch
        if img_cnt > (train_len / config.logging_density * logging_cnt):
            log_summary(init_time=config.init_time, state=train_state)
            logging_cnt += 1

    log_summary(init_time=config.init_time, state=train_state)


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


def update_metrics(
    test_state: MetricController,
    output_abs: Tensor,
    target_abs: Tensor,
    target_mask: Tensor,
) -> None:
    for i in range(output_abs.shape[0]):
        output_abs_i = output_abs[i : i + 1, :, :, :]
        target_abs_i = target_abs[i : i + 1, :, :, :]
        target_mask_i = target_mask[i : i + 1, :, :, :]
        mask_sum = torch.sum(target_mask_i, dim=(1, 2, 3), keepdim=True)
        if not torch.any(mask_sum == 0):
            test_state.add("mse", calculate_mse(output_abs_i, target_abs_i, target_mask_i))


def test_part_predrecon(
    _data: dict[DataKey, Tensor | str],
    test_dir: Path,
    model: NETWORK,
    save_val: bool,
    test_state: MetricController,
    img_cnt: int,
) -> int:
    loss_func = get_loss_func(config.loss_model)

    prior: Tensor = _data[DataKey.Prior].to(config.device)
    target: Tensor = _data[DataKey.Target].to(config.device)
    target_mask: Tensor = _data[DataKey.Mask].to(config.device)

    batch_cnt = prior.shape[0]

    (
        output,
        label_undersample,
        mask,
        mask_prob,
    ) = valid_recon(
        network=model,
        prior=prior,
        target=target,
    )

    validate_tensor_dimensions([output], 4)  # [B, C, H, W]
    validate_tensor_dimensions([target], 5)  # [B, Z, H, W, C]

    c_middle = target.shape[1] // 2
    c_middle = 0

    output_abs = torch.sqrt(output[:, 0, :, :] ** 2 + output[:, 1, :, :] ** 2).unsqueeze(1)  # [B, 1, H, W]
    target_slice = target[:, c_middle : c_middle + 1, :, :, :]  # [B, 1, H, W, C]
    target_abs = torch.sqrt(target_slice[:, :, :, :, 0] ** 2 + target_slice[:, :, :, :, 1] ** 2)  # [B, 1, H, W]

    loss = loss_func(output_abs, target_abs)
    loss = torch.mean(loss, dim=(1, 2, 3), keepdim=True)
    test_state.add("loss_recon", loss)

    update_metrics(
        test_state=test_state,
        output_abs=output_abs,
        target_abs=target_abs,
        target_mask=target_mask,
    )

    prior = prior[:, c_middle : c_middle + 1, :, :]
    target = target[:, c_middle : c_middle + 1, :, :, :]

    if save_val:
        save_result_to_npy(
            test_dir=test_dir,
            batch_cnt=batch_cnt,
            tesner_dict={
                "out": output,
                "target": target,
                "prior": prior,
                # "label_undersample": label_undersample,
                # "mask": mask,
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
        if ModelType.from_string(config.model_type) == ModelType.PredRecon:
            batch_cnt = test_part_predrecon(
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
