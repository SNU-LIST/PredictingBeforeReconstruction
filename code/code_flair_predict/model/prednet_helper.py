import torch
from torch import Tensor

from common.logger import logger
from common.utils import validate_tensor_dimensions, validate_tensors
from model.prednet import PredNet


def train_part(
    network: PredNet,
    img_fl: Tensor,
    img_t1: Tensor,
    img_t2: Tensor,
    meta: Tensor,
) -> tuple[
    Tensor,
    Tensor,
]:
    net = network.module if isinstance(network, torch.nn.DataParallel) else network

    num_timesteps = net.num_timesteps
    device = net.device

    validate_tensors([img_fl, img_t1, img_t2, meta])
    validate_tensor_dimensions([img_fl, img_t1, img_t2], 4)

    time = torch.randint(low=0, high=num_timesteps, size=(img_fl.shape[0], 1), device=device)

    B, C, H, W = img_fl.shape
    label = img_fl

    noise = torch.randn(B, 1, H, W).to(device)

    t = time.squeeze(1).long()
    t_n = t / num_timesteps

    x_t = ((1 - t_n).view(B, 1, 1, 1) * label + t_n.view(B, 1, 1, 1) * noise).type(torch.float32)

    x = [x_t, img_t1, img_t2]

    pred = network.forward(
        x=x,
        t=t_n,
        noise=noise,
        m=meta,
    )

    target = noise - label

    return (
        pred,
        target,
    )


def validate_part(
    network: PredNet,
    img_t1: Tensor,
    img_t2: Tensor,
    meta: Tensor,
    step: int = 25,
) -> Tensor:
    net = network.module if isinstance(network, torch.nn.DataParallel) else network

    num_timesteps = net.num_timesteps
    device = net.device

    validate_tensors([img_t1, img_t2, meta])
    validate_tensor_dimensions([img_t1, img_t2], 4)

    B, C, W, H = img_t1.shape

    times = torch.linspace(1000, 0, step + 1, dtype=torch.long)
    dts = (times[:-1] - times[1:]) / num_timesteps
    times = times[:-1]

    noise = torch.randn(B, 1, H, W).to(device)

    t = torch.tensor(num_timesteps, device=device, dtype=torch.long)
    t_n = t / num_timesteps
    t_n = t_n.repeat(B).view(B, 1, 1, 1)
    x_t = (t_n * noise).type(torch.float32)

    for dt, time in zip(dts, times, strict=True):
        logger.trace(f"Diffusion time : {time}")
        t_batch = (
            torch.full(
                size=(B,),
                fill_value=time.item() if isinstance(time, torch.Tensor) else time,
                device=device,
                dtype=torch.long,
            )
            / num_timesteps
        )

        x = [x_t, img_t1, img_t2]

        network_out = network.forward(
            x=x,
            t=t_batch,
            noise=noise,
            m=meta,
        )

        x_next = x_t - dt * network_out
        x_t = x_next

    return x_t
