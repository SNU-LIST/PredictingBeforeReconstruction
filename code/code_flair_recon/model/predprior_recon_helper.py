from functools import lru_cache

import torch
from torch import Tensor

from common.utils import validate_tensor_channels, validate_tensor_dimensions, validate_tensors
from model.predprior_recon import PredPriorRecon


def train_forward(
    network: PredPriorRecon,
    prior: Tensor,
    target: Tensor,
    time: Tensor,
) -> tuple[
    Tensor,
    Tensor,
    Tensor | None,
]:
    """
    Args:
        prior: [B, Z, H, W]
        target: [B, Z, H, W, C]
        time: [B, 1]
    """
    net = network.module if isinstance(network, torch.nn.DataParallel) else network

    num_timesteps = net.num_timesteps
    device = net.device
    _undersample_img = net._undersample_img

    validate_tensors([prior, target, time])
    validate_tensor_dimensions([prior], 4)  # [B, Z, H, W]
    validate_tensor_dimensions([target], 5)  # [B, Z, H, W, C]

    B, Z, H, W, _ = target.shape

    label = target[:, 0:1, :, :, :]  # [B, 1, H, W, C]
    label = label.transpose(1, 4).squeeze(4)  # [B, C, H, W]
    noise = torch.randn_like(label).to(device)

    t = time.squeeze(1).long()
    t_n = t / num_timesteps

    x_t = ((1 - t_n).view(B, 1, 1, 1) * label + t_n.view(B, 1, 1, 1) * noise).type(torch.float32)

    (
        label_undersample,  # [B, 1, H, W, C]
        mask,
        mask_prob,
    ) = _undersample_img(
        img=target[:, 0:1, :, :, :],
    )

    prior_cond = prior

    pred = network(
        label_undersample=label_undersample,
        prior=prior_cond,
        x_t=x_t,
        t_n=t_n,
        mask=mask,
    )

    return (
        pred,
        noise - label,
        mask_prob,
    )


def ifft2c(img_k: Tensor) -> Tensor:
    img = torch.fft.ifftn(torch.fft.ifftshift(img_k, dim=(-2, -1)), dim=(-2, -1))
    return img


def fft2c(img: Tensor) -> Tensor:
    img_k = torch.fft.fftshift(torch.fft.fftn(img, dim=(-2, -1)), dim=(-2, -1))
    return img_k


def channl_2_complex(img: Tensor) -> Tensor:
    return (img[:, 0, :, :] + 1j * img[:, 1, :, :]).unsqueeze(1)


def complex_2_channel(img: Tensor) -> Tensor:
    return torch.stack([img.real, img.imag], dim=1).squeeze(2)


@lru_cache(maxsize=16)
def gauss_filter(
    length: int,
    img_dim: int,
    sigma: float = 0.99,
) -> Tensor:
    x = (torch.arange(length, dtype=torch.float32) - length // 2) / length
    gauss = torch.exp(-0.5 * (x / sigma) ** 2)
    gauss = gauss / gauss.max()
    dims_to_unsqueeze = max(0, img_dim - 2)
    for _ in range(dims_to_unsqueeze):
        gauss = gauss.unsqueeze(0)

    return gauss.unsqueeze(-1)


def apply_consistency(
    x0_pred: Tensor,
    label: Tensor,
    mask: Tensor,
    sigma: float,
) -> Tensor:
    label_complex = channl_2_complex(label)
    x0_pred_complex = channl_2_complex(x0_pred)
    con_k = fft2c(label_complex - x0_pred_complex) * mask
    filter = (
        gauss_filter(
            length=label.shape[-1],
            img_dim=label.dim(),
            sigma=sigma,
        )
        .to(con_k.device)
        .type(torch.complex64)
    )
    con_k = con_k * filter

    x0_pred_kspace_complex = fft2c(x0_pred_complex) + con_k

    x0_pred_complex = ifft2c(x0_pred_kspace_complex)
    x0_pred = complex_2_channel(x0_pred_complex)
    return x0_pred


def tensor_5d_2_complex(
    img: Tensor,
) -> Tensor:
    validate_tensor_dimensions([img], 5)  # [B, Z, H, W, C]
    if img.shape[-1] != 2:
        raise ValueError(f"Expected last dimension to be 2, got {img.shape[-1]} instead.")

    img_complex = img[..., 0] + 1j * img[..., 1]
    return img_complex.type(torch.complex64)


def flow_reverse(
    network: PredPriorRecon,
    label_undersample: Tensor,
    label_cond: Tensor,
    prior_cond: Tensor,
    mask: Tensor,
    step: int,
    sigma: float = 0.99,
) -> Tensor:
    """
    Args:
        label_cond: [B, C, H, W].
        prior_cond: [B, Z, H, W].
        mask: [B, 1, H, W].
    """
    net = network.module if isinstance(network, torch.nn.DataParallel) else network

    num_timesteps = net.num_timesteps
    device = net.device

    validate_tensors([label_cond, prior_cond, mask])
    validate_tensor_dimensions([label_cond, prior_cond, mask], 4)
    validate_tensor_channels(label_cond, 2)

    B, _, H, W = label_cond.shape

    times = torch.linspace(1000, 0, step + 1, dtype=torch.long)
    dts = (times[:-1] - times[1:]) / num_timesteps
    times = times[:-1]

    t = torch.tensor(num_timesteps, device=device, dtype=torch.long)
    noise = torch.randn_like(label_cond).to(device)

    t_n = t / num_timesteps
    t_n = t_n.repeat(B).view(B, 1, 1, 1)
    x_t = (t_n * noise).type(torch.float32)

    for dt, time_item in zip(dts, times, strict=True):
        t_batch = (
            torch.full(
                size=(B,),
                fill_value=time_item.item() if isinstance(time_item, torch.Tensor) else time_item,
                device=device,
                dtype=torch.long,
            )
            / num_timesteps
        )

        network_out = network(
            label_undersample=label_undersample,
            prior=prior_cond,
            x_t=x_t,
            t_n=t_batch,
            mask=mask,
        )

        x0_pred = x_t - (time_item / num_timesteps) * network_out

        pred = apply_consistency(x0_pred=x0_pred, label=label_cond, mask=mask, sigma=sigma)

        network_output = noise - pred
        network_output = network_output.type_as(x_t)

        validate_tensors([network_output])

        x_next = x_t - dt * network_output
        x_t = x_next

    x_t = apply_consistency(x0_pred=x_t, label=label_cond, mask=mask, sigma=sigma)
    return x_t.type(torch.float32)


def valid_recon(
    network: PredPriorRecon,
    prior: Tensor,
    target: Tensor,
    step: int = 25,
) -> tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor | None,
]:
    """
    Args:
        prior: [B, Z, H, W, C].
        prior_rot: [B, Z, H, W].
        target: [B, Z, H, W, C].
    """
    net = network.module if isinstance(network, torch.nn.DataParallel) else network

    undersample_img = net.undersample_img

    validate_tensors([prior, target])
    validate_tensor_dimensions([prior], 4)  # [B, Z, H, W]
    validate_tensor_dimensions([target], 5)  # [B, Z, H, W, C]

    (
        target_undersample,
        mask,
        mask_prob,
    ) = undersample_img(
        img=target,
    )

    z_middle = target.shape[1] // 2

    label_undersample = target_undersample[:, z_middle : z_middle + 1, :, :, :]  # [B, 1, H, W, C]
    label_cond = label_undersample.transpose(1, 4).squeeze(4)  # [B, C, H, W]

    prior_cond = prior

    x_t = flow_reverse(
        network=network,
        label_undersample=label_undersample,
        label_cond=label_cond,
        prior_cond=prior_cond,
        mask=mask,
        step=step,
    )

    return (
        x_t.type(torch.float32),
        label_undersample.type(torch.float32),
        mask,
        mask_prob,
    )
