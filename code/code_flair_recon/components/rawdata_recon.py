from functools import lru_cache

import torch
from torch import Tensor

from common.logger import logger
from common.utils import validate_tensor_dimensions, validate_tensors
from model.predprior_recon import PredPriorRecon


def tensor_5d_2_complex(
    img: Tensor,
) -> Tensor:
    validate_tensor_dimensions([img], 5)  # [B, Z, H, W, C]
    if img.shape[-1] != 2:
        raise ValueError(f"Expected last dimension to be 2, got {img.shape[-1]} instead.")

    img_complex = img[..., 0] + 1j * img[..., 1]
    return img_complex.type(torch.complex64)


def tensor_complex_2_5d(
    img: Tensor,
) -> Tensor:
    validate_tensor_dimensions([img], 4)  # [B, Z, H, W]

    img_real = img.real.unsqueeze(-1)
    img_imag = img.imag.unsqueeze(-1)
    img_5d = torch.cat([img_real, img_imag], dim=-1)
    return img_5d.type(torch.float32)


def run_undersample(
    long_recon: PredPriorRecon,
    target: Tensor,
    slice_num: int,
    z_middle: int,
    B: int,
    Z: int,
) -> tuple[
    Tensor,
    Tensor,
    Tensor | None,
]:
    target_pad = torch.nn.functional.pad(target, (0, 0, 0, 0, 0, 0, z_middle, z_middle), mode="constant", value=0)

    # Under-sampling
    target_undersample = torch.zeros_like(target)  # [B, Z, H, W, C]

    for i in range(Z):
        target_slice = target_pad[:, i : i + slice_num, :, :, :]
        (
            target_undersample_slice,
            mask,
            mask_prob,
        ) = long_recon.undersample_img(
            target_slice,
        )
        target_undersample[:, i, :, :, :] = target_undersample_slice[:, z_middle : z_middle + 1, :, :, :]

    return (
        target_undersample,
        mask,
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


def run_recon_net(
    long_recon: PredPriorRecon,
    target: Tensor,
    prior_conds: Tensor,
    target_undersample_pad: Tensor,
    mask: Tensor,
    slice_num: int,
    Z: int,
    batch_size: int,
) -> Tensor:
    target_reconstruction = torch.zeros_like(target)
    if target_undersample_pad.shape[0] != 1:
        raise ValueError("Batch size must be 1 for whole brain reconstruction.")

    # Batch acceleration
    for i in range(0, Z, batch_size):
        actual_batch = min(batch_size, Z - i)
        idx_list = [i + j for j in range(actual_batch)]

        # Build batch
        # [B_infer, slice_num, H, W]
        # prior_batch = torch.stack([prior_cond_pad[:, j : j + slice_num, :, :].squeeze(0) for j in idx_list])
        prior_batch_list = []
        for prior_cond in prior_conds:
            _prior_batch = torch.stack([prior_cond[:, j : j + slice_num, :, :].squeeze(0) for j in idx_list])
            prior_batch_list.append(_prior_batch)
        prior_batch = torch.cat(prior_batch_list, dim=1)  # [B_infer, input_num * slice_num, H, W]

        # [B_infer, slice_num, H, W, C]
        target_us_batch = torch.stack([target_undersample_pad[:, j : j + slice_num, :, :, :].squeeze(0) for j in idx_list])

        z_middle = slice_num // 2
        label_batch = target_us_batch[:, z_middle : z_middle + 1, :, :, :]
        label_cond = label_batch.transpose(1, 4).squeeze(-1)  # [B_infer, C, H, W]

        mask_batch = mask.expand(actual_batch, -1, -1, -1)  # [B_infer, 1, H, W]

        sig = 0.9
        # logger.info(f"Sigma: {sig}")
        recon_batch = flow_reverse(
            network=long_recon,
            label_undersample=label_batch,
            label_cond=label_cond,
            prior_cond=prior_batch,
            mask=mask_batch,
            step=25,
            sigma=sig,
        )  # [B_infer, C, H, W]

        for j, z_idx in enumerate(idx_list):
            target_reconstruction[:, z_idx : z_idx + 1, :, :, :] = recon_batch[j : j + 1, ...].unsqueeze(-1).transpose(1, 4)

    return target_reconstruction


def longitudinal_recon_wholebrain(
    long_recon: PredPriorRecon,
    prior: Tensor,
    target: Tensor,
    batch_size: int,
) -> tuple[
    Tensor,
    Tensor,
    Tensor | None,
]:
    """
    Args:
        long_recon (PredPriorRecon): Longitudinal reconstruction model.
        prior (Tensor): Prior image tensor of shape [B, Z, H, W].
        target (Tensor): Target image tensor of shape [B, Z, H, W, C].
        meta (Tensor): Meta information tensor of shape [B, A].
    """
    validate_tensors([prior, target])
    validate_tensor_dimensions([prior], 4)  # [B, Z, H, W]
    validate_tensor_dimensions([target], 5)  # [B, Z, H, W, C]

    B, Z, _, _, _ = target.shape
    if B != 1:
        raise ValueError("Batch size must be 1 for whole brain reconstruction.")

    slice_num = 1
    z_middle = slice_num // 2

    # Under-sampling
    (
        target_undersample,
        mask,
        mask_prob,
    ) = run_undersample(
        long_recon=long_recon,
        target=target,
        slice_num=slice_num,
        z_middle=z_middle,
        B=B,
        Z=Z,
    )

    # prior_cond = prior_registration if using_registration else prior
    # prior_cond = prior
    # prior_cond_pad = torch.nn.functional.pad(prior_cond, (0, 0, 0, 0, z_middle, z_middle), mode="constant", value=0)
    prior_conds = []
    for i in range(prior.shape[1] // Z):
        _prior = prior[:, i * Z : (i + 1) * Z, :, :]
        _prior_cond_pad = torch.nn.functional.pad(_prior, (0, 0, 0, 0, z_middle, z_middle), mode="constant", value=0)
        prior_conds.append(_prior_cond_pad)
    target_undersample_pad = torch.nn.functional.pad(target_undersample, (0, 0, 0, 0, 0, 0, z_middle, z_middle), mode="constant", value=0)

    # Reconstruction
    target_reconstruction = run_recon_net(
        long_recon=long_recon,
        target=target,
        prior_conds=prior_conds,
        target_undersample_pad=target_undersample_pad,
        mask=mask,
        slice_num=slice_num,
        Z=Z,
        batch_size=batch_size,
    )

    return (
        target_reconstruction,
        mask,
        mask_prob,
    )
