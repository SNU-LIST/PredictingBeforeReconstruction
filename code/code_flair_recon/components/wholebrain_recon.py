import torch
from torch import Tensor

from common.utils import validate_tensor_dimensions, validate_tensors
from model.predprior_recon import PredPriorRecon
from model.predprior_recon_helper import flow_reverse


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


def run_recon_net(
    long_recon: PredPriorRecon,
    target: Tensor,
    prior_conds: list[Tensor],
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

        recon_batch = flow_reverse(
            network=long_recon,
            label_undersample=label_batch,
            label_cond=label_cond,
            prior_cond=prior_batch,
            mask=mask_batch,
            step=25,
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
        target_undersample,
        mask,
        mask_prob,
    )
