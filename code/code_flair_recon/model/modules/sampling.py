from functools import lru_cache

import torch
from torch import Tensor


def ifft2c(img_k: Tensor) -> Tensor:
    img = torch.fft.ifftn(torch.fft.ifftshift(img_k, dim=(-2, -1)), dim=(-2, -1))
    return img


def fft2c(img: Tensor) -> Tensor:
    img_k = torch.fft.fftshift(torch.fft.fftn(img, dim=(-2, -1)), dim=(-2, -1))
    return img_k


@lru_cache
def gen_fixed_mask(
    B: int,
    H: int,
    W: int,
    acs_half: int,
    step: int,
    device: torch.device,
) -> Tensor:

    mask = torch.zeros([B, 1, H, W], dtype=torch.complex64)
    cen = mask.shape[3] // 2
    mask[:, :, cen - acs_half : cen + acs_half, :] = 1

    right_area = mask[..., cen + acs_half :]
    right_idx = torch.arange(0, right_area.shape[-1], step, device=device)
    mask[:, :, cen + acs_half + right_idx, :] = 1

    left_area_len = cen - acs_half
    left_idx = torch.arange(left_area_len - 1, -1, -step, device=device)
    mask[:, :, left_idx, :] = 1

    mask = mask.to(device)

    return mask


def apply_fixed_mask(
    img: Tensor,
    acs_num: int,
    parallel_factor: int,
) -> tuple[
    Tensor,
    Tensor,
    None,
]:

    acs_half = acs_num // 2
    img_k = fft2c(img)
    B, C, H, W = img.shape

    mask = gen_fixed_mask(
        B=B,
        H=H,
        W=W,
        acs_half=acs_half,
        step=parallel_factor,
        device=img.device,
    )

    output = ifft2c(img_k * mask)

    mask = mask.type(torch.float32)

    return (
        output,
        mask.type(torch.float32),
        None,
    )
