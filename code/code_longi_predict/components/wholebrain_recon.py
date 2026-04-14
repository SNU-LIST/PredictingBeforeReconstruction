import torch
from torch import Tensor

from common.utils import validate_tensor_dimensions, validate_tensors
from model.prednet import PredNet
from model.prednet_helper import validate_part


def prediction_wholebrain(
    prednet: PredNet,
    img: Tensor,
    img_prev: Tensor,
    mask: Tensor,
    meta: Tensor,
    batch_size: int,
) -> Tensor:
    validate_tensors([img, img_prev, mask, meta])
    validate_tensor_dimensions([img, img_prev, mask], 4)

    B, Z, _H, _W = img.shape
    if B != 1:
        raise ValueError("Batch size must be 1 for whole brain reconstruction.")

    slice_num = 1
    c_middle = 0

    output = torch.zeros_like(img)

    for i in range(0, Z, batch_size):
        actual_batch = min(batch_size, Z - i)
        idx_list = [i + j for j in range(actual_batch)]

        # [actual_batch, slice_num, H, W]
        img_prev_batch = torch.stack([img_prev[:, j : j + slice_num, :, :].squeeze(0) for j in idx_list])
        img_batch = torch.stack([img[:, j : j + slice_num, :, :].squeeze(0) for j in idx_list])
        mask_batch = torch.stack([mask[:, j : j + slice_num, :, :].squeeze(0) for j in idx_list])

        # Inference
        output_batch = validate_part(
            network=prednet,
            img_prev=img_prev_batch,  # [B', slice_num, H, W]
            meta=meta.expand(actual_batch, -1),  # expand meta for batch
        )  # output: [B', 1, H, W]

        # Normalize using mask
        img_center = img_batch[:, c_middle : c_middle + 1, :, :]
        mask_center = mask_batch[:, c_middle : c_middle + 1, :, :]

        for b in range(actual_batch):
            if torch.any(mask_center[b, ...]):
                scale = img_center[b, ...][mask_center[b, ...] == 1].mean() / output_batch[b, ...][mask_center[b, ...] == 1].mean()
                output[0, i + b : i + b + 1] = output_batch[b, ...] * scale
            else:
                output[0, i + b : i + b + 1] = output_batch[b, ...]

    return output
