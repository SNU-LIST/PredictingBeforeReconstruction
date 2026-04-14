import torch
from torch import Tensor

from common.utils import validate_tensor_dimensions, validate_tensors
from model.prednet import PredNet
from model.prednet_helper import validate_part


def prediction_wholebrain(
    prednet: PredNet,
    img_fl: Tensor,
    img_t1: Tensor,
    img_t2: Tensor,
    mask_f: Tensor,
    meta: Tensor,
    batch_size: int,
) -> Tensor:
    validate_tensors([img_fl, img_t1, img_t2, mask_f, meta])
    validate_tensor_dimensions([img_fl, img_t1, img_t2, mask_f], 4)

    B, Z, _H, _W = img_fl.shape
    if B != 1:
        raise ValueError("Batch size must be 1 for whole brain reconstruction.")

    slice_num = 1
    c_middle = 0

    output = torch.zeros_like(img_fl)

    for i in range(0, Z, batch_size):
        actual_batch = min(batch_size, Z - i)
        idx_list = [i + j for j in range(actual_batch)]

        # [actual_batch, slice_num, H, W]
        t1_batch = torch.stack([img_t1[:, j : j + slice_num, :, :].squeeze(0) for j in idx_list])
        t2_batch = torch.stack([img_t2[:, j : j + slice_num, :, :].squeeze(0) for j in idx_list])
        fl_batch = torch.stack([img_fl[:, j : j + slice_num, :, :].squeeze(0) for j in idx_list])
        mask_batch = torch.stack([mask_f[:, j : j + slice_num, :, :].squeeze(0) for j in idx_list])

        # Inference
        output_batch = validate_part(
            network=prednet,
            img_t1=t1_batch,  # [B', slice_num, H, W]
            img_t2=t2_batch,
            meta=meta.expand(actual_batch, -1),  # expand meta for batch
        )  # output: [B', 1, H, W]

        # Normalize using mask
        fl_center = fl_batch[:, c_middle : c_middle + 1, :, :]
        mask_center = mask_batch[:, c_middle : c_middle + 1, :, :]

        for b in range(actual_batch):
            if torch.any(mask_center[b, ...]):
                scale = fl_center[b, ...][mask_center[b, ...] == 1].mean() / output_batch[b, ...][mask_center[b, ...] == 1].mean()
                output[0, i + b : i + b + 1] = output_batch[b, ...] * scale
            else:
                output[0, i + b : i + b + 1] = output_batch[b, ...]

    return output
