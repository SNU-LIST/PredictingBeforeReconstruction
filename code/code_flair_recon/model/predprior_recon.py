import torch
from torch import Tensor

from common.utils import validate_tensor_dimensions, validate_tensors
from model.backbone.tunet import TimeUnet
from model.modules.sampling import apply_fixed_mask
from params import ModelConfig, config, parse_prior


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


class PredPriorRecon(torch.nn.Module):
    num_timesteps: int
    device: torch.device

    acs_num: int
    parallel_factor: int

    recon_net: TimeUnet

    def __init__(
        self,
        device: torch.device,
        modelconfig: ModelConfig,
    ) -> None:
        super().__init__()
        self.num_timesteps: int = 1000
        self.device: torch.device = device

        self.acs_num: int = modelconfig.acs_num
        self.parallel_factor: int = modelconfig.parallel_factor

        self.recon_net = TimeUnet(
            input_number=(len(parse_prior(config.prior_key).split(","))),
            input_depth=1,
            chans=modelconfig.recon_net_chan,
            num_pool_layers=modelconfig.recon_net_pool,
            time_emb_dim=128,
        )

    def _run_recon_net(
        self,
        input: tuple[Tensor, Tensor, Tensor],
        t: Tensor,
    ) -> Tensor:
        for _input in input:
            validate_tensors([_input])

        pred = self.recon_net.forward(
            x=input,
            t=t.type(torch.float32),
        )
        return pred

    def _undersample_img(
        self,
        img: Tensor,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor | None,
    ]:
        validate_tensors([img])
        validate_tensor_dimensions([img], 5)  # [B, Z, H, W, C]
        B, _, H, _, _ = img.shape
        img_complex = tensor_5d_2_complex(img)

        img_undersample, mask, mask_prob = apply_fixed_mask(
            img=img_complex,
            acs_num=self.acs_num,
            parallel_factor=self.parallel_factor,
        )

        img_undersample = tensor_complex_2_5d(img_undersample)
        validate_tensor_dimensions([img_undersample], 5)  # [B, Z, H, W, C]

        return (
            img_undersample,
            mask.clone().detach(),
            mask_prob,
        )

    def forward(
        self,
        label_undersample: Tensor | None,
        prior: Tensor | None,
        x_t: Tensor | None,
        t_n: Tensor | None,
        mask: Tensor | None,
    ) -> Tensor:
        """
        Args:
            prior: [B, Z, H, W]
            label_undersample: [B, 1, H, W, C]
            x_t: [B, C, H, W]
            t_n: [B]
            label: [B, C, H, W]
            mask: [B, 1, H, W]
        """

        validate_tensors([x_t, label_undersample, prior, t_n, mask])
        prior_cond = prior
        label_cond = label_undersample.transpose(1, 4).squeeze(4)  # [B, C, H, W]
        pred = self._run_recon_net(
            input=[
                x_t,
                label_cond,
                prior_cond,
            ],
            t=t_n,
        )

        return pred[:, : label_cond.shape[1], :, :]

    @torch.inference_mode()
    def undersample_img(
        self,
        img: Tensor,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor | None,
    ]:
        validate_tensors([img])
        validate_tensor_dimensions([img], 5)  # [B, Z, H, W, C]

        img_undersample, mask, mask_prob = self._undersample_img(
            img=img,
        )

        validate_tensor_dimensions([img_undersample], 5)  # [B, Z, H, W, C]

        return (
            img_undersample,
            mask,
            mask_prob,
        )
