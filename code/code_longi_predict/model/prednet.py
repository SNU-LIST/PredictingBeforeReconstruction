import torch
from torch import Tensor

from model.backbone.tunet import TimeUnet
from params import PredNetConfig


class PredNet(torch.nn.Module):
    num_timesteps: int
    device: torch.device
    input_num: int
    input_depth: int
    prednet: TimeUnet

    def __init__(
        self,
        device: torch.device,
        prednetconfig: PredNetConfig,
    ) -> None:
        super().__init__()
        self.num_timesteps: int = 1000
        self.device: torch.device = device

        self.prednet = TimeUnet(
            in_chans=2,
            out_chans=1,
            meta_dim=prednetconfig.meta_dim,
            chans=prednetconfig.recon_net_chan,
            num_pool_layers=prednetconfig.recon_net_pool,
            time_emb_dim=128,
        )

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        noise: Tensor,
        m: Tensor,
    ) -> Tensor:
        pred = self.prednet(
            x=x,
            t=t,
            m=m,
        )

        return noise - pred
