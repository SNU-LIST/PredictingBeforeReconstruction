import math
from enum import StrEnum
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional


class BlockType(StrEnum):
    BLOCK1 = "block1"
    BLOCK2 = "block2"
    BLOCK3 = "block3"

    @classmethod
    def from_string(cls, value: str) -> "BlockType":
        try:
            return cls(value)
        except ValueError as err:
            raise ValueError(f"Invalid BlockType value: {value}. Must be one of {list(cls)} : {err}") from err


def validate_tensors(tensors: list[Tensor]) -> None:
    for i, t in enumerate(tensors):
        if not isinstance(t, Tensor):
            raise TypeError(f"Tensor at index {i} is not a torch.Tensor, got {type(t)} instead.")


def validate_tensor_dimensions(tensors: list[Tensor], expected_dim: int) -> None:
    for i, t in enumerate(tensors):
        if t.dim() != expected_dim:
            raise ValueError(f"Tensor at index {i} has {t.dim()} dimensions, expected {expected_dim} dimensions.")


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

        self.linear = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(inplace=True),
            nn.Linear(dim, dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        half_dim = self.dim // 2
        emb = math.log(100) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        emb = self.linear(emb)

        return emb


class TimeConvAttentionBlock1(nn.Module):
    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        time_emb_dim: int,
        head_dim: int = 32,
    ):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_chans),
            nn.SiLU(inplace=True),
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, out_chans * 2),
            nn.SiLU(inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_chans),
            nn.SiLU(inplace=True),
        )

        self.num_heads = out_chans // head_dim
        self.scale = head_dim**-0.5
        self.qkv = nn.Conv2d(out_chans, out_chans * 3, kernel_size=1)
        self.proj = nn.Conv2d(out_chans, out_chans, kernel_size=1)
        self.attention_norm = nn.GroupNorm(4, out_chans)

        self.layer3 = nn.Sequential(
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_chans),
            nn.SiLU(inplace=True),
        )

    def shifted_window_attention(
        self,
        x: torch.Tensor,
        window_size: int = 8,
    ) -> torch.Tensor:
        B, C, H, W = x.shape

        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size

        x = functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        shift_size = window_size // 2
        x_shifted = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(2, 3))

        x1 = self._apply_window_attention(x, window_size)
        x2 = self._apply_window_attention(x_shifted, window_size)

        x2 = torch.roll(x2, shifts=(shift_size, shift_size), dims=(2, 3))

        x = (x1 + x2) * 0.5
        x = x[:, :, :H, :W]
        return x

    def _apply_window_attention(
        self,
        x: torch.Tensor,
        window_size: int,
    ) -> torch.Tensor:
        B, C, H, W = x.shape

        x = x.unfold(2, window_size, window_size).unfold(3, window_size, window_size)
        x = x.contiguous().view(B, C, -1, window_size, window_size)
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        num_windows = x.shape[1]
        x = x.view(-1, C, window_size, window_size)

        qkv = self.qkv(x).reshape(-1, 3, self.num_heads, C // self.num_heads, window_size * window_size)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).reshape(-1, C, window_size, window_size)

        x = self.proj(x)

        x = x.view(B, num_windows, C, window_size, window_size)
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        x = x.view(B, C, H, W)
        return x

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> torch.Tensor:

        x = self.layer1(x)

        t_emb = self.time_mlp(t_emb)[:, :, None, None]
        shift, bias = t_emb.chunk(2, dim=1)
        x = x * (1 + shift) + bias

        x = self.layer2(x)

        residual = x
        x = self.attention_norm(x)
        x = self.shifted_window_attention(x)
        x = residual + x

        x = self.layer3(x)

        return x


class TimeConvAttentionBlock2(nn.Module):
    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        time_emb_dim: int,
    ) -> None:
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_chans),
            nn.SiLU(inplace=True),
        )

        self.meta_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(inplace=True),
            nn.Linear(time_emb_dim, out_chans * 2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_chans),
            nn.SiLU(inplace=True),
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(inplace=True),
            nn.Linear(time_emb_dim, out_chans * 2),
        )

    def forward(
        self,
        x: Tensor,
        t_emb: Tensor,
        m_emb: Tensor,
    ) -> Tensor:
        x = self.layer1(x)

        t_emb = self.time_mlp(t_emb)[:, :, None, None]
        shift, bias = t_emb.chunk(2, dim=1)
        x = x * (1 + shift) + bias

        x = self.layer2(x)

        m_emb = self.meta_mlp(m_emb)[:, :, None, None]
        shift, bias = m_emb.chunk(2, dim=1)
        x = x * (1 + shift) + bias

        return x


class TimeConvAttentionBlock3(nn.Module):
    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        time_emb_dim: int,
    ) -> None:
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_chans),
            nn.SiLU(inplace=True),
        )

        self.meta_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(inplace=True),
            nn.Linear(time_emb_dim, out_chans * 2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_chans),
            nn.SiLU(inplace=True),
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(inplace=True),
            nn.Linear(time_emb_dim, out_chans * 2),
        )

        self.se_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_chans, out_chans // 4, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_chans // 4, out_chans, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: Tensor,
        t_emb: Tensor,
        m_emb: Tensor,
    ) -> Tensor:
        x = self.layer1(x)

        t_emb = self.time_mlp(t_emb)[:, :, None, None]
        shift, bias = t_emb.chunk(2, dim=1)
        x = x * (1 + shift) + bias

        x = self.layer2(x)

        se_weight = self.se_block(x)
        x = x * se_weight

        m_emb = self.meta_mlp(m_emb)[:, :, None, None]
        shift, bias = m_emb.chunk(2, dim=1)
        x = x * (1 + shift) + bias

        return x


def get_time_conv_attention_block(
    block_type: BlockType,
    in_chans: int,
    out_chans: int,
    time_emb_dim: int,
) -> nn.Module:
    if block_type == BlockType.BLOCK1:
        return TimeConvAttentionBlock1(
            in_chans=in_chans,
            out_chans=out_chans,
            time_emb_dim=time_emb_dim,
        )
    elif block_type == BlockType.BLOCK2:
        return TimeConvAttentionBlock2(
            in_chans=in_chans,
            out_chans=out_chans,
            time_emb_dim=time_emb_dim,
        )
    elif block_type == BlockType.BLOCK3:
        return TimeConvAttentionBlock3(
            in_chans=in_chans,
            out_chans=out_chans,
            time_emb_dim=time_emb_dim,
        )
    else:
        raise ValueError(f"Unknown block type: {block_type}")


class FirstLayer(nn.Module):
    def __init__(
        self,
        in_chans: int,
        chans: int,
        time_emb_dim: int,
        block_type: BlockType = BlockType.BLOCK2,
    ) -> None:
        super().__init__()
        self.in_chans = in_chans

        self.expand = get_time_conv_attention_block(
            block_type=block_type,
            in_chans=2,
            out_chans=chans,
            time_emb_dim=time_emb_dim,
        )

    def forward(
        self,
        x: tuple[Tensor, ...],
        t_emb: Tensor,
        m_emb: Tensor,
    ) -> Tensor:
        assert len(x) == self.in_chans, f"Expected {self.in_chans} input tensors, got {len(x)}"

        x_cat = torch.cat(x, dim=1)
        output = self.expand(x_cat, t_emb, m_emb)

        return output


class FinalLayer(nn.Module):
    def __init__(
        self,
        feature_chans: int,
        out_chans: int,
        time_emb_dim: int,
    ):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(feature_chans, feature_chans, kernel_size=3, padding=1),
            nn.GroupNorm(4, feature_chans),
            nn.SiLU(inplace=True),
        )

        self.meta_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(inplace=True),
            nn.Linear(time_emb_dim, out_chans * 2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(feature_chans, feature_chans, kernel_size=3, padding=1),
            nn.GroupNorm(4, feature_chans),
            nn.SiLU(inplace=True),
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(inplace=True),
            nn.Linear(time_emb_dim, out_chans * 2),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(feature_chans, feature_chans, kernel_size=3, padding=1),
            nn.GroupNorm(4, feature_chans),
            nn.SiLU(inplace=True),
            nn.Conv2d(feature_chans, out_chans, kernel_size=1, padding=0),
        )

    def forward(
        self,
        output: Tensor,
        t_emb: Tensor,
        m_emb: Tensor,
    ) -> Tensor:
        x = self.layer1(output)

        t_emb = self.time_mlp(t_emb)[:, :, None, None]
        shift, bias = t_emb.chunk(2, dim=1)
        x = x * (1 + shift) + bias

        x = self.layer2(x)

        m_emb = self.meta_mlp(m_emb)[:, :, None, None]
        shift, bias = m_emb.chunk(2, dim=1)
        x = x * (1 + shift) + bias

        x = self.layer3(x)

        return x


class TimeUnet(nn.Module):
    blochk_type: BlockType
    time_mlp: TimeEmbedding
    meta_mlp: nn.Sequential
    first_layer: FirstLayer
    down_pool_layers: nn.ModuleList
    down_layers: nn.ModuleList
    bottleneck_conv: nn.Module
    up_conv_layers: nn.ModuleList
    up_layers: nn.ModuleList
    final_conv: FinalLayer

    def __init__(
        self,
        in_chans: int = 2,
        out_chans: int = 1,
        meta_dim: int = 5,
        chans: int = 64,
        num_pool_layers: int = 4,
        time_emb_dim: int = 256,
        block_type: Literal["block1", "block2", "block3"] = "block2",
    ):
        super().__init__()
        block_type = BlockType.from_string(block_type)

        self.time_mlp = TimeEmbedding(time_emb_dim)

        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_dim, time_emb_dim),
            nn.SiLU(inplace=True),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(inplace=True),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.first_layer = FirstLayer(
            in_chans=in_chans,
            chans=chans,
            time_emb_dim=time_emb_dim,
            block_type=block_type,
        )

        self.down_pool_layers = self.create_down_pool_layers(
            chans=chans,
            num_pool_layers=num_pool_layers,
        )

        self.down_layers = self.create_down_layers(
            in_chans=in_chans,
            chans=chans,
            num_pool_layers=num_pool_layers,
            time_emb_dim=time_emb_dim,
            block_type=block_type,
        )

        self.bottleneck_conv = get_time_conv_attention_block(
            block_type=block_type,
            in_chans=chans * (2 ** (num_pool_layers - 1)),
            out_chans=chans * (2 ** (num_pool_layers - 1)),
            time_emb_dim=time_emb_dim,
        )

        self.up_conv_layers = self.create_up_conv_layers(
            chans=chans,
            num_pool_layers=num_pool_layers,
        )

        self.up_layers = self.create_up_layers(
            in_chans=in_chans,
            chans=chans,
            num_pool_layers=num_pool_layers,
            time_emb_dim=time_emb_dim,
            block_type=block_type,
        )

        self.final_conv = FinalLayer(
            feature_chans=chans,
            out_chans=out_chans,
            time_emb_dim=time_emb_dim,
        )

    def create_down_pool_layers(
        self,
        chans: int,
        num_pool_layers: int,
    ):
        layers = nn.ModuleList()
        ch = chans
        for _ in range(num_pool_layers - 1):
            layers.append(nn.Conv2d(ch * 2, ch * 2, kernel_size=4, stride=2, padding=1))
            ch *= 2
        return layers

    def create_down_layers(
        self,
        in_chans: int,
        chans: int,
        num_pool_layers: int,
        time_emb_dim: bool,
        block_type: BlockType = BlockType.BLOCK1,
    ):
        layers = nn.ModuleList([])
        ch = chans
        layers.append(
            get_time_conv_attention_block(
                block_type=block_type,
                in_chans=ch,
                out_chans=ch * 2,
                time_emb_dim=time_emb_dim,
            )
        )
        ch *= 2
        for _ in range(num_pool_layers - 2):
            layers.append(
                get_time_conv_attention_block(
                    block_type=block_type,
                    in_chans=ch,
                    out_chans=ch * 2,
                    time_emb_dim=time_emb_dim,
                )
            )
            ch *= 2
        return layers

    def create_up_conv_layers(
        self,
        chans: int,
        num_pool_layers: int,
    ):
        layers = nn.ModuleList()
        ch = chans * (2 ** (num_pool_layers - 1))
        for _ in range(num_pool_layers - 1):
            layers.append(nn.ConvTranspose2d(ch, ch, kernel_size=2, stride=2))
            ch //= 2
        layers.append(nn.Identity())
        return layers

    def create_up_layers(
        self,
        in_chans: int,
        chans: int,
        num_pool_layers: int,
        time_emb_dim: bool,
        block_type: BlockType = BlockType.BLOCK1,
    ):
        layers = nn.ModuleList()
        ch = chans * (2 ** (num_pool_layers - 1))
        for _ in range(num_pool_layers - 1):
            layers.append(
                get_time_conv_attention_block(
                    block_type=block_type,
                    in_chans=ch * 2,
                    out_chans=ch // 2,
                    time_emb_dim=time_emb_dim,
                )
            )
            ch //= 2
        layers.append(
            get_time_conv_attention_block(
                block_type=block_type,
                in_chans=ch * 2,
                out_chans=ch,
                time_emb_dim=time_emb_dim,
            )
        )
        return layers

    def forward(
        self,
        x: tuple[Tensor, ...],
        t: Tensor,
        m: Tensor,
    ) -> Tensor:
        validate_tensors([t, m])
        validate_tensor_dimensions([m], 2)

        t_emb = self.time_mlp(t)
        m_emb = self.meta_mlp(m)

        stack = []
        output = self.first_layer(x, t_emb, m_emb)
        stack.append(output)

        for down_pool, layer in zip(self.down_pool_layers, self.down_layers, strict=False):
            output = layer(output, t_emb, m_emb)
            stack.append(output)
            output = down_pool(output)

        output = self.bottleneck_conv(output, t_emb, m_emb)

        for up_conv, layer in zip(self.up_conv_layers, self.up_layers, strict=False):
            downsampled_output = stack.pop()
            output = up_conv(output)

            B, C, W, H = downsampled_output.shape
            output = output[:, :, :W, :H]

            output = torch.cat([output, downsampled_output], dim=1)
            output = layer(output, t_emb, m_emb)

        output = self.final_conv(output, t_emb, m_emb)

        return output
