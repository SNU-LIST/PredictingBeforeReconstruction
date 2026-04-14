import glob
from dataclasses import dataclass
from enum import IntEnum

import torch
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset

from params import parse_prior

prob_flip: float = 0.5
scale_fac: float = 0.05


class DataKey(IntEnum):
    Prior = 0
    Target = 1
    Mask = 2
    Time = 3
    Name = 4


@dataclass
class LoaderConfig:
    data_type: str
    batch: int
    num_workers: int
    shuffle: bool
    debug_mode: bool
    rotation_conf: str
    prior_key: str
    target_key: str
    target_mask_key: str


class DataWrapper(Dataset):
    num_timesteps: int
    file_list: list[str]
    training_mode: bool
    prior_key: str
    target_key: str
    target_mask_key: str

    def __init__(
        self,
        file_path: list[str],
        training_mode: bool,
        loader_cfg: LoaderConfig,
    ) -> None:
        super().__init__()

        self.num_timesteps: int = 1000
        self.training_mode = training_mode
        self.prior_key = parse_prior(loader_cfg.prior_key)
        self.prior_keys = self.prior_key.split(",")
        self.target_key = loader_cfg.target_key
        self.target_mask_key = loader_cfg.target_mask_key

        # Initialize the dataset
        self._collect_file_list(
            file_path=file_path,
            cfg=loader_cfg,
        )

    def _collect_file_list(
        self,
        file_path: list[str],
        cfg: LoaderConfig,
    ) -> None:
        total_list: list[str] = []
        for _file_path in file_path:
            total_list += glob.glob(f"{_file_path}/{cfg.data_type}")
        self.file_list = total_list

        if cfg.debug_mode:
            if self.training_mode:
                self.file_list = self.file_list[:500]
            else:
                self.file_list = self.file_list[:100]

        else:
            if self.training_mode:
                self.file_list = self.file_list
            else:
                self.file_list = self.file_list

        self.file_list = sorted(self.file_list)

    def __getitem__(
        self,
        idx: int,
    ):
        file_mat = loadmat(self.file_list[idx])

        target = torch.from_numpy(file_mat[self.target_key]).type(torch.complex64)
        mask = torch.from_numpy(file_mat[self.target_mask_key]).type(torch.float)

        prior_list: list[torch.Tensor] = []
        for pk in self.prior_keys:
            _prior = torch.from_numpy(file_mat[pk]).type(torch.float32) if pk in file_mat else torch.zeros_like(target, dtype=torch.float32)
            # _prior = torch.cat([_prior.real, _prior.imag], dim=0).type(torch.float32)
            prior_list.append(_prior)
        prior = torch.cat(prior_list, dim=0)

        target = torch.stack([target.real, target.imag], dim=-1).type(torch.float32)

        time = torch.randint(low=0, high=self.num_timesteps, size=[1])

        _name = self.file_list[idx].split("/")[-1]

        return (
            prior,
            target,
            mask,
            time,
            _name,
        )

    def __len__(self) -> int:
        return len(self.file_list)


def get_data_wrapper_loader(
    file_path: list[str],
    training_mode: bool,
    loader_cfg: LoaderConfig,
) -> tuple[
    DataLoader,
    DataWrapper,
    int,
]:
    dataset = DataWrapper(
        file_path=file_path,
        training_mode=training_mode,
        loader_cfg=loader_cfg,
    )

    _ = dataset[0]

    dataloader = DataLoader(
        dataset,
        batch_size=loader_cfg.batch,
        num_workers=loader_cfg.num_workers,
        pin_memory=False,
        persistent_workers=True,
        shuffle=loader_cfg.shuffle,
    )

    return (
        dataloader,
        dataset,
        len(dataset),
    )
