import glob
from dataclasses import dataclass
from enum import IntEnum

import torch
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset

from params import parse_prior


class DataKey(IntEnum):
    Prior = 0
    Target = 1
    Mask = 2
    Time = 3
    Name = 4
    FL_BIASFIELD = 5


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
        self.file_list.sort()

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

    def __getitem__(
        self,
        idx: int,
    ):
        # Load the file
        file_mat = loadmat(self.file_list[idx])

        # Extract the target and mask
        target_raw_img = torch.from_numpy(file_mat[self.target_key + "_raw_img"]).type(torch.float32)
        target_raw_sen = torch.from_numpy(file_mat[self.target_key + "_raw_sen"]).type(torch.float32)
        target_raw_img_complex = target_raw_img[:, :, 0, :, :] + 1j * target_raw_img[:, :, 1, :, :]
        target_raw_sen_complex = target_raw_sen[:, :, 0, :, :] + 1j * target_raw_sen[:, :, 1, :, :]
        target = target_raw_img_complex * torch.conj(target_raw_sen_complex)
        target = target.sum(dim=1, keepdim=False)  # .abs().type(torch.complex64)

        # target = torch.abs(target).type(torch.complex64)

        mask = torch.from_numpy(file_mat[self.target_mask_key]).type(torch.float)
        # mask = mask[mask.shape[0] // 2 : mask.shape[0] // 2 + 1, :, :]

        # Extract the prior
        # if self.prior_key in file_mat:
        #     prior = torch.from_numpy(file_mat[self.prior_key]).type(torch.float)
        # else:
        #     prior = torch.zeros_like(target, dtype=torch.float)

        prior_list: list[torch.Tensor] = []
        for pk in self.prior_keys:
            _prior = torch.from_numpy(file_mat[pk]).type(torch.float) if pk in file_mat else torch.zeros_like(target, dtype=torch.float)
            # _prior = _prior / (_prior.mean() + 1e-8)
            prior_list.append(_prior)
        prior = torch.cat(prior_list, dim=0)

        # Convert the prior to complex
        target_phase = torch.angle(target)
        target = target * torch.exp(-1j * target_phase)
        target = torch.stack([target.real, target.imag], dim=-1).type(torch.float32)

        # Generate meta
        time = torch.randint(low=0, high=self.num_timesteps, size=[1])
        _name = self.file_list[idx].split("/")[-1]

        fl_biasfield = torch.from_numpy(file_mat["fl_biasfield"]).type(torch.float)

        return (prior, target, mask, time, _name, fl_biasfield)

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
