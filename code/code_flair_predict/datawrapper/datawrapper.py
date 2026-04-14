import glob
import random
from dataclasses import dataclass
from enum import IntEnum

import torch
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset

prob_flip: float = 0.5


class DataKey(IntEnum):
    FL = 0
    FL_Mask = 1
    T1 = 2
    T2 = 3
    T1Weight = 4
    T2Weight = 5
    Meta = 6
    Name = 7


@dataclass
class LoaderConfig:
    data_type: str
    batch: int
    num_workers: int
    shuffle: bool
    debug_mode: bool
    source_contrast: str


class DataWrapper(Dataset):

    num_timesteps: int
    file_list: list[str]
    training_mode: bool
    source_contrast: list[str]

    def __init__(
        self,
        file_path: list[str],
        data_type: str,
        training_mode: bool,
        debug_mode: bool,
        source_contrast: str = "t1,t2",
    ):
        self.num_timesteps: int = 1000

        super().__init__()
        total_list: list[str] = []
        for _file_path in file_path:
            total_list += glob.glob(f"{_file_path}/{data_type}")

        self.file_list = total_list
        self.training_mode = training_mode
        self.source_contrast = source_contrast
        self.source_contrast = self.source_contrast.split(",")

        if debug_mode:
            if training_mode:
                self.file_list = self.file_list[::10]
            else:
                self.file_list = self.file_list[:100]

        else:
            if training_mode:
                self.file_list = self.file_list
            else:
                self.file_list = self.file_list
        self.len = len(self.file_list)

    @staticmethod
    def _load_from_mat(
        file_mat: dict,
        key: str,
    ) -> torch.Tensor:
        img = torch.from_numpy(file_mat[key])
        return img

    def _get_image(
        self,
        file_mat: dict,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:

        fl = self._load_from_mat(file_mat, "fl")

        fl_mask = self._load_from_mat(file_mat, "fl_mask")
        fl_mask = fl_mask[fl_mask.shape[0] // 2 : fl_mask.shape[0] // 2 + 1, :, :]

        t1 = self._load_from_mat(file_mat, "t1_reg") if "t1_reg" in file_mat else torch.zeros_like(fl)
        t2 = self._load_from_mat(file_mat, "t2_reg") if "t2_reg" in file_mat else torch.zeros_like(fl)

        t1_exist = torch.tensor(1.0) if "t1_reg" in file_mat else torch.tensor(0.0)
        t2_exist = torch.tensor(1.0) if "t2_reg" in file_mat else torch.tensor(0.0)

        fl = fl.type(torch.float)
        t1 = t1.type(torch.float)
        t2 = t2.type(torch.float)

        return fl, fl_mask, t1, t2, t1_exist, t2_exist

    def _get_meta(
        self,
        file_mat: dict,
    ) -> torch.Tensor:
        meta_keys = [
            "ScanOptions",
            "EchoTime",
            "RepetitionTime",
            "InversionTime",
            "T2_EchoTime",
            "T2_RepetitionTime",
            "T1_EchoTime",
            "T1_RepetitionTime",
        ]
        meta = [float(file_mat.get(key, 0)) for key in meta_keys]

        for idx in [4, 5, 6, 7]:
            if random.random() < 0.5:
                meta[idx] = 0

        meta_tensor = torch.tensor(meta, dtype=torch.float32).squeeze()
        return meta_tensor

    def _augment_image(
        self,
        fl: torch.Tensor,
        t1: torch.Tensor,
        t2: torch.Tensor,
        t1_exist: torch.Tensor,
        t2_exist: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        weight_factor = 10.0

        if random.random() > prob_flip:
            fl = torch.flip(fl, dims=[1])
            t1 = torch.flip(t1, dims=[1])
            t2 = torch.flip(t2, dims=[1])

        if random.random() > prob_flip:
            fl = torch.flip(fl, dims=[2])
            t1 = torch.flip(t1, dims=[2])
            t2 = torch.flip(t2, dims=[2])

        t1_exists = t1_exist * weight_factor
        t2_exists = t2_exist * weight_factor

        return fl, t1, t2, t1_exists, t2_exists

    def __getitem__(
        self,
        idx: int,
    ):
        file_mat = loadmat(self.file_list[idx])
        fl, fl_mask, t1, t2, t1_exist, t2_exist = self._get_image(file_mat)
        meta_tensor = self._get_meta(file_mat)

        if self.training_mode:
            fl, t1, t2, t1_exist, t2_exist = self._augment_image(fl, t1, t2, t1_exist, t2_exist)

        _name = self.file_list[idx].split("/")[-1]

        return (
            fl,
            fl_mask,
            t1,
            t2,
            t1_exist,
            t2_exist,
            meta_tensor,
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
        data_type=loader_cfg.data_type,
        training_mode=training_mode,
        debug_mode=loader_cfg.debug_mode,
        source_contrast=loader_cfg.source_contrast,
    )

    _ = dataset[0]

    dataloader = DataLoader(
        dataset,
        batch_size=loader_cfg.batch,
        num_workers=loader_cfg.num_workers,
        pin_memory=True,
        persistent_workers=True,
        shuffle=loader_cfg.shuffle,
    )

    return (
        dataloader,
        dataset,
        dataset.len,
    )
