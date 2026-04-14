import glob
import random
from dataclasses import dataclass
from enum import IntEnum

import torch
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset

prob_flip: float = 0.5


class DataKey(IntEnum):
    IMG = 0
    IMG_Mask = 1
    PREV = 2
    Meta = 3
    Name = 4


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
        source_contrast: str = "prev",
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
    ]:

        img = self._load_from_mat(file_mat, "visit2").type(torch.float)
        img_mask = self._load_from_mat(file_mat, "visit2_mask").type(torch.uint8)
        prev = self._load_from_mat(file_mat, "visit1").type(torch.float)

        return img, img_mask, prev

    def _get_meta(
        self,
        file_mat: dict,
    ) -> torch.Tensor:
        meta_keys = [
            "TE_visit2_ms",
            "TR_visit2_ms",
            "TE_visit1_ms",
            "TR_visit1_ms",
            "time_diff",
            "Age",
            "CDRSB",
        ]
        meta = [float(file_mat.get(key, 0)) for key in meta_keys]

        meta_tensor = torch.tensor(meta, dtype=torch.float32).squeeze()
        return meta_tensor

    def _augment_image(
        self,
        img: torch.Tensor,
        prev: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

        if random.random() > prob_flip:
            img = torch.flip(img, dims=[1])
            prev = torch.flip(prev, dims=[1])

        if random.random() > prob_flip:
            img = torch.flip(img, dims=[2])
            prev = torch.flip(prev, dims=[2])

        return img, prev

    def __getitem__(
        self,
        idx: int,
    ):
        file_mat = loadmat(self.file_list[idx])
        img, img_mask, prev = self._get_image(file_mat)
        meta_tensor = self._get_meta(file_mat)

        if self.training_mode:
            img, prev = self._augment_image(img, prev)

        _name = self.file_list[idx].split("/")[-1]

        return (
            img,
            img_mask,
            prev,
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
