import glob
from dataclasses import dataclass
from enum import IntEnum

import torch
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset


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
                self.file_list = self.file_list[:1000]
            else:
                self.file_list = self.file_list[:100]

        else:
            if training_mode:
                self.file_list = self.file_list
            else:
                self.file_list = self.file_list

    @staticmethod
    def _load_from_mat(
        file_mat: dict,
        key: str,
    ) -> torch.Tensor:
        img = torch.from_numpy(file_mat[key]).type(torch.float)
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
        meta_default = [
            0.0,  # TE_visit2_ms
            0.0,  # TR_visit2_ms
            0.0,  # TE_visit1_ms
            0.0,  # TR_visit1_ms
            0.0,  # time_diff
            0.0,  # Age
            0.0,  # CDRSB
        ]
        meta = [float(file_mat.get(key, default)) for key, default in zip(meta_keys, meta_default, strict=False)]
        meta_tensor = torch.tensor(meta, dtype=torch.float32).squeeze()
        return meta_tensor

    def __getitem__(
        self,
        idx: int,
    ):
        file_mat = loadmat(self.file_list[idx])
        img, img_mask, prev = self._get_image(file_mat)
        meta_tensor = self._get_meta(file_mat)

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
        len(dataset),
    )
