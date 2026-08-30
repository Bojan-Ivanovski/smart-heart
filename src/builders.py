from functools import partial
from pathlib import Path

from opentslm.model_config import PATCH_SIZE
from opentslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)
from torch.utils.data import DataLoader, Dataset

from .loaders.children import ADHDChildren
from .loaders.gameplay import ADHDGameplay

DATASET_TYPES = {
    "children": ADHDChildren,
    "gameplay": ADHDGameplay,
}


class DatasetPipelineBuilder:
    def __init__(self, datasets_root: Path):
        self.datasets_root = datasets_root

    def build_dataset(
        self,
        dataset_name: str,
        *,
        window_size: int | None,
        window_stride: int | None,
    ) -> Dataset:
        try:
            dataset_type = DATASET_TYPES[dataset_name]
        except KeyError as exc:
            available = ", ".join(sorted(DATASET_TYPES))
            raise KeyError(
                f"Unknown dataset '{dataset_name}'. Available datasets: {available}"
            ) from exc

        dataset = dataset_type(
            self.datasets_root,
            window_size=window_size,
            window_stride=window_stride,
        )
        if len(dataset) == 0:
            raise ValueError(
                f"Dataset '{dataset_name}' produced no samples under "
                f"'{self.datasets_root}'."
            )
        return dataset

    @staticmethod
    def build_dataloader(
        dataset: Dataset,
        *,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        collate_fn = partial(
            extend_time_series_to_match_patch_size_and_aggregate,
            patch_size=PATCH_SIZE,
            normalize=True,
        )
        return DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            collate_fn=collate_fn,
        )
