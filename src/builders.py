from dataclasses import dataclass
from functools import partial
from pathlib import Path

from opentslm.model_config import PATCH_SIZE
from opentslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)
from torch import Generator
from torch.utils.data import DataLoader, Dataset, Subset

from .loaders.smartheart import SmartHeartDataset

DATASET_TYPES = {
    "all": None,
    "children": "adhd_children",
    "cognitive_function": "adhd_cognitive_function",
    "gameplay": "adhd_gameplay",
}


@dataclass(frozen=True)
class DatasetSplits:
    train: Subset
    validation: Subset
    test: Subset
    train_groups: int
    validation_groups: int
    test_groups: int


class DatasetPipelineBuilder:
    def __init__(self, dataset_root: Path):
        self.dataset_root = dataset_root

    def build_dataset(
        self,
        dataset_name: str,
        *,
        window_size: int | None,
        window_stride: int | None,
    ) -> Dataset:
        try:
            source_dataset = DATASET_TYPES[dataset_name]
        except KeyError as exc:
            available = ", ".join(sorted(DATASET_TYPES))
            raise KeyError(
                f"Unknown dataset '{dataset_name}'. Available datasets: {available}"
            ) from exc

        dataset = SmartHeartDataset(
            self.dataset_root,
            source_dataset=source_dataset,
            window_size=window_size,
            window_stride=window_stride,
        )
        if len(dataset) == 0:
            raise ValueError(
                f"Dataset '{dataset_name}' produced no samples under "
                f"'{self.dataset_root}'."
            )
        return dataset

    @staticmethod
    def split_dataset(dataset: Dataset) -> DatasetSplits:
        samples = getattr(dataset, "samples", None)
        if samples is None:
            raise TypeError("Dataset must expose canonical recording-window samples")

        split_names = ("train", "validation", "test")
        split_indices: dict[str, list[int]] = {name: [] for name in split_names}
        split_patients: dict[str, set[str]] = {name: set() for name in split_names}
        for sample_index, (recording, _) in enumerate(samples):
            patient = recording.patient
            if patient.split not in split_indices:
                raise ValueError(
                    f"Patient '{patient.patient_id}' has invalid split "
                    f"'{patient.split}'."
                )
            split_indices[patient.split].append(sample_index)
            split_patients[patient.split].add(patient.patient_id)

        return DatasetSplits(
            train=Subset(dataset, split_indices["train"]),
            validation=Subset(dataset, split_indices["validation"]),
            test=Subset(dataset, split_indices["test"]),
            train_groups=len(split_patients["train"]),
            validation_groups=len(split_patients["validation"]),
            test_groups=len(split_patients["test"]),
        )

    @staticmethod
    def build_dataloader(
        dataset: Dataset,
        *,
        batch_size: int,
        shuffle: bool,
        seed: int | None = None,
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
            generator=Generator().manual_seed(seed) if seed is not None else None,
        )
