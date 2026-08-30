import random
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from opentslm.model_config import PATCH_SIZE
from opentslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)
from torch.utils.data import DataLoader, Dataset, Subset

from .classes.patient import Patient
from .loaders.children import ADHDChildren
from .loaders.gameplay import ADHDGameplay

DATASET_TYPES = {
    "children": ADHDChildren,
    "gameplay": ADHDGameplay,
}

TRAIN_RATIO = 0.60
VALIDATION_RATIO = 0.20
TEST_RATIO = 0.20


@dataclass(frozen=True)
class DatasetSplits:
    train: Subset
    validation: Subset
    test: Subset
    train_groups: int
    validation_groups: int
    test_groups: int


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
    def split_dataset(dataset: Dataset, *, seed: int) -> DatasetSplits:
        samples = getattr(dataset, "samples", None)
        if samples is None:
            raise TypeError("Dataset must expose patient-window samples for splitting")

        group_indices: dict[str, list[int]] = {}
        group_labels: dict[str, bool] = {}
        for sample_index, (patient, _) in enumerate(samples):
            if not isinstance(patient, Patient):
                raise TypeError("Dataset samples must reference Patient instances")
            split_key = patient.split_key
            existing_label = group_labels.setdefault(split_key, patient.adhd)
            if existing_label != patient.adhd:
                raise ValueError(
                    f"Patient split group '{split_key}' contains conflicting labels"
                )
            group_indices.setdefault(split_key, []).append(sample_index)

        groups_by_label: dict[bool, list[str]] = {False: [], True: []}
        for split_key, label in group_labels.items():
            groups_by_label[label].append(split_key)

        split_groups: list[list[str]] = [[], [], []]
        randomizer = random.Random(seed)
        for label, label_groups in groups_by_label.items():
            if len(label_groups) < 3:
                label_name = "ADHD" if label else "non-ADHD"
                raise ValueError(
                    f"At least 3 {label_name} patient groups are required to create "
                    "stratified train, validation, and test splits"
                )
            label_groups.sort()
            randomizer.shuffle(label_groups)
            counts = DatasetPipelineBuilder._split_counts(len(label_groups))
            start = 0
            for destination, count in zip(split_groups, counts):
                destination.extend(label_groups[start : start + count])
                start += count

        split_indices = []
        for keys in split_groups:
            indices = [index for key in keys for index in group_indices[key]]
            split_indices.append(sorted(indices))

        return DatasetSplits(
            train=Subset(dataset, split_indices[0]),
            validation=Subset(dataset, split_indices[1]),
            test=Subset(dataset, split_indices[2]),
            train_groups=len(split_groups[0]),
            validation_groups=len(split_groups[1]),
            test_groups=len(split_groups[2]),
        )

    @staticmethod
    def _split_counts(group_count: int) -> tuple[int, int, int]:
        ratios = (TRAIN_RATIO, VALIDATION_RATIO, TEST_RATIO)
        exact_counts = [group_count * ratio for ratio in ratios]
        counts = [int(count) for count in exact_counts]
        remainder = group_count - sum(counts)
        priority = sorted(
            range(len(ratios)),
            key=lambda index: (exact_counts[index] - counts[index], -index),
            reverse=True,
        )
        for index in priority[:remainder]:
            counts[index] += 1
        return counts[0], counts[1], counts[2]

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
