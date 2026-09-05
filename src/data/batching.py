from functools import partial

from torch import Generator
from torch.utils.data import DataLoader, Dataset


def build_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int | None = None,
    patch_size: int | None = None,
) -> DataLoader:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if patch_size is not None and patch_size < 1:
        raise ValueError("patch_size must be at least 1 when provided.")

    from opentslm.model_config import PATCH_SIZE
    from opentslm.time_series_datasets.util import (
        extend_time_series_to_match_patch_size_and_aggregate,
    )

    collate = partial(
        extend_time_series_to_match_patch_size_and_aggregate,
        patch_size=patch_size if patch_size is not None else PATCH_SIZE,
        normalize=False,
    )
    generator = None
    if seed is not None:
        generator = Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        generator=generator,
    )
