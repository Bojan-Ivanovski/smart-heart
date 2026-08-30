from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from torch import Tensor

from ..utility.windowing import build_window_spans


@dataclass(frozen=True)
class Patient:
    path: Path
    patient_id: str
    adhd: bool = False
    split_group: str | None = None

    @property
    def split_key(self) -> str:
        return self.split_group or self.patient_id

    def to_dict(self, loader_function: Callable) -> dict[str, object]:
        data: Tensor = loader_function(self.path, self.patient_id)
        if data.numel() == 0 or data.ndim != 2 or data.shape[1] == 0:
            raise ValueError(f"Loaded empty time series for {self.path}")

        mean = data.mean()
        std = data.std(unbiased=False)
        return {
            "answer": "YES" if self.adhd else "NO",
            "pre_prompt": (
                "Given the following time series data, determine if the patient "
                "has ADHD."
            ),
            "post_prompt": "Does this patient have ADHD?",
            "time_series": data,
            "time_series_text": [
                "This is the time series data. It has a mean of {} and a standard "
                "deviation of {}.".format(mean, std)
            ]
            * data.shape[0],
        }

    def get_window_spans(
        self,
        loader_function: Callable,
        window_size: int | None,
        window_stride: int | None,
    ) -> list[tuple[int, int]]:
        data: Tensor = loader_function(self.path, self.patient_id)
        if data.numel() == 0 or data.ndim != 2 or data.shape[1] == 0:
            return []

        return build_window_spans(data.shape[1], window_size, window_stride)
