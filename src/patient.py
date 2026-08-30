from pathlib import Path
from torch import Tensor
from typing import Callable
import math

class Patient:

    def __init__(self, path: Path, patient_id: str, adhd: bool = False):
        self.path = path
        self.patient_id = patient_id
        self.adhd = adhd

    def to_dict(self, loader_function : Callable):
        data : Tensor = loader_function(self.path, self.patient_id)
        if data.numel() == 0 or data.ndim != 2 or data.shape[1] == 0:
            raise ValueError(f"Loaded empty time series for {self.path}")

        mean = data.mean()
        std = data.std(unbiased=False)
        return {
            'answer': "YES" if self.adhd else "NO",
            'pre_prompt': "Given the following time series data, determine if the patient has ADHD.",
            'post_prompt': "Does this patient have ADHD?",
            'time_series': data,
            'time_series_text': ["This is the time series data. It has a mean of {} and a standard deviation of {}.".format(mean, std)]*data.shape[0]
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

        total_steps = data.shape[1]
        if window_size is None or total_steps <= window_size:
            return [(0, total_steps)]

        stride = window_stride or window_size
        window_count = math.ceil(max(total_steps - window_size, 0) / stride) + 1
        spans = []
        for window_index in range(window_count):
            start = window_index * stride
            end = min(start + window_size, total_steps)
            if end <= start:
                continue
            spans.append((start, end))

        last_end = spans[-1][1] if spans else 0
        if last_end < total_steps:
            spans.append((max(total_steps - window_size, 0), total_steps))
        return spans
