from pathlib import Path
from torch import Tensor
from typing import Callable

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
