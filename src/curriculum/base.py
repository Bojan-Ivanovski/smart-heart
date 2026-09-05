from abc import ABC
from collections import defaultdict
from collections.abc import Sequence
from typing import TypedDict

import torch
from torch import Tensor
from torch.utils.data import Dataset

from ..data.loader import SmartHeartDataset


class OpenTSLMSample(TypedDict, total=False):
    answer: str
    pre_prompt: str
    post_prompt: str
    time_series: list[Tensor]
    time_series_text: list[str]
    patient_id: str
    recording_ids: list[str]
    segment_ids: list[str]
    split: str
    source_dataset: str
    question_id: str
    reference_analyses: list[dict[str, object]]


class CurriculumDataset(Dataset, ABC):
    name: str
    scope: str
    target_key: str

    def __init__(self, dataset: SmartHeartDataset, eos_token: str) -> None:
        if not eos_token:
            raise ValueError("The curriculum requires a non-empty EOS token.")
        self.dataset = dataset
        self.eos_token = eos_token

    def _target(self, index: int) -> object | None:
        return self.dataset.curriculum_target(
            index,
            self.scope,
            self.target_key,
        )

    def _matching_indices(self) -> list[int]:
        return [
            index
            for index in range(len(self.dataset))
            if self._target(index) is not None
        ]

    def _grouped_indices(self) -> list[list[int]]:
        groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
        for index in self._matching_indices():
            patient_id, recording_id, _ = self.dataset.sample_identity(index)
            key = (
                (patient_id, recording_id)
                if self.scope == "recording"
                else (patient_id,)
            )
            groups[key].append(index)
        return [groups[key] for key in sorted(groups)]

    def _representative_indices(self, indices: Sequence[int]) -> list[int]:
        by_segment: dict[tuple[str, str, str], list[int]] = defaultdict(list)
        for index in indices:
            by_segment[self.dataset.sample_identity(index)].append(index)
        return [
            values[len(values) // 2]
            for _, values in sorted(by_segment.items())
        ]

    def _format_sample(
        self,
        indices: Sequence[int],
        *,
        pre_prompt: str,
        post_prompt: str,
        answer: str,
        source_samples: Sequence[dict[str, object]] | None = None,
    ) -> OpenTSLMSample:
        if not indices:
            raise ValueError("A curriculum sample requires at least one window.")

        if source_samples is None:
            source_samples = [self.dataset[index] for index in indices]
        elif len(source_samples) != len(indices):
            raise ValueError("Source samples must match the supplied indices.")
        time_series, time_series_text = self._prepare_time_series(source_samples)
        first = source_samples[0]
        return {
            "answer": self._with_eos(answer),
            "pre_prompt": pre_prompt.strip(),
            "post_prompt": post_prompt.strip(),
            "time_series": time_series,
            "time_series_text": time_series_text,
            "patient_id": str(first["patient_id"]),
            "recording_ids": list(
                dict.fromkeys(str(sample["recording_id"]) for sample in source_samples)
            ),
            "segment_ids": list(
                dict.fromkeys(str(sample["segment_id"]) for sample in source_samples)
            ),
            "split": str(first["split"]),
            "source_dataset": str(first["source_dataset"]),
            "reference_analyses": [
                dict(sample["window_analysis"]) for sample in source_samples
            ],
        }

    @staticmethod
    def _prepare_time_series(
        samples: Sequence[dict[str, object]],
    ) -> tuple[list[Tensor], list[str]]:
        series_by_channel: dict[tuple[str, str], list[Tensor]] = defaultdict(list)
        context_by_channel: dict[tuple[str, str], tuple[str, str]] = {}

        for sample in samples:
            recording_id = str(sample["recording_id"])
            channel_names = list(sample["channel_names"])
            time_series = sample["time_series"]
            if not isinstance(time_series, Tensor):
                raise TypeError("time_series must be a torch.Tensor.")
            if time_series.ndim != 2 or time_series.shape[0] != len(channel_names):
                raise ValueError("Channel metadata does not match the time series.")

            for channel_name, channel in zip(channel_names, time_series):
                key = recording_id, str(channel_name)
                series_by_channel[key].append(channel.to(dtype=torch.float32))
                context_by_channel[key] = (
                    str(sample["representation"]),
                    str(sample["condition"]),
                )

        prepared: list[Tensor] = []
        descriptions: list[str] = []
        for key, chunks in series_by_channel.items():
            recording_id, channel_name = key
            series = torch.cat(chunks)
            mean = series.mean()
            standard_deviation = series.std(unbiased=False)
            scale = standard_deviation.clamp_min(1e-6)
            prepared.append((series - mean) / scale)
            representation, condition = context_by_channel[key]
            descriptions.append(
                f"Recording {recording_id}, channel {channel_name}, contains "
                f"{representation} from the {condition} condition. Its original "
                f"mean is {mean.item():.4f} and standard deviation is "
                f"{standard_deviation.item():.4f}."
            )

        return prepared, descriptions

    def _with_eos(self, answer: str) -> str:
        answer = answer.strip()
        if not answer.endswith(self.eos_token):
            answer += self.eos_token
        return answer
