from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .parser import PatientDocuments, parse_patient
from ..domain.patient import DatasetSplit
from ..domain.recording import Recording
from ..domain.segment import Segment


class SmartHeartDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        *,
        split: DatasetSplit | str | None = None,
        source_dataset: str | None = None,
        include_unusable: bool = False,
    ) -> None:
        self.dataset_root = dataset_root.resolve()
        self.split = DatasetSplit(split) if split is not None else None
        self.source_dataset = source_dataset
        self.include_unusable = include_unusable
        self._documents: dict[str, PatientDocuments] = {}
        self._recordings: dict[tuple[str, str], Recording] = {}
        self._segments: dict[tuple[str, str], Segment] = {}
        self._samples: list[tuple[str, str, str, int, int]] = []

        patients_root = self.dataset_root / "patients"
        if not patients_root.is_dir():
            raise ValueError(f"Patient directory does not exist: '{patients_root}'.")

        for directory in sorted(patients_root.iterdir()):
            if not directory.is_dir() or not (directory / "patient.json").is_file():
                continue
            documents = parse_patient(directory)
            patient, _, recordings, segments = documents
            if self.split is not None and patient.metadata.split != self.split:
                continue
            if source_dataset and patient.metadata.source_dataset != source_dataset:
                continue

            self._documents[patient.patient_id] = documents
            for recording in recordings.recordings:
                self._recordings[(patient.patient_id, recording.recording_id)] = recording
            for segment in segments.segments:
                if not include_unusable and not segment.quality.usable:
                    continue
                recording = self._recordings[(patient.patient_id, segment.recording_id)]
                if not include_unusable and not recording.quality.usable:
                    continue
                self._segments[(patient.patient_id, segment.segment_id)] = segment
                self._add_segment_windows(patient.patient_id, segment)

        if not self._documents:
            raise ValueError("No patients match the requested dataset filters.")
        if not self._samples:
            raise ValueError("No usable segment windows match the requested filters.")

    def __len__(self) -> int:
        return len(self._samples)

    def sample_identity(self, index: int) -> tuple[str, str, str]:
        patient_id, recording_id, segment_id, _, _ = self._samples[index]
        return patient_id, recording_id, segment_id

    def curriculum_target(
        self,
        index: int,
        scope: str,
        target_key: str,
    ) -> object | None:
        patient_id, recording_id, segment_id = self.sample_identity(index)
        patient, _, _, _ = self._documents[patient_id]

        if scope == "segment":
            targets = self._segments[
                (patient_id, segment_id)
            ].curriculum_targets
        elif scope == "recording":
            targets = self._recordings[
                (patient_id, recording_id)
            ].curriculum_targets
        elif scope == "patient":
            targets = patient.curriculum_targets
        else:
            raise ValueError(f"Unknown curriculum scope '{scope}'.")

        target = getattr(targets, target_key, None)
        if target is None:
            return None
        if isinstance(target, list):
            return [
                value.model_dump(mode="json", exclude_none=True)
                for value in target
            ]
        return target.model_dump(mode="json", exclude_none=True)

    def clinical_profile(self, index: int) -> dict[str, object]:
        patient_id, _, _ = self.sample_identity(index)
        _, profile, _, _ = self._documents[patient_id]
        return profile.model_dump(mode="json", exclude_none=True)

    def __getitem__(self, index: int) -> dict[str, object]:
        patient_id, recording_id, segment_id, start, end = self._samples[index]
        patient, profile, _, _ = self._documents[patient_id]
        recording = self._recordings[(patient_id, recording_id)]
        segment = self._segments[(patient_id, segment_id)]
        signal_path = self.dataset_root / "patients" / patient_id / recording.signal_path

        try:
            with np.load(signal_path, allow_pickle=False) as archive:
                if recording.signal_key not in archive:
                    raise ValueError(
                        f"Signal key '{recording.signal_key}' is missing from "
                        f"'{signal_path}'."
                    )
                signal = archive[recording.signal_key]
                if signal.shape != recording.shape:
                    raise ValueError(
                        f"Signal shape {signal.shape} does not match "
                        f"{recording.shape} for '{recording.recording_id}'."
                    )
                window = np.array(signal[:, start:end], dtype=np.float32, copy=True)
        except OSError as error:
            raise ValueError(f"Could not load signal archive '{signal_path}'.") from error

        sample: dict[str, object] = {
            "time_series": torch.from_numpy(window),
            "patient_id": patient_id,
            "recording_id": recording_id,
            "segment_id": segment_id,
            "window_start": start,
            "window_end": end,
            "split": patient.metadata.split.value,
            "source_dataset": patient.metadata.source_dataset,
            "channel_names": recording.channel_names,
            "representation": recording.representation.value,
            "condition": recording.condition,
            "task": recording.task,
            "segment_type": segment.segment_type,
            "clinical_profile": profile.model_dump(mode="json", exclude_none=True),
            "segment_curriculum_targets": segment.curriculum_targets.model_dump(
                mode="json",
                exclude_none=True,
            ),
            "recording_curriculum_targets": recording.curriculum_targets.model_dump(
                mode="json",
                exclude_none=True,
            ),
            "patient_curriculum_targets": patient.curriculum_targets.model_dump(
                mode="json",
                exclude_none=True,
            ),
        }
        if recording.sampling_rate_hz is not None:
            sample["sampling_rate_hz"] = recording.sampling_rate_hz
        return sample

    def _add_segment_windows(self, patient_id: str, segment: Segment) -> None:
        size = segment.window_policy.size_samples
        stride = segment.window_policy.stride_samples
        length = segment.end_timestep - segment.start_timestep
        local_spans = self._window_spans(length, size, stride)
        local_spans = self._limit_spans(
            local_spans,
            segment.window_policy.maximum_windows,
        )
        self._samples.extend(
            (
                patient_id,
                segment.recording_id,
                segment.segment_id,
                segment.start_timestep + start,
                segment.start_timestep + end,
            )
            for start, end in local_spans
        )

    @staticmethod
    def _window_spans(
        length: int,
        size: int,
        stride: int,
    ) -> list[tuple[int, int]]:
        if length <= size:
            return [(0, length)]
        spans = [
            (start, start + size)
            for start in range(0, length - size + 1, stride)
        ]
        final_span = (length - size, length)
        if spans[-1] != final_span:
            spans.append(final_span)
        return spans

    @staticmethod
    def _limit_spans(
        spans: list[tuple[int, int]],
        maximum: int,
    ) -> list[tuple[int, int]]:
        if len(spans) <= maximum:
            return spans
        if maximum == 1:
            return [spans[len(spans) // 2]]
        indices = [
            round(index * (len(spans) - 1) / (maximum - 1))
            for index in range(maximum)
        ]
        return [spans[index] for index in indices]
