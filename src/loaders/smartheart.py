import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from ..utility.windowing import build_window_spans


@dataclass(frozen=True)
class CanonicalPatient:
    patient_id: str
    split: str
    diagnosis: str
    source_dataset: str

    @property
    def adhd(self) -> bool:
        return self.diagnosis == "ADHD"


@dataclass(frozen=True)
class CanonicalRecording:
    patient: CanonicalPatient
    recording_id: str
    signal_path: Path
    signal_key: str
    representation: str
    channel_names: tuple[str, ...]
    timestep_count: int
    condition: str
    task: str

    def load(self) -> Tensor:
        with np.load(self.signal_path, allow_pickle=False) as archive:
            if self.signal_key not in archive:
                raise ValueError(
                    f"Signal key '{self.signal_key}' is missing from "
                    f"'{self.signal_path}'."
                )
            signal = np.asarray(archive[self.signal_key], dtype=np.float32)

        if signal.ndim != 2 or signal.shape[1] == 0:
            raise ValueError(f"Loaded invalid time series from '{self.signal_path}'.")
        if signal.shape != (len(self.channel_names), self.timestep_count):
            raise ValueError(
                f"Signal shape {signal.shape} does not match metadata for "
                f"'{self.recording_id}'."
            )
        return torch.from_numpy(signal).contiguous()


class SmartHeartDataset(Dataset):
    def __init__(
        self,
        path: Path,
        *,
        source_dataset: str | None = None,
        window_size: int | None = None,
        window_stride: int | None = None,
    ) -> None:
        self.path = path
        self.source_dataset = source_dataset
        self.window_size = window_size
        self.window_stride = window_stride
        self.patients: list[CanonicalPatient] = []
        self.samples: list[tuple[CanonicalRecording, tuple[int, int]]] = []

        manifest_path = self.path / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Canonical dataset manifest not found at '{manifest_path}'."
            )

        manifest = self._read_json(manifest_path)
        for entry in manifest.get("patients", []):
            if source_dataset and entry.get("source_dataset") != source_dataset:
                continue
            self._add_patient(entry)

        if source_dataset and not self.patients:
            raise ValueError(
                f"Canonical dataset has no patients for source '{source_dataset}'."
            )

    @staticmethod
    def _read_json(path: Path) -> dict[str, object]:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise ValueError(f"Expected a JSON object in '{path}'.")
        return value

    def _add_patient(self, entry: dict[str, object]) -> None:
        patient_path = self.path / str(entry["path"])
        metadata = self._read_json(patient_path)
        ground_truth = metadata["clinical_ground_truth"]
        if not isinstance(ground_truth, dict):
            raise ValueError(f"Invalid clinical ground truth in '{patient_path}'.")

        patient = CanonicalPatient(
            patient_id=str(metadata["patient_id"]),
            split=str(metadata["split"]),
            diagnosis=str(ground_truth["diagnosis"]),
            source_dataset=str(metadata["source_dataset"]),
        )
        if patient.patient_id != entry["patient_id"]:
            raise ValueError(f"Patient ID mismatch in '{patient_path}'.")
        for field, actual in (
            ("split", patient.split),
            ("source_dataset", patient.source_dataset),
            ("diagnosis", patient.diagnosis),
        ):
            if actual != entry[field]:
                raise ValueError(f"Patient {field} mismatch in '{patient_path}'.")
        if patient.split not in {"train", "validation", "test"}:
            raise ValueError(f"Invalid split in '{patient_path}'.")
        if patient.diagnosis not in {"ADHD", "NOT_ADHD"}:
            raise ValueError(f"Invalid diagnosis in '{patient_path}'.")

        self.patients.append(patient)
        recordings = metadata.get("recordings")
        if not isinstance(recordings, list):
            raise ValueError(f"Invalid recordings list in '{patient_path}'.")
        if len(recordings) != entry["recording_count"]:
            raise ValueError(f"Patient recording count mismatch in '{patient_path}'.")

        for value in recordings:
            if not isinstance(value, dict):
                raise ValueError(f"Invalid recording in '{patient_path}'.")
            quality = value.get("quality", {})
            if isinstance(quality, dict) and not quality.get("usable", True):
                continue
            shape = value["shape"]
            if not isinstance(shape, list) or len(shape) != 2:
                raise ValueError(f"Invalid recording shape in '{patient_path}'.")
            recording = CanonicalRecording(
                patient=patient,
                recording_id=str(value["recording_id"]),
                signal_path=patient_path.parent / str(value["signal_path"]),
                signal_key=str(value["signal_key"]),
                representation=str(value["representation"]),
                channel_names=tuple(str(name) for name in value["channel_names"]),
                timestep_count=int(shape[1]),
                condition=str(value["condition"]),
                task=str(value["task"]),
            )
            spans = build_window_spans(
                recording.timestep_count,
                self.window_size,
                self.window_stride,
            )
            self.samples.extend((recording, span) for span in spans)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        recording, (start, end) = self.samples[index]
        time_series = recording.load()[:, start:end]
        mean = time_series.mean(dim=1)
        standard_deviation = time_series.std(dim=1, unbiased=False)
        answer = "YES" if recording.patient.adhd else "NO"

        return {
            "answer": answer,
            "pre_prompt": (
                "Given the following time series data, determine if the patient "
                "has ADHD."
            ),
            "post_prompt": (
                f"This {recording.representation} recording is from the "
                f"'{recording.condition}' condition and covers timesteps "
                f"{start} to {end}. Does this patient have ADHD? Answer with "
                "exactly YES or NO."
            ),
            "time_series": time_series,
            "time_series_text": [
                f"Channel {name} has a mean of {channel_mean.item():.6g} and a "
                f"standard deviation of {channel_std.item():.6g}."
                for name, channel_mean, channel_std in zip(
                    recording.channel_names,
                    mean,
                    standard_deviation,
                )
            ],
            "patient_id": recording.patient.patient_id,
            "recording_id": recording.recording_id,
            "window_start": start,
            "window_end": end,
            "split": recording.patient.split,
        }
