from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

from ..classes.patient import Patient

BANDPOWER_HEADER_PREFIX = "Theta"
MAX_TIMESTEPS = 4096


def is_bandpower_csv(path: Path) -> bool:
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        header = handle.readline().strip()
    return header.startswith(BANDPOWER_HEADER_PREFIX)


def loader_function(path: Path, patient_id: str) -> Tensor:
    del patient_id
    data = np.genfromtxt(
        path,
        delimiter=",",
        skip_header=1,
        autostrip=True,
        usecols=(0, 1, 2, 3, 4),
        invalid_raise=False,
    )

    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)

    data = data[np.isfinite(data).all(axis=1)]

    data = torch.as_tensor(data, dtype=torch.float32).transpose(0, 1).contiguous()
    if data.shape[1] > MAX_TIMESTEPS:
        data = F.interpolate(
            data.unsqueeze(0),
            size=MAX_TIMESTEPS,
            mode="linear",
            align_corners=False,
        ).squeeze(0)
    return data


class ADHDGameplay(Dataset):
    def __init__(
        self,
        path: Path,
        source: str | None = None,
        window_size: int | None = None,
        window_stride: int | None = None,
    ) -> None:
        self.patients: list[Patient] = []
        self.samples: list[tuple[Patient, tuple[int, int]]] = []
        self.path = path / "adhd_individuals_gameplay_dataset"
        self.groups = ("ADHD", "Non-ADHD")
        self.source = source
        self.window_size = window_size
        self.window_stride = window_stride

        for group in self.groups:
            group_path = self.path / group
            if not group_path.exists() or not group_path.is_dir():
                print(f"Directory {group_path} does not exist or is not a directory.")
                continue

            for subject_path in sorted(group_path.iterdir()):
                if not subject_path.is_dir():
                    continue
                subject_id = subject_path.name

                session_dirs = (
                    [subject_path / self.source]
                    if self.source
                    else sorted(subject_path.iterdir())
                )
                for session_dir in session_dirs:
                    if not session_dir.exists() or not session_dir.is_dir():
                        continue
                    source_name = session_dir.name

                    for file_path in sorted(session_dir.glob("*.csv")):
                        if not is_bandpower_csv(file_path):
                            continue

                        patient = Patient(
                            path=file_path,
                            patient_id=(
                                f"{group}_{subject_id}_{source_name}_{file_path.stem}"
                            ),
                            adhd=(group == "ADHD"),
                            split_group=f"{group}_{subject_id}",
                        )
                        self.patients.append(patient)
                        spans = patient.get_window_spans(
                            loader_function,
                            self.window_size,
                            self.window_stride,
                        )
                        self.samples.extend((patient, span) for span in spans)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        patient, (start, end) = self.samples[idx]
        sample = patient.to_dict(loader_function)
        sample["time_series"] = sample["time_series"][:, start:end]
        sample["post_prompt"] = (
            f"Window covers timesteps {start} to {end}. {sample['post_prompt']}"
        )
        return sample
