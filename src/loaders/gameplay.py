from pathlib import Path
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

from ..patient import Patient

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
    def __init__(self, path: Path, source: str | None = None):
        self.patients: list[Patient] = []
        self.path = (
            path
            / "adhd_individuals_gameplay_dataset"
        )
        self.groups = ["ADHD", "Non-ADHD"]
        self.source = source

        for group in self.groups:
            group_path = self.path / group
            if not group_path.exists() or not group_path.is_dir():
                print(f"Directory {group_path} does not exist or is not a directory.")
                continue

            for subject_dir in sorted(os.listdir(group_path)):
                subject_path = group_path / subject_dir
                if not subject_path.is_dir():
                    continue

                source_dirs = [self.source] if self.source else sorted(os.listdir(subject_path))
                for source_dir in source_dirs:
                    session_dir = subject_path / source_dir
                    if not session_dir.exists() or not session_dir.is_dir():
                        continue

                    for file in sorted(os.listdir(session_dir)):
                        if not file.lower().endswith(".csv"):
                            continue
                        file_path = session_dir / file
                        if not is_bandpower_csv(file_path):
                            continue

                        patient = Patient(
                            path=file_path,
                            patient_id=f"{group}_{subject_dir}_{source_dir}_{Path(file).stem}",
                            adhd=(group == "ADHD"),
                        )
                        self.patients.append(patient)

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient: Patient = self.patients[idx]
        return patient.to_dict(loader_function)
