from pathlib import Path

from scipy.io import loadmat
import torch
from torch import Tensor
from torch.utils.data import Dataset

from ..classes.patient import Patient


def loader_function(path: Path, patient_id: str) -> Tensor:
    data = torch.as_tensor(
        loadmat(path).get(patient_id, []),
        dtype=torch.float32,
    )
    data = data.transpose(0, 1).contiguous()
    return data


class ADHDChildren(Dataset):
    def __init__(
        self,
        path: Path,
        window_size: int | None = None,
        window_stride: int | None = None,
    ) -> None:
        self.patients: list[Patient] = []
        self.samples: list[tuple[Patient, tuple[int, int]]] = []
        self.path = path / "adhd_children_dataset"
        self.directories = (
            "ADHD_part1",
            "ADHD_part2",
            "Control_part1",
            "Control_part2",
        )
        self.window_size = window_size
        self.window_stride = window_stride

        for directory in self.directories:
            dir_path = self.path / directory
            if dir_path.exists() and dir_path.is_dir():
                for file_path in sorted(dir_path.glob("*.mat")):
                    patient = Patient(
                        path=file_path,
                        patient_id=file_path.stem,
                        adhd="ADHD" in directory,
                    )
                    self.patients.append(patient)
                    spans = patient.get_window_spans(
                        loader_function,
                        self.window_size,
                        self.window_stride,
                    )
                    self.samples.extend((patient, span) for span in spans)
            else:
                print(f"Directory {dir_path} does not exist or is not a directory.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        patient, (start, end) = self.samples[idx]
        sample = patient.to_dict(loader_function)
        sample["time_series"] = sample["time_series"][:, start:end]
        sample["post_prompt"] = (
            f"{sample['post_prompt']} Window covers timesteps {start} to {end}."
        )
        return sample
