from ..patient import Patient
from torch.utils.data import Dataset
from pathlib import Path
import os
from scipy.io import loadmat
from torch import Tensor


def loader_function(path: Path, patient_id):
    data : Tensor = loadmat(path).get(patient_id, [])
    data = data.transpose()
    return data


class ADHDChildren(Dataset):

    def __init__(self, path: Path):
        self.patients : list[Patient] = []
        self.path = path / "adhd_children_dataset"
        self.directories = ["ADHD_part1", "ADHD_part2", "Control_part1", "Control_part2"] 

        for directory in self.directories:
            dir_path = self.path / directory
            if dir_path.exists() and dir_path.is_dir():
                for file in os.listdir(dir_path):
                    patient = Patient(path=dir_path/file, patient_id=file.removesuffix(".mat"), adhd="ADHD" in directory)
                    self.patients.append(patient)
            else:
                print(f"Directory {dir_path} does not exist or is not a directory.")

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient : Patient = self.patients[idx]
        return patient.to_dict(loader_function) 