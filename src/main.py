from scipy.io import loadmat

from pathlib import Path

import os

from torch.utils.data import DataLoader, Dataset

from torch import Tensor
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate
from opentslm.model_config import PATCH_SIZE

class Patient:

    def __init__(self, path: Path, patient_id: str, adhd: bool = False):
        self.path = path
        self.patient_id = patient_id
        self.adhd = adhd

    def to_dict(self):
        data : Tensor = loadmat(self.path).get(self.patient_id, [])
        data = data.reshape(data.shape[1], data.shape[0])
        mean = data.mean()
        std = data.std()
        return {
            'answer': "YES" if self.adhd else "NO",
            'pre_prompt': "Given the following time series data, determine if the patient has ADHD.",
            'post_prompt': "Does this patient have ADHD?",
            'time_series': data,
            'time_series_text': "This is the time series data. It has a mean of {} and a standard deviation of {}.".format(mean, std)
        }

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
        return patient.to_dict() 

test_loader = DataLoader(
    ADHDChildren(Path("datasets")),
    shuffle=True,
    batch_size=5,
    collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
         batch, patch_size=PATCH_SIZE, normalize=True
    ),
)

for i, batch in enumerate(test_loader):
    print(f"Batch: {i}")
    for sample in batch:
        print("Question:", sample.get("pre_prompt", "N/A"))
        print("Answer:", sample.get("answer", "N/A"))

