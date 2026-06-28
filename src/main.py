from scipy.io import loadmat

from pathlib import Path

import os

from torch.utils.data import DataLoader, Dataset
from time import perf_counter
from torch.nn.utils import clip_grad_norm_

from torch import Tensor
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate
from opentslm.model_config import PATCH_SIZE
from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP
import torch

WINDOW_SIZE = 256
WINDOW_STRIDE = 256
WINDOW_MICROBATCH_SIZE = 1


class Patient:

    def __init__(self, path: Path, patient_id: str, adhd: bool = False):
        self.path = path
        self.patient_id = patient_id
        self.adhd = adhd

    def to_dict(self):
        data : Tensor = loadmat(self.path).get(self.patient_id, [])
        data = data.transpose()
        mean = data.mean()
        std = data.std()
        return {
            'answer': "YES" if self.adhd else "NO",
            'pre_prompt': "Given the following time series data, determine if the patient has ADHD.",
            'post_prompt': "Does this patient have ADHD?",
            'time_series': data,
            'time_series_text': ["This is the time series data. It has a mean of {} and a standard deviation of {}.".format(mean, std)]*data.shape[0]
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

dataloader = DataLoader(
    ADHDChildren(Path("datasets")),
    shuffle=True,
    batch_size=1,
    collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
         batch, patch_size=PATCH_SIZE, normalize=True
    ),
)

model = OpenTSLMSP(llm_id="google/gemma-3-270m")
model.enable_lora()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.01,
)

checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(parents=True, exist_ok=True)

step_count = 0
final_loss_value = float("nan")
start_time = perf_counter()
model.train()

for epoch_index in range(1, 2):
    epoch_loss_total = 0.0
    epoch_step_count = 0

    for batch_index, batch in enumerate(dataloader, start=1):
        windowed_batch = []
        optimizer.zero_grad(set_to_none=True)
        print(f"[debug] patient_step={batch_index}")
        loss = model.compute_loss(batch)
        loss.backward()
        clip_grad_norm_(
            model.parameters(),
            1.0,
        )
        optimizer.step()

        loss_value = float(loss.detach().item())
        final_loss_value = loss_value
        epoch_loss_total += loss_value
        epoch_step_count += 1
        step_count += 1

        print(
            f"[train] epoch={epoch_index}/1 "
            f"step={batch_index} loss={loss_value:.6f}"
        )

    average_epoch_loss = epoch_loss_total / max(epoch_step_count, 1)
    print(
        f"[train] epoch={epoch_index}/{1} "
        f"avg_loss={average_epoch_loss:.6f}"
    )

checkpoint_path = checkpoint_dir / "google__gemma-3-270m"
model.store_to_file(str(checkpoint_path))
elapsed_seconds = perf_counter() - start_time
# for i, batch in enumerate(dataloader):
#     print(f"Batch: {i}")
#     for sample in batch:
#         print("Question:", sample.get("pre_prompt", "N/A"))
#         print("Answer:", sample.get("answer", "N/A"))
#         print(sample["time_series"].shape)
#         print(sample["time_series_text"])
