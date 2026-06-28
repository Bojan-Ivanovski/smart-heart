

from torch.utils.data import DataLoader, Dataset
from time import perf_counter
from torch.nn.utils import clip_grad_norm_
from .loaders.children import ADHDChildren
from .loaders.gameplay import ADHDGameplay
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate
from opentslm.model_config import PATCH_SIZE
from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP
from pathlib import Path
import torch

dataloader = DataLoader(
    ADHDGameplay(Path("datasets")),
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
#         print(sample["time_series"])
#         print(sample["time_series_text"])
