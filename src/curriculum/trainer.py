from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from ..configs.training import TrainingConfig
from ..utility.runtime import Runtime
from .checkpoints import prepare_training_checkpoint
from .model_factory import OpenTSLMModelFactory


@dataclass(frozen=True)
class TrainingSummary:
    model_id: str
    runtime: str
    epochs: int
    steps: int
    checkpoint_loaded: bool
    final_loss: float
    checkpoint_path: Path
    elapsed_seconds: float


class Trainer:
    def __init__(
        self,
        runtime: Runtime,
        model_factory: OpenTSLMModelFactory,
    ):
        self.runtime = runtime
        self.model_factory = model_factory

    def train(
        self,
        dataloader: DataLoader,
        config: TrainingConfig,
    ) -> TrainingSummary:
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        checkpoint_path, checkpoint_loaded = prepare_training_checkpoint(
            config.model_id,
            config.checkpoint_root,
            fresh_start=config.fresh_start,
        )

        model = self.model_factory.create(
            config.model_id,
            enable_lora=config.enable_lora,
        )
        if checkpoint_loaded:
            print(f"[checkpoint] loading={checkpoint_path}")
            model.load_from_file(str(checkpoint_path))

        eos_token = model.get_eos_token()
        if not eos_token:
            raise ValueError("The selected model tokenizer has no EOS token")

        trainable_parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        if not trainable_parameters:
            raise ValueError("The model has no trainable parameters")

        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        step_count = 0
        final_loss = float("nan")
        start_time = perf_counter()
        model.train()

        for epoch_index in range(1, config.epochs + 1):
            epoch_loss = 0.0
            epoch_steps = 0

            for batch_index, batch in enumerate(dataloader, start=1):
                optimizer.zero_grad(set_to_none=True)
                self._append_eos_token(batch, eos_token)
                loss = model.compute_loss(batch)
                loss.backward()
                clip_grad_norm_(trainable_parameters, config.max_grad_norm)
                self.runtime.optimizer_step(optimizer)

                final_loss = float(loss.detach().item())
                epoch_loss += final_loss
                epoch_steps += 1
                step_count += 1
                print(
                    f"[train] epoch={epoch_index}/{config.epochs} "
                    f"step={batch_index} loss={final_loss:.6f}"
                )

            if epoch_steps == 0:
                raise ValueError("The dataloader produced no training batches")
            print(
                f"[train] epoch={epoch_index}/{config.epochs} "
                f"avg_loss={epoch_loss / epoch_steps:.6f}"
            )

        model.store_to_file(str(checkpoint_path))
        return TrainingSummary(
            model_id=config.model_id,
            runtime=self.runtime.kind,
            epochs=config.epochs,
            steps=step_count,
            checkpoint_loaded=checkpoint_loaded,
            final_loss=final_loss,
            checkpoint_path=checkpoint_path,
            elapsed_seconds=perf_counter() - start_time,
        )

    @staticmethod
    def _append_eos_token(batch: list[dict[str, object]], eos_token: str) -> None:
        for sample in batch:
            answer = str(sample["answer"]).strip()
            if not answer.endswith(eos_token):
                sample["answer"] = answer + eos_token
