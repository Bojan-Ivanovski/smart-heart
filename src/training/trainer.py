from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch
from torch.nn.utils import clip_grad_norm_

from ..curriculum import Curriculum, CurriculumStage
from ..data import SmartHeartDataset, build_dataloader
from ..models import OpenTSLMModel, OpenTSLMModelFactory
from ..runtime import Runtime

from .checkpoints import CheckpointManager
from .config import TrainingConfig


@dataclass(frozen=True)
class StageTrainingSummary:
    stage: str
    epochs: int
    steps: int
    final_loss: float
    initialized_from: Path | None
    checkpoint: Path
    elapsed_seconds: float


@dataclass(frozen=True)
class TrainingSummary:
    model_id: str
    architecture: str
    runtime: str
    stages: tuple[StageTrainingSummary, ...]

    @property
    def total_steps(self) -> int:
        return sum(stage.steps for stage in self.stages)


class Trainer:
    def __init__(
        self,
        runtime: Runtime,
        model_factory: OpenTSLMModelFactory,
    ) -> None:
        self.runtime = runtime
        self.model_factory = model_factory

    def train(
        self,
        dataset: SmartHeartDataset,
        config: TrainingConfig,
        *,
        stages: tuple[str, ...] | None = None,
    ) -> TrainingSummary:
        self._set_seed(config.seed)
        print(
            f"[train] loading model={config.model_id} "
            f"runtime={self.runtime.kind.value}",
            flush=True,
        )
        model = self.model_factory.create(
            config.model_id,
            architecture=config.architecture,
            enable_lora=config.enable_lora,
            gradient_checkpointing=config.gradient_checkpointing,
        )
        print("[train] model ready", flush=True)
        curriculum = Curriculum(dataset, self._eos_token(model))
        selected = self._select_stages(curriculum, stages)
        empty_stages = tuple(stage.name for stage in selected if len(stage) == 0)
        if empty_stages and stages is not None:
            names = ", ".join(empty_stages)
            raise ValueError(
                f"The selected dataset has no samples for: {names}. "
                "Choose a stage available for this source."
            )
        if empty_stages:
            print(
                f"[train] skipping stages with no samples: {', '.join(empty_stages)}",
                flush=True,
            )
            selected = tuple(stage for stage in selected if len(stage) > 0)
        if not selected:
            raise ValueError("The selected dataset has no curriculum samples.")
        all_stage_names = tuple(stage.name for stage in curriculum)
        manager = CheckpointManager(
            config.checkpoint_root,
            config.model_id,
            config.architecture,
        )
        if config.fresh_start:
            if selected[0].name != all_stage_names[0]:
                raise ValueError("A fresh curriculum must begin with stage1_mcq.")
            manager.clear()
        summaries: list[StageTrainingSummary] = []
        for stage in selected:
            initialized_from = manager.latest()
            if initialized_from is None and stage.name != all_stage_names[0]:
                raise FileNotFoundError(
                    f"Stage '{stage.name}' requires an existing numbered checkpoint. "
                    "Begin a new curriculum with stage1_mcq."
                )
            if initialized_from is not None:
                print(
                    f"[train] stage={stage.name} loading_checkpoint="
                    f"{initialized_from}",
                    flush=True,
                )
                model.load_from_file(str(initialized_from))

            checkpoint = manager.next_path(stage.name)
            print(
                f"[train] stage={stage.name} saving_checkpoint={checkpoint}",
                flush=True,
            )

            print(
                f"[train] stage={stage.name} samples={len(stage)} "
                f"epochs={config.epochs_per_stage} "
                f"batch_size={config.batch_size}",
                flush=True,
            )

            summaries.append(
                self._train_stage(
                    model,
                    stage,
                    config,
                    manager,
                    initialized_from,
                    checkpoint,
                )
            )

        return TrainingSummary(
            model_id=config.model_id,
            architecture=config.architecture.value,
            runtime=self.runtime.kind.value,
            stages=tuple(summaries),
        )

    def _train_stage(
        self,
        model: OpenTSLMModel,
        stage: CurriculumStage,
        config: TrainingConfig,
        manager: CheckpointManager,
        initialized_from: Path | None,
        checkpoint: Path,
    ) -> StageTrainingSummary:
        dataloader = build_dataloader(
            stage,
            batch_size=config.batch_size,
            shuffle=True,
            seed=config.seed,
        )
        parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        if not parameters:
            raise ValueError("The model has no trainable parameters.")
        optimizer = torch.optim.AdamW(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        model.train()
        started_at = perf_counter()
        steps = 0
        final_loss = float("nan")
        for epoch in range(1, config.epochs_per_stage + 1):
            epoch_loss = 0.0
            epoch_steps = 0
            for batch_index, batch in enumerate(dataloader, start=1):
                optimizer.zero_grad(set_to_none=True)
                loss = model.compute_loss(batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Stage '{stage.name}' produced a non-finite loss."
                    )
                loss.backward()
                clip_grad_norm_(parameters, config.max_grad_norm)
                self.runtime.optimizer_step(optimizer)

                final_loss = float(loss.detach().item())
                epoch_loss += final_loss
                epoch_steps += 1
                steps += 1
                print(
                    f"[train] stage={stage.name} "
                    f"epoch={epoch}/{config.epochs_per_stage} "
                    f"step={batch_index} loss={final_loss:.6f}",
                    flush=True,
                )
            if epoch_steps == 0:
                raise ValueError(f"Stage '{stage.name}' produced no batches.")
            print(
                f"[train] stage={stage.name} "
                f"epoch={epoch}/{config.epochs_per_stage} "
                f"average_loss={epoch_loss / epoch_steps:.6f}",
                flush=True,
            )
            manager.save(model, checkpoint)

        return StageTrainingSummary(
            stage=stage.name,
            epochs=config.epochs_per_stage,
            steps=steps,
            final_loss=final_loss,
            initialized_from=initialized_from,
            checkpoint=checkpoint,
            elapsed_seconds=perf_counter() - started_at,
        )

    @staticmethod
    def _select_stages(
        curriculum: Curriculum,
        requested: tuple[str, ...] | None,
    ) -> tuple[CurriculumStage, ...]:
        available = tuple(curriculum)
        if requested is None:
            return available
        if not requested:
            raise ValueError("At least one curriculum stage must be selected.")
        if len(set(requested)) != len(requested):
            raise ValueError("Curriculum stages cannot be repeated.")

        selected = tuple(curriculum.get(name) for name in requested)
        positions = [available.index(stage) for stage in selected]
        if positions != sorted(positions):
            raise ValueError("Curriculum stages must follow their canonical order.")
        return selected

    @staticmethod
    def _set_seed(seed: int) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _eos_token(model: OpenTSLMModel) -> str:
        eos_token = model.get_eos_token()
        if not eos_token:
            raise ValueError("The selected model tokenizer has no EOS token.")
        return str(eos_token)
