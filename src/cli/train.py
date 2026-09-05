from pathlib import Path
from typing import Annotated

import typer

from ..data import SmartHeartDataset
from ..models import ModelArchitecture, OpenTSLMModelFactory
from ..runtime import RuntimeKind, resolve_runtime
from ..training import Trainer, TrainingConfig

from .common import (
    DEFAULT_CHECKPOINT_ROOT,
    DEFAULT_DATASET_ROOT,
    CurriculumStageName,
    DatasetSource,
    stage_names,
)


def train(
    model_id: Annotated[
        str,
        typer.Option(help="Hugging Face ID of the base language model."),
    ] = "google/gemma-3-270m",
    architecture: Annotated[
        ModelArchitecture,
        typer.Option(help="OpenTSLM integration architecture."),
    ] = ModelArchitecture.SP,
    device: Annotated[
        RuntimeKind,
        typer.Option(help="Execution device; auto prefers CUDA when available."),
    ] = RuntimeKind.AUTO,
    dataset_root: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=False,
            resolve_path=True,
            help="Canonical dataset directory.",
        ),
    ] = DEFAULT_DATASET_ROOT,
    source: Annotated[
        DatasetSource,
        typer.Option(help="Restrict training to one source dataset."),
    ] = DatasetSource.ALL,
    window_size: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Override window size while preserving each segment's overlap ratio.",
        ),
    ] = None,
    stage: Annotated[
        list[CurriculumStageName] | None,
        typer.Option(
            "--stage",
            help="Stage to train; repeat in curriculum order. Defaults to all.",
        ),
    ] = None,
    epochs: Annotated[
        int,
        typer.Option(min=1, help="Epochs to run for every selected stage."),
    ] = 1,
    batch_size: Annotated[
        int,
        typer.Option(min=1, help="Samples per optimization batch."),
    ] = 1,
    learning_rate: Annotated[
        float,
        typer.Option(min=1e-12, help="AdamW learning rate."),
    ] = 1e-4,
    weight_decay: Annotated[
        float,
        typer.Option(min=0.0, help="AdamW weight decay."),
    ] = 0.01,
    max_grad_norm: Annotated[
        float,
        typer.Option(min=1e-12, help="Gradient clipping norm."),
    ] = 1.0,
    enable_lora: Annotated[
        bool,
        typer.Option(help="Enable LoRA for OpenTSLMSP."),
    ] = True,
    gradient_checkpointing: Annotated[
        bool,
        typer.Option(help="Enable Flamingo gradient checkpointing."),
    ] = False,
    checkpoint_root: Annotated[
        Path,
        typer.Option(file_okay=False, resolve_path=True),
    ] = DEFAULT_CHECKPOINT_ROOT,
    fresh_start: Annotated[
        bool,
        typer.Option(help="Delete this model's curriculum checkpoints first."),
    ] = False,
    seed: Annotated[int, typer.Option(help="Training and shuffle seed.")] = 42,
) -> None:
    """Train one or more curriculum stages in canonical order."""
    dataset = SmartHeartDataset(
        dataset_root,
        split="train",
        source_dataset=source.filter_value,
        window_size=window_size,
    )
    runtime = resolve_runtime(device)
    config = TrainingConfig(
        model_id=model_id,
        architecture=architecture,
        epochs_per_stage=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        enable_lora=enable_lora,
        gradient_checkpointing=gradient_checkpointing,
        checkpoint_root=checkpoint_root,
        fresh_start=fresh_start,
        seed=seed,
    )
    factory = OpenTSLMModelFactory(runtime)
    summary = Trainer(runtime, factory).train(
        dataset,
        config,
        stages=stage_names(stage),
    )
    typer.echo(
        f"Training complete: model={summary.model_id} "
        f"runtime={summary.runtime} steps={summary.total_steps}"
    )
    for result in summary.stages:
        typer.echo(
            f"{result.stage}: loss={result.final_loss:.6f} "
            f"checkpoint={result.checkpoint}"
        )
