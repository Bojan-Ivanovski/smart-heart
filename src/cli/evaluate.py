from pathlib import Path
from typing import Annotated

import typer

from ..data import SmartHeartDataset
from ..evaluation import EvaluationConfig, Evaluator
from ..models import ModelArchitecture, OpenTSLMModelFactory
from ..runtime import RuntimeKind, resolve_runtime

from .common import (
    DEFAULT_CHECKPOINT_ROOT,
    DEFAULT_DATASET_ROOT,
    DEFAULT_OUTPUT_ROOT,
    CurriculumStageName,
    EvaluationSplit,
    stage_names,
)


def evaluate(
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
    split: Annotated[
        EvaluationSplit,
        typer.Option(help="Held-out patient split to evaluate."),
    ] = EvaluationSplit.VALIDATION,
    dataset_root: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=False,
            resolve_path=True,
            help="Canonical dataset directory.",
        ),
    ] = DEFAULT_DATASET_ROOT,
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
            help="Stage to evaluate; repeat to select several. Defaults to all.",
        ),
    ] = None,
    batch_size: Annotated[
        int,
        typer.Option(min=1, help="Samples per generation batch."),
    ] = 1,
    enable_lora: Annotated[
        bool,
        typer.Option(help="Enable LoRA before loading an SP checkpoint."),
    ] = True,
    gradient_checkpointing: Annotated[
        bool,
        typer.Option(help="Enable Flamingo gradient checkpointing."),
    ] = False,
    checkpoint_root: Annotated[
        Path,
        typer.Option(file_okay=False, resolve_path=True),
    ] = DEFAULT_CHECKPOINT_ROOT,
    output_root: Annotated[
        Path,
        typer.Option(file_okay=False, resolve_path=True),
    ] = DEFAULT_OUTPUT_ROOT,
) -> None:
    """Evaluate exact stage checkpoints on a held-out patient split."""
    dataset = SmartHeartDataset(
        dataset_root,
        split=split.value,
        window_size=window_size,
    )
    runtime = resolve_runtime(device)
    config = EvaluationConfig(
        model_id=model_id,
        architecture=architecture,
        batch_size=batch_size,
        enable_lora=enable_lora,
        gradient_checkpointing=gradient_checkpointing,
        checkpoint_root=checkpoint_root,
        output_root=output_root,
    )
    factory = OpenTSLMModelFactory(runtime)
    summaries = Evaluator(runtime, factory).evaluate(
        dataset,
        config,
        split=split.value,
        stages=stage_names(stage),
    )
    for summary in summaries:
        metrics = " ".join(
            f"{name}={value:.4f}"
            for name, value in summary.metrics.items()
        )
        typer.echo(
            f"{summary.stage}: split={summary.split} "
            f"samples={summary.samples} {metrics}"
        )
        typer.echo(f"predictions: {summary.predictions_path}")
