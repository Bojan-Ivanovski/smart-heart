import json
from pathlib import Path
from typing import Annotated

import typer

from ..data import DatasetSplit, SmartHeartDataset
from ..inference import PredictionConfig, Predictor
from ..models import ModelArchitecture, OpenTSLMModelFactory
from ..runtime import RuntimeKind, resolve_runtime

from .common import (
    DEFAULT_CHECKPOINT_ROOT,
    DEFAULT_DATASET_ROOT,
    CurriculumStageName,
)


def predict(
    model_id: Annotated[
        str,
        typer.Option(help="Hugging Face ID of the base language model."),
    ] = "google/gemma-3-1b-pt",
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
    split: Annotated[
        DatasetSplit,
        typer.Option(help="Patient split from which to select one input."),
    ] = DatasetSplit.TEST,
    stage: Annotated[
        CurriculumStageName,
        typer.Option(help="Checkpoint and curriculum prompt to use."),
    ] = CurriculumStageName.DIAGNOSTIC_COT,
    index: Annotated[
        int,
        typer.Option(min=0, help="Curriculum sample index to predict."),
    ] = 0,
    window_size: Annotated[
        int,
        typer.Option(min=1, help="Window size used to train this stage checkpoint."),
    ] = 128,
    prompt: Annotated[
        str | None,
        typer.Option(help="Append an instruction to the stage's standard prompt."),
    ] = None,
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
    output: Annotated[
        Path | None,
        typer.Option(
            dir_okay=False,
            resolve_path=True,
            help="Write the prediction and prompt context to a JSON file.",
        ),
    ] = None,
) -> None:
    """Generate one checkpoint prediction without running evaluation metrics."""
    dataset = SmartHeartDataset(
        dataset_root,
        split=split,
        window_size=window_size,
    )
    runtime = resolve_runtime(device)
    result = Predictor(runtime, OpenTSLMModelFactory(runtime)).predict(
        dataset,
        PredictionConfig(
            model_id=model_id,
            architecture=architecture,
            enable_lora=enable_lora,
            gradient_checkpointing=gradient_checkpointing,
            checkpoint_root=checkpoint_root,
        ),
        stage_name=stage.value,
        index=index,
        prompt=prompt,
    )

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(f"{output.suffix}.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                {
                    "stage": result.stage,
                    "patient_id": result.patient_id,
                    "recording_ids": result.recording_ids,
                    "segment_ids": result.segment_ids,
                    "checkpoint": str(result.checkpoint),
                    "pre_prompt": result.pre_prompt,
                    "post_prompt": result.post_prompt,
                    "prediction": result.prediction,
                },
                handle,
                indent=2,
                ensure_ascii=True,
            )
            handle.write("\n")
        temporary.replace(output)

    typer.echo(f"stage: {result.stage}")
    typer.echo(f"patient: {result.patient_id}")
    typer.echo(f"checkpoint: {result.checkpoint}")
    typer.echo("\nPROMPT")
    typer.echo(result.pre_prompt)
    typer.echo(result.post_prompt)
    typer.echo("\nPREDICTION")
    typer.echo(result.prediction)
