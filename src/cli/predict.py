import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..data import DatasetSplit, SmartHeartDataset
from ..inference import PredictionConfig, PredictionResult, Predictor
from ..inference import save_time_series_plot
from ..models import ModelArchitecture, OpenTSLMModelFactory
from ..runtime import RuntimeKind, resolve_runtime

from .common import (
    DEFAULT_CHECKPOINT_ROOT,
    DEFAULT_DATASET_ROOT,
    DEFAULT_OUTPUT_ROOT,
    CurriculumStageName,
    MODEL_SOURCE_DATASET,
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
        typer.Option(help="Curriculum prompt to use with the latest checkpoint."),
    ] = CurriculumStageName.DIAGNOSTIC_COT,
    index: Annotated[
        int | None,
        typer.Option(
            min=0,
            help="Curriculum sample index; omit to select one randomly.",
        ),
    ] = None,
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
    plot_output: Annotated[
        Path,
        typer.Option(
            dir_okay=False,
            resolve_path=True,
            help="Save a visualization of the time-series inputs.",
        ),
    ] = DEFAULT_OUTPUT_ROOT / "prediction_time_series.png",
) -> None:
    """Generate one checkpoint prediction without running evaluation metrics."""
    dataset = SmartHeartDataset(
        dataset_root,
        split=split,
        source_dataset=MODEL_SOURCE_DATASET,
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
    plot_path = save_time_series_plot(
        result.time_series,
        result.time_series_text,
        plot_output,
        patient_id=result.patient_id,
    )

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(f"{output.suffix}.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                {
                    "stage": result.stage,
                    "index": result.index,
                    "patient_id": result.patient_id,
                    "recording_ids": result.recording_ids,
                    "segment_ids": result.segment_ids,
                    "checkpoint": str(result.checkpoint),
                    "pre_prompt": result.pre_prompt,
                    "post_prompt": result.post_prompt,
                    "prediction": result.prediction,
                    "expected_answer": result.expected_answer,
                    "time_series_count": len(result.time_series),
                    "time_series_text": result.time_series_text,
                    "plot": str(plot_path),
                },
                handle,
                indent=2,
                ensure_ascii=True,
            )
            handle.write("\n")
        temporary.replace(output)

    _render_prediction(result, plot_path)


def _render_prediction(result: PredictionResult, plot_path: Path) -> None:
    console = Console()
    metadata = Table.grid(padding=(0, 2))
    metadata.add_column(style="bold cyan", no_wrap=True)
    metadata.add_column()
    metadata.add_row("Stage", result.stage)
    metadata.add_row("Sample index", str(result.index))
    metadata.add_row("Patient", result.patient_id)
    metadata.add_row("Checkpoint", str(result.checkpoint))
    metadata.add_row("Input series", str(len(result.time_series)))
    metadata.add_row("Visualization", str(plot_path))

    shown = result.time_series_text[:6]
    series_summary = "\n".join(
        f"{index}. {description}"
        for index, description in enumerate(shown, start=1)
    )
    if len(result.time_series_text) > len(shown):
        series_summary += (
            f"\n... {len(result.time_series_text) - len(shown)} additional "
            "series are included in the model input and heatmap."
        )

    console.rule("[bold]OpenTSLM EEG Prediction")
    console.print(metadata)
    console.print(
        Panel(
            Text(series_summary),
            title="Time-Series Input",
            border_style="cyan",
        )
    )
    console.print(
        Panel(
            Text(result.pre_prompt),
            title="Prompt: Context",
            border_style="blue",
        )
    )
    console.print(
        Panel(
            Text(result.post_prompt),
            title="Prompt: Question and Output Contract",
            border_style="blue",
        )
    )
    console.print(
        Panel(
            Text(result.prediction),
            title="Model Prediction",
            border_style="green",
        )
    )
    console.print(
        Panel(
            Text(result.expected_answer),
            title="Expected Answer",
            border_style="yellow",
        )
    )
