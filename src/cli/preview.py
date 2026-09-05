import json
from pathlib import Path
from typing import Annotated

import typer

from ..curriculum import Curriculum
from ..data import SmartHeartDataset
from ..domain.patient import DatasetSplit

from .common import (
    DEFAULT_DATASET_ROOT,
    CurriculumStageName,
    DatasetSource,
)


_PREVIEW_EOS_TOKEN = "<EOS>"


def preview(
    dataset_root: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help="Canonical dataset directory.",
        ),
    ] = DEFAULT_DATASET_ROOT,
    split: Annotated[
        DatasetSplit | None,
        typer.Option(help="Restrict the preview to one persisted patient split."),
    ] = None,
    source: Annotated[
        DatasetSource,
        typer.Option(help="Restrict the preview to one source dataset."),
    ] = DatasetSource.ALL,
    window_size: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Override window size while preserving each segment's overlap ratio.",
        ),
    ] = None,
    stage: Annotated[
        CurriculumStageName,
        typer.Option(help="Curriculum stage whose model input and target are shown."),
    ] = CurriculumStageName.MCQ,
    index: Annotated[
        int,
        typer.Option(min=0, help="Curriculum sample index to inspect."),
    ] = 0,
) -> None:
    """Inspect the complete model input and target for one curriculum sample."""
    dataset = SmartHeartDataset(
        dataset_root,
        split=split,
        source_dataset=source.filter_value,
        window_size=window_size,
    )
    curriculum_stage = Curriculum(dataset, _PREVIEW_EOS_TOKEN).get(stage.value)
    if not curriculum_stage:
        raise typer.BadParameter(
            f"stage '{stage.value}' has no samples for the selected filters.",
            param_hint="--stage",
        )
    if index >= len(curriculum_stage):
        raise typer.BadParameter(
            f"index must be smaller than the stage size ({len(curriculum_stage)}).",
            param_hint="--index",
        )
    sample = curriculum_stage[index]
    typer.echo(f"stage: {curriculum_stage.name}")
    typer.echo(f"stage_samples: {len(curriculum_stage)}")
    typer.echo(f"signal_windows: {len(dataset)}")
    typer.echo(f"patient: {sample['patient_id']}")
    typer.echo(f"recordings: {', '.join(sample['recording_ids'])}")
    typer.echo(f"segments: {', '.join(sample['segment_ids'])}")
    typer.echo(f"split: {sample['split']}")
    typer.echo(f"source: {sample['source_dataset']}")
    typer.echo("\nPRE-PROMPT")
    typer.echo(sample["pre_prompt"])
    typer.echo("\nTIME SERIES")
    for series_index, (series, description) in enumerate(
        zip(sample["time_series"], sample["time_series_text"]),
        start=1,
    ):
        typer.echo(f"[{series_index}] shape={tuple(series.shape)} {description}")
    typer.echo("\nPOST-PROMPT")
    typer.echo(sample["post_prompt"])
    typer.echo("\nREFERENCE ANALYSIS (NOT INCLUDED IN MODEL PROMPT)")
    for analysis_index, analysis in enumerate(
        sample.get("reference_analyses", []),
        start=1,
    ):
        typer.echo(f"[{analysis_index}] {json.dumps(analysis, sort_keys=True)}")
    typer.echo("\nEXPECTED ANSWER")
    typer.echo(sample["answer"].removesuffix(_PREVIEW_EOS_TOKEN))
