import json
from dataclasses import dataclass
from pathlib import Path

import torch

from ..curriculum import Curriculum, CurriculumStage
from ..data import SmartHeartDataset, build_dataloader
from ..domain.patient import DatasetSplit
from ..models import OpenTSLMModel, OpenTSLMModelFactory
from ..runtime import Runtime
from ..training import CheckpointManager

from .config import EvaluationConfig
from .metrics import MetricValues, PredictionPair, evaluate_stage


_GENERATION_TOKEN_LIMIT = 512


@dataclass(frozen=True)
class EvaluationSummary:
    stage: str
    split: str
    samples: int
    checkpoint: Path
    predictions_path: Path
    metrics_path: Path
    metrics: MetricValues


class Evaluator:
    def __init__(
        self,
        runtime: Runtime,
        model_factory: OpenTSLMModelFactory,
    ) -> None:
        self.runtime = runtime
        self.model_factory = model_factory

    def evaluate(
        self,
        dataset: SmartHeartDataset,
        config: EvaluationConfig,
        *,
        split: DatasetSplit | str,
        stages: tuple[str, ...] | None = None,
    ) -> tuple[EvaluationSummary, ...]:
        manager = CheckpointManager(
            config.checkpoint_root,
            config.model_id,
            config.architecture,
        )
        model = self.model_factory.create(
            config.model_id,
            architecture=config.architecture,
            enable_lora=config.enable_lora,
            gradient_checkpointing=config.gradient_checkpointing,
        )
        eos_token = model.get_eos_token()
        if not eos_token:
            raise ValueError("The selected model tokenizer has no EOS token.")
        curriculum = Curriculum(dataset, str(eos_token))
        split_name = DatasetSplit(split).value
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
                "[evaluate] skipping stages with no samples: "
                f"{', '.join(empty_stages)}"
            )
            selected = tuple(stage for stage in selected if len(stage) > 0)
        if not selected:
            raise ValueError("The selected dataset has no curriculum samples.")

        summaries = []
        for stage in selected:
            checkpoint = manager.require(stage.name)
            model.load_from_file(str(checkpoint))
            model.eval()
            summaries.append(
                self._evaluate_stage(
                    model,
                    stage,
                    config,
                    split_name,
                    checkpoint,
                )
            )
        return tuple(summaries)

    def _evaluate_stage(
        self,
        model: OpenTSLMModel,
        stage: CurriculumStage,
        config: EvaluationConfig,
        split: str,
        checkpoint: Path,
    ) -> EvaluationSummary:
        dataloader = build_dataloader(
            stage,
            batch_size=config.batch_size,
            shuffle=False,
        )
        prediction_pairs: list[PredictionPair] = []
        rows: list[dict[str, object]] = []
        total_batches = len(dataloader)
        print(
            f"[evaluate] stage={stage.name} split={split} "
            f"samples={len(stage)} batches={total_batches}",
            flush=True,
        )
        with torch.inference_mode():
            for batch_index, batch in enumerate(dataloader, start=1):
                generated = model.generate(
                    batch,
                    max_new_tokens=_GENERATION_TOKEN_LIMIT,
                )
                if len(generated) != len(batch):
                    raise ValueError(
                        f"Stage '{stage.name}' generated {len(generated)} outputs "
                        f"for a batch of {len(batch)} samples."
                    )
                for sample, output in zip(batch, generated):
                    predicted = self._without_eos(str(output), stage.eos_token)
                    expected = self._without_eos(
                        str(sample["answer"]),
                        stage.eos_token,
                    )
                    prediction_pairs.append((predicted, expected))
                    rows.append(
                        {
                            "patient_id": sample.get("patient_id"),
                            "recording_ids": sample.get("recording_ids"),
                            "segment_ids": sample.get("segment_ids"),
                            "question_id": sample.get("question_id"),
                            "pre_prompt": sample["pre_prompt"],
                            "post_prompt": sample["post_prompt"],
                            "expected": expected,
                            "predicted": predicted,
                        }
                    )
                if (
                    batch_index == 1
                    or batch_index % 10 == 0
                    or batch_index == total_batches
                ):
                    running_metrics = evaluate_stage(
                        stage.name,
                        prediction_pairs,
                    )
                    metric_text = " ".join(
                        f"{name}={value:.4f}"
                        for name, value in running_metrics.items()
                    )
                    print(
                        f"[evaluate] stage={stage.name} split={split} "
                        f"batch={batch_index}/{total_batches} "
                        f"samples={len(prediction_pairs)}/{len(stage)} "
                        f"{metric_text}",
                        flush=True,
                    )

        metrics = evaluate_stage(stage.name, prediction_pairs)
        output_directory = (
            config.output_root
            / checkpoint.parent.parent.name
            / checkpoint.parent.name
            / stage.name
            / split
        )
        predictions_path = output_directory / "predictions.jsonl"
        metrics_path = output_directory / "metrics.json"
        self._write_jsonl(predictions_path, rows)
        self._write_json(
            metrics_path,
            {
                "stage": stage.name,
                "split": split,
                "samples": len(rows),
                "checkpoint": str(checkpoint),
                "metrics": metrics,
            },
        )
        return EvaluationSummary(
            stage=stage.name,
            split=split,
            samples=len(rows),
            checkpoint=checkpoint,
            predictions_path=predictions_path,
            metrics_path=metrics_path,
            metrics=metrics,
        )

    @staticmethod
    def _select_stages(
        curriculum: Curriculum,
        requested: tuple[str, ...] | None,
    ) -> tuple[CurriculumStage, ...]:
        if requested is None:
            return tuple(curriculum)
        if not requested:
            raise ValueError("At least one curriculum stage must be selected.")
        if len(set(requested)) != len(requested):
            raise ValueError("Evaluation stages cannot be repeated.")
        return tuple(curriculum.get(name) for name in requested)

    @staticmethod
    def _without_eos(value: str, eos_token: str) -> str:
        if eos_token in value:
            value = value.split(eos_token, maxsplit=1)[0]
        return value.strip()

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
        temporary.replace(path)

    @staticmethod
    def _write_json(path: Path, value: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=True)
            handle.write("\n")
        temporary.replace(path)
