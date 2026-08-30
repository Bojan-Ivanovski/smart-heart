import re
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..configs.evaluation import EvaluationConfig
from ..utility.runtime import Runtime
from .checkpoints import require_checkpoint
from .model_factory import OpenTSLMModelFactory


@dataclass(frozen=True)
class EvaluationSummary:
    split: str
    evaluated_samples: int
    correct_predictions: int
    accuracy: float
    checkpoint_path: Path


class Evaluator:
    def __init__(
        self,
        runtime: Runtime,
        model_factory: OpenTSLMModelFactory,
    ):
        self.runtime = runtime
        self.model_factory = model_factory

    def evaluate(
        self,
        dataloaders: dict[str, DataLoader],
        config: EvaluationConfig,
    ) -> list[EvaluationSummary]:
        path = require_checkpoint(config.model_id, config.checkpoint_root)
        model = self.model_factory.create(
            config.model_id,
            enable_lora=config.enable_lora,
        )
        print(f"[checkpoint] loading={path}")
        model.load_from_file(str(path))
        model.eval()

        summaries = []
        with torch.inference_mode():
            for split_name, dataloader in dataloaders.items():
                summaries.append(
                    self._evaluate_split(
                        model,
                        dataloader,
                        split_name,
                        path,
                        config,
                    )
                )
        return summaries

    @staticmethod
    def _evaluate_split(
        model: object,
        dataloader: DataLoader,
        split_name: str,
        path: Path,
        config: EvaluationConfig,
    ) -> EvaluationSummary:
        evaluated_samples = 0
        correct_predictions = 0

        for batch in dataloader:
            if config.max_samples is not None:
                remaining = config.max_samples - evaluated_samples
                if remaining <= 0:
                    break
                batch = batch[:remaining]

            outputs = model.generate(batch, max_new_tokens=config.max_new_tokens)
            for sample, output in zip(batch, outputs):
                expected = str(sample["answer"]).upper()
                predicted = Evaluator._extract_label(output)
                is_correct = predicted == expected
                correct_predictions += int(is_correct)
                evaluated_samples += 1
                print(
                    f"[evaluate] split={split_name} sample={evaluated_samples} "
                    f"expected={expected} predicted={predicted or 'UNKNOWN'} "
                    f"output={output!r}"
                )

        if evaluated_samples == 0:
            raise ValueError(f"The {split_name} dataloader produced no samples")
        return EvaluationSummary(
            split=split_name,
            evaluated_samples=evaluated_samples,
            correct_predictions=correct_predictions,
            accuracy=correct_predictions / evaluated_samples,
            checkpoint_path=path,
        )

    @staticmethod
    def _extract_label(output: str) -> str | None:
        normalized_output = output.strip().upper()
        match = re.match(r"(YES|NO)", normalized_output)
        if match is None:
            match = re.search(r"\b(YES|NO)\b", normalized_output)
        return match.group(1) if match else None
