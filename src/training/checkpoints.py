import hashlib
import re
from pathlib import Path

from ..models import ModelArchitecture, OpenTSLMModel


class CheckpointManager:
    def __init__(
        self,
        root: Path,
        model_id: str,
        architecture: ModelArchitecture,
    ) -> None:
        digest = hashlib.sha256(model_id.encode("utf-8")).hexdigest()[:8]
        model_name = re.sub(r"[^A-Za-z0-9._-]+", "__", model_id).strip("._-")
        if not model_name:
            raise ValueError("model_id must produce a valid checkpoint name.")
        self.root = root.resolve()
        self.directory = (
            self.root / f"{model_name}__{digest}" / architecture.value
        ).resolve()
        if not self.directory.is_relative_to(self.root):
            raise ValueError("Checkpoint directory must be inside checkpoint_root.")

    def path_for(self, stage_name: str) -> Path:
        if not re.fullmatch(r"stage[1-9][0-9]*_[a-z0-9_]+", stage_name):
            raise ValueError(f"Invalid curriculum stage name '{stage_name}'.")
        path = (self.directory / f"{stage_name}.pt").resolve()
        if path.parent != self.directory:
            raise ValueError("Checkpoint path escaped the model directory.")
        return path

    def find_initial_checkpoint(
        self,
        stage_name: str,
        previous_stage_name: str | None,
    ) -> Path | None:
        current = self.path_for(stage_name)
        self._validate_checkpoint_path(current)
        if current.is_file():
            return current
        if previous_stage_name is None:
            return None
        previous = self.path_for(previous_stage_name)
        self._validate_checkpoint_path(previous)
        if previous.is_file():
            return previous
        raise FileNotFoundError(
            f"Stage '{stage_name}' requires checkpoint '{previous}'."
        )

    def require(self, stage_name: str) -> Path:
        path = self.path_for(stage_name)
        self._validate_checkpoint_path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"No checkpoint exists for stage '{stage_name}' at '{path}'."
            )
        return path

    def save(self, model: OpenTSLMModel, stage_name: str) -> Path:
        path = self.path_for(stage_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(".tmp")
        if temporary_path.exists():
            temporary_path.unlink()
        model.store_to_file(str(temporary_path))
        temporary_path.replace(path)
        return path

    def clear(self, stage_names: tuple[str, ...]) -> None:
        for stage_name in stage_names:
            path = self.path_for(stage_name)
            if path.is_file():
                path.unlink()

    @staticmethod
    def _validate_checkpoint_path(path: Path) -> None:
        if path.exists() and not path.is_file():
            raise ValueError(f"Checkpoint path is not a file: '{path}'.")
