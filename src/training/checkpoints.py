import hashlib
import re
from pathlib import Path

from ..models import ModelArchitecture, OpenTSLMModel


_CHECKPOINT_PATTERN = re.compile(
    r"checkpoint_(?P<sequence>[1-9][0-9]*)_stage_(?P<stage>[1-9][0-9]*)[.]pt"
)
_STAGE_PATTERN = re.compile(r"stage(?P<number>[1-9][0-9]*)_[a-z0-9_]+")


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

    def latest(self) -> Path | None:
        checkpoints = self._numbered_checkpoints()
        if checkpoints:
            return checkpoints[-1][1]
        legacy = self._legacy_checkpoints()
        if legacy:
            names = ", ".join(path.name for path in legacy)
            raise FileNotFoundError(
                "Found legacy stage-named checkpoints but no numbered checkpoint. "
                "Rename the latest model to 'checkpoint_1_stage_N.pt' before "
                f"continuing: {names}."
            )
        return None

    def require_latest(self) -> Path:
        path = self.latest()
        if path is None:
            raise FileNotFoundError(
                f"No numbered curriculum checkpoint exists in '{self.directory}'."
            )
        return path

    def next_path(self, stage_name: str) -> Path:
        stage_match = _STAGE_PATTERN.fullmatch(stage_name)
        if stage_match is None:
            raise ValueError(f"Invalid curriculum stage name '{stage_name}'.")
        checkpoints = self._numbered_checkpoints()
        sequence = checkpoints[-1][0] + 1 if checkpoints else 1
        path = (
            self.directory
            / f"checkpoint_{sequence}_stage_{stage_match.group('number')}.pt"
        ).resolve()
        self._validate_checkpoint_path(path)
        return path

    def save(self, model: OpenTSLMModel, path: Path) -> Path:
        path = path.resolve()
        self._validate_checkpoint_path(path)
        if _CHECKPOINT_PATTERN.fullmatch(path.name) is None:
            raise ValueError(f"Invalid numbered checkpoint path '{path}'.")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(".tmp")
        if temporary_path.exists():
            temporary_path.unlink()
        model.store_to_file(str(temporary_path))
        temporary_path.replace(path)
        return path

    def clear(self) -> None:
        if not self.directory.is_dir():
            return
        for path in self.directory.glob("*.pt"):
            self._validate_checkpoint_path(path)
            if path.is_file():
                path.unlink()

    def _numbered_checkpoints(self) -> list[tuple[int, Path]]:
        if not self.directory.is_dir():
            return []
        checkpoints: list[tuple[int, Path]] = []
        sequences: set[int] = set()
        for path in self.directory.glob("checkpoint_*_stage_*.pt"):
            match = _CHECKPOINT_PATTERN.fullmatch(path.name)
            if match is None:
                continue
            self._validate_checkpoint_path(path)
            sequence = int(match.group("sequence"))
            if sequence in sequences:
                raise ValueError(
                    f"Duplicate checkpoint sequence number {sequence} in "
                    f"'{self.directory}'."
                )
            sequences.add(sequence)
            checkpoints.append((sequence, path))
        return sorted(checkpoints, key=lambda item: item[0])

    def _legacy_checkpoints(self) -> list[Path]:
        if not self.directory.is_dir():
            return []
        return sorted(
            path for path in self.directory.glob("stage*.pt") if path.is_file()
        )

    def _validate_checkpoint_path(self, path: Path) -> None:
        if path.parent != self.directory:
            raise ValueError("Checkpoint path escaped the model directory.")
        if path.exists() and not path.is_file():
            raise ValueError(f"Checkpoint path is not a file: '{path}'.")
