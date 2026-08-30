from pathlib import Path


def checkpoint_path(model_id: str, checkpoint_root: Path) -> Path:
    safe_model_id = model_id.replace("/", "__").replace("\\", "__")
    path = checkpoint_root / safe_model_id
    if path.resolve().parent != checkpoint_root.resolve():
        raise ValueError("Checkpoint path must be inside checkpoint_root")
    return path


def prepare_training_checkpoint(
    model_id: str,
    checkpoint_root: Path,
    *,
    fresh_start: bool,
) -> tuple[Path, bool]:
    path = checkpoint_path(model_id, checkpoint_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    _validate_file_path(path)
    if fresh_start and path.is_file():
        path.unlink()
        print(f"[checkpoint] removed={path}")
    return path, path.is_file()


def require_checkpoint(model_id: str, checkpoint_root: Path) -> Path:
    path = checkpoint_path(model_id, checkpoint_root)
    _validate_file_path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"No checkpoint found for model '{model_id}' at '{path}'. "
            "Run training before evaluation."
        )
    return path


def _validate_file_path(path: Path) -> None:
    if path.exists() and not path.is_file():
        raise ValueError(f"Checkpoint path is not a file: {path}")
