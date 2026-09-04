# Source Code

SmartHeart separates dataset loading, runtime selection, model construction, and training so each concern can evolve independently.

## Modules

- `main.py`: command-line entry point for dataset previews and training.
- `builders.py`: canonical source selection, persisted split selection, and OpenTSLM `DataLoader` construction.
- `configs/`: validated, immutable training configuration.
- `curriculum/`: OpenTSLM model construction, training, metrics, and checkpoints.
- `loaders/`: canonical manifest and NPZ dataset adapter.
- `utility/`: accelerator selection and reusable time-series windowing helpers.

## Dataset Splits

The canonical dataset stores label-stratified assignments of 60% training, 20%
validation, and 20% testing. Splits are made by patient rather than by time-series
window, so every recording and window belonging to one patient remains in one
split. Runtime code reads these persisted assignments; `--seed` controls shuffled
sample order, not split membership.

## Commands

Previewing a dataset is the safe default:

```powershell
python -m src.main
python -m src.main --mode preview --dataset children
python -m src.main --mode preview --dataset cognitive_function
python -m src.main --mode preview --dataset all
```

Training must be requested explicitly:

```powershell
python -m src.main --mode train --dataset gameplay --device auto
```

Each model ID maps to one file under `--checkpoint-root`. If that file exists,
training loads its model weights automatically and overwrites it after the run.
Use `--fresh-start` to delete the matching file before initializing the model:

```powershell
python -m src.main --mode train --model-id google/gemma-3-270m --fresh-start
```

The checkpoint format stores model weights but not optimizer or completed-epoch
state, so loading it warm-starts another configured training run.

Training targets append the selected tokenizer's EOS token to the `YES` and `NO`
labels. This teaches generation to stop after one classification token. Checkpoints
created before this behavior should be retrained once with `--fresh-start`.

Evaluate a checkpoint with generated `YES` or `NO` predictions. Validation and
test results are reported separately, and the model is loaded only once:

```powershell
python -m src.main --mode evaluate --dataset gameplay --evaluation-split both
```

Use `--max-evaluation-samples` to cap the number of samples evaluated per split.
Capped evaluation uses a deterministic shuffle instead of taking adjacent windows
from the first patient. Evaluation uses `--seed` for that sample order. Generation
stops when the model emits the EOS token learned during training.

Run `python -m src.main --help` for the complete set of dataset, windowing, optimizer, accelerator, LoRA, seed, and checkpoint options.
