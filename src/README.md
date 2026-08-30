# Source Code

SmartHeart separates dataset loading, runtime selection, model construction, and training so each concern can evolve independently.

## Modules

- `main.py`: command-line entry point for dataset previews and training.
- `builders.py`: dataset registry and OpenTSLM `DataLoader` construction.
- `classes/`: domain objects such as patient recordings.
- `configs/`: validated, immutable training configuration.
- `curriculum/`: OpenTSLM model construction, training, metrics, and checkpoints.
- `loaders/`: source-specific children and gameplay dataset adapters.
- `utility/`: accelerator selection and reusable time-series windowing helpers.

## Dataset Splits

Training uses deterministic, label-stratified splits of 60% training, 20%
validation, and 20% testing. Splits are made by patient rather than by time-series
window. For gameplay data, every recording and session belonging to the same
subject remains in one split. `--seed` controls the assignment.

## Commands

Previewing a dataset is the safe default:

```powershell
python -m src.main
python -m src.main --mode preview --dataset children
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

Evaluate a checkpoint with generated `YES` or `NO` predictions. Validation and
test results are reported separately, and the model is loaded only once:

```powershell
python -m src.main --mode evaluate --dataset gameplay --evaluation-split both
```

Use `--max-evaluation-samples` to cap the number of samples evaluated per split
and `--max-new-tokens` to control the generated answer length. Capped evaluation
uses a deterministic shuffle instead of taking adjacent windows from the first
patient. Evaluation uses the same `--seed` as training to reconstruct the patient
assignments and sample order.

Run `python -m src.main --help` for the complete set of dataset, windowing, optimizer, accelerator, LoRA, seed, and checkpoint options.
