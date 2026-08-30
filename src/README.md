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

Run `python -m src.main --help` for the complete set of dataset, windowing, optimizer, accelerator, LoRA, seed, and checkpoint options.
