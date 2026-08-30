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

Run `python -m src.main --help` for the complete set of dataset, windowing, optimizer, accelerator, LoRA, seed, and checkpoint options.
