# SmartHeart

## Project Goal

The goal of SmartHeart is to train a model using OpenTSLM technology so it can understand time-series data. In this project, the focus is on brain-signal time-series data, enabling the model to classify patterns and support reasoning about why a given individual might have ADHD.

## Current Status

The repository includes a canonical patient-level EEG dataset and an OpenTSLM training pipeline. It supports raw EEG and gameplay band-power recordings, configurable time-series windows, persisted patient-level dataset splits, LoRA training, deterministic seeds, automatic checkpoint reuse, and automatic XLA, CUDA, MPS, or CPU selection.

## Environment

The project currently targets Python `3.12`, as defined in `.python-version`. Base CPU and CUDA dependencies are listed in `requirements.txt`. TPU environments should install `requirements-tpu.txt`, which adds PyTorch/XLA and the Cloud TPU runtime.

## Datasets

The canonical dataset contract, layout, and source summary are documented in [dataset/README.md](dataset/README.md).

## Source Code

More information about the current implementation can be found in [src/README.md](src/README.md).

## Repository Layout

- `README.md`: project overview and workspace notes.
- `.gitignore`: keeps Python cache files, local environment files, and temporary artifacts out of version control.
- `.python-version`: defines the Python version used for local development.
- `requirements.txt`: lists the current Python dependencies for the project.
- `dataset/`: version-controlled canonical metadata and compressed patient signals.
- `src/`: package containing the CLI, data pipeline, runtime selection, model factory, and trainer.
- `tmp/`: temporary workspace files used during local experimentation.

## Usage

Preview the active gameplay dataset without initializing a language model:

```powershell
python -m src.main
```

Start training explicitly:

```powershell
python -m src.main --mode train --dataset gameplay --device auto
```

Evaluate the saved checkpoint on both held-out partitions:

```powershell
python -m src.main --mode evaluate --dataset gameplay --evaluation-split both
```

Training automatically loads the checkpoint matching `--model-id` when that file
already exists under `--checkpoint-root`. To discard it and train from the base
model, add `--fresh-start`; this deletes only the selected model's checkpoint.
Checkpoints contain model weights, so reuse is a warm start rather than an optimizer
or epoch resume.

See [src/README.md](src/README.md) or run `python -m src.main --help` for additional options.
