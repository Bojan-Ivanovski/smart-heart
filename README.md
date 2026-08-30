# SmartHeart

## Project Goal

The goal of SmartHeart is to train a model using OpenTSLM technology so it can understand time-series data. In this project, the focus is on brain-signal time-series data, enabling the model to classify patterns and support reasoning about why a given individual might have ADHD.

## Current Status

The repository includes an early implementation for preparing EEG-based ADHD data for OpenTSLM. It supports the children `.mat` recordings and gameplay bandpower CSV recordings, configurable time-series windows, dataset previews, LoRA training, deterministic seeds, checkpoint output, and automatic XLA, CUDA, MPS, or CPU selection.

## Environment

The project currently targets Python `3.12`, as defined in `.python-version`. Python dependencies are listed in `requirements.txt`, including PyTorch, SciPy, Hugging Face tooling, and OpenTSLM.

## Datasets

More information about the datasets, including the current dataset map and naming conventions, can be found in [datasets/README.md](datasets/README.md).

## Source Code

More information about the current implementation can be found in [src/README.md](src/README.md).

## Repository Layout

- `README.md`: project overview and workspace notes.
- `.gitignore`: keeps local datasets, Python cache files, and temporary artifacts out of version control.
- `.python-version`: defines the Python version used for local development.
- `requirements.txt`: lists the current Python dependencies for the project.
- `datasets/`: not committed; all datasets are stored here.
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

See [src/README.md](src/README.md) or run `python -m src.main --help` for additional options.
