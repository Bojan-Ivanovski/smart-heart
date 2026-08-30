import argparse
from pathlib import Path

from .builders import DATASET_TYPES, DatasetPipelineBuilder
from .configs.training import TrainingConfig
from .curriculum.model_factory import OpenTSLMModelFactory
from .curriculum.trainer import Trainer
from .utility.runtime import resolve_runtime

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview SmartHeart EEG datasets or train an OpenTSLM model."
    )
    parser.add_argument("--mode", choices=["preview", "train"], default="preview")
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_TYPES),
        default="gameplay",
    )
    parser.add_argument("--datasets-root", type=Path, default=PROJECT_ROOT / "datasets")
    parser.add_argument("--window-size", type=int, default=4096)
    parser.add_argument("--window-stride", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--model-id", default="google/gemma-3-270m")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--device",
        choices=["auto", "xla", "cuda", "mps", "cpu"],
        default="auto",
    )
    parser.add_argument(
        "--enable-lora",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=PROJECT_ROOT / "checkpoints",
    )
    parser.add_argument(
        "--fresh-start",
        action="store_true",
        help="Delete the selected model's checkpoint before training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline_builder = DatasetPipelineBuilder(args.datasets_root)
    dataset = pipeline_builder.build_dataset(
        args.dataset,
        window_size=args.window_size,
        window_stride=args.window_stride,
    )

    if args.mode == "preview":
        sample = dataset[0]
        print(f"dataset: {args.dataset}")
        print(f"samples: {len(dataset)}")
        print(f"answer: {sample['answer']}")
        print(f"time_series_shape: {tuple(sample['time_series'].shape)}")
        print(f"prompt: {sample['post_prompt']}")
        return

    config = TrainingConfig(
        model_id=args.model_id,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        enable_lora=args.enable_lora,
        checkpoint_root=args.checkpoint_root,
        fresh_start=args.fresh_start,
        seed=args.seed,
    )
    splits = pipeline_builder.split_dataset(dataset, seed=config.seed)
    print(
        f"[split] train samples={len(splits.train)} groups={splits.train_groups}; "
        f"validation samples={len(splits.validation)} "
        f"groups={splits.validation_groups}; "
        f"test samples={len(splits.test)} groups={splits.test_groups}"
    )
    dataloader = pipeline_builder.build_dataloader(
        splits.train,
        batch_size=config.batch_size,
        shuffle=True,
    )
    runtime = resolve_runtime(args.device)
    trainer = Trainer(runtime, OpenTSLMModelFactory(runtime))
    summary = trainer.train(dataloader, config)

    print("Training complete")
    print(f"model_id: {summary.model_id}")
    print(f"runtime: {summary.runtime}")
    print(f"epochs: {summary.epochs}")
    print(f"steps: {summary.steps}")
    print(f"checkpoint_loaded: {summary.checkpoint_loaded}")
    print(f"final_loss: {summary.final_loss:.6f}")
    print(f"checkpoint_path: {summary.checkpoint_path}")
    print(f"elapsed_seconds: {summary.elapsed_seconds:.2f}")


if __name__ == "__main__":
    main()
