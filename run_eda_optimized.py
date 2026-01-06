import argparse
from pathlib import Path

from llm_parallelization.new_processor import NEMO, NewProcessor
import yaml

from eda_pipeline_optimized import (
    EDAPipelineOptimized,
    get_checkpoint_status,
    inspect_live_results,
    load_dataframe,
)

# Model mapping
MODEL_MAPPING = {
    "NEMO": NEMO,
}


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Map model string to actual object
    if "processor" in config and "llm" in config["processor"]:
        model_name = config["processor"]["llm"]
        if model_name in MODEL_MAPPING:
            config["processor"]["llm"] = MODEL_MAPPING[model_name]
        else:
            raise ValueError(
                f"Unknown model: {model_name}. Available: {list(MODEL_MAPPING.keys())}"
            )

    return config


def main(config_path: str = "config.yaml", resume: bool = True, clear_checkpoint: bool = False):
    # Load config
    print(f"📋 Loading config from: {config_path}")
    config = load_config(config_path)

    # Setup checkpoint directory
    output_folder = config.get("output", {}).get("output_folder", "./output")
    checkpoint_dir = Path(output_folder) / "checkpoints"

    # Clear checkpoint if requested
    if clear_checkpoint and checkpoint_dir.exists():
        import shutil

        print(f"🧹 Clearing existing checkpoint at {checkpoint_dir}")
        shutil.rmtree(checkpoint_dir)

    # Load data
    print("📂 Loading input data...")
    input_df = load_dataframe(config["input"]["dataframe_path"])

    print("📂 Loading reference data...")
    reference_df = load_dataframe(config["reference"]["dataframe_path"])

    # Filter reference by length
    reference_df = reference_df[
        reference_df[config["reference"]["text_column"]]
        .str.len()
        .between(config["reference"]["min_length"], config["reference"]["max_length"])
    ].reset_index(drop=True)

    print(f"✅ Loaded {len(input_df)} input texts")
    print(f"✅ Loaded {len(reference_df)} reference texts")
    print(f"📋 Input columns: {list(input_df.columns)}")

    quality_config = config.get("quality", {})
    min_similarity = quality_config.get("min_semantic_similarity", 0.65)
    enable_validation = quality_config.get("enable_validation", False)

    # Get checkpoint save frequency
    checkpoint_config = config.get("checkpointing", {})
    save_frequency = checkpoint_config.get("save_frequency", 1)
    print(f"💾 Checkpoint save frequency: every {save_frequency} batch(es)")

    # Initialize processor
    processor = NewProcessor(**config["processor"])

    try:
        # Initialize optimized pipeline with checkpointing
        pipeline = EDAPipelineOptimized(
            processor=processor,
            reference_df=reference_df,
            reference_text_col=config["reference"]["text_column"],
            min_semantic_similarity=min_similarity,
            enable_validation=enable_validation,
            checkpoint_dir=str(checkpoint_dir),
            checkpoint_save_frequency=save_frequency,
        )

        # Run pipeline
        output_df = pipeline.run_pipeline(
            input_df=input_df,
            config=config,
            text_column=config["input"]["text_column"],
            resume=resume,
        )

        # Save final output
        output_path = Path(output_folder) / "final_augmented_dataset.csv"
        if not output_path.exists() or len(output_df) > 0:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_df.to_csv(output_path, index=False)

        print(f"\n🎉 Final dataset saved to: {output_path}")
        print(f"📊 Total texts: {len(output_df)}")
        print(f"📈 Augmentation multiplier: {len(output_df) / len(input_df):.1f}x")
        print(f"📋 Output columns: {list(output_df.columns)}")

    finally:
        processor.terminate()


def inspect(config_path: str = "config.yaml"):
    """Inspect live results from a running or interrupted pipeline."""
    print(f"📋 Loading config from: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    output_folder = config.get("output", {}).get("output_folder", "./output")
    checkpoint_dir = Path(output_folder) / "checkpoints"

    # Get status
    status = get_checkpoint_status(str(checkpoint_dir))

    print("\n📊 Checkpoint Status:")
    print(f"   - Checkpoint exists: {status['checkpoint_exists']}")
    print(f"   - Live file exists: {status['live_file_exists']}")
    print(f"   - Live results rows: {status['live_rows']}")

    if status["last_modified"]:
        print(f"   - Last modified: {status['last_modified']}")

    if status["state"]:
        state = status["state"]
        print("\n📈 Pipeline State:")
        print(f"   - Completed chunks: {len(state.get('completed_chunks', []))}")
        print(f"   - Current chunk: {state.get('current_chunk', 'N/A')}")
        print(f"   - Current round: {state.get('current_round', 'N/A')}")
        print(f"   - Global text ID: {state.get('global_text_id', 0)}")
        print(f"   - Total output count: {state.get('total_output_count', 0)}")
        print(f"   - Last checkpoint: {state.get('timestamp', 'N/A')}")

    # Load and show sample of live results
    if status["live_file_exists"] and status["live_rows"] > 0:
        print(f"\n📄 Live results file: {checkpoint_dir / 'live_results.csv'}")
        df = inspect_live_results(str(checkpoint_dir))
        if df is not None:
            print(f"\n📋 Columns: {list(df.columns)}")
            print("\n🔍 Sample (last 5 rows):")
            print(df.tail())

            # Show round distribution
            if "round" in df.columns:
                print("\n📊 Rows per round:")
                print(df["round"].value_counts().to_string())


def tail(config_path: str = "config.yaml", n: int = 10):
    """Show the last N rows of live results."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    output_folder = config.get("output", {}).get("output_folder", "./output")
    checkpoint_dir = Path(output_folder) / "checkpoints"

    df = inspect_live_results(str(checkpoint_dir))
    if df is not None:
        print(f"📊 Live results: {len(df)} total rows")
        print(f"\n🔍 Last {n} rows:")
        print(df.tail(n).to_string())
    else:
        print("❌ No live results found")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EDA augmentation pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command (default)
    run_parser = subparsers.add_parser("run", help="Run the augmentation pipeline")
    run_parser.add_argument(
        "config",
        nargs="?",
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    run_parser.add_argument(
        "--no-resume", action="store_true", help="Start fresh, don't resume from checkpoint"
    )
    run_parser.add_argument(
        "--clear-checkpoint",
        action="store_true",
        help="Clear existing checkpoint before starting",
    )

    # Inspect command
    inspect_parser = subparsers.add_parser("inspect", help="Inspect live results")
    inspect_parser.add_argument(
        "config",
        nargs="?",
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )

    # Tail command
    tail_parser = subparsers.add_parser("tail", help="Show last N rows of live results")
    tail_parser.add_argument(
        "config",
        nargs="?",
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    tail_parser.add_argument(
        "-n", type=int, default=10, help="Number of rows to show (default: 10)"
    )

    args = parser.parse_args()

    # Default to run if no command specified
    if args.command is None:
        # Check if first positional arg looks like a config file
        import sys

        if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
            config_path = sys.argv[1]
        else:
            config_path = "config.yaml"

        # Parse remaining args for run command
        main(
            config_path=config_path,
            resume="--no-resume" not in sys.argv,
            clear_checkpoint="--clear-checkpoint" in sys.argv,
        )
    elif args.command == "run":
        main(
            config_path=args.config,
            resume=not args.no_resume,
            clear_checkpoint=args.clear_checkpoint,
        )
    elif args.command == "inspect":
        inspect(config_path=args.config)
    elif args.command == "tail":
        tail(config_path=args.config, n=args.n)
