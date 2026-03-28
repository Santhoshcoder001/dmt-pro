"""CLI-style entrypoint for running the full ABC-HP pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from abc_hp.pipeline import ABCHPPipeline
from abc_hp.utils.logging_config import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ABC-HP end-to-end pipeline")
    parser.add_argument("accident_csv_path", type=Path, help="Path to accident CSV file")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train model before prediction",
    )
    return parser


def main() -> None:
    setup_logging()
    parser = build_parser()
    args = parser.parse_args()

    pipeline = ABCHPPipeline()

    if args.train:
        train_info = pipeline.train(args.accident_csv_path)
        print("Training complete:", train_info)

    predictions = pipeline.predict(args.accident_csv_path)
    map_path = pipeline.generate_map(predictions)

    print("Predictions generated:", len(predictions))
    print("Hotspot map:", map_path)


if __name__ == "__main__":
    main()
