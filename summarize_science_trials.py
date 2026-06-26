from __future__ import annotations

import argparse
import os

from src.report_generator import build_science_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build science summary CSV/charts from science_trials.csv")
    parser.add_argument("--output", default="output", help="Root output directory")
    parser.add_argument("--csv", default=None, help="Optional path to science_trials.csv")
    args = parser.parse_args()

    csv_path = args.csv or os.path.join(args.output, "science_trials.csv")
    artifacts = build_science_outputs(args.output, csv_path)
    if not artifacts:
        print(f"No science outputs created. Check CSV path: {csv_path}")
        return
    print("Science outputs created:")
    for name, path in artifacts.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
