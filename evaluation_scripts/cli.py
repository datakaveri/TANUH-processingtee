#!/usr/bin/env python3
"""Standalone CLI: compute TANUH benchmark metrics from model outputs + ground truth.

Examples:

    # Single CSV with both columns (notes.md Option A/B format)
    python cli.py --dataset oral_cancer --input predictions.csv

    # Model outputs and ground truth as two separate files, joined by id
    python cli.py --dataset breast_cancer \\
        --ground-truth ground_truth.csv --predictions model_outputs.csv \\
        --output results.json
"""

import argparse
import json
import sys

from eval_lib import DATASET_REGISTRY, evaluate_from_files
from eval_lib.io_utils import CsvFormatError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_REGISTRY), help="Which dataset/problem statement to score against.")
    parser.add_argument("--input", help="Single CSV containing both ground truth and predictions columns.")
    parser.add_argument("--ground-truth", help="Ground truth CSV (paired with --predictions).")
    parser.add_argument("--predictions", help="Model outputs CSV (paired with --ground-truth).")
    parser.add_argument("--id-col", help="Override the auto-detected id column.")
    parser.add_argument("--gt-col", help="Override the auto-detected ground-truth column.")
    parser.add_argument("--pred-col", help="Override the auto-detected predictions column.")
    parser.add_argument("--score-col", help="Override the auto-detected score/probability column (enables AUC).")
    parser.add_argument("--output", help="Write the metrics JSON to this path (also printed to stdout).")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if bool(args.input) == bool(args.ground_truth or args.predictions):
        print("error: pass either --input, or --ground-truth together with --predictions", file=sys.stderr)
        return 2
    if bool(args.ground_truth) != bool(args.predictions):
        print("error: --ground-truth and --predictions must be given together", file=sys.stderr)
        return 2

    try:
        metrics = evaluate_from_files(
            args.dataset,
            input_csv=args.input,
            ground_truth_csv=args.ground_truth,
            predictions_csv=args.predictions,
            id_col=args.id_col,
            gt_col=args.gt_col,
            pred_col=args.pred_col,
            score_col=args.score_col,
        )
    except (CsvFormatError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    output = json.dumps(metrics, indent=2)
    print(output)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
