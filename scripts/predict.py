#!/usr/bin/env python3
"""CLI script for running TriFrame inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from triframe.inference import TriFramePredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TriFrame inference on DNA reads."
    )

    parser.add_argument(
        "--checkpoint", required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--input", required=True, help="Path to input FASTA file"
    )
    parser.add_argument(
        "--output", required=True, help="Path to output TSV file"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Inference batch size"
    )
    parser.add_argument(
        "--device", default="auto", help="Device (auto/cuda/mps/cpu)"
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print(f"Loading model from {args.checkpoint}")
    predictor = TriFramePredictor.from_checkpoint(args.checkpoint, device=args.device)

    print(f"Running inference on {args.input}")
    results = predictor.predict_fasta(
        fasta_path=args.input,
        batch_size=args.batch_size,
        output_path=args.output,
    )

    # Print summary
    coding_count = sum(1 for r in results if r["is_coding"])
    non_coding_count = len(results) - coding_count

    print(f"\nInference complete!")
    print(f"Total reads: {len(results)}")
    print(f"Coding reads: {coding_count} ({100*coding_count/len(results):.1f}%)")
    print(f"Non-coding reads: {non_coding_count} ({100*non_coding_count/len(results):.1f}%)")
    print(f"\nResults saved to: {args.output}")

    # Print sample predictions
    print("\nSample predictions:")
    for i, result in enumerate(results[:3]):
        print(f"\n  Read: {result['read_id']}")
        print(f"    Coding: {result['is_coding']} (conf: {result['coding_confidence']:.3f})")
        print(f"    Frame: {result['predicted_frame']} (gates: {[f'{g:.3f}' for g in result['frame_confidences']]})")
        if result['ec_number']:
            print(f"    EC: {result['ec_number']} (conf: {result['ec_confidence']:.3f})")
        if result['kegg_orthologs']:
            print(f"    KEGG: {', '.join(result['kegg_orthologs'][:3])}")
        if result['cog_categories']:
            print(f"    COG: {', '.join(result['cog_categories'])}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
