#!/usr/bin/env python3
"""Quickstart example for TriFrame inference."""

from triframe.model import TriFrameModel, TriFrameConfig
from triframe.inference import TriFramePredictor

import random


def main():
    print("TriFrame Quickstart Example")
    print("=" * 40)

    # Create a small model
    print("\n1. Creating model...")
    config = TriFrameConfig.small()
    model = TriFrameModel(config)
    print(f"   Model parameters: {model.count_parameters():,}")

    # Create predictor
    print("\n2. Creating predictor...")
    predictor = TriFramePredictor(model)

    # Generate random DNA read
    print("\n3. Generating random DNA read...")
    read = "".join(random.choices(['A', 'C', 'G', 'T'], k=200))
    print(f"   Sequence: {read[:50]}...")
    print(f"   Length: {len(read)} bp")

    # Run prediction
    print("\n4. Running inference...")
    results = predictor.predict_reads([read])
    result = results[0]

    # Print results
    print("\n5. Prediction Results:")
    print(f"   Read ID: {result['read_id']}")
    print(f"   Is Coding: {result['is_coding']} (confidence: {result['coding_confidence']:.3f})")
    print(f"   Predicted Frame: {result['predicted_frame']}")
    print(f"   Frame Confidences (6 frames): {[f'{c:.3f}' for c in result['frame_confidences']]}")

    if result['ec_number']:
        print(f"   EC Number: {result['ec_number']} (confidence: {result['ec_confidence']:.3f})")
    else:
        print(f"   EC Number: None")

    if result['kegg_orthologs']:
        print(f"   KEGG Orthologs: {', '.join(result['kegg_orthologs'][:5])}")
    else:
        print(f"   KEGG Orthologs: None")

    if result['cog_categories']:
        print(f"   COG Categories: {', '.join(result['cog_categories'])}")
    else:
        print(f"   COG Categories: None")

    print("\n" + "=" * 40)
    print("Quickstart complete!")


if __name__ == "__main__":
    main()
