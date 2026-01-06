#!/usr/bin/env python3
"""
Download and convert SFT training data from HuggingFace.
Downloads IIGroup/X-Coder-SFT-376k and converts hybrid_376k to jsonl format.
"""

import os
import json
import argparse
from huggingface_hub import snapshot_download, hf_hub_download
from datasets import load_dataset


REPO_ID = "IIGroup/X-Coder-SFT-376k"
REPO_TYPE = "dataset"


def download_and_convert(output_dir: str = ".", output_filename: str = "hybrid_376k.jsonl"):
    """
    Download the dataset from HuggingFace and convert to jsonl format.
    Only keeps 'query' and 'response' fields.

    Args:
        output_dir: Directory to save the output jsonl file
        output_filename: Name of the output jsonl file
    """
    print(f"Loading dataset from {REPO_ID}...")

    # Load the dataset
    dataset = load_dataset(REPO_ID, "hybrid_376k", split="train")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    print(f"Converting to jsonl format...")
    print(f"Total samples: {len(dataset)}")

    # Convert to jsonl with only query and response fields
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, item in enumerate(dataset):
            # Extract only query and response fields
            record = {
                "query": item["query"],
                "response": item["response"]
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

            if (i + 1) % 10000 == 0:
                print(f"Processed {i + 1}/{len(dataset)} samples...")

    print(f"Successfully saved {len(dataset)} samples to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Download and convert X-Coder SFT data to jsonl format"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to save the output jsonl file (default: current directory)"
    )
    parser.add_argument(
        "--output-filename",
        type=str,
        default="hybrid_376k.jsonl",
        help="Name of the output jsonl file (default: hybrid_376k.jsonl)"
    )

    args = parser.parse_args()
    download_and_convert(args.output_dir, args.output_filename)


if __name__ == "__main__":
    main()
