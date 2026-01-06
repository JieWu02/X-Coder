#!/usr/bin/env python3
"""
Download X-Coder RL training data from HuggingFace.

Usage:
    python download_data.py                    # Download all data
    python download_data.py --syn-only         # Download only synthetic data
    python download_data.py --real-only        # Download only real data
    python download_data.py --output-dir ./data  # Custom output directory
"""

import argparse
import os
from pathlib import Path

try:
    from huggingface_hub import snapshot_download, hf_hub_download
except ImportError:
    print("Please install huggingface_hub: pip install huggingface_hub")
    exit(1)


REPO_ID = "IIGroup/X-Coder-RL-40k"
REPO_TYPE = "dataset"


def download_data(output_dir: str = "rl-recipe", syn_only: bool = False, real_only: bool = False):
    """
    Download training data from HuggingFace.

    Args:
        output_dir: Directory to save the data (default: rl-recipe)
        syn_only: Only download synthetic data
        real_only: Only download real data
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading data from {REPO_ID}...")
    print(f"Output directory: {output_path.absolute()}")

    if syn_only:
        # Download only synthetic data
        print("\nDownloading synthetic RL data...")
        snapshot_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            local_dir=output_path,
            allow_patterns=["syn_rl_data/**"],
        )
        print("Synthetic data downloaded successfully!")

    elif real_only:
        # Download only real data
        print("\nDownloading real RL data...")
        snapshot_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            local_dir=output_path,
            allow_patterns=["real_rl_data/**"],
        )
        print("Real data downloaded successfully!")

    else:
        # Download all data
        print("\nDownloading all RL data (synthetic + real)...")
        snapshot_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            local_dir=output_path,
        )
        print("All data downloaded successfully!")

    # Print summary
    print("\n" + "=" * 50)
    print("Download Complete!")
    print("=" * 50)

    syn_path = output_path / "syn_rl_data"
    real_path = output_path / "real_rl_data"

    if syn_path.exists():
        print(f"\nSynthetic data: {syn_path}")
        for f in syn_path.rglob("*.parquet"):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  - {f.relative_to(syn_path)}: {size_mb:.1f} MB")

    if real_path.exists():
        print(f"\nReal data: {real_path}")
        for f in real_path.rglob("*.parquet"):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  - {f.relative_to(real_path)}: {size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Download X-Coder RL training data from HuggingFace"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="rl-recipe",
        help="Output directory for downloaded data (default: rl-recipe)"
    )
    parser.add_argument(
        "--syn-only",
        action="store_true",
        help="Only download synthetic data"
    )
    parser.add_argument(
        "--real-only",
        action="store_true",
        help="Only download real data"
    )

    args = parser.parse_args()

    if args.syn_only and args.real_only:
        print("Error: Cannot specify both --syn-only and --real-only")
        exit(1)

    download_data(
        output_dir=args.output_dir,
        syn_only=args.syn_only,
        real_only=args.real_only,
    )


if __name__ == "__main__":
    main()
