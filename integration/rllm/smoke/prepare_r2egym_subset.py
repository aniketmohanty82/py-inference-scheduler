"""Materialize a small, fixed r2egym task subset and register it as 'r2egym_smoke'.

The plain `rllm dataset pull r2egym` materializes all 4,578 task dirs; the smoke
needs a frozen handful whose docker images can be pre-pulled. Task order comes
from the upstream HF split, so `limit=N` is deterministic for a pinned dataset
revision.

Usage:
    python prepare_r2egym_subset.py [--limit 8] [--out-dir ~/.rllm/tasks/r2egym_smoke]
"""

import argparse
import json
from pathlib import Path

from rllm.data.dataset import DatasetRegistry
from rllm.data.r2egym_builder import build_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=Path.home() / ".rllm" / "tasks" / "r2egym_smoke")
    args = parser.parse_args()

    build_benchmark(
        name="r2egym_smoke",
        split="train",
        out_dir=args.out_dir,
        limit=args.limit,
        register=True,
    )

    dataset = DatasetRegistry.load_dataset("r2egym_smoke", "train")
    if dataset is None:
        raise RuntimeError("build_benchmark did not register r2egym_smoke")
    manifest = [
        {"id": row["id"], "docker_image": row["docker_image"]}
        for row in dataset
    ]
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=1))
    print(f"registered r2egym_smoke with {len(manifest)} tasks")
    print(f"manifest (freeze this + pre-pull these images): {manifest_path}")
    for entry in manifest:
        print(f"  {entry['id']}  {entry['docker_image']}")


if __name__ == "__main__":
    main()
