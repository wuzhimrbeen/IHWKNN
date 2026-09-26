"""Extract the public benchmark archives into the repository data directory."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, default=ROOT / "data" / "archives")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    archives = sorted(args.archive_dir.glob("*.zip"))
    if not archives:
        raise FileNotFoundError(f"No ZIP archives found in {args.archive_dir}")

    for archive in archives:
        marker = args.output_dir / archive.stem
        if marker.exists() and not args.force:
            print(f"skip existing: {marker}")
            continue
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(args.output_dir)
        print(f"extracted: {archive.name}")


if __name__ == "__main__":
    main()

