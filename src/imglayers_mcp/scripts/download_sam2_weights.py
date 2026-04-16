"""Download SAM2 checkpoints into ./weights/sam2 (or $SAM2_WEIGHTS_DIR).

Usage:
  imglayers-download-weights                 # downloads small + base_plus
  imglayers-download-weights --all           # downloads tiny/small/base_plus/large
  imglayers-download-weights --only tiny     # downloads just one
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

_URL_BASE = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
_FILES = {
    "tiny":      "sam2.1_hiera_tiny.pt",
    "small":     "sam2.1_hiera_small.pt",
    "base_plus": "sam2.1_hiera_base_plus.pt",
    "large":     "sam2.1_hiera_large.pt",
}


def _resolve_dir() -> Path:
    return Path(os.environ.get("SAM2_WEIGHTS_DIR", "./weights/sam2")).resolve()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"→ {url}\n  into {dest}")
    with urllib.request.urlopen(url) as resp, open(dest, "wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        read = 0
        chunk = 1024 * 1024
        while True:
            buf = resp.read(chunk)
            if not buf:
                break
            out.write(buf)
            read += len(buf)
            if total:
                pct = read / total * 100
                sys.stdout.write(f"\r  {read / 1e6:7.1f} / {total / 1e6:.1f} MB  ({pct:5.1f}%)")
                sys.stdout.flush()
    print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="download all four checkpoints")
    ap.add_argument("--only", choices=list(_FILES.keys()), help="download a single checkpoint")
    ap.add_argument("--force", action="store_true", help="redownload if a file already exists")
    args = ap.parse_args()

    dest_dir = _resolve_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)

    if args.only:
        wanted = [args.only]
    elif args.all:
        wanted = list(_FILES.keys())
    else:
        wanted = ["small", "base_plus"]

    print(f"Target directory: {dest_dir}")
    print(f"Checkpoints: {wanted}\n")

    for kind in wanted:
        fname = _FILES[kind]
        dest = dest_dir / fname
        if dest.exists() and not args.force:
            print(f"✓ {fname} already exists ({dest.stat().st_size / 1e6:.1f} MB) — skipping")
            continue
        url = _URL_BASE + fname
        try:
            _download(url, dest)
            print(f"✓ {fname}")
        except Exception as exc:
            print(f"✗ {fname}: {exc}", file=sys.stderr)
            return 1
    print("\nAll done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
