#!/usr/bin/env python3
"""Subset the Barlow fonts the pages embed (dev-time only; needs fonttools + brotli).

The simulator, report and landing page are self-contained HTML files that must
open offline, so their fonts are inlined as base64 WOFF2. This script cuts the
upstream TTFs (https://github.com/google/fonts, ofl/barlow and
ofl/barlowsemicondensed, SIL OFL 1.1) down to the characters the pages use.

    python scripts/subset_fonts.py /path/to/downloaded/ttf/dir
"""
import sys
from pathlib import Path

from fontTools import subset

OUT = Path(__file__).resolve().parent.parent / "web" / "fonts"
FACES = {
    "Barlow-Regular": "barlow-400",
    "Barlow-Italic": "barlow-400i",
    "Barlow-SemiBold": "barlow-600",
    "BarlowSemiCondensed-Medium": "barlow-sc-500",
    "BarlowSemiCondensed-SemiBold": "barlow-sc-600",
}
UNICODES = ("U+0020-007E,U+00A0-00FF,U+2013,U+2014,U+2018,U+2019,U+201C,U+201D,U+2022,"
            "U+2026,U+2032,U+2033,U+2190,U+2192,U+2212,U+2248,U+2260,U+2264,U+2265,U+00D7")


def main(src_dir: str) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for src, dst in FACES.items():
        out = OUT / f"{dst}.woff2"
        subset.main([
            str(Path(src_dir) / f"{src}.ttf"), f"--unicodes={UNICODES}",
            "--layout-features=kern,liga,ccmp,locl,tnum,pnum,sups,frac",
            "--flavor=woff2", f"--output-file={out}", "--no-hinting", "--desubroutinize",
        ])
        print(f"{out.name}: {out.stat().st_size / 1024:.1f} kB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
