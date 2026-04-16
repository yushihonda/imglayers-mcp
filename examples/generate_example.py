"""Generate a small synthetic banner for local development.

Run `python examples/generate_example.py` to write `examples/banner.png`.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    out = Path(__file__).resolve().parent / "banner.png"
    img = Image.new("RGB", (1200, 600), (250, 250, 248))
    draw = ImageDraw.Draw(img)
    draw.rectangle([(60, 90), (1140, 190)], fill=(30, 80, 200))   # headline band
    draw.rectangle([(60, 220), (760, 280)], fill=(180, 180, 180))  # subcopy band
    draw.rectangle([(60, 420), (320, 510)], fill=(230, 110, 40))   # CTA
    img.save(out, format="PNG", optimize=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
