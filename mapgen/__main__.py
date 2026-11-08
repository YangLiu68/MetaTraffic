"""Command line: python -m mapgen --seed 7 -o map.svg --png map.png"""

from __future__ import annotations

import argparse
import sys
import time

from .export import to_png, to_svg
from .generator import CityConfig, generate_city


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mapgen", description="Generate a procedural city map.")
    ap.add_argument("-o", "--out", default="map.svg", help="SVG output path (default: map.svg)")
    ap.add_argument("--png", help="also write a PNG preview to this path")
    ap.add_argument("--png-scale", type=float, default=1.0, help="PNG pixels per SVG unit")
    ap.add_argument("--seed", type=int, help="random seed (default: random)")
    ap.add_argument("--width", type=float, default=1139.0, help="view width in SVG units")
    ap.add_argument("--height", type=float, default=678.0, help="view height in SVG units")
    ap.add_argument("--zoom", type=float, default=0.638, help="SVG units per world unit")
    ap.add_argument("--no-water", action="store_true", help="skip coastline and river")
    ap.add_argument("--no-buildings", action="store_true", help="skip building lots")
    args = ap.parse_args(argv)

    cfg = CityConfig(
        seed=args.seed,
        width=args.width,
        height=args.height,
        zoom=args.zoom,
        water=not args.no_water,
        buildings=not args.no_buildings,
    )
    t = time.perf_counter()
    city = generate_city(cfg)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(to_svg(city))
    if args.png:
        to_png(city, args.png, scale=args.png_scale)
    steps = ", ".join(f"{k} {v:.1f}s" for k, v in city.timings.items())
    print(
        f"seed {city.seed}: {len(city.main_roads)} main, {len(city.major_roads)} major, "
        f"{len(city.minor_roads)} minor roads, {len(city.parks)} parks, {len(city.lots)} lots "
        f"-> {args.out} ({time.perf_counter() - t:.1f}s: {steps})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
