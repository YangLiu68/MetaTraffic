# mapgen: procedural city maps in Python

A Python port of [ProbableTrain/MapGenerator](https://github.com/ProbableTrain/MapGenerator)
(TypeScript, in `../third_party/MapGenerator`). It produces the same kind of map and writes the same SVG
format, so MetaTraffic loads the output directly.

How it works (same pipeline as the original):

1. A tensor field is built from four grid fields and one radial field placed at random.
2. A coastline and a river are traced through the field (with noise); the river banks become roads.
3. Main, major and minor roads are traced as streamlines of the field, each layer keeping its
   distance from the roads before it. Loose road ends are joined to nearby roads.
4. Two blocks between the bigger roads become parks (their minor roads are bent by noise).
5. Blocks are found between the roads, shrunk back from the road centrelines and cut into
   building lots.

## Usage

```bash
pip install -r mapgen/requirements.txt
python -m mapgen --seed 7 -o map.svg --png map.png
```

```python
from mapgen import CityConfig, generate_city, to_svg

city = generate_city(CityConfig(seed=7))
svg_text = to_svg(city)  # e.g. DrivingEnv(svg_text) in legacy2d/rl/
```

The same seed always gives the same map. A full map takes about 3 seconds.

Options: `--width/--height` (view size in SVG units), `--zoom` (SVG units per world unit),
`--no-water`, `--no-buildings`. The defaults (1139 x 678 at zoom 0.638) match the size and
scale of the MapGenerator SVGs MetaTraffic was developed with.

## Differences from the TypeScript version

- No browser UI, animation, colour schemes other than "AppleDark", or 3D/STL/heightmap export.
- Random numbers come from a seeded `random.Random`, so maps are reproducible.
- Blocks are found with shapely (node the road lines, then `polygonize`) instead of the
  original graph walk. The faces are the same; blocks with a dead-end road inside are now found
  too.
- Geometry operations use shapely (GEOS) where the original used jsts (both are ports of JTS).
  If shrinking a block splits it in two, the larger part is kept.
- When a lot-cutting line misses the block, the original retried until JavaScript's stack
  overflowed and then dropped the block; here it retries up to 200 times.

## License

LGPL-3.0, like the original (see `COPYING` and `COPYING.LESSER`).
