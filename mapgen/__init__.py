"""mapgen: procedural city maps in Python.

A Python port of ProbableTrain/MapGenerator (https://github.com/ProbableTrain/MapGenerator),
the TypeScript generator in ../MapGenerator. Roads are streamlines of a tensor field
(Chen et al., "Interactive Procedural Street Modeling", 2008), then water, parks and building
lots are added. Licensed under the LGPL-3.0, like the original (see COPYING.LESSER).

    from mapgen import CityConfig, generate_city, to_svg
    city = generate_city(CityConfig(seed=7))
    open("map.svg", "w").write(to_svg(city))
"""

from .export import to_png, to_svg
from .generator import CityConfig, CityMap, generate_city

__all__ = ["CityConfig", "CityMap", "generate_city", "to_png", "to_svg"]
