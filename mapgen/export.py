"""SVG and PNG output in MapGenerator's "AppleDark" style (ui/style.ts DefaultStyle.draw).

The SVG uses the same element order, colours and stroke widths as the SVGs the web app
exports, so MetaTraffic's SVG parsers (Python and JS) read it unchanged.
"""

from __future__ import annotations

from .generator import CityMap

Vec = tuple[float, float]

APPLE_DARK = {
    "bg": "#2b2d2f",
    "building": "#343638",
    "building_stroke": "#2f3133",
    "sea": "#374464",
    "grass": "#283838",
    "minor": "#414447",
    "minor_outline": "#2b2d2f",
    "major": "#4e5154",
    "major_outline": "#2b2d2f",
    "main": "#956c3e",
    "main_outline": "#333333",
    "outline_size": 1.0,
    "minor_width": 2.0,
    "major_width": 4.0,
    "main_width": 5.0,
}


def _fmt(v: float) -> str:
    s = f"{v:.3f}".rstrip("0").rstrip(".")
    return "0" if s == "-0" else s


def _layers(city: CityMap, s: dict) -> list[tuple[str, list[list[Vec]], dict]]:
    """(kind, shapes, style) in drawing order; kind is 'fill' (polygon) or 'line'."""
    z = city.config.zoom
    o = s["outline_size"]
    return [
        ("fill", [city.sea], {"fill": s["sea"], "stroke": s["sea"], "width": 0.1}),
        ("line", [city.coastline], {"stroke": s["bg"], "width": 30 * z}),
        ("fill", city.parks, {"fill": s["grass"], "stroke": s["bg"], "width": 1.0}),
        ("fill", [city.river], {"fill": s["sea"], "stroke": s["sea"], "width": 1.0}),
        ("line", city.minor_roads,
         {"stroke": s["minor_outline"], "width": o + s["minor_width"] * z}),
        ("line", city.major_roads + [city.secondary_river],
         {"stroke": s["major_outline"], "width": o + s["major_width"] * z}),
        ("line", city.main_roads + city.coastline_roads,
         {"stroke": s["main_outline"], "width": o + s["main_width"] * z}),
        ("line", city.minor_roads, {"stroke": s["minor"], "width": s["minor_width"] * z}),
        ("line", city.major_roads + [city.secondary_river],
         {"stroke": s["major"], "width": s["major_width"] * z}),
        ("line", city.main_roads + city.coastline_roads,
         {"stroke": s["main"], "width": s["main_width"] * z}),
        ("fill", city.lots, {"fill": s["building"], "stroke": s["building_stroke"], "width": 1.0}),
    ]  # fmt: skip


def to_svg(city: CityMap, style: dict | None = None) -> str:
    s = style or APPLE_DARK
    z = city.config.zoom
    ox, oy = city.view_origin

    def pts(shape: list[Vec]) -> str:
        return " ".join(f"{_fmt((x - ox) * z)},{_fmt((y - oy) * z)}" for x, y in shape)

    fx, fy = (city.domain_origin[0] - ox) * z, (city.domain_origin[1] - oy) * z
    fw, fh = city.domain_dims[0] * z, city.domain_dims[1] * z
    out = [
        '<?xml version="1.0" standalone="no"?>',
        f'<svg id="map-svg" xmlns="http://www.w3.org/2000/svg" version="1.1" '
        f'width="{_fmt(fw)}" height="{_fmt(fh)}" '
        f'viewBox="{_fmt(fx)} {_fmt(fy)} {_fmt(fw)} {_fmt(fh)}">',
        f'<rect width="{_fmt(fw)}" height="{_fmt(fh)}" fill="{s["bg"]}" fill-opacity="1" '
        f'stroke="#000000" stroke-width="1" x="{_fmt(fx)}" y="{_fmt(fy)}"/>',
    ]
    for kind, shapes, st in _layers(city, s):
        for shape in shapes:
            if len(shape) < 2:
                continue
            if kind == "fill":
                paint = f'fill="{st["fill"]}" fill-opacity="1"'
            else:
                paint = 'fill-opacity="0"'
            out.append(
                f'<polyline points="{pts(shape)}" {paint} stroke="{st["stroke"]}" '
                f'stroke-width="{_fmt(st["width"])}"/>'
            )
    out.append("</svg>")
    return "\n".join(out)


def to_png(city: CityMap, path: str, scale: float = 1.0, style: dict | None = None) -> None:
    """Raster preview of the whole generated area (needs Pillow)."""
    from PIL import Image, ImageDraw

    s = style or APPLE_DARK
    z = city.config.zoom * scale
    x0, y0 = city.domain_origin
    w = max(1, round(city.domain_dims[0] * z))
    h = max(1, round(city.domain_dims[1] * z))
    img = Image.new("RGB", (w, h), s["bg"])
    draw = ImageDraw.Draw(img)

    def px(shape: list[Vec]) -> list[tuple[float, float]]:
        return [((x - x0) * z, (y - y0) * z) for x, y in shape]

    for kind, shapes, st in _layers(city, s):
        width = max(1, round(st["width"] * scale))
        for shape in shapes:
            if len(shape) < 2:
                continue
            if kind == "fill":
                if len(shape) >= 3:
                    draw.polygon(px(shape), fill=st["fill"], outline=st["stroke"])
            else:
                draw.line(px(shape), fill=st["stroke"], width=width, joint="curve")
    img.save(path)
