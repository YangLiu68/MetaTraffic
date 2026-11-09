"""Whole-city generation (port of MainGUI.generateEverything in ui/main_gui.ts and its helpers)."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field, replace

from .integrator import RK4Integrator
from .polygon_finder import PolygonParams, divide_polygons, find_polygons, shrink_polygons
from .streamlines import StreamlineGenerator, StreamlineParams
from .tensor_field import NoiseParams, Radial, TensorField
from .water import NoiseStreamlineParams, WaterGenerator, WaterParams

Vec = tuple[float, float]

DRAW_INFLATE_AMOUNT = 1.2  # roads are generated 20% beyond the view so buildings reach the edge


@dataclass
class CityConfig:
    """Generation settings. Defaults reproduce the original app's settings.

    ``width``/``height`` are the size of the view in output (SVG) units and ``zoom`` is output
    units per world unit. The defaults match the MapGenerator SVGs MetaTraffic was built with
    (view 1139 x 678 at zoom 0.638, so the SVG frame is 1366.8 x 813.6).
    """

    seed: int | None = None
    width: float = 1139.0
    height: float = 678.0
    zoom: float = 0.638
    water: bool = True
    coast: bool = True  # with water=True: generate the coastline / the river separately
    river: bool = True
    buildings: bool = True
    num_big_parks: int = 2
    num_small_parks: int = 0
    cluster_big_parks: bool = False
    minor: StreamlineParams = field(
        default_factory=lambda: StreamlineParams(
            dsep=20, dtest=15, dstep=1, dlookahead=40, dcirclejoin=5, joinangle=0.1,
            path_iterations=1000, seed_tries=300, simplify_tolerance=0.5, collide_early=0,
        )
    )  # fmt: skip
    major_dsep: float = 100.0
    major_dtest: float = 30.0
    major_dlookahead: float = 200.0
    main_dsep: float = 400.0
    main_dtest: float = 200.0
    main_dlookahead: float = 500.0
    coast_noise: NoiseStreamlineParams = field(default_factory=NoiseStreamlineParams)
    river_noise: NoiseStreamlineParams = field(default_factory=NoiseStreamlineParams)
    river_bank_size: float = 10.0
    river_size: float = 30.0
    park_params: PolygonParams = field(
        default_factory=lambda: PolygonParams(max_length=20, min_area=80, shrink_spacing=4,
                                              chance_no_divide=1)
    )  # fmt: skip
    building_params: PolygonParams = field(
        default_factory=lambda: PolygonParams(max_length=20, min_area=50, shrink_spacing=4,
                                              chance_no_divide=0.05)
    )  # fmt: skip


@dataclass
class CityMap:
    """Generated city in world coordinates (multiply by ``zoom`` for SVG units)."""

    config: CityConfig
    seed: int
    view_origin: Vec
    view_dims: Vec
    domain_origin: Vec
    domain_dims: Vec
    sea: list[Vec] = field(default_factory=list)
    coastline: list[Vec] = field(default_factory=list)
    river: list[Vec] = field(default_factory=list)
    parks: list[list[Vec]] = field(default_factory=list)
    main_roads: list[list[Vec]] = field(default_factory=list)
    major_roads: list[list[Vec]] = field(default_factory=list)
    minor_roads: list[list[Vec]] = field(default_factory=list)
    coastline_roads: list[list[Vec]] = field(default_factory=list)
    secondary_river: list[Vec] = field(default_factory=list)
    lots: list[list[Vec]] = field(default_factory=list)
    radial_centres: list[Vec] = field(default_factory=list)  # centres of radial districts
    timings: dict[str, float] = field(default_factory=dict)


def generate_city(config: CityConfig | None = None) -> CityMap:
    cfg = config or CityConfig()
    seed = cfg.seed if cfg.seed is not None else random.SystemRandom().randrange(2**31)
    rng = random.Random(seed)
    timings: dict[str, float] = {}
    t0 = time.perf_counter()

    def lap(name: str) -> None:
        nonlocal t0
        now = time.perf_counter()
        timings[name] = now - t0
        t0 = now

    # Screen/world geometry (DomainController with the screen origin at world (0, 0))
    view_origin = (0.0, 0.0)
    view_dims = (cfg.width / cfg.zoom, cfg.height / cfg.zoom)
    dom_dims = (view_dims[0] * DRAW_INFLATE_AMOUNT, view_dims[1] * DRAW_INFLATE_AMOUNT)
    dom_origin = (
        view_origin[0] - (dom_dims[0] - view_dims[0]) / 2,
        view_origin[1] - (dom_dims[1] - view_dims[1]) / 2,
    )
    # RoadGUI.setPathIterations: long enough for a road to cross 1.5 screens
    path_iterations = int(1.5 * max(cfg.width, cfg.height) / cfg.minor.dstep)

    noise = NoiseParams(global_noise=False, noise_size_park=20, noise_angle_park=90,
                        noise_size_global=30, noise_angle_global=20)  # fmt: skip
    tf = TensorField(noise, rng)
    tf.set_recommended(view_origin, view_dims, rng)
    integrator = RK4Integrator(tf, cfg.minor.dstep)

    minor_p = replace(cfg.minor, path_iterations=path_iterations)
    major_p = replace(minor_p, dsep=cfg.major_dsep, dtest=cfg.major_dtest,
                      dlookahead=cfg.major_dlookahead)  # fmt: skip
    main_p = replace(minor_p, dsep=cfg.main_dsep, dtest=cfg.main_dtest,
                     dlookahead=cfg.main_dlookahead)  # fmt: skip
    water_p = WaterParams(**vars(minor_p), coast_noise=cfg.coast_noise,
                          river_noise=cfg.river_noise, river_bank_size=cfg.river_bank_size,
                          river_size=cfg.river_size)  # fmt: skip
    water_p.simplify_tolerance = 10

    def new_gen(params: StreamlineParams) -> StreamlineGenerator:
        return StreamlineGenerator(integrator, dom_origin, dom_dims, params, rng)

    # 1. Coastline and river
    water = WaterGenerator(integrator, dom_origin, dom_dims, water_p, tf, rng)
    if cfg.water and cfg.coast:
        water.create_coast()
    if cfg.water and cfg.river:
        water.create_river()
    lap("water")

    # 2. Main roads (allowed to cross the river: bridges)
    tf.ignore_river = True
    main = new_gen(main_p)
    main.add_existing_streamlines(water)
    main.create_all_streamlines()
    lap("main_roads")

    # 3. Major roads, then pick big parks from the blocks they enclose
    major = new_gen(major_p)
    major.add_existing_streamlines(water)
    major.add_existing_streamlines(main)
    major.create_all_streamlines()
    tf.ignore_river = False
    big_parks = _pick_parks(
        main.all_streamlines_simple + major.all_streamlines_simple, cfg, tf, rng, big=True
    )
    tf.parks = big_parks
    lap("major_roads")

    # 4. Minor roads (bent by noise inside parks), then optional small parks
    minor = new_gen(minor_p)
    minor.add_existing_streamlines(water)
    minor.add_existing_streamlines(main)
    minor.add_existing_streamlines(major)
    minor.create_all_streamlines()
    small_parks = _pick_parks(
        major.all_streamlines_simple + main.all_streamlines_simple + minor.all_streamlines_simple,
        cfg, tf, rng, big=False,
    )  # fmt: skip
    tf.parks = big_parks + small_parks
    lap("minor_roads")

    # 5. Buildings: blocks -> shrink -> subdivide into lots
    lots: list[list[Vec]] = []
    if cfg.buildings:
        all_roads = (
            main.all_streamlines_simple
            + major.all_streamlines_simple
            + minor.all_streamlines_simple
            + water.all_streamlines_simple
            + [water.river_secondary_road]
        )
        bp = cfg.building_params
        blocks = find_polygons(all_roads, bp, tf)
        lots = divide_polygons(shrink_polygons(blocks, bp.shrink_spacing), bp, rng)
    lap("buildings")

    return CityMap(
        config=cfg,
        seed=seed,
        view_origin=view_origin,
        view_dims=view_dims,
        domain_origin=dom_origin,
        domain_dims=dom_dims,
        sea=water.sea_polygon,
        coastline=water.coastline,
        river=water.river_polygon,
        parks=big_parks + small_parks,
        main_roads=main.all_streamlines_simple,
        major_roads=major.all_streamlines_simple,
        minor_roads=minor.all_streamlines_simple,
        coastline_roads=water.all_streamlines_simple,
        secondary_river=water.river_secondary_road,
        lots=lots,
        radial_centres=[f.centre for f in tf.basis_fields if isinstance(f, Radial)],
        timings=timings,
    )


def _pick_parks(
    roads: list[list[Vec]], cfg: CityConfig, tf: TensorField, rng: random.Random, big: bool
) -> list[list[Vec]]:
    """MainGUI.addParks: random blocks become parks (indices drawn with replacement)."""
    polygons = find_polygons(roads, cfg.park_params, tf)
    if big:
        n = cfg.num_big_parks
        if len(polygons) <= n:
            return list(polygons)
        if cfg.cluster_big_parks:
            start = int(rng.random() * (len(polygons) - n))
            return polygons[start : start + n]
        return [polygons[int(rng.random() * len(polygons))] for _ in range(n)]
    if not polygons:
        return []
    return [polygons[int(rng.random() * len(polygons))] for _ in range(cfg.num_small_parks)]
