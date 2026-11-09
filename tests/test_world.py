import hashlib

import numpy as np

from metatraffic3d.assets import AssetLibrary
from metatraffic3d.world import build_world

from .conftest import small_cfg


def _digest(w) -> str:
    h = hashlib.sha256()
    for b in w.buildings:
        h.update(np.round(np.asarray(b.footprint.exterior.coords), 3).tobytes())
        h.update(f"{b.floors}{b.style}{b.zone}".encode())
    for e in w.net.edges:
        h.update(np.round(e.center.pts, 3).tobytes())
    for m in w.scene.meshes:
        h.update(np.round(m.vertices, 2).tobytes())
    return h.hexdigest()


def test_same_seed_same_city_and_different_seed_differs():
    lib = AssetLibrary("procedural")
    a = build_world(small_cfg(3), assets=lib)
    b = build_world(small_cfg(3), assets=lib)
    c = build_world(small_cfg(8), assets=lib)
    assert _digest(a) == _digest(b)
    assert _digest(a) != _digest(c)
    assert a.profile == b.profile


def test_profile_can_be_pinned_and_terrain_follows_it():
    lib = AssetLibrary("procedural")
    flat = build_world(small_cfg(3, terrain="flat"), with_meshes=False, assets=lib)
    hilly = build_world(small_cfg(3, terrain="steep"), with_meshes=False, assets=lib)
    assert flat.terrain.stats()["max_height_m"] == 0.0
    assert hilly.profile.terrain == "steep" and hilly.terrain.stats()["max_height_m"] > 5.0


def test_districts_buildings_and_junction_control(small_world):
    w = small_world
    assert {b.zone for b in w.buildings} <= {"cbd", "commercial", "residential", "industrial"}
    assert all(b.height > 0 for b in w.buildings)
    assert set(w.controls) <= {"none", "signal", "all_stop", "two_stop"}
    assert sum(v for k, v in w.controls.items() if k != "none") > 0
    groups = {m.group for m in w.scene.meshes}
    assert {"road", "sidewalk", "marking", "building", "ground"} <= groups
