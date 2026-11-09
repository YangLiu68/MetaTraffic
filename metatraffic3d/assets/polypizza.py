"""Low-poly models from Poly Pizza (https://poly.pizza), downloaded on demand.

``polypizza_manifest.json`` maps each asset role (``sedan``, ``tree_round``, ``house_a``...) to one
Poly Pizza model: its public GLB URL, creator, licence and how to orient and size it. Models are
CC0 or CC-BY; ``credit_line`` / ``python -m metatraffic3d.assets.polypizza credits`` gives the
attribution text.

``load_parts(role)`` turns a cached GLB into ``Part`` objects in the asset frame used by
``procedural.py`` (metres, x forward, y left, z up, origin at the footprint centre on the ground):

1. read the glTF scene with trimesh and apply the node transforms (or, with ``pose``, pose a
   skinned character by sampling that animation, since its bind pose is a T-pose),
2. convert glTF y-up to z-up, rotate by ``yaw_deg`` so the model faces +x,
3. scale uniformly to the manifest ``target`` (``length`` = x extent, ``width`` = y, ``height`` = z,
   ``footprint`` = max(x, y), or a plain ``scale`` factor) and put the origin at the footprint
   centre on the ground (``"origin": "base"``: centre of what touches the ground, for poles),
4. bake one colour per face (material ``baseColorFactor`` times the base-colour texture sampled at
   the face's UV centroid), converted to sRGB display values like the procedural palette,
5. merge similar colours into at most ``MAX_COLOURS`` clusters -> one ``Part`` per cluster, largest
   first. ``"tintable": true`` marks the body-paint Part; ``"light_bar": {"z_min": m}`` splits
   saturated faces above ``z_min`` into extra Parts appended last (roof lamps that can flash);
   ``"tags": {name: [r, g, b]}`` names the Part nearest a colour. ``part_tags(role)`` returns
   these names (``paint``, ``light_bar``, ``housing``, ``lamp_red``...) aligned with the Parts.

The result is cached next to the GLB as ``<role>.npz`` so later loads need only numpy. Processing
a GLB needs ``trimesh`` (and Pillow for textured models); without it, or without a cached file,
``load_parts`` returns ``None`` and ``AssetLibrary`` falls back to the procedural model.
``ALIASES`` maps the procedural names ``car`` and ``person`` to ``sedan`` and ``person_a``.

CLI::

    python -m metatraffic3d.assets.polypizza fetch [ROLE ...]   # download (+ bake) models
    python -m metatraffic3d.assets.polypizza credits            # attribution lines
    python -m metatraffic3d.assets.polypizza table              # role / tris / parts / size
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.request
import warnings
from pathlib import Path

import numpy as np

from ..meshes import Part

MANIFEST_PATH = Path(__file__).with_name("polypizza_manifest.json")
MANIFEST: dict[str, dict] = {e["role"]: e for e in json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))}
CACHE_DIR = Path(os.environ.get("METATRAFFIC3D_ASSET_CACHE") or "~/.cache/metatraffic3d/polypizza").expanduser()
ALIASES = {"car": "sedan", "person": "person_a"}  # names used by the procedural library
MAX_COLOURS = 8
BAKE_VERSION = 2
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/130.0 Safari/537.36 metatraffic3d"
)
LICENCE_URLS = {
    "CC0 1.0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "CC-BY 3.0": "https://creativecommons.org/licenses/by/3.0/",
}


# ------------------------------------------------------------------------------ manifest + cache
def resolve(name: str) -> str:
    return ALIASES.get(name, name)


def entry(name: str) -> dict | None:
    return MANIFEST.get(resolve(name))


def _cache_dir(cache_dir: str | os.PathLike | None) -> Path:
    return Path(cache_dir).expanduser() if cache_dir else CACHE_DIR


def glb_path(name: str, cache_dir: str | os.PathLike | None = None) -> Path:
    e = entry(name)
    if e is None:
        raise KeyError(f"no Poly Pizza model for {name!r}")
    return _cache_dir(cache_dir) / f"{e['id']}.glb"


def _bake_key(e: dict) -> str:
    keys = (
        "id",
        "glb",
        "yaw_deg",
        "target",
        "tintable",
        "tint_rgb",
        "max_colours",
        "pose",
        "pose_time",
        "origin",
        "light_bar",
        "tags",
    )
    blob = json.dumps({"v": BAKE_VERSION, **{k: e.get(k) for k in keys}}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def _download(url: str, dest: Path, timeout: float = 60.0) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    if data[:4] != b"glTF":
        raise ValueError(f"{url} did not return a GLB file")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_bytes(data)
    os.replace(tmp, dest)


def fetch(roles=None, cache_dir=None, pause: float = 0.5, verbose: bool = False) -> dict[str, Path]:
    """Download the GLBs for ``roles`` (default: all) unless cached; return ``{role: path}``.

    Roles whose download fails are left out of the result (a warning names them).
    """
    roles = list(MANIFEST) if roles is None else [resolve(r) for r in roles]
    out: dict[str, Path] = {}
    failed: list[str] = []
    for role in roles:
        e = MANIFEST.get(role)
        if e is None:
            failed.append(f"{role} (not in manifest)")
            continue
        path = glb_path(role, cache_dir)
        if not path.exists():
            try:
                _download(e["glb"], path)
                if verbose:
                    print(f"downloaded {role:16s} {e['title']} ({path.stat().st_size // 1024} KB)")
                time.sleep(pause)
            except Exception as exc:  # noqa: BLE001 - network/HTTP errors, bad payloads
                failed.append(f"{role} ({exc})")
                continue
        out[role] = path
    if failed:
        warnings.warn("Poly Pizza download failed for: " + ", ".join(failed), stacklevel=2)
    return out


# ------------------------------------------------------------------------------ public loading
def load_parts(name: str, cache_dir=None, download: bool = False) -> list[Part] | None:
    """Parts for ``name`` from the cache (baking the GLB if needed), or ``None`` if unavailable."""
    got = _load(name, cache_dir, download)
    return None if got is None else got[0]


def part_tags(name: str, cache_dir=None, download: bool = False) -> list[str] | None:
    """Semantic tag per ``Part`` (same order as ``load_parts``); ``""`` for untagged Parts.

    Tags: ``paint`` (tintable body), ``light_bar`` (roof lamps of emergency vehicles, split off so
    they can flash), and manifest colour tags such as ``housing`` / ``lamp_red`` / ``lamp_yellow`` /
    ``lamp_green`` on the traffic light.
    """
    got = _load(name, cache_dir, download)
    return None if got is None else got[1]


def _load(name: str, cache_dir, download: bool) -> tuple[list[Part], list[str]] | None:
    e = entry(name)
    if e is None:
        return None
    root = _cache_dir(cache_dir)
    role = e["role"]
    key = _bake_key(e)
    npz = root / f"{role}.npz"
    if npz.exists():
        got = _read_npz(npz, key)
        if got is not None:
            return got
    glb = root / f"{e['id']}.glb"
    if not glb.exists() and (not download or not fetch([role], root)):
        return None
    try:
        parts, tags = bake_tagged(glb, e)
    except ImportError:
        warnings.warn("trimesh is needed to process Poly Pizza models (pip install trimesh pillow)", stacklevel=2)
        return None
    except Exception as exc:  # noqa: BLE001 - corrupt or unexpected file: fall back to procedural
        warnings.warn(f"could not process Poly Pizza model {role!r}: {exc}", stacklevel=2)
        return None
    try:
        _write_npz(npz, key, parts, tags)
    except OSError:
        pass
    return parts, tags


def credit_line(name: str) -> str:
    e = entry(name)
    if e is None:
        return name
    lic = e["licence"]
    lic_txt = lic if lic.startswith("CC0") else f"{lic}, {LICENCE_URLS.get(lic, '')}".rstrip(", ")
    return f"{e['title']} by {e['creator']} ({lic_txt}), via Poly Pizza https://poly.pizza/m/{e['id']}"


def credits(roles=None) -> list[str]:
    roles = list(MANIFEST) if roles is None else [resolve(r) for r in roles]
    lines, seen = [], set()
    for r in roles:
        e = MANIFEST.get(r)
        if e and e["id"] not in seen:
            seen.add(e["id"])
            lines.append(credit_line(r))
    return lines


# ------------------------------------------------------------------------------ npz cache
def _write_npz(path: Path, key: str, parts: list[Part], tags: list[str]) -> None:
    vs = [p.vertices.astype(np.float32) for p in parts]
    fs = [p.faces.astype(np.int32) for p in parts]
    tmp = path.with_name(path.name + ".part.npz")
    np.savez_compressed(
        tmp,
        key=np.array(key),
        vertices=np.concatenate(vs) if vs else np.zeros((0, 3), np.float32),
        faces=np.concatenate(fs) if fs else np.zeros((0, 3), np.int32),
        n_vertices=np.array([len(v) for v in vs], np.int64),
        n_faces=np.array([len(f) for f in fs], np.int64),
        rgba=np.array([p.rgba for p in parts], np.float64).reshape(-1, 4),
        tintable=np.array([p.tintable for p in parts], bool),
        tags=np.array(tags, dtype="<U32"),
    )
    os.replace(tmp, path)


def _read_npz(path: Path, key: str) -> tuple[list[Part], list[str]] | None:
    try:
        with np.load(path, allow_pickle=False) as z:
            if str(z["key"]) != key:
                return None
            v, f = z["vertices"].astype(np.float64), z["faces"].astype(np.int64)
            nv, nf, rgba, tint = z["n_vertices"], z["n_faces"], z["rgba"], z["tintable"]
            tags = [str(t) for t in z["tags"]]
    except (OSError, KeyError, ValueError, EOFError):
        return None
    parts, vo, fo = [], 0, 0
    for i in range(len(nv)):
        parts.append(
            Part(v[vo : vo + nv[i]].copy(), f[fo : fo + nf[i]].copy(), tuple(float(c) for c in rgba[i]), bool(tint[i]))
        )
        vo += int(nv[i])
        fo += int(nf[i])
    return parts, tags


# ------------------------------------------------------------------------------ baking
def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def _lab(rgb: np.ndarray) -> np.ndarray:
    """CIE L*a*b* (D65) of sRGB colours in 0..1, for perceptual colour distances."""
    lin = _srgb_to_linear(np.clip(rgb, 0, 1))
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = lin @ m.T / np.array([0.9505, 1.0, 1.089])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], axis=1)


def _face_colours(mesh) -> np.ndarray:
    """(m, 4) sRGB colour per face: baseColorFactor x texture at the UV centroid x vertex colour."""
    n = len(mesh.faces)
    vis = mesh.visual
    lin = np.full((n, 3), 0.5)
    alpha = np.ones(n)
    kind = getattr(vis, "kind", None)
    if kind == "vertex":
        vc = np.asarray(vis.vertex_colors, np.float64)[mesh.faces].mean(axis=1) / 255.0
        return np.column_stack([_linear_to_srgb(vc[:, :3]), vc[:, 3]])
    mat = getattr(vis, "material", None)
    if mat is not None:
        factor = getattr(mat, "baseColorFactor", None)
        image = getattr(mat, "baseColorTexture", None)
        if factor is None and image is None:  # SimpleMaterial
            factor = getattr(mat, "diffuse", None)
            image = getattr(mat, "image", None)
        f = np.ones(4) if factor is None else np.asarray(factor, np.float64).ravel()[:4]
        if f.max() > 1.0:
            f = f / 255.0
        lin = np.tile(f[:3], (n, 1))
        alpha = np.full(n, f[3] if len(f) > 3 else 1.0)
        uv = getattr(vis, "uv", None)
        if image is not None and uv is not None and len(uv) == len(mesh.vertices):
            from trimesh.visual.color import uv_to_color

            tex = uv_to_color(np.asarray(uv, np.float64)[mesh.faces].mean(axis=1), image) / 255.0
            lin = lin * _srgb_to_linear(tex[:, :3])
            alpha = alpha * tex[:, 3]
    va = getattr(vis, "vertex_attributes", None) or {}
    if "color" in va and len(va["color"]) == len(mesh.vertices):
        vc = np.asarray(va["color"], np.float64)
        vc = vc / 255.0 if vc.max() > 1.0 else vc
        vc = vc[mesh.faces].mean(axis=1)
        lin = lin * vc[:, :3]
        if vc.shape[1] > 3:
            alpha = alpha * vc[:, 3]
    return np.column_stack([_linear_to_srgb(lin), alpha])


_GL_DTYPES = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_GL_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def _read_glb(path) -> tuple[dict, bytes]:
    data = Path(path).read_bytes()
    if data[:4] != b"glTF":
        raise ValueError(f"{path} is not a GLB file")
    gltf, binary, pos = None, b"", 12
    while pos + 8 <= len(data):
        length, ctype = int.from_bytes(data[pos : pos + 4], "little"), data[pos + 4 : pos + 8]
        chunk = data[pos + 8 : pos + 8 + length]
        if ctype == b"JSON":
            gltf = json.loads(chunk)
        elif ctype == b"BIN\x00":
            binary = chunk
        pos += 8 + length
    if gltf is None:
        raise ValueError(f"{path} has no JSON chunk")
    return gltf, binary


def _accessor(gltf: dict, binary: bytes, index: int) -> np.ndarray:
    acc = gltf["accessors"][index]
    dtype = np.dtype(_GL_DTYPES[acc["componentType"]])
    width, count = _GL_WIDTH[acc["type"]], acc["count"]
    if "bufferView" not in acc:
        return np.zeros((count, width), np.float64)
    view = gltf["bufferViews"][acc["bufferView"]]
    start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride = view.get("byteStride") or dtype.itemsize * width
    rows = np.ndarray((count, width), dtype, binary, start, (stride, dtype.itemsize))
    out = rows.astype(np.float64)
    if acc.get("normalized") and dtype.kind in "iu":
        out /= np.iinfo(dtype).max
    return out


def _quat_matrix(q) -> np.ndarray:
    x, y, z, w = np.asarray(q, np.float64) / (np.linalg.norm(q) or 1.0)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])  # fmt: skip


def _load_posed(path, pose: str, pose_time: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Skinned GLB posed at ``pose_time`` (s) of the animation whose name contains ``pose``.

    trimesh ignores skins and shows characters in their bind (T) pose; this applies the glTF
    linear-blend skinning by hand. Colours come from material baseColorFactor and COLOR_0.
    """
    gltf, binary = _read_glb(path)
    nodes = gltf["nodes"]
    trs = []
    for nd in nodes:
        trs.append({"translation": np.array(nd.get("translation", [0.0, 0.0, 0.0])),
                    "rotation": np.array(nd.get("rotation", [0.0, 0.0, 0.0, 1.0])),
                    "scale": np.array(nd.get("scale", [1.0, 1.0, 1.0]))})  # fmt: skip
    anims = [a for a in gltf.get("animations", []) if pose.lower() in a.get("name", "").lower()]
    if not anims:
        raise ValueError(f"{path} has no animation matching {pose!r}")
    anim = anims[0]
    for ch in anim["channels"]:
        path_ = ch["target"].get("path")
        if path_ not in ("translation", "rotation", "scale") or "node" not in ch["target"]:
            continue
        smp = anim["samplers"][ch["sampler"]]
        times = _accessor(gltf, binary, smp["input"])[:, 0]
        vals = _accessor(gltf, binary, smp["output"])
        if smp.get("interpolation") == "CUBICSPLINE":
            vals = vals[1::3]
        t = float(np.clip(pose_time, times[0], times[-1]))
        k = int(np.clip(np.searchsorted(times, t, side="right") - 1, 0, len(times) - 1))
        if smp.get("interpolation") == "STEP" or k + 1 >= len(times):
            val = vals[k]
        else:
            a = (t - times[k]) / max(times[k + 1] - times[k], 1e-9)
            v0, v1 = vals[k], vals[k + 1]
            if path_ == "rotation" and np.dot(v0, v1) < 0:
                v1 = -v1
            val = v0 * (1 - a) + v1 * a
        trs[ch["target"]["node"]][path_] = val

    local = []
    for nd, p in zip(nodes, trs):
        m = np.eye(4)
        if "matrix" in nd:
            m = np.array(nd["matrix"], np.float64).reshape(4, 4).T
        else:
            m[:3, :3] = _quat_matrix(p["rotation"]) * p["scale"]
            m[:3, 3] = p["translation"]
        local.append(m)
    parent = {c: i for i, nd in enumerate(nodes) for c in nd.get("children", [])}
    world: dict[int, np.ndarray] = {}

    def glob(i: int) -> np.ndarray:
        if i not in world:
            world[i] = glob(parent[i]) @ local[i] if i in parent else local[i]
        return world[i]

    mats = gltf.get("materials", [])
    vs, fs, cs, off = [], [], [], 0
    for ni, nd in enumerate(nodes):
        if "mesh" not in nd:
            continue
        skin = gltf["skins"][nd["skin"]] if "skin" in nd else None
        if skin is not None:
            ibm = _accessor(gltf, binary, skin["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)
            joint_m = np.stack([glob(j) for j in skin["joints"]]) @ ibm
        for prim in gltf["meshes"][nd["mesh"]]["primitives"]:
            if prim.get("mode", 4) != 4:
                continue
            attr = prim["attributes"]
            pos = np.column_stack(
                [_accessor(gltf, binary, attr["POSITION"]), np.ones(gltf["accessors"][attr["POSITION"]]["count"])]
            )
            if skin is not None and "JOINTS_0" in attr:
                joints = _accessor(gltf, binary, attr["JOINTS_0"]).astype(np.int64)
                weights = _accessor(gltf, binary, attr["WEIGHTS_0"])
                weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-9)
                skin_m = np.einsum("nk,nkij->nij", weights, joint_m[joints])
                p = np.einsum("nij,nj->ni", skin_m, pos)[:, :3]
            else:
                p = (pos @ glob(ni).T)[:, :3]
            if "indices" in prim:
                tri = _accessor(gltf, binary, prim["indices"]).astype(np.int64).reshape(-1, 3)
            else:
                tri = np.arange(len(p)).reshape(-1, 3)
            mat = mats[prim["material"]] if "material" in prim else {}
            f = np.array(mat.get("pbrMetallicRoughness", {}).get("baseColorFactor", [1.0, 1.0, 1.0, 1.0]), np.float64)
            lin = np.tile(f[:3], (len(tri), 1))
            alpha = np.full(len(tri), f[3])
            if "COLOR_0" in attr:
                vc = _accessor(gltf, binary, attr["COLOR_0"])[tri].mean(axis=1)
                lin = lin * vc[:, :3]
            vs.append(p)
            fs.append(tri + off)
            cs.append(np.column_stack([_linear_to_srgb(lin), alpha]))
            off += len(p)
    if not vs:
        raise ValueError(f"{path} has no triangles")
    v = np.vstack(vs)
    v = np.column_stack([v[:, 0], -v[:, 2], v[:, 1]])  # glTF y-up -> z-up
    return v, np.vstack(fs), np.vstack(cs)


def load_scene(path: str | os.PathLike, pose: str | None = None, pose_time: float = 0.0):
    """All triangles of a GLB in z-up metres: (vertices, faces, per-face sRGB rgba).

    With ``pose``, skinned meshes are posed by that animation (see ``_load_posed``); otherwise the
    file is read with trimesh (static models, textures supported).
    """
    if pose:
        return _load_posed(path, pose, pose_time)
    import trimesh

    scene = trimesh.load(str(path), force="scene", process=False)
    vs, fs, cs, off = [], [], [], 0
    for node in scene.graph.nodes_geometry:
        transform, gname = scene.graph[node]
        g = scene.geometry[gname]
        if not isinstance(g, trimesh.Trimesh) or len(g.faces) == 0:
            continue
        faces = np.asarray(g.faces, np.int64)
        if np.linalg.det(transform[:3, :3]) < 0:  # mirrored node: keep counter-clockwise winding
            faces = faces[:, ::-1]
        vs.append(trimesh.transform_points(np.asarray(g.vertices, np.float64), transform))
        fs.append(faces + off)
        cs.append(_face_colours(g))
        off += len(g.vertices)
    if not vs:
        raise ValueError(f"{path} has no triangles")
    v = np.vstack(vs)
    v = np.column_stack([v[:, 0], -v[:, 2], v[:, 1]])  # glTF y-up -> z-up (rotate +90 deg about x)
    return v, np.vstack(fs), np.vstack(cs)


def _cluster(colours: np.ndarray, weights: np.ndarray, k: int, min_delta_e: float = 4.0):
    """Label per face: merge the closest colours (in Lab) until <= k remain and none are near-equal."""
    step = 1 / 63
    while True:
        q = np.round(colours / step).astype(np.int64)
        uniq, labels = np.unique(q, axis=0, return_inverse=True)
        if len(uniq) <= 400 or step > 0.2:
            break
        step *= 2
    labels = labels.ravel()
    w = np.bincount(labels, weights=weights, minlength=len(uniq)) + 1e-12
    cent = np.stack([np.bincount(labels, weights=weights * colours[:, i], minlength=len(uniq)) for i in range(4)], 1)
    cent /= w[:, None]
    ids = np.arange(len(uniq))  # cluster id of each unique colour
    alive = list(range(len(uniq)))
    while len(alive) > 1:
        c = cent[alive]
        lab = _lab(c[:, :3])
        d = np.linalg.norm(lab[:, None] - lab[None], axis=2) + np.abs(c[:, None, 3] - c[None, :, 3]) * 100
        np.fill_diagonal(d, np.inf)
        i, j = np.unravel_index(np.argmin(d), d.shape)
        if len(alive) <= k and d[i, j] >= min_delta_e:
            break
        a, b = alive[i], alive[j]
        cent[a] = (cent[a] * w[a] + cent[b] * w[b]) / (w[a] + w[b])
        w[a] += w[b]
        ids[ids == b] = a
        alive.remove(b)
    remap = np.full(len(uniq), -1, np.int64)
    remap[alive] = np.arange(len(alive))
    return remap[ids][labels], cent[alive]


def _pick_tint(parts: list[Part], areas: list[float], hint=None) -> int | None:
    """Body-paint Part: the largest Part covering >= 15% of the surface that is not dark
    (tyres, tinted glass); a manifest ``tint_rgb`` hint picks the nearest colour instead."""
    if not parts:
        return None
    lab = _lab(np.array([p.rgba[:3] for p in parts]))
    if hint is not None:
        return int(np.argmin(np.linalg.norm(lab - _lab(np.asarray([hint], float)), axis=1)))
    share = np.asarray(areas) / max(sum(areas), 1e-12)
    ok = (share >= 0.15) & (lab[:, 0] > 30)
    return int(np.argmax(np.where(ok, areas, -1))) if ok.any() else None


def bake_tagged(path: str | os.PathLike, e: dict) -> tuple[list[Part], list[str]]:
    """Normalised, colour-clustered Parts for manifest entry ``e`` plus a tag per Part.

    Body Parts come first, largest surface first; light-bar Parts (if any) are appended last.
    """
    v, f, col = load_scene(path, e.get("pose"), e.get("pose_time", 0.0))
    yaw = np.radians(e.get("yaw_deg", 0.0))
    cz, sz = np.cos(yaw), np.sin(yaw)
    v = v @ np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]]).T
    lo, hi = v.min(axis=0), v.max(axis=0)
    size = hi - lo
    target = e.get("target") or {}
    if "scale" in target:
        s = float(target["scale"])
    elif "length" in target:
        s = target["length"] / size[0]
    elif "width" in target:
        s = target["width"] / size[1]
    elif "height" in target:
        s = target["height"] / size[2]
    elif "footprint" in target:
        s = target["footprint"] / max(size[0], size[1])
    else:
        s = 1.0
    xy_lo, xy_hi = lo[:2], hi[:2]
    if e.get("origin") == "base":  # poles with overhanging arms: centre on what touches the ground
        low = v[:, 2] <= lo[2] + 0.02 * size[2]
        xy_lo, xy_hi = v[low, :2].min(axis=0), v[low, :2].max(axis=0)
    centre = np.array([(xy_lo[0] + xy_hi[0]) / 2, (xy_lo[1] + xy_hi[1]) / 2, lo[2]])
    v = (v - centre) * s

    tri = v[f]
    areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    keep = areas > 1e-12
    f, col, areas = f[keep], col[keep], areas[keep]
    labels, cent = _cluster(col, areas, int(e.get("max_colours", MAX_COLOURS)))

    rgbas = [tuple(float(x) for x in np.clip(c, 0, 1)) for c in cent]
    groups = [(np.flatnonzero(labels == c), rgbas[c], "") for c in range(len(cent))]
    lb = e.get("light_bar")
    if lb:  # split roof-lamp faces of saturated colours into their own Parts
        lab = _lab(np.array([g[1][:3] for g in groups]))
        chroma = np.hypot(lab[:, 1], lab[:, 2])
        zc = v[f][:, :, 2].mean(axis=1)
        split = []
        for gi, (idx, rgba, _) in enumerate(groups):
            if chroma[gi] < lb.get("min_chroma", 25.0):
                continue
            up = zc[idx] >= lb["z_min"]
            if up.any():
                groups[gi] = (idx[~up], rgba, "")
                split.append((idx[up], rgba, "light_bar"))
    else:
        split = []

    def build(idx, rgba):
        used, inv = np.unique(f[idx], return_inverse=True)
        return Part(v[used], inv.reshape(-1, 3).astype(np.int64), rgba)

    body = sorted((g for g in groups if len(g[0])), key=lambda g: -areas[g[0]].sum())
    parts = [build(idx, rgba) for idx, rgba, _ in body + split]
    part_area = [float(areas[idx].sum()) for idx, _, _ in body + split]
    tags = [t for _, _, t in body + split]
    if e.get("tintable"):
        i = _pick_tint(parts[: len(body)], part_area[: len(body)], e.get("tint_rgb"))
        if i is not None:
            parts[i].tintable = True
            tags[i] = "paint"
    if e.get("tags"):  # colour hints -> nearest untagged Part
        lab = _lab(np.array([p.rgba[:3] for p in parts]))
        for tag, rgb in e["tags"].items():
            d = np.linalg.norm(lab - _lab(np.asarray([rgb], float)), axis=1)
            d[[t != "" for t in tags]] = np.inf
            i = int(np.argmin(d))
            if d[i] < 12.0:
                tags[i] = tag
    return parts, tags


def bake(path: str | os.PathLike, e: dict) -> list[Part]:
    """Load a GLB and return normalised, colour-clustered Parts for manifest entry ``e``."""
    return bake_tagged(path, e)[0]


# ------------------------------------------------------------------------------ CLI
def _table(roles, cache_dir) -> None:
    print(f"{'role':16s} {'tris':>6s} {'parts':>5s}   {'size x/y/z (m)':22s} source")
    for role in roles:
        got = _load(role, cache_dir, False)
        if got is None:
            print(f"{role:16s} {'-':>6s} {'-':>5s}   not cached")
            continue
        parts, tags = got
        v = np.vstack([p.vertices for p in parts])
        size = v.max(axis=0) - v.min(axis=0)
        tris = sum(len(p.faces) for p in parts)
        tagged = ", ".join(f"{i}:{t}" for i, t in enumerate(tags) if t)
        e = MANIFEST[role]
        print(f"{role:16s} {tris:6d} {len(parts):5d}   {size[0]:6.2f} {size[1]:6.2f} {size[2]:6.2f}   "
              f"{e['title']} / {e['creator']}" + (f"  [{tagged}]" if tagged else ""))  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m metatraffic3d.assets.polypizza", description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=["fetch", "credits", "table"])
    ap.add_argument("roles", nargs="*", help="roles (default: all in the manifest)")
    ap.add_argument("--cache-dir", default=None, help=f"default {CACHE_DIR} (env METATRAFFIC3D_ASSET_CACHE)")
    ap.add_argument("--no-bake", action="store_true", help="fetch: only download, do not pre-process")
    args = ap.parse_args(argv)
    roles = [resolve(r) for r in args.roles] or list(MANIFEST)
    unknown = [r for r in roles if r not in MANIFEST]
    if unknown:
        ap.error(f"unknown roles: {', '.join(unknown)}")
    if args.command == "credits":
        print("\n".join(credits(roles)))
    elif args.command == "fetch":
        got = fetch(roles, args.cache_dir, verbose=True)
        print(f"{len(got)}/{len(roles)} models in {_cache_dir(args.cache_dir)}")
        if not args.no_bake:
            _table(list(got), args.cache_dir)
        return 0 if len(got) == len(roles) else 1
    else:
        _table(roles, args.cache_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
