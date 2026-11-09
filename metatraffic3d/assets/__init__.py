"""Vehicle, tree and prop models.

Two sources: Poly Pizza low-poly models (downloaded on demand, see ``polypizza.py``) and
procedural fallbacks (always available). ``AssetLibrary("auto")`` uses a Poly Pizza model
when it is in the local cache and the procedural one otherwise, so nothing needs network.
"""

from __future__ import annotations

from ..meshes import Part
from .procedural import PROCEDURAL

VEHICLE_ASSETS = ("car", "van")


class AssetLibrary:
    def __init__(self, source: str = "auto", cache_dir: str | None = None) -> None:
        if source not in ("auto", "procedural", "polypizza"):
            raise ValueError(f"unknown asset source {source!r}")
        self.source = source
        self.cache_dir = cache_dir
        self._cache: dict[tuple[str, str], list[Part]] = {}
        self.used: dict[str, str] = {}  # asset name -> where it came from

    def parts(self, name: str, lod: str = "high") -> list[Part]:
        """Parts of an asset. ``lod="low"`` prefers the cheap procedural model (for thousands of copies)."""
        key = (name, lod)
        if key not in self._cache:
            self._cache[key] = self._load(name, lod)
        return self._cache[key]

    def has(self, name: str) -> bool:
        try:
            self.parts(name)
            return True
        except (KeyError, FileNotFoundError):
            return False

    def _load(self, name: str, lod: str = "high") -> list[Part]:
        if lod == "low" and name in PROCEDURAL:
            self.used.setdefault(name, "procedural")
            return PROCEDURAL[name]()
        if self.source != "procedural":
            from . import polypizza

            parts = polypizza.load_parts(name, self.cache_dir, download=self.source == "polypizza")
            if parts is not None:
                self.used[name] = "polypizza"
                return parts
            if self.source == "polypizza":
                raise FileNotFoundError(f"Poly Pizza model for {name!r} not available")
        if name not in PROCEDURAL:
            raise KeyError(f"unknown asset {name!r}")
        self.used[name] = "procedural"
        return PROCEDURAL[name]()

    def credits(self) -> list[str]:
        from . import polypizza

        return [polypizza.credit_line(n) for n, src in sorted(self.used.items()) if src == "polypizza"]
