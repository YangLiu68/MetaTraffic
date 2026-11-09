"""NVIDIA Isaac Sim backend: the city as a USD stage, plus a runner that drives it in Isaac.

* ``usd_export.export_usd(world, "city.usd")`` needs only ``usd-core`` (verified on macOS).
* ``mirror.StageMirror`` / ``mirror.KinematicEgo`` move traffic, people, lamps and the ego on
  the stage with plain ``pxr`` (verified with usd-core).
* ``run_isaac.py`` is the Isaac Sim script (app, rendering, camera capture): untested, see README.

Submodules import ``pxr`` on use, so ``import metatraffic3d.backends.isaac`` works without it.
"""

from __future__ import annotations

__all__ = ["export_usd", "StageMirror", "KinematicEgo"]


def __getattr__(name: str):
    if name == "export_usd":
        from .usd_export import export_usd

        return export_usd
    if name in ("StageMirror", "KinematicEgo"):
        from . import mirror

        return getattr(mirror, name)
    raise AttributeError(name)
