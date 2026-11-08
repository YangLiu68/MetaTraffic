"""Field integrators (port of impl/integrator.ts)."""

from __future__ import annotations

from .tensor_field import TensorField

Vec = tuple[float, float]


class FieldIntegrator:
    def __init__(self, field: TensorField) -> None:
        self.field = field

    def integrate(self, x: float, y: float, major: bool) -> Vec:
        raise NotImplementedError

    def sample_field_vector(self, x: float, y: float, major: bool) -> Vec:
        t = self.field.sample_point(x, y)
        return t.major() if major else t.minor()

    def on_land(self, x: float, y: float) -> bool:
        return self.field.on_land(x, y)


class EulerIntegrator(FieldIntegrator):
    def __init__(self, field: TensorField, dstep: float) -> None:
        super().__init__(field)
        self.dstep = dstep

    def integrate(self, x: float, y: float, major: bool) -> Vec:
        vx, vy = self.sample_field_vector(x, y, major)
        return (vx * self.dstep, vy * self.dstep)


class RK4Integrator(FieldIntegrator):
    """As in the original, the mid and end samples are offset by (h/2, h/2) and (h, h)."""

    def __init__(self, field: TensorField, dstep: float) -> None:
        super().__init__(field)
        self.dstep = dstep

    def integrate(self, x: float, y: float, major: bool) -> Vec:
        h = self.dstep
        k1x, k1y = self.sample_field_vector(x, y, major)
        k2x, k2y = self.sample_field_vector(x + h / 2, y + h / 2, major)
        k4x, k4y = self.sample_field_vector(x + h, y + h, major)
        s = h / 6
        return ((k1x + 4 * k2x + k4x) * s, (k1y + 4 * k2y + k4y) * s)
