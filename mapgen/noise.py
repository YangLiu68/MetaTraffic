"""2-D simplex noise, ported from the `simplex-noise` 2.4 npm package used by MapGenerator."""

from __future__ import annotations

import math
import random

_F2 = 0.5 * (math.sqrt(3.0) - 1.0)
_G2 = (3.0 - math.sqrt(3.0)) / 6.0
_GRAD3 = (
    1, 1, 0, -1, 1, 0, 1, -1, 0, -1, -1, 0,
    1, 0, 1, -1, 0, 1, 1, 0, -1, -1, 0, -1,
    0, 1, 1, 0, -1, 1, 0, 1, -1, 0, -1, -1,
)  # fmt: skip


class SimplexNoise:
    def __init__(self, rng: random.Random) -> None:
        p = list(range(256))
        for i in range(255):
            r = i + int(rng.random() * (256 - i))
            p[i], p[r] = p[r], p[i]
        self.perm = [p[i & 255] for i in range(512)]
        self.perm_mod12 = [v % 12 for v in self.perm]

    def noise2d(self, xin: float, yin: float) -> float:
        perm, pm12, g = self.perm, self.perm_mod12, _GRAD3
        s = (xin + yin) * _F2
        i = math.floor(xin + s)
        j = math.floor(yin + s)
        t = (i + j) * _G2
        x0 = xin - (i - t)
        y0 = yin - (j - t)
        if x0 > y0:
            i1, j1 = 1, 0
        else:
            i1, j1 = 0, 1
        x1 = x0 - i1 + _G2
        y1 = y0 - j1 + _G2
        x2 = x0 - 1.0 + 2.0 * _G2
        y2 = y0 - 1.0 + 2.0 * _G2
        ii = i & 255
        jj = j & 255
        n0 = n1 = n2 = 0.0
        t0 = 0.5 - x0 * x0 - y0 * y0
        if t0 >= 0:
            gi = pm12[ii + perm[jj]] * 3
            t0 *= t0
            n0 = t0 * t0 * (g[gi] * x0 + g[gi + 1] * y0)
        t1 = 0.5 - x1 * x1 - y1 * y1
        if t1 >= 0:
            gi = pm12[ii + i1 + perm[jj + j1]] * 3
            t1 *= t1
            n1 = t1 * t1 * (g[gi] * x1 + g[gi + 1] * y1)
        t2 = 0.5 - x2 * x2 - y2 * y2
        if t2 >= 0:
            gi = pm12[ii + 1 + perm[jj + 1]] * 3
            t2 *= t2
            n2 = t2 * t2 * (g[gi] * x2 + g[gi + 1] * y2)
        return 70.0 * (n0 + n1 + n2)
