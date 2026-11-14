"""
Units: mm, MPa
Managed by Timothy Louis Barus
"""

from __future__ import annotations

from mapdl.specimen import Specimen, rectangles

class Block(Specimen):
    """Rectangular block, corner at the origin, x along the length. plate: a BasePlate to stand on (None = old plate)."""
    name = "block"

    def __init__(self, mapdl, cfg, lx=15.0, ly=10.0, lz=12.0, plate=None):
        super().__init__(mapdl, cfg, plate=plate)
        self.lx, self.ly, self.lz = lx, ly, lz
        self.sections = {"mid_length": (0, lx / 2)}      # sigma_xx on the (y, z) plane

    @property
    def height(self):
        return self.lz

    def dims(self):
        return {"lx": self.lx, "ly": self.ly, "lz": self.lz}

    def footprint(self):
        # split at x = lx/2 so there are nodes on the measurement plane
        rectangles(self.mapdl, [0, self.lx / 2, self.lx], [0, self.ly])

    def skin_strips(self, w):
        # element centroids closer than w to a side face (x = 0, lx or y = 0, ly): (axis, lo, hi) ranges for ESEL,CENT.
        # Config.check makes w a multiple of elem_size, so the limit never falls on an element centre.
        if not 0 < w < min(self.lx, self.ly) / 2:
            raise ValueError(f"skin_width={w:g} must be > 0 and less than half of the smallest side ({min(self.lx, self.ly) / 2:g})")
        return [("X", -1.0, w), ("X", self.lx - w, self.lx + 1.0), ("Y", -1.0, w), ("Y", self.ly - w, self.ly + 1.0)]

    def footprint_rects(self):
        # the same two rectangles as footprint(), for the BasePlate
        return [(0.0, self.lx / 2, 0.0, self.ly), (self.lx / 2, self.lx, 0.0, self.ly)]