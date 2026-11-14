"""
Units: mm, MPa
Managed by Timothy Louis Barus
"""

from __future__ import annotations
 
from mapdl.specimen import Specimen

class Cylinder(Specimen):
    """Solid cylinder on the z axis, centre at x = y = 0."""
    name = "cylinder"
 
    def __init__(self, mapdl, cfg, d=20.0, h=30.0):
        super().__init__(mapdl, cfg)
        self.d, self.h = d, h
        self.sections = {"axial": (1, 0.0)}              # sigma_yy on the (x, z) plane
 
    @property
    def height(self):
        return self.h
 
    def dims(self):
        return {"d": self.d, "h": self.h}
 
    def radii(self):
        return 0.0, self.d / 2
 
    def footprint(self):
        # four quarters -> nodes on the planes x = 0 and y = 0
        r_in, r_out = self.radii()
        for t in (0, 90, 180, 270):
            self.mapdl.cyl4(0, 0, r_in, t, r_out, t + 90)