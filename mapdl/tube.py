"""
Units: mm, MPa
Managed by Timothy Louis Barus
"""

from __future__ import annotations
 
from mapdl.cylinder import Cylinder

class Tube(Cylinder):
    """Hollow tube on the z axis."""
    name = "tube"
 
    def __init__(self, mapdl, cfg, d_out=20.0, d_in=10.0, h=30.0):
        super().__init__(mapdl, cfg, d_out, h)
        self.d_in = d_in
        if not 0 < d_in < d_out:
            raise ValueError("need 0 < d_in < d_out")
 
    def dims(self):
        return {"d_out": self.d, "d_in": self.d_in, "h": self.h}
 
    def radii(self):
        return self.d_in / 2, self.d / 2