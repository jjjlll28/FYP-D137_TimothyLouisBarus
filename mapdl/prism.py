"""
Units: mm, MPa
Managed by Timothy Louis Barus
"""

from __future__ import annotations

from mapdl.block import Block

class Prism(Block):
    """Square prism, 20 x 20 x 30."""
    name = "prism"
 
    def __init__(self, mapdl, cfg, lx=20.0, ly=20.0, lz=30.0):
        super().__init__(mapdl, cfg, lx, ly, lz)
        self.sections = {"mid_section": (0, lx / 2)}