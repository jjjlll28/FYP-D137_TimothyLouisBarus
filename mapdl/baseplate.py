"""Units: mm
Managed by Timothy Louis Barus

Build plate of Zhang et al. 2025 (Sect. 2.2, 2.5, Fig. 3a): 25 x 25 x 15 mm, the part in the middle, held only at four bottom
corners. Plate and part are glued bricks (BLOCK, VGLUE) with a mapped hex mesh, sharing nodes at z = 0. Details: README.txt.

    plate = BasePlate(25, 25)                  # x, y [, thickness=15, dz=None, support="corners"]
    block = Block(mapdl, cfg, plate=plate)     # plate=None: the old plate of cfg.base_t
"""

from __future__ import annotations

SUPPORTS = ("corners", "face")


def _multiple(a, b):
    return abs(a / b - round(a / b)) < 1e-6


def _breaks(values, tol=1e-9):
    """Sorted values without duplicates."""
    out = []
    for v in sorted(values):
        if not out or v - out[-1] > tol:
            out.append(v)
    return out


class BasePlate:
    """Rectangular plate under the part, centred under its footprint.
    x, y: size; thickness: z from -thickness to 0; dz: element size through the thickness (None = elem_size);
    support: "corners" (Zhang's four points) or "face" (whole bottom clamped).
    """

    def __init__(self, x: float = 25.0, y: float = 25.0, thickness: float = 15.0, dz: float | None = None,
                 support: str = "corners"):
        if min(x, y, thickness) <= 0:
            raise ValueError("the plate size and thickness must be positive")
        if dz is not None and dz <= 0:
            raise ValueError("dz must be positive (or None for cfg.elem_size)")
        if support not in SUPPORTS:
            raise ValueError(f"support must be one of {SUPPORTS}")
        self.x, self.y, self.thickness, self.dz, self.support = float(x), float(y), float(thickness), dz, support
        self._bounds = None                         # (xmin, xmax, ymin, ymax), set by build()

    @classmethod
    def zhang(cls, dz: float | None = None) -> "BasePlate":
        """Zhang et al. 2025: 25 x 25 x 15 mm, four corner points held."""
        return cls(25.0, 25.0, 15.0, dz=dz, support="corners")

    # ---- description (also used to tell saved calibration runs apart) ----------------------------------
    def describe(self) -> dict:
        return dict(x=self.x, y=self.y, thickness=self.thickness, dz=self.dz, support=self.support)

    def __repr__(self):
        dz = "elem_size" if self.dz is None else f"{self.dz:g}"
        return f"BasePlate({self.x:g} x {self.y:g} x {self.thickness:g}, dz={dz}, support={self.support!r})"

    # ---- geometry --------------------------------------------------------------------------------------
    def check(self, cfg, rects):
        """Raise if the plate cannot be made with this footprint and mesh."""
        dz = self.dz or cfg.ez
        if not _multiple(self.thickness, dz):
            raise ValueError(f"the plate thickness {self.thickness:g} is not a multiple of its element size {dz:g} "
                             f"(dz, or the element size in the build direction when dz is None)")
        fx0, fx1 = min(r[0] for r in rects), max(r[1] for r in rects)
        fy0, fy1 = min(r[2] for r in rects), max(r[3] for r in rects)
        x0, x1, y0, y1 = self._bounds
        if x0 > fx0 + 1e-9 or x1 < fx1 - 1e-9 or y0 > fy0 + 1e-9 or y1 < fy1 - 1e-9:
            raise ValueError(f"the plate {self.x:g} x {self.y:g} is smaller than the footprint of the part "
                             f"({fx1 - fx0:g} x {fy1 - fy0:g})")

    def build(self, spec):
        """Make plate and part as glued bricks and mesh them (called by Specimen.make_mesh, in /PREP7).

        spec must have footprint_rects() (rectangles (x0, x1, y0, y1) at z = 0) and height."""
        m, c = spec.mapdl, spec.cfg
        rects = spec.footprint_rects() if hasattr(spec, "footprint_rects") else None
        if not rects:
            raise NotImplementedError(f"{spec.name}: no footprint_rects(), so it cannot stand on a BasePlate yet "
                                      f"(only rectangular footprints, for now the Block)")
        t, h = self.thickness, float(spec.height)

        # centred under the footprint, so the part stands in the middle of the plate (as in Zhang's Fig. 3)
        cx = (min(r[0] for r in rects) + max(r[1] for r in rects)) / 2
        cy = (min(r[2] for r in rects) + max(r[3] for r in rects)) / 2
        self._bounds = (cx - self.x / 2, cx + self.x / 2, cy - self.y / 2, cy + self.y / 2)
        self.check(c, rects)

        # one common grid for the plate and the part, so every brick meets its neighbours face to face
        xs = _breaks([self._bounds[0], self._bounds[1]] + [v for r in rects for v in r[:2]])
        ys = _breaks([self._bounds[2], self._bounds[3]] + [v for r in rects for v in r[2:]])
        cells = [(xs[i], xs[i + 1], ys[j], ys[j + 1]) for i in range(len(xs) - 1) for j in range(len(ys) - 1)]
        under_part = [any(r[0] - 1e-9 <= (a + b) / 2 <= r[1] + 1e-9 and r[2] - 1e-9 <= (p + q) / 2 <= r[3] + 1e-9
                          for r in rects) for a, b, p, q in cells]

        m.wpcsys(-1, 0)                             # BLOCK works on the working plane: make it the global one
        for a, b, p, q in cells:
            m.block(a, b, p, q, -t, 0)              # the plate, from z = -thickness to 0
        for (a, b, p, q), on in zip(cells, under_part):
            if on:
                m.block(a, b, p, q, 0, h)           # the part, from z = 0 to the height
        m.allsel()
        m.vglue("ALL")                              # bricks that touch share their faces, lines and keypoints

        # mesh: mapped bricks, in-plane size = elem_size, through the plate thickness = dz (None = elem_z), through the
        # part = elem_z (None = elem_size)
        m.allsel()
        m.type(1)                                   # ET_SOLID of specimen.py
        m.esize(c.elem_size)
        m.mshape(0, "3D")
        m.mshkey(1)
        dz = self.dz or c.ez
        if abs(dz - c.elem_size) > 1e-9:
            m.lsel("S", "LOC", "Z", -t / 2 - 1e-3, -t / 2 + 1e-3)      # the vertical lines of the plate (centre at -t/2)
            m.lesize("ALL", dz)
            m.allsel()
        if abs(c.ez - c.elem_size) > 1e-9:
            m.lsel("S", "LOC", "Z", h / 2 - 1e-3, h / 2 + 1e-3)        # the vertical lines of the part (centre at h/2)
            m.lesize("ALL", c.ez)
            m.allsel()
        m.vsel("S", "LOC", "Z", -t, 0)              # volumes whose centre is in the plate ...
        m.vatt(c.MAT_BASE, "", 1)
        m.vsel("S", "LOC", "Z", 0, h)               # ... and in the part
        m.vatt(c.MAT_PART, "", 1)
        m.allsel()
        try:
            m.vmesh("ALL")
        except Exception as err:
            raise RuntimeError(f"{spec.name}: meshing the plate and the part failed ({err}). Every brick needs a "
                               f"mapped hexahedral mesh: check elem_size / layer_h / the plate dz against the "
                               f"dimensions.") from err
        m.allsel()
        m.nummrg("NODE")                            # safety: nodes at the same place are one node
        self._verify(spec)

    def _verify(self, spec):
        """Stop at once, with a clear message, if the part is not on the plate (it would float free) or is missing."""
        m, c = spec.mapdl, spec.cfg
        m.allsel()
        if int(m.get_value("ELEM", 0, "COUNT")) == 0:
            raise RuntimeError(f"{spec.name}: meshing produced no elements")
        n = {}
        for label, mat in (("plate", c.MAT_BASE), ("part", c.MAT_PART)):
            m.esel("S", "MAT", "", mat)
            n[label] = int(m.get_value("ELEM", 0, "COUNT"))
            if n[label] == 0:
                m.allsel()
                raise RuntimeError(f"{spec.name}: no {label} elements after meshing")
            m.nsle("S")
            n[label + "_zmin"] = float(m.get_value("NODE", 0, "MNLOC", "Z"))
            n[label + "_zmax"] = float(m.get_value("NODE", 0, "MXLOC", "Z"))
        t, h = self.thickness, float(spec.height)
        if abs(n["plate_zmin"] + t) > 1e-6 or abs(n["part_zmax"] - h) > 1e-6:
            m.allsel()
            raise RuntimeError(f"{spec.name}: the mesh covers z = {n['plate_zmin']:g} ... {n['part_zmax']:g}, expected "
                               f"{-t:g} ... {h:g}")
        # nodes of the part on z = 0, and how many of them also belong to a plate element
        m.esel("S", "MAT", "", c.MAT_PART)
        m.nsle("S")
        m.nsel("R", "LOC", "Z", 0)
        n_bottom = int(m.get_value("NODE", 0, "COUNT"))
        m.esel("S", "MAT", "", c.MAT_BASE)
        m.nsle("R")
        n_shared = int(m.get_value("NODE", 0, "COUNT"))
        m.allsel()
        if n_bottom == 0:
            raise RuntimeError(f"{spec.name}: the part has no nodes on z = 0, so it does not stand on the plate")
        if n_shared != n_bottom:
            raise RuntimeError(f"{spec.name}: the part is not joined to the plate: {n_shared} of the {n_bottom} nodes "
                               f"on the underside of the part are shared with the plate")

    # ---- boundary conditions -----------------------------------------------------------------------------
    def fix(self, mapdl):
        """Hold the plate (in /SOLU). "face": all DOFs of the bottom face. "corners": z at all four corners, plus a 3-2-1
        in the plane (corner 1 x, y; corner 2 y). Zhang's y is z here.
        """
        if self._bounds is None:
            raise RuntimeError("fix() before build(): the plate has not been made yet")
        m, t = mapdl, self.thickness
        x0, x1, y0, y1 = self._bounds
        m.nsel("S", "LOC", "Z", -t)
        if self.support == "face":
            m.d("ALL", "ALL")
        else:
            corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            nodes = [int(m.queries.node(x, y, -t)) for x, y in corners]
            if len(set(nodes)) < 4:
                raise RuntimeError(f"the four corner nodes of the plate are not distinct: {nodes}")
            for (x, y), nd in zip(corners, nodes):         # really at the corners, not just the nearest nodes
                found = (float(m.get_value("NODE", nd, "LOC", "X")), float(m.get_value("NODE", nd, "LOC", "Y")))
                if abs(found[0] - x) > 1e-6 or abs(found[1] - y) > 1e-6:
                    raise RuntimeError(f"node {nd} at {found} is not the plate corner ({x:g}, {y:g})")
            n1, n2, n3, n4 = nodes
            m.d(n1, "UX"); m.d(n1, "UY"); m.d(n1, "UZ")
            m.d(n2, "UY"); m.d(n2, "UZ")
            m.d(n3, "UZ")
            m.d(n4, "UZ")
        m.allsel()