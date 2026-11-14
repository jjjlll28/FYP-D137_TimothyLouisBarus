"""Units: mm, MPa
Managed by Timothy Louis Barus

Layer-by-layer loop after am_cube.py: NROPT,FULL before the first EKILL, dead nodes clamped, one solve per layer.
"""

from __future__ import annotations

class Solution:
    def __init__(self, mapdl, cfg, plate=None, skin=None, top=None):   # skin and top may both be given
        self.mapdl = mapdl
        self.cfg = cfg
        self.plate = plate                  # a BasePlate (baseplate.py) or None = clamp the whole bottom face of cfg.base_t
        self.skin = skin                    # None, or dict(strips=[(axis, lo, hi) ...], factor=f): see define_skin()
        self.top = top                      # None, or dict(z0=height above which the layers get the factor, factor=f)
        self.time = 0.0                     # load-step counter (TIME value)
 
    # ---- full sequence ------------------------------------------------------
    def run(self, height: float, release=None):
        """start, fix plate, kill part, build layer by layer, release, finish. release: optional release(sol), default release_part()."""
        self.start()
        self.fix_plate()
        if self.skin is not None:
            self.define_skin()
        self.kill_part()
        self.apply_layer_by_layer(height)
        if self.cfg.release:
            self.mapdl.outres("ALL", "LAST")    # write results for the release steps
            if release is None:
                self.release_part()
            else:
                release(self)
        self.finish()
 
    # ---- steps ----------------------------------------------------------------
    def start(self):
        """Enter /SOLU and set the analysis options (same order as am_cube.py)."""
        m, c = self.mapdl, self.cfg
        m.finish()
        m.slashsolu()
        m.antype("STATIC")
        m.nlgeom("ON" if c.nlgeom else "OFF")
        m.run("SOLCONTROL,ON")
        m.nropt("FULL", "", "ON")           # birth and death needs this BEFORE the first EKILL / SOLVE
        m.autots("ON")
        m.nsubst(*c.nsubst)
        m.outres("ERASE")
        m.outres("ALL", "NONE")             # write nothing until the last solves
        m.bfunif("TEMP", c.air_temp)        # every node starts at the reference temperature
        self.time = 0.0
 
    def fix_plate(self):
        """Hold the build plate: with a BasePlate as it says (Zhang: four corner points), otherwise clamp all DOFs on
        the bottom face of the plate."""
        if self.plate is not None:
            self.plate.fix(self.mapdl)
            return
        m = self.mapdl
        m.nsel("S", "LOC", "Z", -self.cfg.base_t)
        m.d("ALL", "ALL")
        m.allsel()
 
    def kill_part(self):
        """Deactivate everything above the plate (part and supports) and clamp the nodes
        that belong only to dead elements, so they cannot drift (as am_cube.py does).
        The nodes shared with the live plate stay free."""
        m, c = self.mapdl, self.cfg
        m.esel("S", "MAT", "", c.MAT_PART, c.MAT_SUPPORT)
        m.ekill("ALL")
        m.nsle("S", "ALL")                  # nodes of the dead elements ...
        m.esel("S", "MAT", "", c.MAT_BASE)
        m.nsle("U")                         # ... minus the nodes of the live plate
        m.d("ALL", "ALL", 0)
        m.allsel()
 
    def apply_layer_by_layer(self, height: float):
        """Activate one layer at a time and give it its inherent strain (two_step: activate and heat in two solves)."""
        m, c = self.mapdl, self.cfg
        n = c.n_layers(height)
        for k in range(n):
            zlo, zhi = k * c.layer_h, (k + 1) * c.layer_h
            self.select_layer(zlo, zhi)
            m.ealive("ALL")                 # the layer is born
            m.nsle("S", "ALL")              # its nodes ...
            m.ddele("ALL", "ALL")           # ... are free from now on
            if c.two_step:
                self.step()
                self.select_layer(zlo, zhi)
            f = self.top["factor"] if self.top is not None and zlo >= self.top["z0"] - 1e-9 else 1.0
            if self.skin is None:
                m.bfe("ALL", "TEMP", "", c.air_temp + f * c.delta_t)    # alpha * f * delta_t = inherent strain
            else:
                self.heat_layer(zlo, zhi, f)
            if k == n - 1 and not c.release:
                m.outres("ALL", "LAST")     # last solve of the build: keep it
            self.step()
 
    def define_skin(self):
        """Element component SKIN: the part elements in the skin zone (union of the strips, by element centroid).
        Stops with a clear message if the zone is empty or is the whole part."""
        m, c = self.mapdl, self.cfg
        m.allsel()
        m.esel("NONE")
        for axis, lo, hi in self.skin["strips"]:
            m.esel("A", "CENT", axis, lo, hi)
        m.esel("R", "MAT", "", c.MAT_PART)                  # the strips also cut through the plate
        n_skin = int(m.get_value("ELEM", 0, "COUNT"))
        m.cm("SKIN", "ELEM")
        m.esel("S", "MAT", "", c.MAT_PART)
        n_part = int(m.get_value("ELEM", 0, "COUNT"))
        m.allsel()
        if n_skin == 0:
            raise ValueError(f"the skin zone (skin_width={c.skin_width:g} mm) contains no element: it must be about "
                             f"half an element or more (elem_size={c.elem_size:g} mm)")
        if n_skin >= n_part:
            raise ValueError(f"the skin zone (skin_width={c.skin_width:g} mm) is the whole part: use a smaller width")

    def heat_layer(self, zlo, zhi, f=1.0):
        """The BFE load of one layer in two groups: skin elements get the factor fs * f, the others f (f = the top
        factor when the layer is in the top zone, else 1). Temperature air_temp + factor * delta_t."""
        m, c = self.mapdl, self.cfg
        self.select_layer(zlo, zhi)
        m.cmsel("R", "SKIN", "ELEM")
        m.bfe("ALL", "TEMP", "", c.air_temp + self.skin["factor"] * f * c.delta_t)
        self.select_layer(zlo, zhi)
        m.cmsel("U", "SKIN", "ELEM")
        m.bfe("ALL", "TEMP", "", c.air_temp + f * c.delta_t)

    def release_part(self):
        """Cut the part off the plate: kill the plate and hold the part with a
        3-2-1 constraint on its bottom face, so it springs back freely."""
        a, b, c = self.pick_321_nodes()
        self.kill_material(self.cfg.MAT_BASE)
        m = self.mapdl
        m.d(a, "UX"); m.d(a, "UY"); m.d(a, "UZ")   # 3: no translation
        m.d(b, "UY"); m.d(b, "UZ")                 # 2: no rotation about z and y
        m.d(c, "UZ")                               # 1: no rotation about x
        self.step()
 
    def finish(self):
        self.mapdl.finish()
 
    # ---- helpers ----------------------------------------------------------------
    def step(self):
        """Solve one load step with everything selected."""
        self.mapdl.allsel()
        self.time += 1.0
        self.mapdl.time(self.time)
        self.mapdl.solve()
 
    def select_layer(self, zlo, zhi, tol=1e-6):
        """Select part and support elements whose centroid lies in (zlo, zhi)."""
        m, c = self.mapdl, self.cfg
        m.esel("S", "MAT", "", c.MAT_PART, c.MAT_SUPPORT)
        m.esel("R", "CENT", "Z", zlo + tol, zhi - tol)
 
    def kill_material(self, mat, x_range=None):
        """Kill all elements of one material, optionally only those with
        centroid x in x_range. Call step() afterwards to solve."""
        m = self.mapdl
        m.esel("S", "MAT", "", mat)
        if x_range is not None:
            m.esel("R", "CENT", "X", *x_range)
        m.ekill("ALL")
        m.allsel()
 
    def pick_321_nodes(self):
        """Three nodes on the part's bottom face (z = 0): near the middle, at
        max x and at max y. Uses *GET queries, so it works inside /SOLU."""
        m = self.mapdl
        m.esel("S", "MAT", "", self.cfg.MAT_PART)
        m.nsle("S")
        m.nsel("R", "LOC", "Z", 0)
        if int(m.get_value("NODE", 0, "COUNT")) < 3:
            raise RuntimeError("not enough nodes on the part's bottom face for 3-2-1")
        xmin = m.get_value("NODE", 0, "MNLOC", "X")
        xmax = m.get_value("NODE", 0, "MXLOC", "X")
        ymin = m.get_value("NODE", 0, "MNLOC", "Y")
        ymax = m.get_value("NODE", 0, "MXLOC", "Y")
        xc, yc = (xmin + xmax) / 2, (ymin + ymax) / 2
        a = int(m.queries.node(xc, yc, 0))
        b = int(m.queries.node(xmax, yc, 0))
        c = int(m.queries.node(xc, ymax, 0))
        m.allsel()
        if len({a, b, c}) < 3:
            raise RuntimeError(f"3-2-1 nodes are not distinct: {a}, {b}, {c}")
        return a, b, c