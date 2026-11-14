"""
Units: mm, MPa
Managed by Timothy Louis Barus
"""

from __future__ import annotations
 
from mapdl.specimen import Specimen
 
 
class Bridge(Specimen):
    """Bridge of Zhang et al., 120 x 20 x 30, with four windows under a deck.
 
        x:  leg 7.5 | window 15 | solid 15 | window 15 | solid 15 | window 15 | solid 15 | window 15 | leg 7.5
        z:  windows are 18 high (deck 12 on top)
 
    Each window holds 3 thin fins (the supports, 1 mm thick, 3 mm gaps).
    Nothing is deleted after meshing: the bridge is built from boxes, and the
    gaps between the fins are simply never created.
 
    window_h: Fig. 6 says 20, the maps in Fig. 11 show ~18.  fin_t: read from the photo.
 
    Release (release()), as in the paper:
      1. off the plate, supports still in   -> warpage
      2. remove the supports (the fins)     -> plane IV (x = L/2)
      3. cut in half at x = L/2, keep x<=L/2 -> plane V (y = W/2)
    """
    name = "bridge"
 
    def __init__(self, mapdl, cfg, length=120.0, width=20.0, height=30.0,
                 leg_w=7.5, window_w=15.0, window_h=18.0, n_windows=4,
                 n_fins=3, fin_t=1.0):
        super().__init__(mapdl, cfg)
        self.L, self.W, self.H = length, width, height
        self.leg_w, self.window_w, self.window_h, self.n_windows = leg_w, window_w, window_h, n_windows
        self.n_fins, self.fin_t = n_fins, fin_t
        self.solid_w = (length - 2 * leg_w - n_windows * window_w) / (n_windows - 1)
        self.gap_w = (window_w - n_fins * fin_t) / (n_fins + 1)
        if self.solid_w <= 0 or self.gap_w <= 0 or not 0 < window_h < height:
            raise ValueError("bridge dimensions do not fit")
        self.sections = {
            "plane_IV": (0, length / 2),     # sigma_xx on (y, z)
            "plane_V": (1, width / 2),       # sigma_yy on (x, z), x <= L/2
        }
        self.stages = {}
 
    @property
    def height(self):
        return self.H
 
    def dims(self):
        return {"L": self.L, "W": self.W, "H": self.H, "leg_w": self.leg_w,
                "window_w": self.window_w, "window_h": self.window_h,
                "n_fins": self.n_fins, "fin_t": self.fin_t}
 
    def z_breaks(self):
        return [self.window_h]
 
    # ---- layout along x ---------------------------------------------------------
    def x_pieces(self):
        """The bridge cut into slices along x: [(x_start, x_end, kind)],
        kind = 'solid' (legs and the parts between windows), 'gap' or 'fin'.
        The slice crossing x = L/2 is split there (plane IV and the half cut)."""
        p = [(0.0, self.leg_w, "solid")]
        for i in range(self.n_windows):
            a = self.leg_w + i * (self.window_w + self.solid_w)       # window start
            if i > 0:
                p.append((a - self.solid_w, a, "solid"))
            x = a
            for k in range(self.n_fins):
                f = a + (k + 1) * self.gap_w + k * self.fin_t         # fin start
                p += [(x, f, "gap"), (f, f + self.fin_t, "fin")]
                x = f + self.fin_t
            p.append((x, a + self.window_w, "gap"))
        p.append((self.L - self.leg_w, self.L, "solid"))
        out = []
        for a, b, kind in p:
            if a < self.L / 2 < b:
                out += [(a, self.L / 2, kind), (self.L / 2, b, kind)]
            else:
                out.append((a, b, kind))
        return out
 
    # ---- PREP7 ------------------------------------------------------------------
    def make_mesh(self):
        m, c = self.mapdl, self.cfg
        pieces = self.x_pieces()
        xs = sorted({round(v, 6) for a, b, _ in pieces for v in (a, b)})
        zs = [-c.base_t, 0.0, self.window_h, self.H]          # plate | lower part | deck
        ys = [0.0, self.W]
 
        # keypoints on a shared grid, so neighbouring boxes share faces (no gluing)
        kp = {(i, j, k): m.k("", x, y, z)
              for i, x in enumerate(xs) for j, y in enumerate(ys) for k, z in enumerate(zs)}
 
        def box(a, b, k):                                    # k = 0 plate, 1 lower part, 2 deck
            i, j = xs.index(round(a, 6)), xs.index(round(b, 6))
            m.v(kp[i, 0, k], kp[j, 0, k], kp[j, 1, k], kp[i, 1, k],
                kp[i, 0, k + 1], kp[j, 0, k + 1], kp[j, 1, k + 1], kp[i, 1, k + 1])
 
        # volumes are numbered in creation order: plate, part, fins
        for a, b, _ in pieces:                               # plate under everything
            box(a, b, 0)
        n_plate = len(pieces)
        for a, b, kind in pieces:                            # legs and solids, full height
            if kind == "solid":
                box(a, b, 1)
        for a, b, _ in pieces:                               # deck over everything
            box(a, b, 2)
        n_part = n_plate + sum(k == "solid" for _, _, k in pieces) + len(pieces)
        for a, b, kind in pieces:                            # the fins (supports)
            if kind == "fin":
                box(a, b, 1)
        n_all = n_part + sum(k == "fin" for _, _, k in pieces)
 
        for first, last, mat in ((1, n_plate, c.MAT_BASE),
                                 (n_plate + 1, n_part, c.MAT_PART),
                                 (n_part + 1, n_all, c.MAT_SUPPORT)):
            m.vsel("S", "VOLU", "", first, last)
            m.vatt(mat, "", 1)
        m.allsel()
 
        m.esize(c.elem_size)
        m.mshape(0, "3D")
        m.mshkey(1)
        m.vmesh("ALL")
        m.allsel()
        if int(m.get_value("ELEM", 0, "COUNT")) == 0:
            raise RuntimeError("bridge: meshing produced no elements")
 
    # ---- release ----------------------------------------------------------------
    def release(self, sol):
        c, m = self.cfg, self.mapdl
        sol.release_part()                                   # 1. off the plate
        self.stages["warpage"] = sol.time
        sol.kill_material(c.MAT_SUPPORT)                     # 2. remove supports
        sol.step()
        self.stages["plane_IV"] = sol.time
        sol.kill_material(c.MAT_PART, x_range=(self.L / 2, self.L + 1))   # 3. half cut
        m.ddele("ALL", "ALL")                                # new 3-2-1 hold on the kept half
        m.esel("S", "MAT", "", c.MAT_PART)
        m.esel("R", "CENT", "X", 0, self.L / 2)
        m.nsle("S")
        m.nsel("R", "LOC", "Z", 0)
        a = int(m.queries.node(self.L / 4, self.W / 2, 0))
        b = int(m.queries.node(0, self.W / 2, 0))
        d = int(m.queries.node(self.L / 4, self.W, 0))
        m.allsel()
        m.d(a, "UX"); m.d(a, "UY"); m.d(a, "UZ")
        m.d(b, "UY"); m.d(b, "UZ")
        m.d(d, "UZ")
        sol.step()
        self.stages["plane_V"] = sol.time
 