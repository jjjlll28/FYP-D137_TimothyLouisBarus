"""
Units: mm, MPa, celcius
Managed by Timothy Louis Barus
"""

# imports
from __future__ import annotations

import numpy as np

MATERIAL_COLORS = {1: "#9e9e9e", 2: "#4c78a8", 3: "#f58518"}   # plot colours per material ID

# element type IDs
ET_SOLID = 1        # SOLID185, the model
ET_MESH2D = 2       # MESH200, only used to mesh the footprint before extrusion


def as_alpha(alpha):
    """float -> in-plane (a, a, 0); 3-sequence -> (ax, ay, az)."""
    if isinstance(alpha, (int, float)):
        return (float(alpha), float(alpha), 0.0)
    alpha = tuple(float(a) for a in alpha)
    if len(alpha) != 3:
        raise ValueError("alpha must be a float or (alp_x, alp_y, alp_z)")
    return alpha


# Base class: meshing, checks, plotting
class Specimen:
    name = "specimen"

    def __init__(self, mapdl, cfg, plate=None):
        self.mapdl = mapdl
        self.cfg = cfg
        self.plate = plate          # a BasePlate, or None = the old plate of cfg.base_t
        # measurement planes: name -> (normal axis 0/1/2 = x/y/z, position)
        self.sections: dict = {}
        self._built = False

    # ---- defined by each specimen ---------------------------------------------
    @property
    def height(self) -> float:
        raise NotImplementedError

    def footprint(self):
        """Create the footprint area(s) on the current working plane (z = -base_t)."""
        raise NotImplementedError

    def footprint_rects(self):
        """The footprint as a list of rectangles (x0, x1, y0, y1) at z = 0, needed to stand on a BasePlate.
        None = this specimen cannot (yet)."""
        return None

    def skin_strips(self, w):
        """Ranges (axis "X"/"Y", lo, hi) for ESEL,CENT whose union is the skin zone of width w along the side faces.
        Only the Block has one so far."""
        raise NotImplementedError(f"{self.name}: no skin zone defined (only the Block, for now)")

    @property
    def plate_t(self) -> float:
        """Thickness of the plate under the part."""
        return self.plate.thickness if self.plate is not None else self.cfg.base_t

    def z_breaks(self):
        """Heights other than 0 and height that must be layer boundaries."""
        return []

    def assign_extra_materials(self):
        """Hook for specimens with more than plate + part (the bridge's supports)."""

    def release(self, sol):
        """Release sequence after the build. Default: cut the part off the plate."""
        sol.release_part()

    def dims(self) -> dict:
        return {}

    # ---- PREP7 -------------------------------------------------------------------
    def build(self, alpha):
        """Clear the database and build the meshed model. alpha: (alp_x, alp_y, alp_z) or one float for in-plane only."""
        c, m = self.cfg, self.mapdl
        c.check(self.height, self.z_breaks(), plate=self.plate is not None)
        m.clear()
        m.prep7()
        self.element_types()
        self.materials(alpha)
        self.make_mesh()
        self.assign_materials()
        m.finish()
        self._built = True
        self._grid_cache = None                            # new mesh: show() fetches it again

    def element_types(self):
        m = self.mapdl
        m.et(ET_SOLID, "SOLID185")
        m.et(ET_MESH2D, "MESH200")
        m.keyopt(ET_MESH2D, 1, 6)                  # 4-node quad

    def materials(self, alpha):
        """Reference temperature and the three materials (temperature-dependent tables from cfg)."""
        c, m = self.cfg, self.mapdl
        alpha = as_alpha(alpha)
        m.tref(c.air_temp)
        part = c.part_material()                         # read from the digitised csv files
        # plate: 316L, no inherent strain; plastic only if the part is
        self._material(c.MAT_BASE, (0.0, 0.0, 0.0), c.plate_material(), plastic=c.plastic and c.plate_plastic)
        # part: Ni superalloy, carries the inherent strain
        self._material(c.MAT_PART, alpha, part, plastic=c.plastic)
        # supports: same strain, scaled stiffness
        support = dict(part, EX=[(T, E * c.support_E_factor) for T, E in part["EX"]])
        self._material(c.MAT_SUPPORT, alpha, support, plastic=c.plastic)

    def _material(self, mid, alpha, props, plastic):
        """props: {"EX": [(T, E)], "PRXY": [(T, nu)], "BISO": [(T, yield, tangent)]}. ALPX/Y/Z are constant: with
        delta_t = 1 they are the inherent strain.
        """
        c, m = self.cfg, self.mapdl
        for prop in ("EX", "PRXY"):
            m.mptemp()                                   # empty the temperature table
            for i, (T, _) in enumerate(props[prop], start=1):
                m.mptemp(i, T)                           # one value per call (MPTEMP takes at most 6)
            for i, (_, v) in enumerate(props[prop], start=1):
                m.mpdata(prop, mid, i, v)
        m.mptemp()                                       # back to a single temperature for the constants below
        m.mp("ALPX", mid, alpha[0])
        m.mp("ALPY", mid, alpha[1])
        m.mp("ALPZ", mid, alpha[2])
        m.mp("REFT", mid, c.air_temp)                    # reference temperature per material (needed for reborn elements)
        if plastic:
            m.tb("BISO", mid, len(props["BISO"]))
            for T, ys, Et in props["BISO"]:
                m.tbtemp(T)
                m.tbdata(1, ys, Et)                      # yield stress, tangent modulus

    def make_mesh(self):
        m, c = self.mapdl, self.cfg
        if self.plate is not None:                 # plate and part as glued, mapped bricks (baseplate.py)
            self.plate.build(self)
            return

        # 1. footprint at z = -base_t
        m.wpcsys(-1, 0)
        m.wpoffs(0, 0, -c.base_t)
        self.footprint()
        m.wpcsys(-1, 0)
        if int(m.get_value("AREA", 0, "COUNT")) > 1:
            m.aglue("ALL")                         # pieces share their edges

        # 2. 2-D quad mesh of the footprint
        m.type(ET_MESH2D)
        m.esize(c.elem_size)
        m.mshape(0, "2D")
        m.mshkey(2)                                # mapped if possible, else free
        m.amesh("ALL")

        # 3. extrude through plate and part, one element per elem_z (= elem_size unless given)
        total = c.base_t + self.height
        m.type(ET_SOLID)
        m.mat(c.MAT_PART)
        m.extopt("ESIZE", int(round(total / c.ez)), 0)
        m.extopt("ACLEAR", 1)                      # drop the 2-D quads afterwards
        m.extopt("ATTR", 0, 0, 0)                  # use the current TYPE / MAT
        m.vext("ALL", "", "", 0, 0, total)

        m.esel("S", "TYPE", "", ET_MESH2D)         # remove any leftover 2-D elements
        if int(m.get_value("ELEM", 0, "COUNT")) > 0:
            m.edele("ALL")
        m.allsel()
        if int(m.get_value("ELEM", 0, "COUNT")) == 0:
            raise RuntimeError(f"{self.name}: meshing produced no elements")

    def assign_materials(self, tol=1e-6):
        m, c = self.mapdl, self.cfg
        m.esel("S", "CENT", "Z", -self.plate_t, -tol)
        m.emodif("ALL", "MAT", c.MAT_BASE)
        self.assign_extra_materials()
        m.allsel()

    # ---- run and read the results (same method as am_cube.py: run_cube / extract) ----
    def simulate(self, alpha, full=True, fs=1.0, ft=1.0):
        """One run: build, layer by layer, release, read the results (dict like am_cube.run_cube, plus 'alpha').
        full=False reads only node_id, coords, sigma, seqv (enough for calibration).
        fs: strain factor of the skin zone along the side faces (needs cfg.skin_width > 0); ft: of the last layers
        (needs cfg.top_zone > 0). 1 = no effect; both may be used together.
        """
        from mapdl.solution import Solution              # imported here: solution.py does not need specimen.py
        skin = top = None
        if abs(fs - 1.0) > 1e-12:
            if self.cfg.skin_width <= 0:
                raise ValueError("fs needs Config(skin_width > 0)")
            skin = dict(strips=self.skin_strips(self.cfg.skin_width), factor=float(fs))
        if abs(ft - 1.0) > 1e-12:
            if self.cfg.top_zone <= 0:
                raise ValueError("ft needs Config(top_zone > 0)")
            top = dict(z0=self.height - self.cfg.top_zone, factor=float(ft))
        self.build(alpha)
        Solution(self.mapdl, self.cfg, plate=self.plate, skin=skin, top=top).run(self.height, release=self.release)
        res = self.extract(full)
        res["alpha"] = np.asarray(as_alpha(alpha), dtype=float)
        smax = np.abs(res["sigma"]).max()
        if smax < 1.0:
            print(f"WARNING: max |stress| = {smax:.3g} MPa for alpha={alpha} - eigenstrain not applied?")
        return res

    def _vget(self, item, comp="", idx=None):
        arr = np.asarray(self.mapdl.get_array("NODE", item1=item, it1num=comp), dtype=float)
        return arr if idx is None else arr[idx]

    def _node_index(self, nid):
        """Index that lines up *VGET arrays with the selected node list nid (None if they already do).
        Checked against single-node queries, so wrong data cannot pass silently.
        """
        probe = [self._vget("LOC", c) for c in "XYZ"]
        if all(p.size == nid.size for p in probe):
            return None
        size = probe[0].size
        candidates = []
        if size >= nid.max():
            candidates.append(nid - 1)                     # one value per node number 1, 2, 3 ...
        if size == nid.max() - nid.min() + 1:
            candidates.append(nid - nid.min())             # one value per node number from the lowest selected one
        m = self.mapdl
        for idx in candidates:
            ok = True
            for i in sorted({0, len(nid) // 2, len(nid) - 1}):
                want = [float(m.get_value("NODE", int(nid[i]), "LOC", c)) for c in "XYZ"]
                if any(abs(p[idx[i]] - w) > 1e-6 for p, w in zip(probe, want)):
                    ok = False
                    break
            if ok:
                return idx
        raise RuntimeError(f"the nodal results do not line up with the node list: {nid.size} selected nodes "
                           f"(numbers {nid.min()}..{nid.max()}) but the coordinate arrays have {size} values, and neither "
                           f"'one per node number' reading matches the coordinates MAPDL gives for single nodes")

    def diagnose(self) -> str:
        """The state of the model, in a few lines: for every material the elements (live / total) and the extent of the
        nodes of the live ones. Used in error messages; it never raises."""
        m, c = self.mapdl, self.cfg
        out = []
        try:
            m.allsel()
            for mid, label in c.MAT_NAMES.items():
                m.esel("S", "MAT", "", mid)
                total = int(m.get_value("ELEM", 0, "COUNT"))
                if total == 0:
                    continue
                m.esel("R", "LIVE")
                live = int(m.get_value("ELEM", 0, "COUNT"))
                if live:
                    m.nsle("S")
                    zr = (m.get_value("NODE", 0, "MNLOC", "Z"), m.get_value("NODE", 0, "MXLOC", "Z"))
                    n = int(m.get_value("NODE", 0, "COUNT"))
                    out.append(f"{label}: {live} of {total} elements live; their {n} nodes have z = {zr[0]:g} ... {zr[1]:g}")
                else:
                    out.append(f"{label}: 0 of {total} elements live")
            m.allsel()
        except Exception as err:                           # diagnosis only, never hide the real error
            out.append(f"(diagnosis failed: {err})")
        return "\n".join(out)

    def extract(self, full=True):
        """Nodal results of the live elements, sorted by node number (full=False: node_id, coords, sigma, seqv)."""
        m = self.mapdl
        m.post1()
        m.set("LAST")
        m.rsys(0)                                          # global axes
        m.allsel()
        m.esel("S", "LIVE")                                # killed plate / supports are not read
        m.nsle("S", "ALL")
        comps = ("X", "Y", "Z", "XY", "YZ", "XZ")
        nid = self._vget("NLIST").astype(int)
        idx = self._node_index(nid)                        # None unless *VGET and NLIST do not line up (see there)
        v = lambda item, comp="": self._vget(item, comp, idx)
        out = dict(node_id=nid,
                   coords=np.column_stack([v("LOC", c) for c in "XYZ"]),               # undeformed
                   sigma=np.column_stack([v("S", c) for c in comps]),                  # MPa
                   seqv=v("S", "EQV"))
        if full:
            out["u"] = np.column_stack([v("U", c) for c in "XYZ"])
            out["eps_el"] = np.column_stack([v("EPEL", c) for c in comps])             # engineering shear
            out["eps_pl"] = np.column_stack([v("EPPL", c) for c in comps])
        m.allsel()
        m.finish()
        order = np.argsort(nid)
        return {k: v[order] for k, v in out.items()}

    def plane_stress(self, res, name):
        """(u, z, s) on a measurement plane: horizontal coordinate, height, normal stress of the part nodes on it."""
        axis, pos = self.sections[name]
        xyz = res["coords"]
        on = (np.abs(xyz[:, axis] - pos) < 1e-4) & (xyz[:, 2] > -1e-6)     # on the plane, part only (no plate)
        u_axis = 1 if axis == 0 else 0
        return xyz[on, u_axis], xyz[on, 2], res["sigma"][on, axis]

    # ---- checks ----------------------------------------------------------------------
    def info(self) -> dict:
        """Element counts per material, number of layers and part bounds."""
        m, c = self.mapdl, self.cfg
        out = {"specimen": repr(self), "layers": c.n_layers(self.height)}
        if self.plate is not None:
            out["base plate"] = repr(self.plate)
        if c.skin_width > 0:
            out["skin zone"] = f"{c.skin_width:g} mm from the side faces"
        if c.top_zone > 0:
            out["top zone"] = f"the last {c.top_zone:g} mm of layers"
        for mid, label in c.MAT_NAMES.items():
            m.esel("S", "MAT", "", mid)
            out[f"elements_{label}"] = int(m.get_value("ELEM", 0, "COUNT"))
        if self.plate is not None:                     # where the part stands on the plate (nodes of each material)
            b = {}
            for label, mid in (("part", c.MAT_PART), ("plate", c.MAT_BASE)):
                m.esel("S", "MAT", "", mid)
                m.nsle("S")
                b[label] = [float(m.get_value("NODE", 0, k, ax)) for ax in "XY" for k in ("MNLOC", "MXLOC")]
            p_, q_ = b["part"], b["plate"]
            out["plate margin x"] = f"{p_[0] - q_[0]:g} mm (-x side) / {q_[1] - p_[1]:g} mm (+x side)"
            out["plate margin y"] = f"{p_[2] - q_[2]:g} mm (-y side) / {q_[3] - p_[3]:g} mm (+y side)"
        m.allsel()
        out["elements_total"] = int(m.get_value("ELEM", 0, "COUNT"))
        out["nodes_total"] = int(m.get_value("NODE", 0, "COUNT"))
        for k, v in out.items():
            print(f"{k:>18}: {v}")
        return out

    def show(self, color="material", clip=None, sections=True, hide=(),
             jupyter_backend=None, **show_kw):
        """Plot the mesh. color: "material" or "layer"; clip: None, "x", "y" or "z" cuts the model at its centre
        (on a base plate this cuts through the middle of the part, so use clip=None to see the part centred);
        sections: draw the measurement plane; hide: materials to leave out; jupyter_backend: passed to pyvista.
        """
        import pyvista as pv

        grid = self.grid()
        if hide:
            ids = [i for i, name in self.cfg.MAT_NAMES.items() if name in hide]
            grid = grid.extract_cells(np.flatnonzero(~np.isin(grid.cell_data["material"], ids)))
        zc = grid.cell_centers().points[:, 2]

        if color == "material":
            key = grid.cell_data["material"]
            labels = self.cfg.MAT_NAMES
            colors = {v: MATERIAL_COLORS.get(v, "#cccccc") for v in labels}
        elif color == "layer":
            layer = np.floor(zc / self.cfg.layer_h + 1e-9).astype(int)
            key = np.where(zc < 0, -1, layer % 2)
            colors = {-1: "#9e9e9e", 0: "#4c78a8", 1: "#a0cbe8"}
            labels = {-1: "plate", 0: "even layer", 1: "odd layer"}
        else:
            raise ValueError('color must be "material" or "layer"')
        present = [v for v in colors if np.any(key == v)]
        keys = sorted(colors)                              # vectorised colour lookup
        palette = np.array([pv.Color(colors[k]).int_rgb for k in keys], dtype=np.uint8)
        grid.cell_data["rgb"] = palette[np.searchsorted(keys, key)]

        shown = grid
        if clip is not None:
            # keep the elements whose centre is on the low side of the mid-plane
            axis = "xyz".index(clip)
            centres = grid.cell_centers().points[:, axis]
            shown = grid.extract_cells(np.where(centres < grid.center[axis])[0])

        p = pv.Plotter()
        p.add_mesh(shown, scalars="rgb", rgb=True, show_edges=True,
                   edge_color="#1a1a1a", edge_opacity=0.35, line_width=0.5)
        legend = [(labels[v], colors[v]) for v in present]
        if sections and self.sections:
            is_part = np.asarray(grid.cell_data["material"]) == self.cfg.MAT_PART      # the plane is drawn on the part only
            pb = grid.extract_cells(np.flatnonzero(is_part)).bounds if is_part.any() else None
            for name, (axis, pos) in self.sections.items():
                p.add_mesh(self._section_box(shown, axis, pos, bounds=pb), color="red", opacity=0.3)
            legend.append(("measurement plane", "red"))
        p.add_legend(legend, bcolor="white", face="rectangle", size=(0.22, 0.05 * len(legend) + 0.02))
        p.add_axes()
        p.add_text(repr(self), font_size=10)
        p.show(jupyter_backend=jupyter_backend, **show_kw)

    def grid(self):
        """pyvista grid of the whole model with a 'material' cell array."""
        if not self._built:
            raise RuntimeError("call build(alpha) first")
        if getattr(self, "_grid_cache", None) is None:     # fetch from MAPDL once per build
            m = self.mapdl
            m.prep7()
            m.allsel()
            grid = m.mesh.grid.copy()
            if "ansys_material_type" in grid.cell_data:
                mats = np.asarray(grid.cell_data["ansys_material_type"])
            else:
                mats = np.asarray(m.mesh.material_type)
            m.finish()
            grid.cell_data["material"] = mats.astype(int)
            self._grid_cache = grid
        return self._grid_cache.copy()

    @staticmethod
    def _section_box(grid, axis, pos, half=0.02, bounds=None):
        import pyvista as pv
        b = list(bounds if bounds is not None else grid.bounds)
        b[4] = max(b[4], 0.0)                     # part only, not the plate
        margin = 0.15 * max(b[1] - b[0], b[3] - b[2], b[5] - b[4])
        for i in range(3):                        # stick out a little so it stays visible
            if i != axis:
                b[2 * i] -= margin if i != 2 else 0.0
                b[2 * i + 1] += margin
        b[2 * axis], b[2 * axis + 1] = pos - half, pos + half
        return pv.Box(bounds=b)

    def __repr__(self):
        dims = ", ".join(f"{k}={v:g}" for k, v in self.dims().items())
        plate = f", plate={self.plate!r}" if self.plate is not None else ""
        return f"{type(self).__name__}({dims}{plate})"


# Footprint helper
def rectangles(mapdl, xs, ys):
    """Grid of rectangles between consecutive xs and ys (on the working plane)."""
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            mapdl.rectng(xs[i], xs[i + 1], ys[j], ys[j + 1])