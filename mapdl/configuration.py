"""
Units: mm, MPa, celcius
Managed by Timothy Louis Barus
"""

# imports
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

 
@dataclass
class Config:

    # material IDs and their names (class constants, not settings)
    MAT_BASE = 1                    # build plate
    MAT_PART = 2                    # printed part, carries the inherent strain
    MAT_SUPPORT = 3                 # support structure (bridge only)
    MAT_NAMES = {1: "plate", 2: "part", 3: "support"}
 
    # meshing
    elem_size: float = 0.5          # in-plane element size (and through the height too, unless elem_z is given)
    elem_z: float | None = None     # element size in the build direction (None = elem_size). Lets the layer be thinner than
                                    # the in-plane element: elem_size=1, elem_z=0.5, layer_h=0.5 (README section 8d)
 
    # AM
    layer_h: float = 0.5            # computational layer thickness, multiple of elem_size

    base_t: float = 2.0             # old plate thickness (no BasePlate); ignored when a BasePlate is given

    # temperature
    air_temp: float = 25.0          # reference temperature (TREF)
    delta_t: float = 1.0            # keep 1 so that alpha == inherent strain
 
    # booleans
    two_step: bool = False          # False (as am_cube.py): activate the layer and heat it in ONE solve
                                    # True: activate at air_temp and solve, then heat it and solve again
    release: bool = True            # cut the part off the base before reading results
    plastic: bool = True            # False: elastic part (standard inherent strain method)
    nlgeom: bool = True             # True: a reactivated layer starts strain-free in its CURRENT shape.
                                    # False: starts in its original shape (Ansys warns, results not meaningful)
 
    # ---- material properties, replicated from Zhang et al. 2025 (Additive Manufacturing 102, 104721) ----
    support_E_factor: float = 1.0   # support stiffness = E * factor
    hardening_strain: float = 0.35  # paper Sec. 2.2: strength rises linearly from yield to UTS at 0.35 plastic strain

    # PART: Zhang's Ni superalloy, read as digitised from the csv files of Fig. 2, searched below material_dir.
    # Nothing is interpolated; see part_material() and the README.
    material_dir: str = "data"

    # PLATE: 316L from Ref [26] (Leggatt et al.); csv files, plate_nu constant. plate_tangent is NOT from Zhang:
    # 0 = perfectly plastic at the proof stress, 747 = fit to the reference curve (see README).
    plate_nu: float = 0.290
    plate_plastic: bool = True      # False: elastic plate even when plastic=True (plastic=False makes the part elastic too)
    plate_tangent: float = 0.0      # BISO tangent modulus of the plate, MPa, same at every temperature
    # BISO tables hold at most 6 temperatures: these rows of the proof-stress csv are used (each must be in the csv)
    plate_yield_temps: tuple = (20, 100, 200, 400, 800, 1400)

    # zones (README): skin = part elements whose centroid is < skin_width from a side face (a multiple of elem_size);
    # top = the last top_zone mm of layers (a multiple of layer_h). Their strain factors fs, ft are set per run. 0 = off.
    skin_width: float = 0.0         # mm
    top_zone: float = 0.0           # mm

    # solver
    nsubst: tuple = (2, 25, 1)      # substeps per load step (n, nmax, nmin), as am_cube.py
 
    # ---- derived ------------------------------------------------------------
    @property
    def ez(self) -> float:
        """Element size in the build direction."""
        return self.elem_size if self.elem_z is None else self.elem_z

    def n_layers(self, height: float) -> int:
        """Number of computational layers for a part of this height."""
        return int(round(height / self.layer_h))
 
    def check(self, height: float, breaks=(), plate=False):
        """Raise if layers and plate do not line up with element boundaries. breaks: other heights that must be layer boundaries;
        plate: a BasePlate is used, so base_t (the old plate) does not matter."""
        if self.elem_z is not None and self.elem_z <= 0:
            raise ValueError(f"elem_z={self.elem_z} must be positive (or None = elem_size)")
        if not _multiple(self.layer_h, self.ez):
            raise ValueError(f"layer_h={self.layer_h} is not a multiple of the element size in the build direction "
                             f"{self.ez:g} (elem_z, or elem_size when elem_z is None)")
        if not plate and not _multiple(self.base_t, self.ez):
            raise ValueError(f"base_t={self.base_t} is not a multiple of the element size in the build direction {self.ez:g}")
        if self.skin_width < 0:
            raise ValueError(f"skin_width={self.skin_width} must be >= 0 (0 = no skin zone)")
        if self.skin_width and not _multiple(self.skin_width, self.elem_size):
            raise ValueError(f"skin_width={self.skin_width:g} must be a multiple of elem_size={self.elem_size:g}, otherwise "
                             f"the edge of the zone falls on the centre of an element row (try {max(1, round(self.skin_width / self.elem_size)) * self.elem_size:g})")
        if self.top_zone < 0:
            raise ValueError(f"top_zone={self.top_zone} must be >= 0 (0 = no top zone)")
        if self.top_zone:
            if not _multiple(self.top_zone, self.layer_h):
                raise ValueError(f"top_zone={self.top_zone:g} must be a multiple of layer_h={self.layer_h:g}")
            if self.top_zone >= height:
                raise ValueError(f"top_zone={self.top_zone:g} must be below the part height {height:g}")
        for z in (height, *breaks):
            if not _multiple(z, self.layer_h):
                raise ValueError(f"height {z} is not a multiple of layer_h={self.layer_h}")
 
    def part_material(self) -> dict:
        """Part properties from the digitised csv files, as lists of (temperature, value) rows: "EX" (MPa), "PRXY",
        "BISO" (T, yield, tangent = (UTS - yield) / hardening_strain). Fig2b_CTE.csv is not read: alpha is the inherent strain.
        """
        d = self.material_dir
        E = [(T, v * 1000.0) for T, v in _read_xy(d, "Fig2a_moduli_scaled.csv")]
        nu = _read_xy(d, "Fig2b_possion_ratio.csv", "Fig2b_poisson_ratio.csv")        # either spelling
        ys = _read_xy(d, "Fig2a_yield_strength.csv")
        uts = _read_xy(d, "Fig2a_UTS.csv")
        if len(ys) != len(uts) or any(abs(a[0] - b[0]) > 2.0 for a, b in zip(ys, uts)):
            raise ValueError(f"yield and UTS csv do not have the same markers: {ys} / {uts}")
        biso = [(T, y, (u - y) / self.hardening_strain) for (T, y), (_, u) in zip(ys, uts)]
        return {"EX": E, "PRXY": nu, "BISO": biso}

    def plate_material(self) -> dict:
        """Plate (316L) properties from the Ref 26 csv files, same form as part_material(); BISO rows from
        plate_yield_temps, tangent = plate_tangent.
        """
        d = self.material_dir
        E = [(T, v * 1000.0) for T, v in _read_xy(d, "Ref26_316L_youngs_modulus.csv")]
        ys = _read_xy(d, "Ref26_316L_proof_stress.csv")
        keep = {float(t) for t in self.plate_yield_temps}
        ys = [r for r in ys if r[0] in keep]
        if len(ys) != len(keep) or len(ys) > 6:
            raise ValueError(f"plate_yield_temps {sorted(keep)} must be at most 6 temperatures that all appear "
                             f"in Ref26_316L_proof_stress.csv")
        return {"EX": E, "PRXY": [(E[0][0], self.plate_nu)],
                "BISO": [(T, y, self.plate_tangent) for T, y in ys]}

    def to_dict(self) -> dict:
        return asdict(self)
 
def _multiple(a, b):
    return abs(a / b - round(a / b)) < 1e-6


def _read_xy(folder, *names):
    """Rows (x, y) of a two-column csv (one header line) found anywhere below folder, sorted by x. Several names = alternative spellings."""
    for name in names:
        p = next(Path(folder).rglob(name), None)
        if p is not None:
            break
    else:
        raise FileNotFoundError(f"{' or '.join(names)} not found below {Path(folder).resolve()}")
    with open(p, newline="") as f:
        rows = [r for r in csv.reader(f) if len(r) >= 2]
    return sorted((float(a), float(b)) for a, b in rows[1:])        # rows[0] is the header