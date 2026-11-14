"""Units: mm, MPa
Managed by Timothy Louis Barus

Lean calibration of the inherent strain (ax = a, ay = a * r, az fixed) with optional top and skin factors (ft, fs).
Every run is saved; fit() builds a surrogate of the whole stress field from the runs and minimises the RMSE on it.
Usage and method: README.txt.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.optimize import minimize
from scipy.stats import qmc

NAMES = ("a", "ft", "fs", "r")                                        # r = ay / ax
DEF = dict(a=-0.0044, ft=1.0, fs=1.0, r=1.0)                          # default of every parameter (1 = no effect)
WIN = dict(a=(-0.012, -0.002), ft=(0.5, 2.0), fs=(0.5, 2.0), r=(0.5, 1.5))      # search windows
LEVELS = (("L0  a", ["a"]), ("L1  a, top, skin", ["a", "ft", "fs"]), ("L2  a, top, skin, y/x", ["a", "ft", "fs", "r"]))
rms = lambda e: float(np.sqrt(np.mean(np.square(e))))


class Calibration:
    def __init__(self, spec, csv, edge_mm=1.0, az=0.005, section="mid_length", save_dir="calibration_runs", tag=""):
        """spec: a Block; csv: digitised map (u_mm, z_mm, measured_MPa, zhang_GB_predicted_MPa); edge_mm: points closer
        to the border are left out of the loss; az: fixed alp_z; tag: change it to start fresh saved runs."""
        t = pd.read_csv(csv)
        self.spec, self.az, self.section = spec, az, section
        self.U, self.Z, self.M, self.ZH = (t[c].to_numpy() for c in ("u_mm", "z_mm", "measured_MPa", "zhang_GB_predicted_MPa"))
        self.mask = np.minimum.reduce([self.U, self.U.max() + 0.05 - self.U, self.Z, self.Z.max() + 0.05 - self.Z]) >= edge_mm
        z = self._rec((), self.ZH)
        print(f"{len(t)} target points, {self.mask.sum()} in the loss | Zhang GB model vs measured: loss {z['loss']:.1f}, "
              f"all {z['all']:.1f}, within 100 MPa {z['w100']:.0f} %")
        c = spec.cfg
        info = dict(spec=type(spec).__name__, dims=spec.dims(), cfg={k: v for k, v in c.to_dict().items() if k != "material_dir"},
                    part=c.part_material(), plate=c.plate_material(), az=az, tag=tag, section=section,
                    grid=hashlib.sha1(np.round(np.r_[self.U, self.Z], 6).tobytes()).hexdigest()[:10],
                    base=spec.plate.describe() if getattr(spec, "plate", None) else None)
        text = json.dumps(info, sort_keys=True, default=str)
        self.dir = Path(save_dir) / f"{type(spec).__name__}_{hashlib.sha1(text.encode()).hexdigest()[:10]}"
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "setup.json").write_text(text)
        fl = self.dir / "failed.txt"                                      # vectors MAPDL could not solve: never tried again
        self.bad = {tuple(map(float, l.split())) for l in fl.read_text().splitlines()} if fl.exists() else set()
        self.runs = {}
        for f in self.dir.glob("run_*.npz"):
            with np.load(f) as d:
                k, sig = tuple(float(v) for v in d["th"]), d["sig"]
            if sig.shape == self.U.shape:
                self.runs[k] = self._rec(k, sig)
        print(f"{len(self.runs)} saved runs in {self.dir}  (layer {spec.cfg.layer_h:g} mm, elements {spec.cfg.elem_size:g} mm)")

    @staticmethod
    def find(folder, name):
        """A csv anywhere below a folder."""
        return next(Path(folder).rglob(name))

    def _rec(self, k, sig):
        e = sig - self.M
        return dict(th=k, sig=sig, loss=rms(e[self.mask]), all=rms(e), bias=float(e.mean()), w100=float(100 * np.mean(np.abs(e) < 100)))

    # ---- one simulation ------------------------------------------------------------------------------
    def run(self, th=None):
        """Simulate (or load) one parameter set; th: dict with any of a, ft, fs, r (others at their default)."""
        k = tuple(round(float((th or {}).get(n, DEF[n])), 6) for n in NAMES)
        if k in self.bad:
            return None
        if k not in self.runs:
            a, ft, fs, r = k
            t0 = time.time()
            try:
                u, z, s = self.spec.plane_stress(self.spec.simulate((a, a * r, self.az), full=False, fs=fs, ft=ft), self.section)
            except Exception as e:
                self.bad.add(k)
                (self.dir / "failed.txt").open("a").write(" ".join(map(str, k)) + "\n")
                print(f"run failed ({type(e).__name__}: {str(e)[:80]}), this vector is skipped from now on: a={a:+.5f} ft={ft:.3f} fs={fs:.3f} r={r:.3f}")
                return None
            sig = griddata((u, z), s, (self.U, self.Z), method="linear")
            bad = np.isnan(sig)
            if bad.any():
                sig[bad] = griddata((u, z), s, (self.U[bad], self.Z[bad]), method="nearest")
            self.runs[k] = q = self._rec(k, sig)
            np.savez(self.dir / "tmp.npz", th=np.array(k), sig=sig)
            os.replace(self.dir / "tmp.npz", self.dir / ("run_%+.6f_%.4f_%.4f_%.4f.npz" % k))
            print(f"run {len(self.runs):03d}  layer={self.spec.cfg.layer_h:g}  a={a:+.5f} ft={ft:.3f} fs={fs:.3f} r={r:.3f}  loss {q['loss']:5.1f}  all {q['all']:5.1f}  "
                  f"bias {q['bias']:+5.1f}  within 100 MPa {q['w100']:4.1f} %  ({time.time() - t0:.0f}s)")
        return self.runs[k]

    # ---- surrogate fit -------------------------------------------------------------------------------
    def fit(self, free, n_design=None, rounds=5, tol=0.1, seed=0):
        """Minimise the loss over the parameters in `free` (names from a, ft, fs, r; the others stay at their default).
        A Latin-hypercube design of runs, then rounds of: quadratic surrogate of every target point in the parameters
        (fitted to the best runs), minimise the RMSE on it, run that proposal for real. Runs already saved count."""
        d, nf = len(free), (len(free) + 1) * (len(free) + 2) // 2
        lo, hi = np.array([WIN[n][0] for n in free]), np.array([WIN[n][1] for n in free])
        c, w = (lo + hi) / 2, (hi - lo) / 2
        fam = lambda: [q for q in self.runs.values() if all(abs(q["th"][i] - DEF[n]) < 1e-9 for i, n in enumerate(NAMES) if n not in free)]
        vec = lambda q: np.array([q["th"][NAMES.index(n)] for n in free])
        phi = lambda x: np.column_stack([np.ones(len(x))] + [x[:, j] for j in range(d)] + [x[:, j] * x[:, k] for j in range(d) for k in range(j, d)])
        self.run()
        for x in lo + (hi - lo) * qmc.LatinHypercube(d=d, seed=seed).random(max(0, (n_design or max(8, 2 * nf)) - len(fam()))):
            self.run(dict(zip(free, x)))
        best, stall, target = min(q["loss"] for q in fam()), 0, self.M[self.mask]
        for i in range(rounds):
            R = sorted(fam(), key=lambda q: q["loss"])[:max(int(1.5 * nf), 12)]
            Xn = (np.array([vec(q) for q in R]) - c) / w
            beta = np.linalg.lstsq(phi(Xn), np.array([q["sig"][self.mask] for q in R]), rcond=None)[0]
            pad = 0.2 * (Xn.max(0) - Xn.min(0)) + 0.05                           # stay near the runs the surrogate was fitted to
            bnd = list(zip(np.maximum(Xn.min(0) - pad, -1), np.minimum(Xn.max(0) + pad, 1)))
            cand = np.array([b[0] for b in bnd]) + np.array([b[1] - b[0] for b in bnd]) * np.random.default_rng(seed + i).random((600, d))
            L = lambda x: rms(phi(x[None]) @ beta - target)
            xs = [minimize(L, x0, bounds=bnd, method="L-BFGS-B") for x0 in cand[np.argsort([L(x) for x in cand])[:5]]]
            xb = min(xs, key=lambda s: s.fun)
            th = dict(zip(free, c + w * xb.x))
            print(f"fit round {i + 1}: surrogate (fit error {rms(phi(Xn) @ beta - np.array([q['sig'][self.mask] for q in R])):.1f} MPa, "
                  f"{len(R)} runs) proposes {', '.join(f'{n}={v:.5f}' for n, v in th.items())}, predicted loss {xb.fun:.1f}")
            kk = tuple(round(float(th.get(n, DEF[n])), 6) for n in NAMES)
            if kk in self.runs or kk in self.bad:
                print("fit: that vector was already run (or failed), converged")
                break
            q = self.run(th)
            best, stall = (q["loss"], 0) if q and q["loss"] < best - tol else (best, stall + 1)
            if stall >= 2:
                print(f"fit: no gain over {tol} MPa in 2 rounds, stopped")
                break
        return min(fam(), key=lambda q: q["loss"])

    def ladder(self, levels=LEVELS, **kw):
        """fit() for each level (each adds parameters to the one before) and a table: what each addition gained."""
        rows = [(name, self.fit(free, **kw)) for name, free in levels]
        print(f"\n{'level':24}{'loss':>7}{'all':>7}{'bias':>7}{'within 100':>12}   parameters")
        for name, q in rows:
            print(f"{name:24}{q['loss']:7.1f}{q['all']:7.1f}{q['bias']:+7.1f}{q['w100']:10.1f} %   "
                  + ", ".join(f"{n}={v:.5f}" for n, v in zip(NAMES, q["th"])))
        z = self._rec((), self.ZH)
        print(f"{'Zhang GB model':24}{z['loss']:7.1f}{z['all']:7.1f}{z['bias']:+7.1f}{z['w100']:10.1f} %")
        return dict(rows)

    # ---- result --------------------------------------------------------------------------------------
    def report(self, th=None):
        """Maps and numbers of one run (th as in run(); default the best run so far)."""
        q = self.run(th) if th is not None else min(self.runs.values(), key=lambda q: q["loss"])
        print("a, ft, fs, r =", q["th"], f"| loss {q['loss']:.1f}, all {q['all']:.1f}, bias {q['bias']:+.1f}, within 100 MPa {q['w100']:.1f} %"
              f" | model vs Zhang GB: {rms((q['sig'] - self.ZH)[self.mask]):.1f}")
        fig, ax = plt.subplots(1, 3, figsize=(13, 4.3))
        for a, v, t, kw in ((ax[0], self.M, "measured", dict(cmap="jet", vmin=-450, vmax=850)), (ax[1], q["sig"], "FE model", dict(cmap="jet", vmin=-450, vmax=850)),
                            (ax[2], q["sig"] - self.M, "model - measured", dict(cmap="RdBu_r", vmin=-200, vmax=200))):
            fig.colorbar(a.scatter(self.U, self.Z, c=v, s=7, marker="s", **kw), ax=a, shrink=.8)
            a.set(title=t, xlabel="u (mm)", ylabel="z (mm)", aspect="equal")
        plt.show()
        return q