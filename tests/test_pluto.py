# radiosynth - From MHD to MHz: synthetic radio images of MHD models of the Sun and stars
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
"""PLUTO reader + SphericalGrid tests.

Writes the analytic corona as PLUTO output (3D spherical .dbl single_file,
2.5D axisymmetric .flt multiple_files, 3D spherical legacy VTK) using
non-trivial code units, reads it back with radiosynth.load_pluto and compares
the radio images with those computed on a fine Cartesian grid.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples"))
import radiosynth as rs                                        # noqa: E402
from radiosynth.constants import M_P, K_B                     # noqa: E402
from synthetic_corona import model_fields, make_model         # noqa: E402

OUT = os.path.join(HERE, "output", "pluto")
os.makedirs(OUT, exist_ok=True)
CONST_mp, CONST_kB, CONST_Rsun = 1.67262171e-24, 1.3806505e-16, 6.96e10
UD, UL, UV = CONST_mp * 1e8, CONST_Rsun, 1.0e7          # code units used below
DEFS = """#define  PHYSICS                        MHD
#define  DIMENSIONS                     3
#define  GEOMETRY                       SPHERICAL
/* -- user-defined parameters (labels) -- */
#define  UNIT_DENSITY      (CONST_mp*1.e8)
#define  UNIT_LENGTH       CONST_Rsun
#define  UNIT_VELOCITY     1.e7
"""
BLOB = ((0.3, -1.9, 0.6), 0.45, 4.0)
TE, TI = 1.5e6, 1.5e6
ok = []


def write_grid_out(path, r_e, t_e, p_e, geom="SPHERICAL"):
    with open(path, "w") as fh:
        fh.write("# ******************************************************\n")
        fh.write("# PLUTO 4.4 Grid File\n# DIMENSIONS: 3\n")
        fh.write(f"# GEOMETRY:   {geom}\n")
        fh.write("# ******************************************************\n")
        for e in (r_e, t_e, p_e):
            fh.write(f"{e.size - 1}\n")
            for i in range(e.size - 1):
                fh.write(f" {i + 1}  {e[i]:.12e}  {e[i + 1]:.12e}\n")


def fields_on(r, th, ph, blob=BLOB, streamer=True):
    R, TH, PH = np.meshgrid(r, th, ph, indexing="ij")
    X = R * np.sin(TH) * np.cos(PH)
    Y = R * np.sin(TH) * np.sin(PH)
    Z = R * np.cos(TH)
    ne, te = model_fields(X, Y, Z, TE, blob=blob, streamer=streamer)
    rho = ne * M_P * 1e-3 / UD                               # code density (mu_e=1)
    prs = 2 * ne * 1e-6 * CONST_kB * te / (UD * UV**2)      # code pressure, Te=Ti
    return rho, prs


def write_dbl(d, r_e, t_e, p_e, fmt="dbl", mode="single_file", blob=BLOB):
    os.makedirs(d, exist_ok=True)
    write_grid_out(os.path.join(d, "grid.out"), r_e, t_e, p_e)
    open(os.path.join(d, "definitions.h"), "w").write(DEFS)
    rc, tc, pc = (0.5 * (e[1:] + e[:-1]) for e in (r_e, t_e, p_e))
    rho, prs = fields_on(rc, tc, pc, blob=blob)
    dt = "<f8" if fmt == "dbl" else "<f4"
    names = ["rho", "vx1", "vx2", "vx3", "prs"]
    arrs = [rho, 0 * rho, 0 * rho, 0 * rho, prs]
    with open(os.path.join(d, f"{fmt}.out"), "w") as fh:
        fh.write(f"0 0.000000e+00 1.0e-04 0 {mode} little {' '.join(names)}\n")
        fh.write(f"1 1.000000e+00 1.0e-04 100 {mode} little {' '.join(names)}\n")
    for n in (0, 1):
        if mode == "single_file":
            with open(os.path.join(d, f"data.{n:04d}.{fmt}"), "wb") as fh:
                for a in arrs:
                    fh.write(a.transpose(2, 1, 0).astype(dt).tobytes())
        else:
            for nm, a in zip(names, arrs):
                a.transpose(2, 1, 0).astype(dt).tofile(os.path.join(d, f"{nm}.{n:04d}.{fmt}"))


def write_vtk(path, r_e, t_e, p_e):
    R, TH, PH = np.meshgrid(r_e, t_e, p_e, indexing="ij")
    P = np.stack([R * np.sin(TH) * np.cos(PH), R * np.sin(TH) * np.sin(PH), R * np.cos(TH)], -1)
    rc, tc, pc = (0.5 * (e[1:] + e[:-1]) for e in (r_e, t_e, p_e))
    rho, prs = fields_on(rc, tc, pc)
    with open(path, "wb") as fh:
        fh.write(b"# vtk DataFile Version 2.0\nPLUTO 4.4 VTK Data\nBINARY\nDATASET STRUCTURED_GRID\n")
        fh.write(b"FIELD FieldData 1\nTIME 1 1 double\n")
        fh.write(np.array([1.0], ">f8").tobytes())
        fh.write(b"\n")
        fh.write(f"DIMENSIONS {r_e.size} {t_e.size} {p_e.size}\n".encode())
        fh.write(f"POINTS {P[..., 0].size} float\n".encode())
        fh.write(P.transpose(2, 1, 0, 3).astype(">f4").tobytes())
        fh.write(f"\nCELL_DATA {rho.size}\n".encode())
        for nm, a in (("rho", rho), ("prs", prs)):
            fh.write(f"SCALARS {nm} float\nLOOKUP_TABLE default\n".encode())
            fh.write(a.transpose(2, 1, 0).astype(">f4").tobytes())
            fh.write(b"\n")


# grids: log-spaced radius 1..8 R_sun
r_e = np.geomspace(1.0, 8.0, 121)
t_e = np.linspace(0, np.pi, 97)
p_e = np.linspace(0, 2 * np.pi, 129)

d3 = os.path.join(OUT, "sph3d_dbl")
write_dbl(d3, r_e, t_e, p_e)
d2 = os.path.join(OUT, "sph2d_flt")
write_dbl(d2, r_e, t_e, np.array([0.0, 2 * np.pi]), fmt="flt", mode="multiple_files",
          blob=None)
vtk = os.path.join(OUT, "data.0001.vtk")
open(os.path.join(OUT, "definitions.h"), "w").write(DEFS)
write_vtk(vtk, r_e, t_e, p_e)

# --- 1. units / fields read back exactly
g3 = rs.load_mhd(d3)
rc = 0.5 * (r_e[1:] + r_e[:-1])
tc = 0.5 * (t_e[1:] + t_e[:-1])
pc = 0.5 * (p_e[1:] + p_e[:-1])
R, TH, PH = np.meshgrid(rc, tc, pc, indexing="ij")
ne_true, _ = model_fields(R * np.sin(TH) * np.cos(PH), R * np.sin(TH) * np.sin(PH),
                          R * np.cos(TH), TE, blob=BLOB, streamer=True)
ne_read = g3._flat[:, 0].reshape(g3.shape)
te_read = g3._flat[:, 4].reshape(g3.shape)
e1 = np.max(np.abs(ne_read / ne_true - 1))
e2 = np.max(np.abs(te_read / TE - 1))
good = e1 < 1e-5 and e2 < 1e-5   # float32 storage
ok.append(good)
print(f"[{'PASS' if good else 'FAIL'}] PLUTO dbl single_file, units from definitions.h: "
      f"rel err n_e {e1:.1e}, T_e {e2:.1e}")

# --- 2. sampler: interpolated n_e and gradient vs analytic at random points
rng = np.random.default_rng(3)
pts = rng.normal(size=(4000, 3))
pts *= (rng.uniform(1.2, 5.0, 4000) / np.linalg.norm(pts, axis=1))[:, None]
quiet = rs.SphericalGrid(rc, tc, pc, *[a for a in model_fields(
    R * np.sin(TH) * np.cos(PH), R * np.sin(TH) * np.sin(PH), R * np.cos(TH), TE)],
    r_edges=r_e, theta_edges=t_e, phi_edges=p_e)
ne_s, gr_s, _, _, ins = quiet.sample(pts)
rr = np.linalg.norm(pts, axis=1)
ne_a = 4.2e4 * 10**(4.32 / rr) * 1e6
gr_a = (ne_a * np.log(10) * (-4.32 / rr**2) / rr)[:, None] * pts
e_n = np.median(np.abs(ne_s / ne_a - 1))
e_g = np.median(np.linalg.norm(gr_s - gr_a, axis=1) / np.linalg.norm(gr_a, axis=1))
good = ins.all() and e_n < 0.01 and e_g < 0.02
ok.append(good)
print(f"[{'PASS' if good else 'FAIL'}] SphericalGrid sampling: median rel err n_e {e_n:.1e}, "
      f"|grad| {e_g:.1e}")

# --- 3. images: spherical PLUTO vs fine Cartesian reference
OBS = (-215.0, 0.0, 0.0)
kw = dict(x_size=8, y_size=8, nx=64, ny=64, n_workers=os.cpu_count(), verbose=False)
cart = make_model(half=4.0, n=241, te=TE, streamer=True, blob=BLOB)
freqs = "80MHz, 327MHz, 1GHz"
ref = rs.make_radio_image(cart, freqs, OBS, **kw)
sph = rs.make_radio_image(g3, freqs, OBS, r_integration=7.0, **kw)
for k, f in enumerate(ref.freqs):
    a, b = ref.intensity[k], sph.intensity[k]
    dflux = sph.flux_density[k] / ref.flux_density[k] - 1
    corr = np.corrcoef(np.log10(a[a > 0].ravel() + 1e-40),
                       np.log10(b[a > 0].ravel() + 1e-40))[0, 1]
    good = abs(dflux) < 0.05 and corr > 0.98
    ok.append(good)
    print(f"[{'PASS' if good else 'FAIL'}] {f / 1e6:6.0f} MHz spherical PLUTO vs Cartesian: "
          f"flux diff {dflux:+.2%}, log-image correlation {corr:.4f}")

# --- 4. axisymmetric 2.5D (flt, multiple_files)
g2 = rs.load_mhd(d2, nout=1)
cart2 = make_model(half=4.0, n=241, te=TE, streamer=True)
ref2 = rs.make_radio_image(cart2, "150MHz", OBS, **kw)
im2 = rs.make_radio_image(g2, "150MHz", OBS, r_integration=7.0, **kw)
dflux = im2.flux_density[0] / ref2.flux_density[0] - 1
good = abs(dflux) < 0.05
ok.append(good)
print(f"[{'PASS' if good else 'FAIL'}] 2.5D axisymmetric PLUTO (flt, multiple_files): "
      f"flux diff {dflux:+.2%} vs Cartesian")

# --- 5. VTK
gv = rs.load_mhd(vtk)
imv = rs.make_radio_image(gv, "327MHz", OBS, r_integration=7.0, **kw)
dflux = imv.flux_density[0] / sph.flux_density[1] - 1
good = abs(dflux) < 0.01
ok.append(good)
print(f"[{'PASS' if good else 'FAIL'}] PLUTO legacy VTK (3D spherical): flux diff vs dbl {dflux:+.2%}")

# --- 6. pyPLUTO route (bundled classic pyPLUTO.pload) gives the same fields
for d in (d3, d2):
    a = rs.load_pluto(d, nout=1, verbose=False)
    b = rs.load_pluto(d, nout=1, reader="pypluto", verbose=False)
    e = np.abs(a._flat - b._flat).max() / np.abs(a._flat).max()
    good = e < 1e-5
    ok.append(good)
    print(f"[{'PASS' if good else 'FAIL'}] pyPLUTO reader ({os.path.basename(d)}): max rel diff {e:.1e}")

sph.plot("intensity", path=os.path.join(OUT, "pluto_sph_images.png"))
print("Summary:", sum(ok), "/", len(ok), "passed")
