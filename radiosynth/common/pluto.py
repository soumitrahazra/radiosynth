# radiosynth - From MHD to MHz: synthetic radio images of MHD models of the Sun and stars
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
"""Reader for PLUTO (Mignone et al. 2007) output.

Supported
---------
* binary ``.dbl`` (float64) and ``.flt`` (float32) output, ``single_file`` or
  ``multiple_files``, described by ``grid.out`` and ``dbl.out`` / ``flt.out``;
* legacy VTK ``.vtk`` (RECTILINEAR_GRID for Cartesian, STRUCTURED_GRID for
  3D spherical), cell data;
* geometries CARTESIAN (-> MHDGrid) and SPHERICAL (-> SphericalGrid; 3D or
  2.5D axisymmetric r-theta).

Units
-----
PLUTO stores normalised variables.  Physical values are
    rho [g/cm^3]      = rho * UNIT_DENSITY
    p   [dyn/cm^2]    = prs * UNIT_DENSITY * UNIT_VELOCITY^2
    length [cm]       = x   * UNIT_LENGTH
The three units are read from ``definitions.h`` when it is found (the
expressions may use PLUTO's CONST_* names), or can be passed explicitly.
PLUTO's own defaults are UNIT_DENSITY = m_p, UNIT_LENGTH = 1 AU,
UNIT_VELOCITY = 1 km/s.

Plasma model
------------
n_e = rho / (mu_e m_p)     (mu_e = mass per electron in m_p; 1 for pure H as
                            in BATS-R-US, ~1.17 for solar H+He)
T_e = p * mu m_p / (rho k_B)  (single-fluid temperature, mu = mean molecular
                            weight; 0.5 for fully ionised H, i.e. T_e = T_i =
                            p / (2 n k_B)).  A 'Te' or 'T' variable in the
                            output takes precedence, or use te_const.
"""
import glob
import math
import os
import re
import struct

import numpy as np

from .constants import K_B, M_P, R_SUN
from .grid import MHDGrid, SphericalGrid

# PLUTO pluto.h physical constants (cgs)
PLUTO_CONST = dict(
    CONST_AH=1.008, CONST_AHe=4.004, CONST_AZ=30.0, CONST_amu=1.66053886e-24,
    CONST_au=1.49597892e13, CONST_c=2.99792458e10, CONST_e=4.80320425e-10,
    CONST_eV=1.602176463158e-12, CONST_G=6.6726e-8, CONST_h=6.62606876e-27,
    CONST_kB=1.3806505e-16, CONST_ly=0.9461e18, CONST_mp=1.67262171e-24,
    CONST_mn=1.67492728e-24, CONST_me=9.1093826e-28, CONST_mH=1.6733e-24,
    CONST_Lsun=3.826e33, CONST_Mearth=5.9736e27, CONST_Msun=2.e33,
    CONST_NA=6.0221367e23, CONST_pc=3.0856775807e18, CONST_PI=np.pi,
    CONST_Rearth=6.378136e8, CONST_Rgas=8.3144598e7, CONST_Rsun=6.96e10,
    CONST_sigma=5.67051e-5, CONST_sigmaT=6.6524e-25,
)
PLUTO_DEFAULT_UNITS = dict(UNIT_DENSITY=PLUTO_CONST["CONST_mp"],
                           UNIT_LENGTH=PLUTO_CONST["CONST_au"],
                           UNIT_VELOCITY=1.0e5)


# ----------------------------------------------------------------------------
def parse_definitions(path):
    """Evaluate UNIT_DENSITY/UNIT_LENGTH/UNIT_VELOCITY (and other numeric
    #defines) from a PLUTO definitions.h."""
    env = dict(PLUTO_CONST)
    env.update(sqrt=np.sqrt, pow=pow, exp=np.exp, log=np.log)
    out = {}
    with open(path) as fh:
        for line in fh:
            m = re.match(r"\s*#define\s+(\w+)\s+(.+?)\s*(/\*.*)?$", line)
            if not m:
                continue
            name, expr = m.group(1), m.group(2)
            try:
                val = eval(expr, {"__builtins__": {}}, env)       # noqa: S307
            except Exception:
                continue
            if isinstance(val, (int, float, np.floating)) and not isinstance(val, bool):
                env[name] = float(val)
                out[name] = float(val)
    return out


def read_grid_out(path):
    """Parse PLUTO grid.out -> (geometry, [(left_edges, right_edges, centres)] x 3)."""
    geom = "CARTESIAN"
    dims = []
    with open(path) as fh:
        lines = fh.readlines()
    k = 0
    while k < len(lines) and lines[k].lstrip().startswith("#"):
        m = re.search(r"GEOMETRY:\s*(\w+)", lines[k])
        if m:
            geom = m.group(1).upper()
        k += 1
    while k < len(lines) and len(dims) < 3:
        tok = lines[k].split()
        k += 1
        if not tok:
            continue
        n = int(tok[0])
        rows = np.array([[float(v) for v in lines[k + i].split()] for i in range(n)])
        k += n
        if rows.shape[1] >= 5:          # PLUTO 3: i xL xc xR dx
            xl, xc, xr = rows[:, 1], rows[:, 2], rows[:, 3]
        else:                            # PLUTO 4: i xL xR
            xl, xr = rows[:, 1], rows[:, 2]
            xc = 0.5 * (xl + xr)
        dims.append((xl, xr, xc))
    while len(dims) < 3:
        dims.append((np.array([0.0]), np.array([1.0]), np.array([0.5])))
    return geom, dims


def _read_out_file(path, nout):
    """Parse dbl.out/flt.out: returns (nout, time, mode, endian, varnames)."""
    rows = [ln.split() for ln in open(path) if ln.strip()]
    if nout is None or nout < 0:
        row = rows[nout if nout is not None else -1]
    else:
        row = next((r for r in rows if int(r[0]) == nout), None)
        if row is None:
            raise ValueError(f"output {nout} not listed in {path}")
    return int(row[0]), float(row[1]), row[4], row[5], row[6:]


def _stag_shape(name, n):
    """Size of staggered fields (bx1s etc.) in PLUTO single-file output."""
    n1, n2, n3 = n
    low = name.lower()
    if low.endswith("1s"):
        return (n1 + 1, n2, n3)
    if low.endswith("2s"):
        return (n1, n2 + 1, n3)
    if low.endswith("3s"):
        return (n1, n2, n3 + 1)
    return n


def read_pluto_binary(directory, nout=-1, fmt=None, variables=("rho", "prs", "Te", "T")):
    """Read selected variables of a PLUTO .dbl/.flt output.

    Returns dict(geometry, dims, time, nout, vars={name: (n1,n2,n3) array})."""
    if fmt is None:
        fmt = "dbl" if os.path.exists(os.path.join(directory, "dbl.out")) else "flt"
    outf = os.path.join(directory, f"{fmt}.out")
    nout, time, mode, endian, names = _read_out_file(outf, nout)
    geom, dims = read_grid_out(os.path.join(directory, "grid.out"))
    n = tuple(len(d[2]) for d in dims)
    dt = np.dtype(("<" if endian.startswith("l") else ">") + ("f8" if fmt == "dbl" else "f4"))
    want = {v.lower() for v in variables}
    out = {}
    if mode == "single_file":
        fn = os.path.join(directory, f"data.{nout:04d}.{fmt}")
        with open(fn, "rb") as fh:
            off = 0
            for name in names:
                shp = _stag_shape(name, n)
                cnt = int(np.prod(shp))
                if name.lower() in want:
                    fh.seek(off)
                    a = np.fromfile(fh, dtype=dt, count=cnt)
                    out[name] = a.reshape(shp[::-1]).transpose(2, 1, 0).astype(float)
                off += cnt * dt.itemsize
    else:
        for name in names:
            if name.lower() not in want:
                continue
            fn = os.path.join(directory, f"{name}.{nout:04d}.{fmt}")
            shp = _stag_shape(name, n)
            a = np.fromfile(fn, dtype=dt, count=int(np.prod(shp)))
            out[name] = a.reshape(shp[::-1]).transpose(2, 1, 0).astype(float)
    return dict(geometry=geom, dims=dims, time=time, nout=nout, vars=out, names=names)


# ----------------------------------------------------------------------------
def read_pluto_vtk(path, variables=("rho", "prs", "Te", "T")):
    """Read a PLUTO legacy-VTK file (binary, big-endian).  Returns the same dict
    as read_pluto_binary; for STRUCTURED_GRID the node coordinates are
    converted back to spherical (r, theta, phi) edges."""
    data = open(path, "rb").read()
    pos = 0

    def line():
        nonlocal pos
        e = data.index(b"\n", pos)
        s = data[pos:e].decode("ascii", "ignore").strip()
        pos = e + 1
        return s

    def arr(count, typ):
        nonlocal pos
        dt = np.dtype(">f4") if typ.startswith("float") else np.dtype(">f8") \
            if typ.startswith("double") else np.dtype(">i4")
        a = np.frombuffer(data, dtype=dt, count=count, offset=pos).astype(float)
        pos += count * dt.itemsize
        return a

    line(); line()
    if line().upper() != "BINARY":
        raise ValueError("only BINARY legacy VTK is supported")
    kind = line().split()[1].upper()
    dims = None
    coords = {}
    pts = None
    want = {v.lower() for v in variables}
    out = {}
    time = 0.0
    b_cartesian = False
    while pos < len(data):
        s = line()
        if not s:
            continue
        t = s.split()
        key = t[0].upper()
        if key == "FIELD":
            for _ in range(int(t[2])):
                h = line().split()
                while not h:
                    h = line().split()
                v = arr(int(h[1]) * int(h[2]), h[3].lower())
                if h[0].upper() == "TIME":
                    time = float(v[0])
        elif key == "DIMENSIONS":
            dims = tuple(int(v) for v in t[1:4])
        elif key in ("X_COORDINATES", "Y_COORDINATES", "Z_COORDINATES"):
            coords[key[0]] = arr(int(t[1]), t[2].lower())
        elif key == "POINTS":
            pts = arr(3 * int(t[1]), t[2].lower()).reshape(-1, 3)
        elif key in ("CELL_DATA", "POINT_DATA"):
            pass
        elif key == "SCALARS":
            name, typ = t[1], t[2].lower()
            line()                                  # LOOKUP_TABLE default
            nc = [max(d - 1, 1) for d in dims]
            a = arr(int(np.prod(nc)), typ)
            if name.lower() in want:
                out[name] = a.reshape(nc[::-1]).transpose(2, 1, 0)
        elif key == "VECTORS":
            nc = [max(d - 1, 1) for d in dims]
            a = arr(3 * int(np.prod(nc)), t[2].lower())
            # VTK_VECTOR_DUMP: PLUTO writes vectors in Cartesian components
            if "bx1" in want and t[1].lower().startswith(("magnetic", "bfield", "b_field")):
                a = a.reshape(nc[::-1] + [3])
                for c in range(3):
                    out[f"bx{c + 1}"] = a[..., c].transpose(2, 1, 0)
                b_cartesian = True
        else:
            continue
    if kind == "RECTILINEAR_GRID":
        geom = "CARTESIAN"
        dd = []
        for c in "XYZ":
            e = coords[c]
            if e.size == 1:
                e = np.array([e[0] - 0.5, e[0] + 0.5])
            dd.append((e[:-1], e[1:], 0.5 * (e[:-1] + e[1:])))
    elif kind == "STRUCTURED_GRID":
        geom = "SPHERICAL"
        P = pts.reshape(dims[2], dims[1], dims[0], 3).transpose(2, 1, 0, 3)
        r = np.linalg.norm(P, axis=-1)
        th = np.arccos(np.clip(P[..., 2] / np.maximum(r, 1e-300), -1, 1))
        ph = np.mod(np.arctan2(P[..., 1], P[..., 0]), 2 * np.pi)
        re_ = r[:, 0, 0]
        te_ = th[0, :, 0] if dims[1] > 1 else np.array([0.0, np.pi])
        pe_ = np.unwrap(ph[0, dims[1] // 2, :]) if dims[2] > 1 else np.array([0.0, 2 * np.pi])
        dd = [(e[:-1], e[1:], 0.5 * (e[:-1] + e[1:])) for e in (re_, te_, pe_)]
    else:
        raise ValueError(f"unsupported VTK dataset {kind}")
    return dict(geometry=geom, dims=dd, time=time, nout=None, vars=out, names=list(out),
                b_cartesian=b_cartesian)


# ----------------------------------------------------------------------------
def _import_pypluto(pypluto_path=None):
    """Import pyPLUTO: from ``pypluto_path`` if given, else an installed pyPLUTO
    (classic ``pload`` version shipped with PLUTO, or the newer PyPLUTO package
    with ``Load``), else the copy bundled in radiosynth/common/contrib/pyPLUTO.py."""
    import importlib
    import sys
    if pypluto_path:
        p = os.path.abspath(pypluto_path)
        if os.path.isfile(p):
            p = os.path.dirname(p)
        if p not in sys.path:
            sys.path.insert(0, p)
        return importlib.import_module("pyPLUTO")
    try:                                    # a pyPLUTO installed by the user
        return importlib.import_module("pyPLUTO")
    except ImportError:                     # the copy bundled with radiosynth
        return importlib.import_module("radiosynth.common.contrib.pyPLUTO")


def _to_n123(a, n):
    """Return array a as (n1, n2, n3) whatever the reader's axis order/dimension."""
    a = np.asarray(a, dtype=float)
    if a.shape == tuple(n):
        return a
    if a.ndim == 3 and a.shape == tuple(n[::-1]):
        return a.transpose(2, 1, 0)
    sq = tuple(k for k in n if k > 1)
    if a.shape == sq:
        return a.reshape(n)
    if a.shape == sq[::-1]:
        return a.T.reshape(n)
    raise ValueError(f"cannot match array shape {a.shape} to grid {n}")


def read_pluto_pypluto(directory, nout=-1, fmt=None, pypluto_path=None,
                       variables=("rho", "prs", "Te", "T")):
    """Read PLUTO output through pyPLUTO; returns the same dict as
    read_pluto_binary.  Works with the classic ``pyPLUTO.pload(ns, w_dir=...,
    datatype=...)`` and with the newer ``pyPLUTO.Load(nout, path=...)``."""
    pp = _import_pypluto(pypluto_path)
    d = directory if directory.endswith(os.sep) else directory + os.sep
    if fmt is None:
        fmt = "dbl" if os.path.exists(os.path.join(d, "dbl.out")) else "flt"
    if nout is None or nout < 0:
        nout = _read_out_file(os.path.join(d, f"{fmt}.out"), nout)[0]
    geom, _ = read_grid_out(os.path.join(d, "grid.out"))
    if hasattr(pp, "pload"):                                   # classic pyPLUTO
        D = pp.pload(nout, w_dir=d, datatype="float" if fmt == "flt" else "double")
        names = list(getattr(D, "vars", []))
        time = float(getattr(D, "time", getattr(D, "SimTime", 0.0)))
    elif hasattr(pp, "Load"):                                  # PyPLUTO >= 1.0
        D = pp.Load(nout, path=d, datatype=fmt)
        names = list(getattr(D, "varsfile", getattr(D, "vars", [])) or [])
        geom = str(getattr(D, "geom", geom)).upper()
        time = float(np.atleast_1d(getattr(D, "timelist", [0.0]))[0])
    else:
        raise ImportError("the imported pyPLUTO has neither pload nor Load")
    x1r, x2r, x3r = (np.atleast_1d(np.asarray(getattr(D, f"x{i}r"), float)) for i in (1, 2, 3))
    if x3r.size < 2:
        x3r = np.array([0.0, 2 * np.pi]) if geom == "SPHERICAL" else np.array([-0.5, 0.5])
    if x2r.size < 2:
        x2r = np.array([0.0, np.pi]) if geom == "SPHERICAL" else np.array([-0.5, 0.5])
    dims = [(e[:-1], e[1:], 0.5 * (e[:-1] + e[1:])) for e in (x1r, x2r, x3r)]
    n = tuple(len(dd[2]) for dd in dims)
    want = {v.lower() for v in variables}
    out = {}
    for v in want:
        for cand in (v, v.capitalize(), v.upper()):
            if hasattr(D, cand):
                out[cand] = _to_n123(getattr(D, cand), n)
                break
    return dict(geometry=geom, dims=dims, time=time, nout=nout, vars=out,
                names=names or list(out))


# ----------------------------------------------------------------------------
def _find_var(vars_, *names):
    low = {k.lower(): k for k in vars_}
    for n in names:
        if n.lower() in low:
            return vars_[low[n.lower()]]
    return None


def load_pluto(path, nout=-1, fmt=None, definitions=None, unit_density=None,
               unit_length=None, unit_velocity=None, mu_e=1.0, mean_mol_weight=0.5,
               te_const=None, r_body=None, dtype=np.float32, verbose=True,
               reader="builtin", pypluto_path=None, r_star=None, with_b=False):
    """Load PLUTO output and return an MHDGrid (Cartesian) or SphericalGrid.

    path        : output directory (with grid.out, dbl.out/flt.out) or a .vtk file
    nout        : output number (default: last one listed in dbl.out/flt.out)
    fmt         : 'dbl' or 'flt' (default: whichever .out file exists)
    definitions : path to definitions.h (default: look in the directory and its
                  parent); explicit unit_* arguments override it
    mu_e        : mass per electron in m_p:  n_e = rho / (mu_e m_p)
    mean_mol_weight : mu in T = p mu m_p / (rho k_B)
    te_const    : use a constant T_e [K] (isothermal / no pressure output)
    r_body      : inner boundary in R_sun (default: inner radial edge)
    reader      : 'builtin' (radiosynth's own .dbl/.flt reader) or 'pypluto'
                  (use the pyPLUTO module shipped with PLUTO; falls back to the
                  built-in reader if pyPLUTO fails)
    pypluto_path: directory containing pyPLUTO.py (if not on sys.path)
    r_star      : stellar radius in R_sun.  If given, coordinates are expressed in
                  stellar radii (x * UNIT_LENGTH / R_star) and the grid's
                  length_unit is R_star; default: coordinates in R_sun
    with_b      : also read the cell-centred magnetic field bx1, bx2, bx3 and
                  attach it in gauss, B = b sqrt(4 pi UNIT_DENSITY) UNIT_VELOCITY
                  (needed for gyroresonance)
    """
    want = ("rho", "prs", "Te", "T") + (("bx1", "bx2", "bx3") if with_b else ())
    if os.path.isdir(path):
        d = None
        if reader == "pypluto":
            try:
                d = read_pluto_pypluto(path, nout, fmt, pypluto_path, variables=want)
                if verbose:
                    print("radiosynth: PLUTO data read with pyPLUTO")
            except Exception as exc:                      # noqa: BLE001
                print(f"radiosynth: pyPLUTO failed ({type(exc).__name__}: {exc}); "
                      f"using the built-in reader")
        if d is None:
            d = read_pluto_binary(path, nout, fmt, variables=want)
        base = path
    else:
        d = read_pluto_vtk(path, variables=want)
        base = os.path.dirname(os.path.abspath(path))

    units = dict(PLUTO_DEFAULT_UNITS)
    src = "PLUTO defaults"
    if definitions is None:
        for cand in (os.path.join(base, "definitions.h"),
                     os.path.join(base, "..", "definitions.h")):
            if os.path.exists(cand):
                definitions = cand
                break
    if definitions is not None:
        defs = parse_definitions(definitions)
        for k in units:
            if k in defs:
                units[k] = defs[k]
        src = os.path.relpath(definitions)
    for k, v in (("UNIT_DENSITY", unit_density), ("UNIT_LENGTH", unit_length),
                 ("UNIT_VELOCITY", unit_velocity)):
        if v is not None:
            units[k] = float(v)
            src = "user arguments"
    UD, UL, UV = units["UNIT_DENSITY"], units["UNIT_LENGTH"], units["UNIT_VELOCITY"]

    v = d["vars"]
    rho = _find_var(v, "rho")
    if rho is None:
        raise KeyError(f"no 'rho' in PLUTO output (variables: {d['names']})")
    rho_cgs = rho * UD
    ne = rho_cgs * 1e3 / (mu_e * M_P)                        # m^-3
    T = _find_var(v, "Te", "T")
    if te_const is not None:
        te = np.full_like(ne, float(te_const))
        tsrc = f"constant {te_const} K"
    elif T is not None:
        te = T
        tsrc = "T/Te variable [K]"
    else:
        prs = _find_var(v, "prs")
        if prs is None:
            raise KeyError("no 'prs' in output: pass te_const= for isothermal runs")
        p_cgs = prs * UD * UV**2
        te = p_cgs * mean_mol_weight * M_P * 1e3 / (np.maximum(rho_cgs, 1e-300) * K_B * 1e7)
        tsrc = f"p mu m_p/(rho k_B), mu={mean_mol_weight}"

    Lu = (r_star or 1.0) * R_SUN                              # metres per output unit
    L = UL / (Lu * 100.0)                                     # code length -> R_sun or R_star
    (x1l, x1r, x1), (x2l, x2r, x2), (x3l, x3r, x3) = d["dims"]
    geom = d["geometry"]
    if geom == "SPHERICAL":
        r, r_e = x1 * L, np.concatenate([x1l, x1r[-1:]]) * L
        nph = ne.shape[2]
        one_d = ne.shape[1] == 1 and nph == 1
        if ne.shape[1] == 1 and not one_d:
            raise ValueError("spherical output with one theta cell but several phi cells "
                             "is not supported")
        phi = x3 if nph > 1 else None
        if one_d:                                  # 1D radial model: spherical symmetry
            from .symmetric import grid_from_profile
            g = grid_from_profile(r, ne[:, 0, 0], te[:, 0, 0], r_edges=r_e, r_body=r_body,
                                  length_unit=Lu, dtype=dtype)
            if verbose:
                print("radiosynth: 1D spherical PLUTO output - assuming spherical symmetry")
        else:
            g = SphericalGrid(r, x2, phi, ne, te, r_edges=r_e,
                              theta_edges=np.concatenate([x2l, x2r[-1:]]),
                              phi_edges=np.concatenate([x3l, x3r[-1:]]) if nph > 1 else None,
                              r_body=r_body, dtype=dtype, length_unit=Lu)
    elif geom == "CARTESIAN":
        if min(ne.shape) < 2:
            raise ValueError("Cartesian PLUTO output must be 3D for imaging")
        g = MHDGrid(x1 * L, x2 * L, x3 * L, ne, te,
                    r_body=1.0 if r_body is None else r_body, dtype=dtype, length_unit=Lu)
    else:
        raise ValueError(f"PLUTO geometry {geom} not supported (use CARTESIAN or SPHERICAL)")

    if with_b:
        b = [_find_var(v, f"bx{i}") for i in (1, 2, 3)]
        if any(c is None for c in b):
            raise KeyError(f"with_b=True but bx1/bx2/bx3 not found (variables: {d['names']})")
        bfac = math.sqrt(4.0 * math.pi * UD) * UV                 # code units -> gauss
        b = [c * bfac for c in b]
        if geom == "SPHERICAL":
            g.add_field(*b, components="cartesian" if d.get("b_cartesian") else "spherical")
        else:
            g.add_field(*b)

    if verbose:
        if with_b:
            bmag = np.sqrt(sum(c**2 for c in b))
            print(f"radiosynth: magnetic field bx1..bx3 x {bfac:.4e} G, |B| max {bmag.max():.3e} G")
        print(f"radiosynth: PLUTO {geom.lower()} output, t={d['time']:g} (code units), "
              f"variables {d['names']}")
        print(f"            units from {src}: UNIT_DENSITY={UD:.4e} g/cm^3, "
              f"UNIT_LENGTH={UL:.4e} cm (= {L:.4g} {'R_star' if r_star else 'R_sun'}), "
              f"UNIT_VELOCITY={UV:.4e} cm/s")
        print(f"            n_e = rho/({mu_e} m_p); T_e from {tsrc}")
        print(f"            {g}")
        if geom == "SPHERICAL" and not (0.5 < g.r_lo < 3.0):
            print(f"WARNING: inner radius is {g.r_lo:.3g} {'R_star' if r_star else 'R_sun'} - "
                  f"check UNIT_LENGTH (PLUTO's default is 1 AU; pass definitions= or "
                  f"unit_length=), or r_star= for a star")
        rr = g.r if geom == "SPHERICAL" else None
        if rr is not None:
            i = np.searchsorted(rr, 1.1 * max(g.r_lo, 1.0))
            i = min(i, rr.size - 1)
            print(f"            at r={rr[i]:.3g} R_sun: n_e = {np.median(ne[i]):.3e} m^-3, "
                  f"T_e = {np.median(te[i]):.3e} K (median over angles)")
    g.info = dict(units=units, time=d["time"], nout=d["nout"], te=tsrc)

    return g
