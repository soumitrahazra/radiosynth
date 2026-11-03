# radiosynth - From MHD to MHz: synthetic radio images of MHD models of the Sun and stars
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
"""Readers that turn MHD model output into an MHDGrid.

Supported inputs
----------------
* numpy arrays                         -> ``grid_from_arrays``
* Tecplot ASCII (.dat/.tec), ordered (I,J,K) zones or FE point zones
                                       -> ``read_tecplot_ascii``
* Tecplot binary (.plt/.szplt) via pytecplot (needs Tecplot 360)
                                       -> ``read_pytecplot``
* BATS-R-US / SWMF IDL plot files (.out, ascii / real4 / real8)
                                       -> ``read_batsrus_idl``
* ``load_mhd(path)`` dispatches on the extension.

All readers return a dict ``{name: array}`` of point data, which
``build_grid`` converts to an MHDGrid (computing n_e [m^-3] and T_e [K]).
If the points are not on a tensor-product grid (e.g. BATS-R-US
unstructured/AMR '3d' output) they are resampled to a uniform Cartesian box.
"""
import os
import re
import struct
import warnings

import numpy as np

from .constants import K_B, M_P, R_SUN
from .grid import MHDGrid


# ============================================================================
# Variable identification and unit handling
# ============================================================================
def _base(name):
    """'Rho [g/cm^3]' -> 'rho'"""
    return re.split(r"[\s\[(]", name.strip().strip('"'), maxsplit=1)[0].lower()


def _unit(name):
    m = re.search(r"\[([^\]]*)\]", name)
    return m.group(1).strip().lower().replace(" ", "") if m else None


def _find(vars_, candidates):
    for c in candidates:
        for k in vars_:
            if _base(k) == c:
                return k
    return None


_MASS_UNITS = {"g/cm^3": 1e3, "g/cm3": 1e3, "g/cc": 1e3, "kg/m^3": 1.0, "kg/m3": 1.0}
_NUM_UNITS = {"cm^-3": 1e6, "/cm^3": 1e6, "1/cm^3": 1e6, "/cc": 1e6, "n/cc": 1e6,
              "mp/cc": 1e6, "amu/cc": 1e6, "amu/cm^3": 1e6, "amu/cm3": 1e6, "cm-3": 1e6,
              "m^-3": 1.0, "/m^3": 1.0, "1/m^3": 1.0, "m-3": 1.0, "/m3": 1.0}
_PRES_UNITS = {"dyne/cm^2": 0.1, "dyn/cm^2": 0.1, "dyne/cm2": 0.1, "dyn/cm2": 0.1,
               "pa": 1.0, "npa": 1e-9, "j/m^3": 1.0}


_B_UNITS = {"gauss": 1.0, "g": 1.0, "nt": 1e-5, "t": 1e4, "tesla": 1e4, "mt": 10.0}


def find_magnetic_field(vars_, b_unit=None, verbose=True):
    """Return (bx, by, bz) in gauss from BATS-R-US-style variables Bx/By/Bz
    (also B_x, b1..b3).  Units from the variable name ([Gauss], [nT], [T]) or
    ``b_unit`` (number = gauss per output unit, or one of 'G', 'nT', 'T');
    if neither is known gauss is assumed with a warning."""
    names = None
    for cand in (("bx", "by", "bz"), ("b_x", "b_y", "b_z"), ("b1", "b2", "b3")):
        ks = [_find(vars_, [c]) for c in cand]
        if all(k is not None for k in ks):
            names = ks
            break
    if names is None:
        raise KeyError("no magnetic field (Bx, By, Bz) in the MHD output; "
                       f"variables: {list(vars_)}")
    if b_unit is not None:
        fac = float(b_unit) if not isinstance(b_unit, str) else _B_UNITS[b_unit.lower()]
        src = f"b_unit={b_unit}"
    else:
        u = _unit(names[0])
        if u in _B_UNITS:
            fac, src = _B_UNITS[u], f"[{u}] in the variable name"
        else:
            fac, src = 1.0, "assumed gauss (no unit in the file; pass b_unit=)"
            warnings.warn("magnetic field unit unknown: assuming gauss (pass b_unit=)")
    if verbose:
        print(f"radiosynth: magnetic field from {names}, {src}")
    return tuple(np.asarray(vars_[k], float) * fac for k in names)


def _density_to_ne(vals, unit, r=None, kind=None, mu=1.0):
    """Convert a density variable to electron number density [m^-3].

    kind: 'mass' or 'number' (None -> from the unit string, else guessed from
    the magnitude near r ~ 1.1 R_sun).  For mass density n_e = rho/(mu m_p)
    (BATS-R-US uses n_e = rho/m_p for a hydrogen plasma, mu=1)."""
    u = unit
    if u in _MASS_UNITS:
        return vals * _MASS_UNITS[u] / (mu * M_P), f"mass density [{u}]"
    if u in _NUM_UNITS:
        return vals * _NUM_UNITS[u], f"number density [{u}]"
    # guess from magnitude (typical low corona n_e ~ 1e8-1e9 cm^-3)
    if r is not None:
        sel = (r > 1.05) & (r < 1.3)
        ref = np.nanmedian(np.abs(vals[sel])) if sel.any() else np.nanmedian(np.abs(vals))
    else:
        ref = np.nanmedian(np.abs(vals))
    guesses = [(1e3 / (mu * M_P), "mass density [g/cm^3] (guessed)"),
               (1.0 / (mu * M_P), "mass density [kg/m^3] (guessed)"),
               (1e6, "number density [cm^-3] (guessed)"),
               (1.0, "number density [m^-3] (guessed)")]
    if kind == "mass":
        guesses = guesses[:2]
    elif kind == "number":
        guesses = guesses[2:]
    best = min(guesses, key=lambda g: abs(np.log10(max(ref * g[0], 1e-300)) - 14.5))
    return vals * best[0], best[1]


def _pressure_si(vals, unit):
    if unit in _PRES_UNITS:
        return vals * _PRES_UNITS[unit]
    # BATS-R-US dimensional output default: dyne/cm^2
    return vals * 0.1


def derive_ne_te(vars_, density_var=None, density_unit=None, te_var=None,
                 pe_var=None, p_var=None, mu=1.0, te_const=None, verbose=True):
    """Return (ne [m^-3], te [K], info) from a dict of point variables.

    Priority for T_e:  te_var / 'te'  >  Pe/(n_e k)  >  P/(2 n_e k) (Te=Ti)
                       >  'T'/'temp'  >  te_const.
    """
    x, y, z = vars_["x"], vars_["y"], vars_["z"]
    r = np.sqrt(x**2 + y**2 + z**2)
    info = {}
    dk = density_var or _find(vars_, ["ne", "n_e", "rho", "n", "dens", "density", "nrho"])
    if dk is None:
        raise KeyError(f"no density variable found among {list(vars_)}; use density_var=")
    unit = density_unit.lower().replace(" ", "") if density_unit else _unit(dk)
    if unit is None and "_units" in vars_:
        unit = vars_["_units"].get(dk)
    kind = None
    if density_unit is None and unit is None:
        kind = "number" if _base(dk) in ("ne", "n_e", "n") else ("mass" if _base(dk) == "rho" else None)
    ne, how = _density_to_ne(np.asarray(vars_[dk], float), unit, r, kind, mu)
    info["density"] = f"{dk}: {how}"

    units = vars_.get("_units", {})
    te = None
    tk = te_var or _find(vars_, ["te", "t_e"])
    if tk is not None:
        te = np.asarray(vars_[tk], float)
        info["te"] = f"{tk} [K]"
    if te is None:
        pk = pe_var or _find(vars_, ["pe", "p_e"])
        if pk is not None:
            pe = _pressure_si(np.asarray(vars_[pk], float), _unit(pk) or units.get(pk))
            te = pe / (np.maximum(ne, 1.0) * K_B)
            info["te"] = f"{pk}/(n_e k_B)"
    if te is None:
        pk = p_var or _find(vars_, ["p", "pres", "pressure"])
        if pk is not None:
            p = _pressure_si(np.asarray(vars_[pk], float), _unit(pk) or units.get(pk))
            te = p / (2.0 * np.maximum(ne, 1.0) * K_B)
            info["te"] = f"{pk}/(2 n_e k_B)  (assumes T_e = T_i, no Pe in file)"
    if te is None:
        tk = _find(vars_, ["t", "temp", "temperature", "ti"])
        if tk is not None:
            te = np.asarray(vars_[tk], float)
            info["te"] = f"{tk} [K]"
    if te is None:
        if te_const is None:
            raise KeyError("no temperature/pressure variable found; pass te_const= or te_var=")
        te = np.full_like(ne, float(te_const))
        info["te"] = f"constant {te_const} K"
    if verbose:
        sel = (r > 1.05) & (r < 1.15)
        msg = ""
        if sel.any():
            msg = (f" (median at r=1.05-1.15: n_e={np.nanmedian(ne[sel]):.3e} m^-3, "
                   f"T_e={np.nanmedian(te[sel]):.3e} K)")
        print(f"radiosynth: n_e from {info['density']}; T_e from {info['te']}{msg}")
    return ne, te, info


# ============================================================================
# Point cloud -> rectilinear grid
# ============================================================================
def _unique_axis(v, rtol=1e-6):
    v = np.asarray(v, float)
    scale = max(np.ptp(v), 1e-30)
    q = np.round(v / (scale * rtol)).astype(np.int64)
    uq, inv = np.unique(q, return_inverse=True)
    # representative coordinate per unique value
    ax = np.zeros(uq.size)
    np.add.at(ax, inv, v)
    ax /= np.bincount(inv)
    return ax, inv


def points_to_grid(vars_, names, resample_shape=None, resample_box=None,
                   method="linear", verbose=True):
    """Arrange point data on a tensor-product grid.

    If the points already form a full tensor grid, they are simply reshaped
    (any ordering).  Otherwise they are resampled to a uniform box of shape
    ``resample_shape`` (default 160^3) covering ``resample_box``
    ((xmin,xmax),(ymin,ymax),(zmin,zmax)); ``method`` is 'linear'
    (Delaunay, accurate, slow for >1e6 points) or 'nearest' (fast).
    Returns (x, y, z, {name: 3D array}).
    """
    x, y, z = (np.asarray(vars_[k], float) for k in ("x", "y", "z"))
    ax_x, ix = _unique_axis(x)
    ax_y, iy = _unique_axis(y)
    ax_z, iz = _unique_axis(z)
    n = x.size
    if ax_x.size * ax_y.size * ax_z.size == n and resample_shape is None:
        flat = (ix * ax_y.size + iy) * ax_z.size + iz
        if np.unique(flat).size == n:
            out = {}
            for k in names:
                a = np.empty(n)
                a[flat] = np.asarray(vars_[k], float)
                out[k] = a.reshape(ax_x.size, ax_y.size, ax_z.size)
            return ax_x, ax_y, ax_z, out

    # ---- resample scattered data
    shape = resample_shape or (160, 160, 160)
    if np.isscalar(shape):
        shape = (int(shape),) * 3
    if resample_box is None:
        resample_box = ((x.min(), x.max()), (y.min(), y.max()), (z.min(), z.max()))
    gx = np.linspace(*resample_box[0], shape[0])
    gy = np.linspace(*resample_box[1], shape[1])
    gz = np.linspace(*resample_box[2], shape[2])
    if verbose:
        print(f"radiosynth: {n} scattered points -> resampling ({method}) to {shape} "
              f"uniform grid; this can take a while")
    from scipy.spatial import cKDTree
    pts = np.column_stack([x, y, z])
    G = np.stack(np.meshgrid(gx, gy, gz, indexing="ij"), axis=-1).reshape(-1, 3)
    # density spans orders of magnitude: interpolate its logarithm
    vals = []
    for k in names:
        v = np.asarray(vars_[k], float)
        vals.append(np.log(np.maximum(v, 1e-300)) if k == "ne" else v)
    vals = np.column_stack(vals)
    if method == "nearest":
        _, idx = cKDTree(pts).query(G, k=1, workers=-1)
        res = vals[idx]
    else:
        from scipy.interpolate import LinearNDInterpolator
        res = LinearNDInterpolator(pts, vals)(G)
        bad = ~np.isfinite(res).all(axis=1)
        if bad.any():
            _, idx = cKDTree(pts).query(G[bad], k=1, workers=-1)
            res[bad] = vals[idx]
    out = {}
    for c, k in enumerate(names):
        a = res[:, c]
        out[k] = (np.exp(a) if k == "ne" else a).reshape(shape)
    return gx, gy, gz, out


def build_grid(vars_, r_body=1.0, resample_shape=None, resample_box=None,
               resample_method="linear", dtype=np.float32, verbose=True, r_star=None,
               with_b=False, b_unit=None, **derive_kw):
    """dict of point variables -> MHDGrid (see derive_ne_te for keywords).

    r_star : stellar radius in R_sun when the coordinates are in stellar radii
             (BATS-R-US stellar runs, #STAR); default 1 (the Sun).
    with_b : also attach the magnetic field Bx, By, Bz (needed for
             gyroresonance); b_unit overrides the unit (see find_magnetic_field)."""
    ne, te, info = derive_ne_te(vars_, verbose=verbose, **derive_kw)
    pv = {"x": vars_["x"], "y": vars_["y"], "z": vars_["z"], "ne": ne, "te": te}
    names = ["ne", "te"]
    if with_b:
        pv["bx"], pv["by"], pv["bz"] = find_magnetic_field(vars_, b_unit, verbose)
        names += ["bx", "by", "bz"]
    x, y, z, f = points_to_grid(pv, names, resample_shape, resample_box,
                                resample_method, verbose)
    g = MHDGrid(x, y, z, f["ne"], f["te"], r_body=r_body, dtype=dtype,
                length_unit=(r_star or 1.0) * R_SUN)
    if with_b:
        g.add_field(f["bx"], f["by"], f["bz"])
    g.info = info
    return g


def grid_from_arrays(x, y, z, ne, te=None, r_body=1.0, **kw):
    """Convenience wrapper: 1D axes [R_sun] + 3D n_e [m^-3], T_e [K]."""
    return MHDGrid(x, y, z, ne, te, r_body=r_body, **kw)


# ============================================================================
# Tecplot ASCII
# ============================================================================
def _standardise_xyz(vars_):
    out = dict(vars_)
    for c in ("x", "y", "z"):
        if c not in out:
            k = _find(vars_, [c])
            if k is None:
                raise KeyError(f"coordinate '{c}' not found in {list(vars_)}")
            out[c] = out[k]
    return out


def read_tecplot_ascii(path, zone=0):
    """Read an ASCII Tecplot file (POINT or BLOCK packing, ordered or FE zone).

    Only nodal variables of the requested zone are returned; FE connectivity
    is ignored (the points are re-gridded by points_to_grid)."""
    with open(path, "r") as fh:
        text = fh.read()
    # header -----------------------------------------------------------------
    mvar = re.search(r"VARIABLES\s*=\s*(.*?)(?=\n\s*(ZONE|DATASETAUXDATA|AUXDATA)\b)",
                     text, re.S | re.I)
    if not mvar:
        raise ValueError("VARIABLES line not found")
    raw = mvar.group(1)
    names = re.findall(r'"([^"]*)"', raw) or re.split(r"[,\s]+", raw.strip())
    names = [n.strip() for n in names if n.strip()]
    zones = [m for m in re.finditer(r"^\s*ZONE\b", text, re.M | re.I)]
    if not zones:
        raise ValueError("no ZONE found")
    zstart = zones[zone].start()
    zend = zones[zone + 1].start() if zone + 1 < len(zones) else len(text)
    ztext = text[zstart:zend]
    # zone header spans until the first line that starts with a number
    lines = ztext.split("\n")
    hdr = []
    i = 0
    num_re = re.compile(r"^\s*[-+]?(\d|\.\d)")
    while i < len(lines) and not num_re.match(lines[i]):
        hdr.append(lines[i])
        i += 1
    header = " ".join(hdr)

    def geti(key):
        m = re.search(r"\b" + key + r"\s*=\s*(\d+)", header, re.I)
        return int(m.group(1)) if m else None

    I, J, K = geti("I"), geti("J"), geti("K")
    N = geti("N") or geti("NODES")
    E = geti("E") or geti("ELEMENTS")
    block = bool(re.search(r"(DATAPACKING\s*=\s*BLOCK|F\s*=\s*BLOCK|F\s*=\s*FEBLOCK)", header, re.I))
    npts = N if N is not None else (I or 1) * (J or 1) * (K or 1)
    nv = len(names)
    body = "\n".join(lines[i:])
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        try:
            arr = np.fromstring(body.replace("D", "E").replace("d", "e"), sep=" ")
        except Exception:
            arr = np.array(body.split(), dtype=float)
    data = arr[: npts * nv]
    if data.size < npts * nv:
        raise ValueError(f"expected {npts * nv} values, found {data.size}")
    data = data.reshape(nv, npts) if block else data.reshape(npts, nv).T
    out = {n: data[c] for c, n in enumerate(names)}
    _ = E
    return _standardise_xyz(out)


# ============================================================================
# Tecplot binary through pytecplot
# ============================================================================
def read_pytecplot(path, zone=0):
    """Read .plt/.szplt with pytecplot (requires a Tecplot 360 installation)."""
    import tecplot as tp
    ds = tp.data.load_tecplot(path)
    z = ds.zone(zone)
    out = {v.name: z.values(v.index).as_numpy_array().astype(float) for v in ds.variables()}
    return _standardise_xyz(out)


# ============================================================================
# BATS-R-US / SWMF IDL format
# ============================================================================
def _read_records(fh):
    """Iterate Fortran unformatted sequential records, auto-detecting endianness."""
    data = fh.read()
    pos = 0
    endian = None
    recs = []
    while pos < len(data):
        if endian is None:
            for e in ("<", ">"):
                (L,) = struct.unpack(e + "i", data[pos:pos + 4])
                if 0 < L < len(data) and data[pos + 4 + L:pos + 8 + L] == data[pos:pos + 4]:
                    endian = e
                    break
            if endian is None:
                raise ValueError("not a Fortran unformatted file")
        (L,) = struct.unpack(endian + "i", data[pos:pos + 4])
        recs.append(data[pos + 4:pos + 4 + L])
        pos += 8 + L
    return recs, endian


def read_batsrus_idl(path):
    """Read a BATS-R-US IDL plot file (.out): ascii, real4 or real8.

    Returns a dict of point variables (coordinates as 'x','y','z') plus
    '_units' when the header line lists units.  Works for regular Cartesian
    (cut/box with dx>0) and unstructured (dx<=0, AMR) 3D output."""
    with open(path, "rb") as fh:
        first = fh.read(4)
    binary = False
    if len(first) == 4:
        for e in ("<", ">"):
            if 0 < struct.unpack(e + "i", first)[0] <= 4096:
                binary = True
    if binary:
        with open(path, "rb") as fh:
            recs, e = _read_records(fh)
        head = recs[0].decode("ascii", "ignore").strip()
        r2 = recs[1]
        rsize = len(r2) - 16
        rfmt = "f" if rsize == 4 else "d"
        nstep, = struct.unpack(e + "i", r2[0:4])
        time_, = struct.unpack(e + rfmt, r2[4:4 + rsize])
        ndim, npar, nvar = struct.unpack(e + "3i", r2[4 + rsize:16 + rsize])
        gencoord = ndim < 0
        ndim = abs(ndim)
        n_d = struct.unpack(e + f"{ndim}i", recs[2][:4 * ndim])
        ir = 3
        params = []
        if npar > 0:
            params = list(struct.unpack(e + f"{npar}{rfmt}", recs[ir][:npar * rsize]))
            ir += 1
        names = recs[ir].decode("ascii", "ignore").split()
        ir += 1
        npts = int(np.prod(n_d))
        dt = np.dtype(e + rfmt)
        coord = np.frombuffer(recs[ir], dtype=dt, count=npts * ndim).astype(float)
        coord = coord.reshape((npts, ndim), order="F")
        ir += 1
        vals = []
        for _ in range(nvar):
            vals.append(np.frombuffer(recs[ir], dtype=dt, count=npts).astype(float))
            ir += 1
    else:
        with open(path, "r") as fh:
            head = fh.readline().strip()
            tok = fh.readline().split()
            nstep, time_ = int(tok[0]), float(tok[1])
            ndim, npar, nvar = int(tok[2]), int(tok[3]), int(tok[4])
            gencoord = ndim < 0
            ndim = abs(ndim)
            n_d = [int(v) for v in fh.readline().split()[:ndim]]
            params = [float(v) for v in fh.readline().split()] if npar > 0 else []
            names = fh.readline().split()
            npts = int(np.prod(n_d))
            arr = np.loadtxt(fh, max_rows=npts)
        coord = arr[:, :ndim]
        vals = [arr[:, ndim + c] for c in range(nvar)]

    cnames = names[:ndim]
    vnames = names[ndim:ndim + nvar]
    out = {}
    for c, cn in enumerate(cnames):
        out[cn] = coord[:, c]
    for c, vn in enumerate(vnames):
        out[vn] = vals[c]
    # units from the header line, if present  ("R R R g/cm3 km/s ...")
    toks = head.split()
    units = {}
    if len(toks) >= ndim + nvar:
        cand = toks[-(ndim + nvar + len(params)):] if len(toks) >= ndim + nvar + len(params) else toks
        cand = cand[: ndim + nvar]
        for n, u in zip(cnames + vnames, cand):
            units[n] = u.lower().replace("cm3", "cm^3").replace("cm2", "cm^2")
    out["_units"] = units
    out["_meta"] = dict(header=head, nstep=nstep, time=time_, ndim=ndim,
                        gencoord=gencoord, n_d=tuple(n_d), params=params,
                        param_names=names[ndim + nvar:])
    if ndim != 3:
        raise ValueError(f"need a 3D file, got ndim={ndim}")
    # coordinate names in BATS-R-US are x y z (or r lon lat for spherical)
    if all(_base(c) in ("x", "y", "z") for c in cnames):
        out["x"], out["y"], out["z"] = (out[k] for k in cnames)
    else:
        raise ValueError(f"non-Cartesian coordinates {cnames}: convert to x,y,z first")
    return out


# ============================================================================
def load_mhd(path, reader=None, **build_kw):
    """Read an MHD output file and return an MHDGrid.

    reader: 'tecplot', 'pytecplot', 'idl', 'pluto', 'npz' or None (inferred:
            a directory containing grid.out or a .vtk file -> PLUTO).
    PLUTO keywords (nout, fmt, definitions, unit_density, unit_length,
    unit_velocity, mu_e, mean_mol_weight, te_const, r_body, with_b) go to load_pluto.
    with_b=True also reads the magnetic field (Bx, By, Bz; PLUTO bx1..bx3), in
    gauss, for gyroresonance; b_unit sets its unit if the file does not.
    build_kw are passed to build_grid (density_var, density_unit, te_var,
    pe_var, p_var, te_const, mu, r_body, resample_shape, resample_box,
    resample_method, dtype)."""
    ext = os.path.splitext(path)[1].lower()
    if reader is None and (ext == ".vtk" or (os.path.isdir(path) and
                           os.path.exists(os.path.join(path, "grid.out")))):
        reader = "pluto"
    if reader == "pluto":
        from .pluto import load_pluto
        keep = ("nout", "fmt", "definitions", "unit_density", "unit_length",
                "unit_velocity", "mu_e", "mean_mol_weight", "te_const", "r_body", "dtype",
                "pluto_reader", "pypluto_path", "r_star", "with_b")
        if "pluto_reader" in build_kw and build_kw["pluto_reader"] is not None:
            build_kw["reader"] = build_kw.pop("pluto_reader")
            keep = keep + ("reader",)
        return load_pluto(path, **{k: v for k, v in build_kw.items()
                                   if k in keep and v is not None})
    build_kw = {k: v for k, v in build_kw.items()
                if k not in ("nout", "fmt", "definitions", "unit_density", "unit_length",
                             "unit_velocity", "mu_e", "mean_mol_weight", "pluto_reader",
                             "pypluto_path")}
    if not build_kw.get("with_b"):
        build_kw.pop("b_unit", None)
    if build_kw.get("r_body") is None:
        build_kw.pop("r_body", None)
    if reader is None:
        reader = {".dat": "tecplot", ".tec": "tecplot", ".plt": "pytecplot",
                  ".szplt": "pytecplot", ".out": "idl", ".outs": "idl",
                  ".npz": "npz"}.get(ext)
    if reader == "tecplot":
        v = read_tecplot_ascii(path)
    elif reader == "pytecplot":
        v = read_pytecplot(path)
    elif reader == "idl":
        v = read_batsrus_idl(path)
    elif reader == "npz":
        z = np.load(path)
        if "geometry" in z.files and str(z["geometry"]) == "spherical":
            from .grid import SphericalGrid
            ax = bool(z["axisym"])
            g = SphericalGrid(z["r"], z["theta"], None if ax else z["phi"], z["ne"], z["te"],
                                 r_edges=z["r_edges"], theta_edges=z["theta_edges"],
                                 phi_edges=None if ax else z["phi_edges"],
                                 r_body=float(z["r_body"]),
                                 length_unit=float(z["length_unit"]) if "length_unit" in z.files
                                 else R_SUN)
            if "b" in z.files:
                g.add_field(*np.moveaxis(z["b"], -1, 0), components=str(z["b_components"]))
            return g
        if {"x", "y", "z", "ne"} <= set(z.files) and z["ne"].ndim == 3:
            g = MHDGrid(z["x"], z["y"], z["z"], z["ne"],
                           z["te"] if "te" in z.files else None,
                           r_body=float(z["r_body"]) if "r_body" in z.files
                           else build_kw.get("r_body", 1.0),
                           length_unit=float(z["length_unit"]) if "length_unit" in z.files
                           else (build_kw.get("r_star") or 1.0) * R_SUN)
            if "b" in z.files:
                g.add_field(*np.moveaxis(z["b"], -1, 0))
            return g
        v = {k: z[k] for k in z.files}
    else:
        raise ValueError(f"cannot infer reader for '{path}'")
    v = {k: (np.ravel(a) if isinstance(a, np.ndarray) else a) for k, a in v.items()}
    return build_grid(v, **build_kw)
