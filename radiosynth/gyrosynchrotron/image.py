# radiosynth - From MHD to MHz: synthetic radio images of MHD models of the Sun and stars
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
"""Gyrosynchrotron (+ free-free) images computed with the fast GS codes.

radiosynth supplies the geometry and the plasma along every line of sight
(common/image.py, common/straight.py): the voxels of each straight ray with
their length, n_e, T_e, |B|, the angle between B and the line of sight, and the
fast-electron parameters from a FastElectrons prescription.  The radiative
transfer along the ray - gyrosynchrotron emission and absorption of the fast
electrons, free-free emission of the thermal plasma, the two magneto-ionic
modes and their coupling - is done by the fast GS codes (GET_MW_SLICE,
Kuznetsov & Fleishman 2021), many lines of sight at once.

Conventions checked against the library (tests/test_gyrosynchrotron.py):
  * the viewing angle theta is the angle between B and the direction to the
    observer; voxel 0 of a line of sight is the far end;
  * RL rows: 1/2 left/right (weak coupling), 3/4 (strong), 5/6 (exact);
    right-hand circular polarisation dominates the x-mode emission where B
    points towards the observer, so V = R - L follows the IAU/IEEE convention
    used everywhere in radiosynth;
  * intensities come out as flux densities [sfu] at 1 AU from a source of area
    Rparms[0]; with the pixel area as source area this gives the intensity
    I = F AU^2 / S, independent of the actual observer distance.
"""
import math
import time

import numpy as np

from ..common.constants import AU, C_LIGHT, K_B, R_SUN
from ..common.image import freq_label, image_geometry, parse_frequencies, start_points
from ..common.raytrace import HIT_BODY
from ..common.straight import default_step, iter_samples, segment_ends
from ..gyroresonance.image import GRImage
from .electrons import ENERGY_INDEX, PITCH_INDEX, FastElectrons
from .library import load_library, library_version

PC = 3.0856775814913673e16
SFU = 1e-22
COUPLING_ROWS = {"weak": (1, 2), "strong": (3, 4), "exact": (5, 6)}
ABUNDANCE = {"dulk": -1, "coronal": 0, "photospheric": 1}


class GSImage(GRImage):
    """Gyrosynchrotron image: Stokes I and V (pol1 = R, pol2 = L).

    gs_fraction (alias gr_fraction): 1 - I_ff/I, the fraction of the intensity
    due to the fast electrons (only with diagnostics=True, else NaN).
    tau is not returned by the library (NaN)."""

    fraction_label = "fast-electron fraction"

    @property
    def gs_fraction(self):
        return self.gr_fraction

    def summary(self):
        lines = []
        for k, f in enumerate(self.freqs):
            I = self.flux_density[k]
            V = self.flux_density_v[k]
            star = self.meta.get("distance_pc") is not None
            fl = (f"{I / 1e-29:.4g} mJy at {self.distance_pc:.3g} pc" if star
                  else f"{I / SFU:.4g} sfu")
            lines.append(f"{freq_label(f):>12s}: max Tb={self.brightness_temperature[k].max():.3e} K, "
                         f"flux={fl}, V/I={V / I if I > 0 else 0:+.3f}")
        return "\n".join(lines)

    def plot(self, quantity="tb", **kw):
        if quantity == "gs_fraction":
            quantity = "gr_fraction"
        if quantity in ("harmonic", "radius"):
            raise ValueError(f"'{quantity}' is not available for gyrosynchrotron images")
        return super().plot(quantity, **kw)


def _fields(sm, grid, idx_scalars):
    """Mid-point quantities of every voxel of a chunk (flattened)."""
    m = lambda a: 0.5 * (a[:, 1:] + a[:, :-1])                       # noqa: E731
    pos = m(sm["p"]).reshape(-1, 3)
    f = dict(pos=pos, r=np.linalg.norm(pos, axis=1),
             ne=m(sm["ne"]).ravel(), te=m(sm["te"]).ravel(),
             b=m(sm["b"]).reshape(-1, 3))
    f["bmag"] = np.linalg.norm(f["b"], axis=1)
    for name in idx_scalars:
        f[name] = grid.sample_scalar(name, pos)
    return f


def voxel_parms(fl, par, dl_cm, valid, nb, theta, free_free, eidx, pidx, abund, is_array):
    """The 24 voxel parameters of the fast GS codes (CallingConventions.pdf) for
    voxels with quantities fl (from _fields), fast-electron parameters par,
    lengths dl_cm [cm], validity mask, n_b [cm^-3] and viewing angle theta [deg]."""
    P = np.zeros((24, dl_cm.size))
    P[0] = np.where(valid, dl_cm, 0.0)
    P[1] = np.where(valid, np.maximum(fl["te"], 1.0), 1e6)
    P[2] = np.where(valid, fl["ne"] * 1e-6, 0.0)
    P[3] = np.maximum(fl["bmag"], 1e-3)
    P[4] = np.where(fl["bmag"] > 0, theta, 90.0)
    P[5] = (0 if free_free else 2) + 4                 # e-neutrals off (no n_HI given)
    P[6] = eidx
    P[7] = nb
    P[8] = par["eps_kappa"]
    P[9], P[10], P[11] = par["emin"], par["emax"], par["ebreak"]
    P[12], P[13] = par["delta"], par["delta2"]
    P[14] = pidx
    P[15], P[16], P[17] = par["alpha_c"], par["dmu"], par["a4"]
    P[18] = P[2]                                       # n_p != 0: n0 is n_e also below 1e5 K
    P[21] = np.where(nb > 0, 2 if is_array else 1, 3)  # analytic / array / free-free only
    P[22] = abund
    return P


def make_gs_image(grid, frequencies, electrons, observer=(-215.0, 0.0, 0.0), x_size=2.0,
                  y_size=2.0, nx=64, ny=64, *, coupling="exact", free_free=True,
                  gyrosynchrotron=True, abundance="coronal", ds=None, t_body=None,
                  r_integration=None, up=(0.0, 0.0, 1.0), distance_pc=None, library=None,
                  fc=12.0, fwh=12.0, nnodes=0, q_optimization=True, diagnostics=False,
                  max_voxels=300000, verbose=True):
    """Gyrosynchrotron + free-free image (Stokes I, V) with the fast GS codes.

    grid        : MHDGrid / SphericalGrid with the magnetic field (gauss)
    frequencies : '3GHz, 17GHz' or numbers [Hz]
    electrons   : FastElectrons (radiosynth.gyrosynchrotron.electrons)
    observer, x_size, y_size, nx, ny, up, distance_pc, r_integration :
                  as in make_radio_image / make_gr_image (straight rays)
    coupling    : 'exact' (default), 'weak' or 'strong' mode coupling (library)
    free_free   : include free-free (electron-ion) emission/absorption
    gyrosynchrotron : include the fast electrons (False: library free-free only)
    abundance   : 'coronal' (Feldman 1992), 'photospheric' (Scott+2015) or
                  'dulk' (Dulk 1985 formulas) for the free-free opacity
    ds          : voxel length along the rays [coordinate units] (default: half
                  the smallest cell)
    t_body      : brightness temperature of the body seen behind the plasma
    library     : path of the compiled GS codes (see library.py)
    fc, fwh     : switch frequencies of the hybrid GS code, in units of the local
                  gyrofrequency (library defaults: 12, 12)
    nnodes, q_optimization : accuracy controls of the continuous GS code
    diagnostics : also compute the free-free-only image -> gs_fraction
    max_voxels  : voxels passed to the library at once (memory control)
    """
    if not getattr(grid, "has_field", False):
        raise ValueError("gyrosynchrotron needs the magnetic field: grid.add_field(...) "
                         "or load_mhd(..., with_b=True)")
    if not isinstance(electrons, FastElectrons):
        raise TypeError("electrons must be a FastElectrons instance")
    if coupling not in COUPLING_ROWS:
        raise ValueError("coupling must be 'exact', 'weak' or 'strong'")
    lib = load_library(library)
    freqs = parse_frequencies(frequencies)
    nf = freqs.size
    L = float(getattr(grid, "length_unit", R_SUN))
    parallel = distance_pc is not None
    xc, yc, dirs, tau_v, xi_v, normal, pix = image_geometry(observer, x_size, y_size, nx, ny,
                                                            up, parallel=parallel)
    diag = math.sqrt((0.5 * x_size) ** 2 + (0.5 * y_size) ** 2)
    if r_integration is None:
        r_integration = float(math.ceil(max(diag + 1.0, 5.0)))
    if parallel:
        distance = distance_pc * PC / L
        far = 4.0 * max(r_integration, getattr(grid, "r_outer", r_integration), diag)
        pos, status0 = start_points(grid, pix + far * normal, dirs, r_integration)
    else:
        distance = float(np.linalg.norm(observer))
        pos, status0 = start_points(grid, observer, dirs, r_integration)
    t_end, status = segment_ends(grid, pos, dirs, status0, r_integration)
    step = default_step(grid, t_end, ds)
    hit_body = status == HIT_BODY
    nray = dirs.shape[0]

    # pixel (source) area in m^2 and the sfu <-> intensity conversion
    s_pix = (x_size / nx * L) * (y_size / ny * L)
    to_int = SFU * AU**2 / s_pix                      # sfu -> W m^-2 Hz^-1 sr^-1

    arr = electrons.array
    ne_, nmu = (arr[0].size, arr[1].size) if arr is not None else (1, 1)
    E_arr = arr[0] if arr is not None else np.zeros(1)
    mu_arr = arr[1] if arr is not None else np.zeros(1)
    log_key = 0
    if arr is not None:
        r = E_arr[1:] / E_arr[:-1]
        log_key = 0 if np.allclose(r, r[0], rtol=1e-3) else 1
    scal = sorted(getattr(grid, "scalars", {}).keys())
    eidx = ENERGY_INDEX[electrons.energy]
    pidx = PITCH_INDEX[electrons.pitch]
    abund = ABUNDANCE[abundance] if isinstance(abundance, str) else int(abundance)
    if verbose:
        ver = library_version(library)
        print(f"radiosynth.gyrosynchrotron: {nx}x{ny} straight rays, coupling={coupling}, "
              f"free-free={'on' if free_free else 'off'}, voxel={step:.3g}, {electrons}")
        print(f"            GS codes: {lib.path}" + (f" ({ver})" if ver else ""))
        print(f"            {grid}")

    I_R = np.zeros((nf, nray))
    I_L = np.zeros((nf, nray))
    I_ff = np.full((nf, nray), np.nan)
    e_ratio = 0.0
    nb_max = 0.0
    t0 = time.time()
    # chunks of rays; each chunk is one GET_MW_SLICE call (OpenMP over rays)
    per_voxel = max(1, (ne_ * nmu) // 24) if arr is not None else 1
    for sm in iter_samples(grid, pos, dirs, t_end, step,
                           max_points=max(1000, max_voxels // per_voxel)):
        idx = sm["idx"]
        npix, nz = sm["dl"].shape
        fl = _fields(sm, grid, scal)
        par = electrons.evaluate(fl)
        dl_cm = (sm["dl"] * L * 100.0).ravel()
        valid = (dl_cm > 0) & (fl["ne"] > 0)
        nb = np.where(valid, par["nb"], 0.0) if gyrosynchrotron else np.zeros_like(dl_cm)
        nb_max = max(nb_max, float(nb.max(initial=0.0)))
        if gyrosynchrotron and nb.max(initial=0) > 0:
            sel = nb > 0
            sub = {k: v[sel] for k, v in fl.items()}
            e_ratio = max(e_ratio, electrons.check_energy(sub, fl["bmag"][sel], verbose=False))
        nobs = np.repeat(-dirs[idx], nz, axis=0)
        cos = np.einsum("ij,ij->i", fl["b"], nobs) / np.maximum(fl["bmag"], 1e-30)
        theta = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

        P = voxel_parms(fl, par, dl_cm, valid, nb, theta, free_free, eidx, pidx, abund,
                        arr is not None)
        # library order: voxel 0 = far end of the line of sight
        P = P.reshape(24, npix, nz)[:, :, ::-1].transpose(0, 2, 1)
        Parms_M = np.asfortranarray(P)
        if arr is not None:
            fz = (arr[2][:, :, None, None] *
                  nb.reshape(npix, nz)[:, ::-1].T[None, None, :, :])
            f_arr = np.asfortranarray(fz)
        else:
            f_arr = np.zeros((1, 1, 1, 1), order="F")
        Lp = np.array([npix, nz, nf, ne_ if arr is not None else 0, nmu if arr is not None else 0,
                       int(nnodes), 0, 0 if q_optimization else 1,
                       1 if arr is None else 2, log_key, 0, 0], np.int32)
        Rp = np.zeros((5, npix), order="F")
        Rp[0] = s_pix * 1e4
        Rp[3], Rp[4] = fc, fwh

        def run(Pm):
            RL = np.zeros((7, nf, npix), order="F")
            RL[0] = (freqs / 1e9)[:, None]
            if t_body is not None:
                body = hit_body[idx]
                Ib = freqs[:, None] ** 2 * K_B * float(t_body) / C_LIGHT**2 / to_int
                RL[1:7] = np.where(body[None, :], Ib, 0.0)[None]
            lib.get_mw_slice(Lp, Rp, Pm, E_arr, mu_arr, f_arr, RL)
            return RL

        RL = run(Parms_M)
        lr, rr = COUPLING_ROWS[coupling]
        I_L[:, idx] = RL[lr] * to_int
        I_R[:, idx] = RL[rr] * to_int
        if diagnostics:
            P2 = Parms_M.copy()
            P2[21] = 3
            RL2 = run(np.asfortranarray(P2))
            I_ff[:, idx] = (RL2[lr] + RL2[rr]) * to_int

    if verbose:
        print(f"   {nf} frequencies: {time.time() - t0:.1f} s; max n_b = {nb_max:.3g} cm^-3; "
              f"max fast-electron/magnetic energy density = {e_ratio:.2g}")
        if e_ratio > 0.1:
            print("WARNING: the fast electrons carry more than 10% of the magnetic energy "
                  "density somewhere - check the prescription")

    shp = (nf, ny, nx)
    I = I_R + I_L
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(I > 0, 1.0 - I_ff / I, 0.0) if diagnostics else np.full_like(I, np.nan)
    meta = dict(x_size=x_size, y_size=y_size, nx=nx, ny=ny, coupling=coupling,
                free_free=free_free, gyrosynchrotron=gyrosynchrotron, abundance=abundance,
                ds=step, t_body=t_body, distance_pc=distance_pc, r_integration=r_integration,
                electrons=repr(electrons), library=lib.path, fc=fc, fwh=fwh,
                tau_vec=tau_v, xi_vec=xi_v, emission="gyrosynchrotron", transfer="full",
                energy_ratio_max=e_ratio)
    R_ = lambda a: a.reshape(shp)                                   # noqa: E731
    nanarr = np.full(shp, np.nan)
    img = GSImage(R_(I), nanarr, R_(np.broadcast_to(status, (nf, nray)).copy()),
                  np.full(shp, int(np.ceil(np.max(t_end, initial=0) / step))), xc, yc, freqs,
                  observer, meta, distance=distance, length_unit=L,
                  stokes_v=R_(I_R - I_L), pol1=R_(I_R), pol2=R_(I_L), tau2=nanarr.copy(),
                  dominant_s=np.full(shp, -1, np.int16), radius=nanarr.copy(),
                  gr_fraction=R_(frac), x_flag=np.zeros(shp, bool), pol_names=("R", "L"))
    if verbose:
        print(img.summary())
    return img
