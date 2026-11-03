# radiosynth - From MHD to MHz: synthetic radio images of MHD models of the Sun and stars
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
"""Combined free-free + gyroresonance + gyrosynchrotron images.

Along every straight line of sight the three mechanisms are put into one
polarised radiative transfer (gyroresonance/transfer.py), so that they emit
and absorb together:

  * free-free (magnetised, o/x modes)        bremsstrahlung/magnetized.py
  * gyroresonance layers nu = s nu_B         gyroresonance/opacity.py
  * gyrosynchrotron of the fast electrons    per-voxel tau and emission from
                                             the fast GS codes (coefficients.py)

In each step the optical depths add and the source function is the
optical-depth-weighted mean of the contributions (S = sum j / sum kappa), so
e.g. an optically thick gyroresonance layer in front of a gyrosynchrotron
source hides it, and dense plasma absorbs what lies behind.

Refraction: gyroresonance and gyrosynchrotron are always computed along
straight rays.  The free-free part can optionally (ff_refraction=True) be
corrected for refraction: the refractive ray tracer (as in
make_radio_image) is run with and without refraction, and in every pixel and
polarisation the free-free contribution of the combined transfer is multiplied
by the ratio refracted/straight of the free-free intensity,

    I_p -> I_p + I_ff,p (R - 1),   R = I_ff(refracted) / I_ff(straight),

i.e. the free-free emission keeps the absorption by the layers and the fast
electrons computed along the straight ray.  This is exact where refraction is
negligible (R = 1) and approximate otherwise; where the straight free-free
intensity vanishes, half of the difference I_ff(refracted) - I_ff(straight)
is added to each polarisation.
"""
import math
import time

import numpy as np

from ..common.constants import R_SUN
from ..common.image import freq_label, image_geometry, parse_frequencies, start_points
from ..common.raytrace import HIT_BODY, MISSED, OVERDENSE
from ..common.straight import default_step, iter_samples, segment_ends
from ..gyroresonance.image import GRImage
from ..gyroresonance.transfer import transfer_chunk
from .coefficients import gs_voxel_coefficients, source_weights

PC = 3.0856775814913673e16
MECH_NAMES = {-1: "none", 0: "free-free", 1: "gyroresonance", 2: "gyrosynchrotron"}


class CombinedImage(GRImage):
    """GRImage with the contributions of the three mechanisms.

    ff_fraction, gr_fraction, gs_fraction : share of the intensity from
        free-free, gyroresonance layers and gyrosynchrotron (sum = 1)
    dominant_mechanism : 0 free-free, 1 gyroresonance, 2 gyrosynchrotron, -1 none
    dominant_s : dominant harmonic among the gyroresonance layers (as GRImage)
    """

    fraction_label = "gyroresonance fraction"

    def __init__(self, *a, gs_fraction=None, **kw):
        super().__init__(*a, **kw)
        self.gs_fraction = gs_fraction
        self.ff_fraction = np.clip(1.0 - self.gr_fraction - gs_fraction, 0.0, 1.0)
        st = np.stack([self.ff_fraction, self.gr_fraction, gs_fraction])
        self.dominant_mechanism = np.where(self.intensity > 0, np.argmax(st, axis=0),
                                           -1).astype(np.int16)

    def summary(self):
        lines = []
        for k, f in enumerate(self.freqs):
            I = self.flux_density[k]
            V = self.flux_density_v[k]
            w = self.intensity[k]
            sh = [float((fr[k] * w).sum() / max(w.sum(), 1e-300))
                  for fr in (self.ff_fraction, self.gr_fraction, self.gs_fraction)]
            star = self.meta.get("distance_pc") is not None
            fl = (f"{I / 1e-29:.4g} mJy" if star else f"{I / 1e-22:.4g} sfu")
            lines.append(f"{freq_label(f):>12s}: max Tb={self.brightness_temperature[k].max():.3e} K, "
                         f"flux={fl}, V/I={V / I if I > 0 else 0:+.3f}; flux from "
                         f"ff/gr/gs = {sh[0]:.2f}/{sh[1]:.2f}/{sh[2]:.2f}")
        return "\n".join(lines)

    def plot(self, quantity="tb", path=None, **kw):
        """Also 'mechanism' (dominant mechanism map), 'ff_fraction', 'gs_fraction'."""
        if quantity in ("mechanism", "ff_fraction", "gs_fraction"):
            return self._plot_extra(quantity, path, **kw)
        return super().plot(quantity, path=path, **kw)

    def _plot_extra(self, quantity, path, ncols=None, show_limb=True, **_):
        import matplotlib
        if path is not None:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap, BoundaryNorm
        nf = len(self.freqs)
        ncols = ncols or min(nf, 3)
        nrows = math.ceil(nf / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.0 * nrows), squeeze=False)
        dx, dy = self.x[1] - self.x[0], self.y[1] - self.y[0]
        ext = [self.x[0] - dx / 2, self.x[-1] + dx / 2, self.y[0] - dy / 2, self.y[-1] + dy / 2]
        for k in range(nrows * ncols):
            ax = axes.flat[k]
            if k >= nf:
                ax.axis("off")
                continue
            if quantity == "mechanism":
                cmap = ListedColormap(["#d9d9d9", "#4c72b0", "#dd8452", "#c44e52"])
                norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5], cmap.N)
                im = ax.imshow(self.dominant_mechanism[k], origin="lower", extent=ext,
                               cmap=cmap, norm=norm)
                cb = fig.colorbar(im, ax=ax, ticks=[-1, 0, 1, 2], shrink=0.85)
                cb.ax.set_yticklabels(["none", "free-free", "gyroresonance", "gyrosynchrotron"])
            else:
                d = getattr(self, quantity)[k]
                im = ax.imshow(d, origin="lower", extent=ext, cmap="magma", vmin=0, vmax=1)
                fig.colorbar(im, ax=ax, label=quantity.replace("_", " "), shrink=0.85)
            if show_limb:
                t = np.linspace(0, 2 * np.pi, 361)
                ax.plot(np.cos(t), np.sin(t), "k--", lw=0.7, alpha=0.6)
            ax.set_xlim(ext[0], ext[1])
            ax.set_ylim(ext[2], ext[3])
            ax.set_title(freq_label(self.freqs[k]))
            ax.set_xlabel(r"X [$R_\odot$]")
            ax.set_ylabel(r"Y [$R_\odot$]")
        fig.tight_layout()
        if path is not None:
            fig.savefig(path, dpi=130)
            plt.close(fig)
        return fig

    def save_npz(self, path):
        super().save_npz(path)
        z = dict(np.load(path, allow_pickle=True))
        z.update(ff_fraction=self.ff_fraction, gs_fraction=self.gs_fraction,
                 dominant_mechanism=self.dominant_mechanism)
        np.savez_compressed(path, **z)


# ----------------------------------------------------------------------------
def make_combined_image(grid, frequencies, electrons=None, observer=(-215.0, 0.0, 0.0),
                        x_size=2.0, y_size=2.0, nx=128, ny=128, *,
                        include=("ff", "gr", "gs"), gaunt="dulk", s_max=10, coupling="weak",
                        ds=None, t_body=None, r_integration=None, up=(0.0, 0.0, 1.0),
                        distance_pc=None, library=None, fc=12.0, fwh=12.0, nnodes=0,
                        q_optimization=True, max_points=200000, verbose=True,
                        ff_refraction=False, trace_kw=None):
    """Free-free + gyroresonance + gyrosynchrotron in one polarised transfer.

    grid        : grid with the magnetic field (load_mhd(..., with_b=True))
    frequencies : '3GHz, 17GHz' or numbers [Hz]
    electrons   : FastElectrons prescription (radiosynth.gyrosynchrotron); None
                  = no fast electrons (then the result equals make_gr_image)
    include     : any subset of ('ff', 'gr', 'gs')
    gaunt, s_max, coupling, ds, t_body : as in make_gr_image
    library, fc, fwh, nnodes, q_optimization : as in make_gs_image
    observer, x_size, y_size, nx, ny, up, distance_pc, r_integration : as usual
    ff_refraction : False (default; free-free along the straight rays like the other
                  two) or True: correct the free-free part for refraction with the
                  refractive ray tracer (see the module docstring); gyroresonance and
                  gyrosynchrotron always use straight rays
    trace_kw    : options of the refractive tracer for ff_refraction (tolerance,
                  step_factor, n_workers, ...)
    """
    if not getattr(grid, "has_field", False):
        raise ValueError("the combined image needs the magnetic field: grid.add_field(...) "
                         "or load_mhd(..., with_b=True)")
    include = tuple(m.lower() for m in include)
    bad = set(include) - {"ff", "gr", "gs"}
    if bad:
        raise ValueError(f"unknown mechanisms {bad}; use 'ff', 'gr', 'gs'")
    use_gs = "gs" in include and electrons is not None
    mech = "+".join(m for m in ("gr", "ff") if m in include)
    if use_gs:
        from ..gyrosynchrotron.electrons import ENERGY_INDEX, PITCH_INDEX, FastElectrons
        from ..gyrosynchrotron.image import _fields, voxel_parms
        from ..gyrosynchrotron.library import load_library, library_version
        if not isinstance(electrons, FastElectrons):
            raise TypeError("electrons must be a FastElectrons instance")
        lib = load_library(library)
        eidx = ENERGY_INDEX[electrons.energy]
        pidx = PITCH_INDEX[electrons.pitch]
        arr = electrons.array
        scal = sorted(getattr(grid, "scalars", {}).keys())
        log_key = 0
        if arr is not None:
            r = arr[0][1:] / arr[0][:-1]
            log_key = 0 if np.allclose(r, r[0], rtol=1e-3) else 1
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
    if verbose:
        print(f"radiosynth.combined: {nx}x{ny} straight rays"
              f"{' (free-free corrected for refraction)' if ff_refraction else ''}, "
              f"include={'+'.join(include)}, "
              f"coupling={coupling}, step={step:.3g}"
              + (f", {electrons}" if use_gs else ", no fast electrons"))
        if use_gs:
            ver = library_version(library)
            print(f"            GS codes: {lib.path}" + (f" ({ver})" if ver else ""))
        print(f"            {grid}")

    keys = ("I1", "I2", "tau1", "tau2", "V", "radius", "gr_fraction", "gs_fraction",
            "ff1", "ff2", "vsign")
    res = {k: np.zeros((nf, nray)) for k in keys}
    res["radius"][:] = np.nan
    dom = np.full((nf, nray), -1, np.int16)
    cut = np.zeros((nf, nray), bool)
    xf = np.zeros((nf, nray), bool)
    t0 = time.time()
    nvox_gs = 0
    e_ratio = 0.0
    for sm in iter_samples(grid, pos, dirs, t_end, step, max_points=max_points):
        idx = sm["idx"]
        nr, nk = sm["dl"].shape
        extra_all = None
        if use_gs:
            fl = _fields(sm, grid, scal)
            par = electrons.evaluate(fl)
            dl_cm = (sm["dl"] * L * 100.0).ravel()
            valid = (dl_cm > 0) & (fl["ne"] > 0)
            nb = np.where(valid, par["nb"], 0.0)
            sel = np.flatnonzero(nb > 0)
            if sel.size:
                nvox_gs += sel.size
                sub = {k: v[sel] for k, v in fl.items()}
                e_ratio = max(e_ratio, electrons.check_energy(sub, fl["bmag"][sel], verbose=False))
                nobs = np.repeat(-dirs[idx], nk, axis=0)[sel]
                cos = np.einsum("ij,ij->i", sub["b"], nobs) / np.maximum(sub["bmag"], 1e-30)
                theta = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
                psub = {k: v[sel] for k, v in par.items()}
                P = voxel_parms(sub, psub, dl_cm[sel], np.ones(sel.size, bool), nb[sel], theta,
                                False, eidx, pidx, -1, arr is not None)
                if arr is not None:
                    f_arr = arr[2][:, :, None] * nb[sel][None, None, :]
                    coef = gs_voxel_coefficients(lib, P, freqs, arr[0], arr[1], f_arr, fc, fwh,
                                                 nnodes, q_optimization, log_key)
                else:
                    coef = gs_voxel_coefficients(lib, P, freqs, fc=fc, fwh=fwh, nnodes=nnodes,
                                                 q_optimization=q_optimization)
                sw = source_weights(coef, freqs)
                extra_all = {}
                for key, val in (("dtau_R", coef["tau_R"]), ("dtau_L", coef["tau_L"]),
                                 ("st_R", sw["st_R"]), ("st_L", sw["st_L"])):
                    a = np.zeros((nf, nr * nk))
                    a[:, sel] = val
                    extra_all[key] = a.reshape(nf, nr, nk)
        for k, f in enumerate(freqs):
            extra = None if extra_all is None else {key: v[k] for key, v in extra_all.items()}
            if extra is None and use_gs:
                z = np.zeros((nr, nk))
                extra = dict(dtau_R=z, dtau_L=z, st_R=z, st_L=z)
            o = transfer_chunk(sm, dirs, f, L, mechanisms=mech, gaunt=gaunt, s_max=s_max,
                               coupling=coupling, t_body=t_body, hit_body=hit_body, extra=extra)
            for key in keys:
                if key in o:
                    res[key][k, idx] = o[key]
            dom[k, idx], cut[k, idx], xf[k, idx] = o["dominant_s"], o["cutoff"], o["x_flag"]
    if verbose:
        print(f"   {nf} frequencies: {time.time() - t0:.1f} s"
              + (f"; {nvox_gs} voxels with fast electrons, max fast-electron/magnetic "
                 f"energy density = {e_ratio:.2g}" if use_gs else ""))
        if e_ratio > 0.1:
            print("WARNING: the fast electrons carry more than 10% of the magnetic energy "
                  "density somewhere - check the prescription")

    ff_factor = None
    if ff_refraction and "ff" in include:
        from ..common.image import make_radio_image
        t1 = time.time()
        common = dict(observer=observer, x_size=x_size, y_size=y_size, nx=nx, ny=ny,
                      emission="bremsstrahlung", gaunt=gaunt, transfer="full", up=up,
                      r_integration=r_integration, distance_pc=distance_pc, verbose=False,
                      **(trace_kw or {}))
        i_ref = make_radio_image(grid, freqs, **common).intensity.reshape(nf, nray)
        i_los = make_radio_image(grid, freqs, no_refraction=True, **common).intensity.reshape(nf, nray)
        good = i_los > 1e-6 * np.maximum(i_los.max(axis=1, keepdims=True), 1e-300)
        with np.errstate(invalid="ignore", divide="ignore"):
            ff_factor = np.where(good, i_ref / np.where(good, i_los, 1.0), np.nan)
        I_old = res["I1"] + res["I2"]
        d1 = np.where(good, res["ff1"] * (ff_factor - 1.0), 0.5 * (i_ref - i_los))
        d2 = np.where(good, res["ff2"] * (ff_factor - 1.0), 0.5 * (i_ref - i_los))
        res["I1"] = np.maximum(res["I1"] + d1, 0.0)
        res["I2"] = np.maximum(res["I2"] + d2, 0.0)
        res["V"] = res["V"] + res["vsign"] * (d1 - d2)
        I_new = res["I1"] + res["I2"]
        with np.errstate(invalid="ignore", divide="ignore"):
            for key in ("gr_fraction", "gs_fraction"):
                res[key] = np.where(I_new > 0, res[key] * I_old / I_new, 0.0)
        if verbose:
            fr_ = i_ref.sum(axis=1) / np.maximum(i_los.sum(axis=1), 1e-300)
            print(f"   free-free refraction correction: {time.time() - t1:.1f} s; refracted/straight "
                  f"free-free flux = {np.round(fr_, 4)}")

    shp = (nf, ny, nx)
    st = np.broadcast_to(status, (nf, nray)).copy()
    st[cut & (st != MISSED)] = OVERDENSE
    meta = dict(x_size=x_size, y_size=y_size, nx=nx, ny=ny, include=include,
                mechanisms=mech, gaunt=gaunt, s_max=s_max, coupling=coupling, ds=step,
                t_body=t_body, distance_pc=distance_pc, r_integration=r_integration,
                electrons=repr(electrons), tau_vec=tau_v, xi_vec=xi_v, emission="combined",
                ff_refraction=bool(ff_refraction and "ff" in include),
                transfer="full", energy_ratio_max=e_ratio)
    R = lambda a: a.reshape(shp)                                   # noqa: E731
    img = CombinedImage(R(res["I1"] + res["I2"]), R(res["tau1"]), R(st),
                        np.full(shp, int(np.ceil(np.max(t_end, initial=0) / step))), xc, yc,
                        freqs, observer, meta, distance=distance, length_unit=L,
                        stokes_v=R(res["V"]), pol1=R(res["I1"]), pol2=R(res["I2"]),
                        tau2=R(res["tau2"]), dominant_s=R(dom), radius=R(res["radius"]),
                        gr_fraction=R(res["gr_fraction"]), x_flag=R(xf),
                        pol_names=("R", "L") if coupling == "weak" else ("x", "o"),
                        gs_fraction=R(res["gs_fraction"]))
    img.ff_refraction_factor = R(ff_factor) if ff_factor is not None else None
    if verbose:
        print(img.summary())
    return img
