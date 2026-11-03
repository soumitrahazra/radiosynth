# radiosynth - From MHD to MHz: synthetic radio images of MHD models of the Sun and stars
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
"""Polarised radiative transfer along straight rays: gyroresonance layers plus
(magnetised) free-free absorption.

For each line of sight, sampled at nodes 0..nk from the observer side
(common/straight.py), and each step k between nodes k and k+1:

* free-free: optical depth  dtau_ff = kappa_sigma(mid-point) dl  for the
  o- and x-modes (bremsstrahlung/magnetized.py);
* gyroresonance: for every harmonic s whose layer |B| = nu/(s nu_B/G) lies
  inside the step (sign change of s nu_B - nu), the layer optical depth
  tau_s (opacity.py) with n_e, T_e, theta interpolated to the crossing and
  L_B = |B| dl / |B_{k+1} - B_k|;
* the step emits S_eff (1 - exp(-dtau)) attenuated by exp(-tau in front),
  with dtau = dtau_ff + sum_s tau_s and S_eff the dtau-weighted mean of
  nu^2 k_B T / c^2 over its free-free part and its layers.

Mode coupling.  Along a ray B . k may change sign (quasi-transverse region).
coupling='weak' (default; usual at microwave frequencies): the sense of
circular polarisation is preserved, so the transfer is done for right- and
left-hand circular polarisation R and L, each using the x-mode coefficients
where it is the x-mode (R: B . k > 0) and the o-mode ones elsewhere;
I = R + L, V = R - L.  coupling='strong': mode identity is preserved, the
transfer is done for the x- and o-modes and V = +/-(I_x - I_o) with the sign
of B . k at the observer end of the ray.

Cutoffs: a polarisation stops where its mode cannot propagate (o: X >= 1,
x: X >= 1 - Y); nothing behind the cutoff contributes.
"""
import numpy as np

from ..common.constants import C_LIGHT, K_B
from ..common.magnetoionic import NU_B_PER_GAUSS, magnetoionic_params
from ..bremsstrahlung.magnetized import magnetized_ff_coefficients
from .opacity import layer_optical_depth

_SRC = K_B / C_LIGHT**2      # S = nu^2 k T / c^2 per mode


def transfer_chunk(sm, dirs, freq, length_unit, *, mechanisms="gr+ff", gaunt="dulk",
                   s_max=10, coupling="weak", t_body=None, hit_body=None,
                   x_warn=0.25, extra=None):
    """Polarised transfer for one chunk of rays from iter_samples.

    Returns dict of per-ray arrays: I1, I2 (the two polarisations: R/L or x/o),
    tau1, tau2 (total optical depth), V, cutoff (bool), dominant_s (0 = free-free,
    -1 = no emission), radius (intensity-weighted emission radius), gr_fraction,
    x_flag (a significant layer had X > x_warn).

    extra : optional additional emission/absorption per step in the circular
            polarisations (used by radiosynth.combined for gyrosynchrotron):
            dict of (nr, nk) arrays 'dtau_R', 'dtau_L' (optical depth of the
            step) and 'st_R', 'st_L' (optical depth x effective temperature
            S/(nu^2 k/c^2) [K]).  Then out['gs_fraction'] is also returned."""
    use_gr = "gr" in mechanisms
    use_ff = "ff" in mechanisms
    nr, nn = sm["ne"].shape
    ne, te, b, dl = sm["ne"], sm["te"], sm["b"], sm["dl"] * length_unit
    nobs = -dirs[sm["idx"]][:, None, :]                       # towards the observer
    bmag = np.sqrt(np.einsum("ijk,ijk->ij", b, b))
    cos = np.where(bmag > 0, np.einsum("ijk,ijk->ij", b, nobs) / np.maximum(bmag, 1e-300), 0.0)
    X, Y = magnetoionic_params(ne, bmag, freq)
    nu_b = NU_B_PER_GAUSS * bmag
    live = sm["inside"]

    # ---- cutoffs at the nodes (polarisation-dependent in weak coupling)
    cut_o = X >= 1.0
    cut_x = X >= 1.0 - Y
    # ---- mid-step quantities
    m = lambda a: 0.5 * (a[:, 1:] + a[:, :-1])                      # noqa: E731
    ne_m, te_m, cos_m, b_m = m(ne), m(te), m(cos), m(bmag)
    step_ok = (dl > 0) & live[:, 1:] & live[:, :-1]
    if use_ff:
        k_o, k_x, _ = magnetized_ff_coefficients(ne_m, te_m, b_m, cos_m, freq, gaunt=gaunt)
        dtau_o = np.where(step_ok, k_o * dl, 0.0)
        dtau_x = np.where(step_ok, k_x * dl, 0.0)
    else:
        dtau_o = np.zeros_like(dl)
        dtau_x = np.zeros_like(dl)
    st_o = dtau_o * te_m                                          # sum tau*T (source weight)
    st_x = dtau_x * te_m
    gr_o = np.zeros_like(dl)
    gst_x = np.zeros_like(dl)                                     # sum tau*T of the layers
    gst_o = np.zeros_like(dl)
    gr_x = np.zeros_like(dl)
    best_tau = np.zeros_like(dl)                                  # largest layer tau in step
    best_s = np.zeros(dl.shape, np.int16)
    x_flag_step = np.zeros(dl.shape, bool)

    if use_gr:
        f0, f1 = nu_b[:, :-1], nu_b[:, 1:]
        for s in range(1, int(s_max) + 1):
            cross = step_ok & ((s * f0 - freq) * (s * f1 - freq) < 0)
            if not cross.any():
                continue
            ii, kk = np.nonzero(cross)
            a0, a1 = f0[ii, kk], f1[ii, kk]
            u = (freq / s - a0) / (a1 - a0)
            lerp = lambda a: a[ii, kk] + u * (a[ii, kk + 1] - a[ii, kk])   # noqa: E731
            ne_c, te_c, cos_c, X_c = lerp(ne), lerp(te), lerp(cos), lerp(X)
            l_b = (freq / (s * NU_B_PER_GAUSS)) * dl[ii, kk] / np.abs(
                bmag[ii, kk + 1] - bmag[ii, kk])
            t_x = layer_optical_depth(s, ne_c, te_c, cos_c, l_b, freq, "x", X=X_c)
            t_o = layer_optical_depth(s, ne_c, te_c, cos_c, l_b, freq, "o", X=X_c)
            # the x-mode does not exist at the s = 1 layer when X > 0 (cutoff X >= 1 - Y = 0)
            if s == 1:
                t_x = np.where(X_c > 0, 0.0, t_x)
            np.add.at(gr_x, (ii, kk), t_x)
            np.add.at(gr_o, (ii, kk), t_o)
            np.add.at(st_x, (ii, kk), t_x * te_c)
            np.add.at(st_o, (ii, kk), t_o * te_c)
            np.add.at(gst_x, (ii, kk), t_x * te_c)
            np.add.at(gst_o, (ii, kk), t_o * te_c)
            tmax = np.maximum(t_x, t_o)
            better = tmax > best_tau[ii, kk]
            best_tau[ii[better], kk[better]] = tmax[better]
            best_s[ii[better], kk[better]] = s
            x_flag_step[ii, kk] |= (X_c > x_warn) & (tmax > 0.1)

    dtx = dtau_x + gr_x
    dto = dtau_o + gr_o
    if coupling == "weak":
        pos = cos_m > 0                       # R is the x-mode where B.k > 0
        d1, d2 = np.where(pos, dtx, dto), np.where(pos, dto, dtx)
        s1, s2 = np.where(pos, st_x, st_o), np.where(pos, st_o, st_x)
        g1, g2 = np.where(pos, gr_x, gr_o), np.where(pos, gr_o, gr_x)
        h1, h2 = np.where(pos, gst_x, gst_o), np.where(pos, gst_o, gst_x)
        posn = cos > 0
        c1, c2 = np.where(posn, cut_x, cut_o), np.where(posn, cut_o, cut_x)
    elif coupling == "strong":
        d1, d2, s1, s2, g1, g2, c1, c2 = dtx, dto, st_x, st_o, gr_x, gr_o, cut_x, cut_o
        h1, h2 = gst_x, gst_o
    else:
        raise ValueError("coupling must be 'weak' or 'strong'")
    e1 = e2 = None
    if extra is not None:
        if coupling == "weak":                    # polarisation 1 = R, 2 = L
            e1d, e2d, e1s, e2s = extra["dtau_R"], extra["dtau_L"], extra["st_R"], extra["st_L"]
        else:                                     # 1 = x (= R where B.k > 0), 2 = o
            pm = cos_m > 0
            e1d = np.where(pm, extra["dtau_R"], extra["dtau_L"])
            e2d = np.where(pm, extra["dtau_L"], extra["dtau_R"])
            e1s = np.where(pm, extra["st_R"], extra["st_L"])
            e2s = np.where(pm, extra["st_L"], extra["st_R"])
        d1, d2, s1, s2 = d1 + e1d, d2 + e2d, s1 + e1s, s2 + e2s
        e1, e2 = e1s, e2s

    out = {}
    contrib_tot = np.zeros_like(dl)
    contrib_gr = np.zeros_like(dl)
    contrib_gs = np.zeros_like(dl)
    cut_any = np.zeros(nr, bool)
    for tag, d, sw, g, c, e, h in (("1", d1, s1, g1, c1, e1, h1), ("2", d2, s2, g2, c2, e2, h2)):
        # polarisation stops at its first cutoff node (steps beyond contribute nothing)
        alive = np.cumprod(~(c & live), axis=1).astype(bool)
        ok = alive[:, 1:]
        d = np.where(ok, d, 0.0)
        sw = np.where(ok, sw, 0.0)
        g = np.where(ok, g, 0.0)
        tau_front = np.cumsum(d, axis=1) - d
        with np.errstate(invalid="ignore", divide="ignore"):
            S = np.where(d > 0, sw / np.where(d > 0, d, 1.0), 0.0) * _SRC * freq**2
        w = np.exp(-tau_front) * -np.expm1(-d)
        contrib = S * w
        I = contrib.sum(axis=1)
        tau = d.sum(axis=1)
        cut_ray = ~alive[:, -1]
        if t_body is not None and hit_body is not None:
            body = hit_body[sm["idx"]] & ~cut_ray
            I = I + np.where(body, _SRC * freq**2 * float(t_body) * np.exp(-tau), 0.0)
        out["I" + tag], out["tau" + tag] = I, tau
        contrib_tot += contrib
        with np.errstate(invalid="ignore", divide="ignore"):
            gsp = 0.0
            if e is None:
                grp = np.where(d > 0, contrib * g / np.where(d > 0, d, 1.0), 0.0)
            else:
                # with an extra source the shares are those of the source function
                hh = np.where(ok, h, 0.0)
                grp = np.where(sw > 0, contrib * hh / np.where(sw > 0, sw, 1.0), 0.0)
                # share of the step's source function due to the extra part:
                # S = sum(tau_i T_i)/sum(tau_i), so the share is st_extra / sw
                ee = np.where(ok, e, 0.0)
                gsp = np.where(sw > 0, contrib * ee / np.where(sw > 0, sw, 1.0), 0.0)
                contrib_gs += gsp
            contrib_gr += grp
        # free-free part of this polarisation (as seen by the observer)
        out["ff" + tag] = (contrib - grp - gsp).sum(axis=1)
        cut_any |= cut_ray

    if coupling == "weak":
        out["V"] = out["I1"] - out["I2"]
        out["vsign"] = np.ones(nr)
    else:
        has_b = bmag > 0
        first = np.argmax(has_b, axis=1)
        sgn = np.sign(cos[np.arange(nr), first])
        out["V"] = sgn * (out["I1"] - out["I2"])
        out["vsign"] = sgn

    tot = contrib_tot.sum(axis=1)
    r_m = np.linalg.norm(0.5 * (sm["p"][:, 1:] + sm["p"][:, :-1]), axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["radius"] = np.where(tot > 0, (contrib_tot * r_m).sum(axis=1) / tot, np.nan)
        out["gr_fraction"] = np.where(tot > 0, contrib_gr.sum(axis=1) / tot, 0.0)
        if extra is not None:
            out["gs_fraction"] = np.where(tot > 0, contrib_gs.sum(axis=1) / tot, 0.0)
    # dominant harmonic: the s whose layers carry most of the intensity
    dom = np.full(nr, -1, np.int16)
    if use_gr:
        share = np.zeros((int(s_max) + 1, nr))
        np.add.at(share, (best_s.ravel(), np.repeat(np.arange(nr), dl.shape[1])),
                  contrib_gr.ravel())
        share[0] = contrib_tot.sum(axis=1) - contrib_gr.sum(axis=1)   # free-free part
        if extra is not None:
            share[0] -= contrib_gs.sum(axis=1)
        dom = np.where(tot > 0, np.argmax(share, axis=0), -1).astype(np.int16)
    elif use_ff:
        dom = np.where(tot > 0, 0, -1).astype(np.int16)
    out["dominant_s"] = dom
    out["cutoff"] = cut_any
    out["x_flag"] = x_flag_step.any(axis=1)
    return out
