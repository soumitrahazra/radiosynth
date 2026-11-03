# radiosynth - From MHD to MHz: synthetic radio images of MHD models of the Sun and stars
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
"""Refractive radio ray tracing with emissivity integration.

Python port of ``ray_path`` in BATS-R-US ModRadioWaveImage.f90
(Benkevitch et al. 2010; Moschou et al. 2018), vectorised over all rays.

Ray equations (Moschou+2018 Eq. 7-13):   dr/ds = v,  dv/ds = v x (grad n / n x v)
with n^2 = eps = 1 - n_e/n_cr.  Each step of trace_rays calls, in order:

1. ``_predictor``        advance half a step along the current direction and
                         sample n_e, grad n_e, T_e and the cell size there;
2. ``_terminations``     find rays that left the box / integration sphere,
                         hit the body or reached over-dense plasma;
   ``_integrate_to_body`` add the last straight piece of rays whose mid point
                         is already inside the body;
3. ``_dielectric``       eps, grad eps and eps half a step back; rays with
                         eps <= 0 there are stopped;
   ``_snell_fallback``   (default mode) step back / reflect rays that
                         trespassed the critical surface instead of stopping them;
4. ``_check_step``       step control: if the curvature is too large shrink ds
                         and redo; if the next step may cross the critical
                         surface mark the ray "steep", store the distance to the
                         critical surface, shrink ds and redo;
5. ``_advance``          straight line (no refraction), ``_parabolic_step``
                         near the critical surface, or ``_boris_step``;
   ``_clip_at_body``     shorten the last segment of rays entering the body;
   ``_add_emission``     add the emission of the segment (thin or full transfer);
6. ``_new_step_size``    smoothly regrow ds towards the cell size.

Lengths are in the grid's coordinate unit (R_sun by default, or R_star; the
conversion to metres is grid.length_unit), physical quantities SI.
"""
import numpy as np

from .constants import R_SUN, critical_density

# Termination codes
ACTIVE, ESCAPED, HIT_BODY, OVERDENSE, NEG_EPS, OPAQUE, MAX_STEPS, MISSED = range(8)
STATUS_NAMES = {ACTIVE: "active", ESCAPED: "escaped", HIT_BODY: "hit body",
                OVERDENSE: "over-dense", NEG_EPS: "eps<=0", OPAQUE: "tau>tau_max",
                MAX_STEPS: "max steps", MISSED: "missed grid"}


# ============================================================================
# building blocks (all vectorised over the rays of the current iteration)
# ============================================================================
def _emission_function(emission, L):
    """emit(ne, te, freq) -> (kappa [1/m], B); passes the length unit to the
    built-in EmissionModel (needed by the simplistic mechanism)."""
    if hasattr(emission, "kind"):                           # built-in EmissionModel
        def emit(ne_, te_, f_):
            return emission(ne_, te_, f_, length_unit=L)
        return emit
    return emission                                         # user function (ne, te, freq)


def _add_emission(intensity, tau, idx, kappa, B, dl, transfer):
    """Add the emission of segments of length dl [m] to rays idx.
    thin: I += kappa B dl;  full: I += B (1 - exp(-dtau)) exp(-tau), tau += dtau."""
    if transfer == "thin":
        intensity[idx] += kappa * B * dl
    else:
        dtau = kappa * dl
        intensity[idx] += B * np.exp(-tau[idx]) * (-np.expm1(-dtau))
        tau[idx] += dtau


def _predictor(grid, p, s, d):
    """Mid point of the new segment and the plasma there."""
    half = 0.5 * d
    pm = p + s * half[:, None]
    ne, gn, te, cell, inside = grid.sample(pm)
    return half, pm, ne, gn, te, cell, inside


def _terminations(pm, ne, inside, ncr, rbody2, rint2):
    """Termination code of every ray from its mid point (0 = continue)."""
    r2 = np.einsum("ij,ij->i", pm, pm)
    term = np.zeros(pm.shape[0], np.int8)
    term[r2 < rbody2] = HIT_BODY
    term[(term == 0) & (~inside | (r2 > rint2))] = ESCAPED
    term[(term == 0) & (ne >= ncr)] = OVERDENSE
    return term


def _integrate_to_body(grid, p, s, d, ncr, rbody2, emit, freq, L, idx, intensity, tau,
                       transfer):
    """Rays whose predictor point is already inside the body: integrate the
    remaining straight piece from the segment start to the surface."""
    bb = np.einsum("ij,ij->i", p, s)
    cc = np.einsum("ij,ij->i", p, p) - rbody2
    tt = np.clip(-bb - np.sqrt(np.maximum(bb * bb - cc, 0.0)), 0.0, d)
    ne_h, _, te_h, _, _ = grid.sample(p + s * (0.5 * tt)[:, None])
    ne_h = np.minimum(ne_h, ncr)
    k_h, B_h = emit(ne_h, te_h, freq)
    _add_emission(intensity, tau, idx, k_h, B_h, tt * L, transfer)


def _dielectric(ne, gn, s, half, ncr):
    """Dielectric permittivity eps = n^2 at the mid point, its gradient
    (per coordinate unit), |grad eps|^2, grad eps . v and eps half a step back."""
    eps = 1.0 - ne / ncr                       # DielPerm
    ge = -gn / ncr                             # GradDielPerm_D  [1/R_sun]
    ge2 = np.einsum("ij,ij->i", ge, ge)
    gs = np.einsum("ij,ij->i", ge, s)          # GradEpsDotSlope
    ehb = eps - gs * half                      # DielPermHalfBack
    return eps, ge, ge2, gs, ehb


def _snell_fallback(term, act, d, s, ge, ge2, gs, ds, slope, steep, min_ds, step_min):
    """Default mode: rays that trespassed the critical surface (over-dense mid
    point or eps <= 0 half a step back) are stepped back with ds/4 and, when
    ds < 4 min_ds, specularly reflected about grad eps (Snell's law).
    Modifies term, ds, slope, steep; returns the mask of such rays."""
    bad = (term == OVERDENSE) | (term == NEG_EPS)
    if not bad.any():
        return np.zeros(act.size, bool)
    back = bad & (d >= 4.0 * min_ds)
    ds[act[back]] = 0.25 * d[back]
    snell = bad & ~back & (ge2 > 0) & (gs < 0)
    if snell.any():
        gh = ge[snell] / np.sqrt(ge2[snell])[:, None]
        sv = s[snell]
        slope[act[snell]] = sv - 2.0 * np.einsum("ij,ij->i", sv, gh)[:, None] * gh
        ds[act[snell]] = np.maximum(d[snell], step_min)
        steep[act[snell]] = False
    term[back | snell] = 0
    return back | snell


def _check_step(ok, act, d, half, ehb, ge2, gs, steep, dist2cr, ds, tolerance, tol2,
                step_min, min_ds, strict_batsrus):
    """BATS-R-US check_step for "gentle" (non-steep) rays.

    (a) curvature too large for the tolerance -> shrink ds;
    (b) the next step may cross the critical surface -> mark steep, store the
        distance to the critical surface, shrink ds.
    The rays stay at the segment start (PositionHalfBack) and are retried.
    Returns the mask of retried rays."""
    chk = ok & ~steep[act]
    with np.errstate(divide="ignore", invalid="ignore"):
        curv = (0.5 * half / ehb)**2 * (ge2 - gs**2)
        r_a = chk & (curv >= tol2)
        newd = d.copy()
        newd[r_a] = 0.99 * d[r_a] / (2.0 * np.sqrt(curv[r_a] / tol2))
        if not strict_batsrus:
            newd[r_a] = np.maximum(newd[r_a], min_ds)
            r_a &= d > min_ds * 1.0000001
        r_b = chk & ~r_a & (3.0 * gs * half <= -ehb)
        d2c = ehb[r_b] / np.sqrt(ge2[r_b])
    idx = act[r_b]
    steep[idx] = True
    dist2cr[idx] = d2c
    newd[r_b] = np.maximum(0.5 * tolerance * d2c, step_min)
    retry = r_a | r_b
    ds[act[retry]] = newd[retry]
    return retry


def _parabolic_step(p_start, s, g, gs, ge2, ehb):
    """Switch to the opposite branch of the parabola near the critical surface
    (reflection).  Returns new direction, new position and path length."""
    lcos = -gs / ge2                                   # L cos(alpha)
    proj = -lcos[:, None] * g                          # v_proj
    stepy = s - proj
    snew = s - 2.0 * proj
    stepy = 4.0 * stepy * (lcos * ehb)[:, None]
    pnew = p_start + stepy                             # PositionHalfBack + StepY
    stepx = (ehb * lcos**2)[:, None] * g
    seg = np.sqrt(np.einsum("ij,ij->i", 2 * stepx, 2 * stepx)
                  + np.einsum("ij,ij->i", stepy, stepy))
    return snew, pnew, seg


def _boris_step(s, pm, half, eps, g):
    """Boris (CYLRAD) rotation of the direction by grad(n)/n over the step, then
    the second half step from the mid point."""
    coef = 0.5 * half / eps
    rel = coef[:, None] * g                            # grad(n)/n * ds/2
    om = np.cross(rel, s)
    s1 = s + np.cross(s, om)
    om = np.cross(rel, s1)
    s1 = s + np.cross(s, om)
    c2 = 2.0 / (1.0 + np.einsum("ij,ij->i", om, om))
    snew = s + c2[:, None] * np.cross(s1, om)
    pnew = pm + snew * half[:, None]
    return snew, pnew


def _advance(p_start, s, pm, half, d, eps, ge, gs, ge2, ehb, no_refraction,
             strict_batsrus, step_min):
    """advance_ray: new direction, position and segment length of every ray."""
    if no_refraction:
        return s, pm + s * half[:, None], d
    refl = 3.0 * gs * half <= -ehb
    if strict_batsrus:
        refl |= (d < step_min) & (gs < 0.0)
    snew = np.empty_like(s)
    pnew = np.empty_like(pm)
    seg = d.copy()
    if refl.any():
        snew[refl], pnew[refl], seg[refl] = _parabolic_step(
            p_start[refl], s[refl], ge[refl], gs[refl], ge2[refl], ehb[refl])
    bo = ~refl
    if bo.any():
        snew[bo], pnew[bo] = _boris_step(s[bo], pm[bo], half[bo], eps[bo], ge[bo])
    return snew, pnew, seg


def _clip_at_body(p_start, pnew, rbody2):
    """Fraction of every segment outside the body (1 if it does not enter it),
    and the mask of segments that enter it."""
    r2e = np.einsum("ij,ij->i", pnew, pnew)
    into_body = r2e < rbody2
    frac = np.ones(pnew.shape[0])
    if into_body.any():
        ps, dd_ = p_start[into_body], pnew[into_body] - p_start[into_body]
        aa = np.einsum("ij,ij->i", dd_, dd_)
        bb = 2.0 * np.einsum("ij,ij->i", ps, dd_)
        cc = np.einsum("ij,ij->i", ps, ps) - rbody2
        disc = np.maximum(bb * bb - 4.0 * aa * cc, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            tt = (-bb - np.sqrt(disc)) / (2.0 * aa)
        frac[into_body] = np.clip(np.nan_to_num(tt, nan=1.0), 0.0, 1.0)
    return frac, into_body


def _new_step_size(ia, d, dsnew, ehb, ge2, steep, dist2cr, ds, no_refraction):
    """get_new_step_size: regrow ds smoothly towards the local cell size, except
    for steep rays still close to the critical surface."""
    cond = ~steep[ia] | (ehb > dist2cr[ia] * np.sqrt(ge2))
    if no_refraction:
        cond[:] = True
    ic = ia[cond]
    steep[ic] = False
    dd, dn = d[cond], dsnew[cond]
    ds[ic] = np.maximum(2.0 - dd / dn, 1.0) * np.minimum(dn, dd)


# ============================================================================
def trace_rays(grid, pos0, dir0, freq, emission, *, transfer="thin",
               no_refraction=False, r_integration=None, tolerance=1.0e-2,
               step_min=1.0e-2, step_factor=1.0, ds_init=None, tau_max=30.0,
               kappa_over_mu=False, max_steps=200000, status0=None,
               record_ids=None, strict_batsrus=False, min_ds=1.0e-6):
    """Trace a bunch of rays and integrate the emission along them.

    Parameters
    ----------
    grid      : MHDGrid
    pos0      : (N,3) starting points [R_sun] (inside the grid box)
    dir0      : (N,3) initial directions (pointing away from the observer)
    freq      : radio frequency [Hz]
    emission  : EmissionModel (or any callable(ne, te, freq) -> (kappa, B))
    transfer  : 'thin'  -> I = sum kappa B ds  (BATS-R-US / Moschou+2018)
                'full'  -> formal solution of dI/ds = kappa (B - I):
                           I = sum B (1 - exp(-dtau)) exp(-tau)
    no_refraction : straight lines of sight (UseNoRefraction in BATS-R-US)
    r_integration : radius of the integration sphere [R_sun]; None -> no limit
    tolerance, step_min : BATS-R-US Tolerance (1e-2) and StepMin (1e-2 R_sun)
    step_factor : DeltaSNew = step_factor * local cell size
    ds_init   : initial step [R_sun]; None -> local cell size * step_factor
    tau_max   : (full transfer only) stop a ray once tau exceeds this
    kappa_over_mu : divide kappa by the refractive index mu=sqrt(eps)
                    (not done in BATS-R-US; physically appropriate near the
                    critical layer, Dulk 1985)
    max_steps : safety cap on iterations per ray
    status0   : optional initial status array (rays with status!=0 are skipped)
    record_ids: optional indices of rays whose paths should be stored
    strict_batsrus : reproduce ModRadioWaveImage.f90 exactly.  Differences
                    of the default (False) mode, following Benkevitch+2010 Sec. 5:
                    (a) the parabolic switch is used only when the linear
                        prediction says the next step would cross the critical
                        surface; the extra Fortran trigger
                        "ds < StepMin .and. grad(eps).v < 0" is dropped, because
                        it is reached after curvature-driven step reduction and
                        makes rays far from the critical surface (eps ~ 0.6)
                        jump along a long, wrong parabola;
                    (b) a ray whose mid point lands in over-dense plasma (or with
                        eps<=0 at the start point) is not discarded: it is stepped
                        back with ds/4 and, when ds < 4*min_ds, specularly
                        reflected about grad(eps) ("linear reflection following
                        Snell's law", Benkevitch+2010; Moschou+2018 Sec. 2.2).
    min_ds    : smallest step used by the curvature control / fallback [R_sun]

    Returns
    -------
    dict with 'intensity' [W m^-2 Hz^-1 sr^-1], 'tau', 'status', 'nstep',
    and 'paths' / 'slopes' (dict id -> (M,3) arrays of positions and unit
    directions after every accepted step) when record_ids is given.
    """
    if transfer not in ("thin", "full"):
        raise ValueError("transfer must be 'thin' or 'full'")
    # ---- set-up
    pos = np.array(pos0, dtype=float)
    slope = np.array(dir0, dtype=float)
    slope /= np.linalg.norm(slope, axis=1)[:, None]
    N = pos.shape[0]
    ncr = float(critical_density(freq))
    tol2 = tolerance**2
    rint2 = np.inf if r_integration is None else r_integration**2 + 0.01
    rbody2 = grid.r_body**2
    L = float(getattr(grid, "length_unit", R_SUN))          # metres per coordinate unit
    emit = _emission_function(emission, L)

    status = np.zeros(N, np.int8) if status0 is None else np.array(status0, np.int8)
    intensity = np.zeros(N)
    tau = np.zeros(N)
    nstep = np.zeros(N, np.int64)
    dist2cr = np.zeros(N)
    steep = np.zeros(N, bool)
    ds = np.zeros(N)

    act = np.flatnonzero(status == ACTIVE)
    if ds_init is None:
        if act.size:
            ds[act] = step_factor * grid.sample(pos[act])[3]
    else:
        ds[:] = ds_init

    paths = None
    if record_ids is not None:
        record_ids = np.atleast_1d(record_ids)
        rec_mask = np.zeros(N, bool)
        rec_mask[record_ids] = True
        paths = {int(i): [pos[i].copy()] for i in record_ids}
        slopes = {int(i): [slope[i].copy()] for i in record_ids}

    # ---- one iteration per loop: every active ray makes (or retries) one step
    while act.size:
        p, s, d = pos[act], slope[act], ds[act]
        half, pm, ne, gn, te, cell, inside = _predictor(grid, p, s, d)
        dsnew = step_factor * cell

        term = _terminations(pm, ne, inside, ncr, rbody2, rint2)
        hb = np.flatnonzero(term == HIT_BODY)
        if hb.size:
            _integrate_to_body(grid, p[hb], s[hb], d[hb], ncr, rbody2, emit, freq, L,
                               act[hb], intensity, tau, transfer)

        eps, ge, ge2, gs, ehb = _dielectric(ne, gn, s, half, ncr)
        term[(term == 0) & (ehb <= 0.0)] = NEG_EPS

        if not strict_batsrus and not no_refraction:
            ok_bad = _snell_fallback(term, act, d, s, ge, ge2, gs, ds, slope, steep,
                                     min_ds, step_min)
        else:
            ok_bad = np.zeros(act.size, bool)
        ok = (term == 0) & ~ok_bad

        retry = np.zeros(act.size, bool)
        if not no_refraction:
            retry = _check_step(ok, act, d, half, ehb, ge2, gs, steep, dist2cr, ds,
                                tolerance, tol2, step_min, min_ds, strict_batsrus)

        adv = np.flatnonzero(ok & ~retry)
        if adv.size:
            ia = act[adv]
            kap, B = emit(ne[adv], te[adv], freq)
            if kappa_over_mu:
                kap = kap / np.sqrt(np.maximum(eps[adv], 1e-6))
            snew, pnew, seg = _advance(p[adv], s[adv], pm[adv], half[adv], d[adv], eps[adv],
                                       ge[adv], gs[adv], ge2[adv], ehb[adv], no_refraction,
                                       strict_batsrus, step_min)
            frac, into_body = _clip_at_body(p[adv], pnew, rbody2)
            _add_emission(intensity, tau, ia, kap, B, seg * frac * L, transfer)
            pos[ia] = pnew
            slope[ia] = snew

            # rays which left the box / integration sphere, hit the body, or are opaque
            r2n = np.einsum("ij,ij->i", pnew, pnew)
            gone = (~grid.inside(pnew) & ~into_body) | (r2n > rint2)
            status[ia[gone]] = ESCAPED
            status[ia[into_body & ~gone]] = HIT_BODY
            if transfer == "full":
                status[ia[(~gone) & (tau[ia] > tau_max)]] = OPAQUE

            _new_step_size(ia, d[adv], dsnew[adv], ehb[adv], ge2[adv], steep, dist2cr, ds,
                           no_refraction)

            if paths is not None:
                for i in ia[rec_mask[ia]]:
                    paths[int(i)].append(pos[i].copy())
                    slopes[int(i)].append(slope[i].copy())

        tmask = (term > 0) & ~ok_bad
        status[act[tmask]] = term[tmask]
        nstep[act] += 1
        over = (nstep[act] >= max_steps) & (status[act] == ACTIVE)
        status[act[over]] = MAX_STEPS
        act = act[status[act] == ACTIVE]

    out = {"intensity": intensity, "tau": tau, "status": status, "nstep": nstep}
    if paths is not None:
        out["paths"] = {k: np.array(v) for k, v in paths.items()}
        out["slopes"] = {k: np.array(v) for k, v in slopes.items()}
    return out
