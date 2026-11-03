# radiosynth - From MHD to MHz: synthetic radio images of MHD models of the Sun and stars
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
"""Description of the fast (non-thermal) electrons.

An MHD model contains no fast electrons, so they have to be prescribed.
FastElectrons holds that prescription: at every sample point of a line of
sight it returns the parameters of an analytical distribution function of
the fast GS codes (Kuznetsov & Fleishman 2021, AnalyticalDistributions.pdf):

    f(E, mu) = u(E) g(mu),   2 pi int u dE = n_b,   int g dmu = 1

energy distributions u(E)   ('energy'):
    'plw'  power law in kinetic energy, N(E) ~ E^-delta, E_min < E < E_max
    'dpl'  double power law (delta below, delta2 above E_break)
    'kap'  kappa distribution (uses kappa; replaces the thermal electrons)
    'tnt'  thermal + power-law tail matched at eps (thermal/non-thermal)
    'tpl'  isotropic thermal + power law (thermal gyro emission included)
    'plp', 'plg'  power law in momentum / Lorentz factor
pitch-angle distributions g(mu)   ('pitch'):
    'iso'  isotropic
    'gau'  gaussian loss cone (alpha_c = loss-cone angle, dmu = width)
    'elc'  exponential loss cone
    'gab'  gaussian beam (alpha_c = beam direction, dmu = width)
    'sga'  super-gaussian beam (also a4)

Every parameter may be a number or a *rule*: a callable taking the dict of
sampled quantities

    pos (N,3) [coordinate units], r (N,), ne [m^-3], te [K], b (N,3) [G],
    bmag [G], and any grid scalar fields (e.g. 'j' [A m^-2] after
    radiosynth.common.fields.current_density(grid))

and returning an (N,) array.  The helper functions below build common rules
for the fast-electron density n_b [cm^-3].

Alternatively, an array-defined distribution can be given (from_array): a
tabulated shape f(E, mu), normalised to unit density, multiplied at every
point by the local n_b (for instance from test-particle or Fokker-Planck
results); it is passed to the codes as their numerically defined
distribution.
"""
import math

import numpy as np

ENERGY_INDEX = {"fff": 0, "thm": 2, "plw": 3, "dpl": 4, "tnt": 5, "kap": 6, "plp": 7,
                "plg": 8, "tnp": 9, "tng": 10, "tpl": 11, "tdp": 12}
PITCH_INDEX = {"iso": 0, "elc": 2, "gau": 3, "gab": 4, "sga": 5}
MEC2_MEV = 0.51099895
_trapz = getattr(np, "trapezoid", None) or np.trapz


class FastElectrons:
    """Prescription of the fast electrons (see module docstring).

    nb       : fast-electron density [cm^-3] (number or rule); 0 = none
    energy   : energy distribution ('plw', 'dpl', 'kap', 'tnt', 'tpl', ...)
    delta    : power-law index (delta_1 for 'dpl')
    emin_mev, emax_mev : energy cutoffs [MeV] (default 10 keV - 10 MeV)
    delta2, ebreak_mev : high-energy index and break energy ('dpl', 'tdp')
    kappa    : kappa index ('kap');  eps : matching parameter ('tnt', 'tnp', 'tng')
    pitch    : pitch-angle distribution ('iso', 'gau', 'elc', 'gab', 'sga')
    alpha_c  : loss-cone angle or beam direction [deg];  dmu : width;  a4 : 'sga'
    nb_min   : n_b below this [cm^-3] is set to 0, so that the (expensive)
               gyrosynchrotron calculation is skipped there (default 1 cm^-3)
    """

    def __init__(self, nb=0.0, energy="plw", delta=4.0, emin_mev=0.01, emax_mev=10.0,
                 delta2=None, ebreak_mev=None, kappa=None, eps=None, pitch="iso",
                 alpha_c=60.0, dmu=0.2, a4=10.0, nb_min=1.0):
        energy, pitch = energy.lower(), pitch.lower()
        if energy not in ENERGY_INDEX:
            raise ValueError(f"energy must be one of {sorted(ENERGY_INDEX)}")
        if pitch not in PITCH_INDEX:
            raise ValueError(f"pitch must be one of {sorted(PITCH_INDEX)}")
        if energy in ("dpl", "tdp") and (delta2 is None or ebreak_mev is None):
            raise ValueError("'dpl'/'tdp' need delta2 and ebreak_mev")
        if energy == "kap" and kappa is None:
            raise ValueError("'kap' needs kappa")
        self.nb, self.energy, self.delta = nb, energy, delta
        self.emin_mev, self.emax_mev = emin_mev, emax_mev
        self.delta2, self.ebreak_mev = delta2, ebreak_mev
        self.kappa, self.eps = kappa, eps
        self.pitch, self.alpha_c, self.dmu, self.a4 = pitch, alpha_c, dmu, a4
        self.nb_min = float(nb_min)
        self.array = None                    # (E [MeV], mu, f normalised) if from_array

    @classmethod
    def from_array(cls, E_mev, mu, f, nb, pitch_average=False):
        """Array-defined distribution: shape f(E, mu) [MeV^-1, arbitrary units] on
        the grid E_mev (NE >= 3, increasing) x mu (Nmu >= 3, from -1 to 1),
        rescaled at every point to the local density n_b [cm^-3] (number or rule)."""
        E = np.asarray(E_mev, float)
        m = np.asarray(mu, float)
        F = np.asarray(f, float)
        if F.shape != (E.size, m.size) or E.size < 3 or m.size < 3:
            raise ValueError("f must have shape (NE, Nmu) with NE, Nmu >= 3")
        norm = 2 * np.pi * _trapz(_trapz(F, m, axis=1), E)
        obj = cls(nb=nb)
        obj.array = (E, m, F / norm)
        obj.pitch_average = pitch_average
        return obj

    def __repr__(self):
        if self.array is not None:
            return f"FastElectrons(array-defined, NE={self.array[0].size}, Nmu={self.array[1].size})"
        return (f"FastElectrons(energy={self.energy!r}, delta={self.delta}, "
                f"E={self.emin_mev}-{self.emax_mev} MeV, pitch={self.pitch!r})")

    @staticmethod
    def _eval(v, fields, n):
        if callable(v):
            return np.broadcast_to(np.asarray(v(fields), float), (n,)).astype(float)
        return np.full(n, float(v))

    def evaluate(self, fields):
        """Dict of per-point parameters for the given sampled fields."""
        n = fields["ne"].shape[0]
        e = lambda v: self._eval(v, fields, n)                          # noqa: E731
        nb = e(self.nb)
        out = dict(nb=np.where(nb >= self.nb_min, nb, 0.0), delta=e(self.delta),
                   emin=e(self.emin_mev), emax=e(self.emax_mev),
                   alpha_c=e(self.alpha_c), dmu=e(self.dmu), a4=e(self.a4))
        out["delta2"] = e(self.delta2) if self.delta2 is not None else np.zeros(n)
        out["ebreak"] = e(self.ebreak_mev) if self.ebreak_mev is not None else np.zeros(n)
        k = self.kappa if self.energy == "kap" else self.eps
        out["eps_kappa"] = e(k) if k is not None else np.zeros(n)
        return out

    def energy_density(self, fields):
        """Kinetic energy density of the fast electrons [erg cm^-3] (power-law
        cases 'plw', 'dpl' analytically; other cases are not estimated: 0)."""
        p = self.evaluate(fields)
        nb, d, a, b = p["nb"], p["delta"], p["emin"], p["emax"]
        if self.array is not None:
            E, m, F = self.array
            mean = 2 * np.pi * _trapz(_trapz(F, m, axis=1) * E, E)
            return nb * mean * 1.602176634e-6
        if self.energy != "plw":
            return np.zeros_like(nb)

        def moment(k):            # int E^(k-delta) dE between a and b
            q = k + 1 - d
            return np.where(np.abs(q) > 1e-9, (b**q - a**q) / np.where(np.abs(q) > 1e-9, q, 1),
                            np.log(b / a))
        mean = moment(1) / moment(0)                                    # MeV
        return nb * mean * 1.602176634e-6

    def check_energy(self, fields, bmag, limit=0.1, verbose=True):
        """Ratio of the fast-electron to the magnetic energy density (B^2/8pi);
        warns where it exceeds ``limit``.  Returns the max ratio."""
        w = self.energy_density(fields)
        wb = np.maximum(bmag, 1e-30) ** 2 / (8 * math.pi)
        r = np.where(w > 0, w / wb, 0.0)
        rmax = float(r.max(initial=0.0))
        if verbose and rmax > limit:
            print(f"WARNING: fast-electron energy density reaches {rmax:.2g} x B^2/8pi "
                  f"(> {limit}); the prescription is probably unphysical")
        return rmax


# ----------------------------------------------------------------------------
# rules for n_b [cm^-3]
# ----------------------------------------------------------------------------
def uniform(nb):
    """Constant n_b wherever there is plasma."""
    return lambda f: np.where(f["ne"] > 0, float(nb), 0.0)


def fraction_of_thermal(fraction):
    """n_b = fraction x n_e (thermal electron density)."""
    return lambda f: float(fraction) * f["ne"] * 1e-6


def power_of(field, nb0, ref, power=1.0, floor=0.0):
    """n_b = nb0 (q/ref)^power for a sampled quantity q = field ('bmag', 'ne',
    'te', 'r' or a grid scalar such as 'j'); values below ``floor`` x ref give 0."""
    def rule(f):
        q = np.maximum(np.asarray(f[field], float), 0.0) / float(ref)
        return np.where(q >= floor, float(nb0) * q ** float(power), 0.0)
    return rule


def in_sphere(center, radius, nb, profile="gaussian", cutoff=3.0):
    """Fast electrons in a blob around ``center`` [coordinate units]:
    profile 'gaussian' n_b = nb exp(-d^2/radius^2) for d < cutoff x radius
    (0 beyond), or 'top-hat' (d < radius)."""
    c = np.asarray(center, float)

    def rule(f):
        d2 = np.sum((f["pos"] - c) ** 2, axis=1) / float(radius) ** 2
        if profile == "gaussian":
            return np.where(d2 < cutoff**2, nb * np.exp(-d2), 0.0)
        return nb * (d2 < 1.0)
    return rule


def in_box(lo, hi, nb):
    """Constant n_b inside the box lo <= pos <= hi."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return lambda f: np.where(np.all((f["pos"] >= lo) & (f["pos"] <= hi), axis=1), float(nb), 0.0)


def product(*rules):
    """Product of rules (e.g. in_sphere(...) x power_of('bmag', ...) / nb0)."""
    def rule(f):
        out = 1.0
        for r in rules:
            out = out * (r(f) if callable(r) else float(r))
        return out
    return rule
