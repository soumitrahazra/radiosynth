# radiosynth - From MHD to MHz: synthetic radio images of MHD models of the Sun and stars
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
"""Emission-model interface used by the refractive ray tracer.

``EmissionModel(kind, **options)`` is a callable returning ``(kappa, S)``
at sample points:

    kappa : absorption coefficient [1/m]  (1/length_unit for 'simplistic')
    S     : source function [W m^-2 Hz^-1 sr^-1]

so that the BATS-R-US-style pixel intensity is I = sum(kappa * S * ds).

The physics lives in the mechanism folders (radiosynth/simplistic,
radiosynth/bremsstrahlung, ...); this class only looks the mechanism up in
the registry (common/mechanism.py) and forwards the call.
"""
import numpy as np

from .constants import R_SUN
from .mechanism import get_mechanism


class EmissionModel:
    """Callable returning (kappa [1/m], S [SI intensity]) for given n_e, T_e, freq.

    kind  : 'bremsstrahlung' or 'simplistic' (any registered mechanism that
            provides coefficients from n_e and T_e alone)
    gaunt : Gaunt factor for bremsstrahlung (number or 'dulk')
    """

    def __init__(self, kind="bremsstrahlung", gaunt=10.0):
        kind = kind.lower()
        mech = get_mechanism(kind)
        if mech.coefficients is None:
            raise ValueError(f"mechanism '{kind}' cannot be used with the refractive ray "
                             f"tracer; use radiosynth.{kind}.make_image instead")
        self.kind = kind
        self.gaunt = gaunt
        self._coef = mech.coefficients

    def __repr__(self):
        return f"EmissionModel(kind={self.kind!r}, gaunt={self.gaunt!r})"

    def __call__(self, ne, te, freq, length_unit=R_SUN):
        if self.kind == "bremsstrahlung":
            return self._coef(ne, te, freq, length_unit=length_unit, gaunt=self.gaunt)
        return self._coef(np.asarray(ne, dtype=float), te, freq, length_unit=length_unit)
