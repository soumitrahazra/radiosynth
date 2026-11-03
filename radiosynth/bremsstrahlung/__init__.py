# radiosynth - From MHD to MHz: synthetic radio images of MHD models of the Sun and stars
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
"""Thermal free-free (bremsstrahlung) emission.

* emission.py   : kappa and source function (Moschou et al. 2018, g=10 or
                  Dulk 1985); used by the refractive ray tracer in common/
* magnetized.py : two-mode (o/x) free-free absorption in a magnetic field,
                  used with the polarized straight-ray transfer (gyroresonance)
* wind.py       : analytic isothermal-wind solution (validation, stars)

>>> from radiosynth import bremsstrahlung as ff
>>> img = ff.make_image(grid, "80MHz, 17GHz", observer=(-215, 0, 0), transfer="full")
"""
from ..common.mechanism import Mechanism, register
from .emission import bremsstrahlung_coefficients
from .magnetized import magnetized_ff_coefficients, magnetoionic_params
from .wind import wind_optical_depth, wind_flux_density


def make_image(grid, frequencies, **kw):
    """Refractive ray-traced free-free image (same keywords as
    radiosynth.make_radio_image; gaunt=10 or 'dulk', transfer='thin'/'full')."""
    from ..common.image import make_radio_image
    kw.pop("emission", None)
    return make_radio_image(grid, frequencies, emission="bremsstrahlung", **kw)


def compare_refraction(grid, frequencies, **kw):
    """Refracted vs straight-line free-free image (see radiosynth.compare_refraction)."""
    from ..common.image import compare_refraction as _cmp
    kw.pop("emission", None)
    return _cmp(grid, frequencies, emission="bremsstrahlung", **kw)


MECHANISM = register(Mechanism(
    "bremsstrahlung", coefficients=bremsstrahlung_coefficients, make_image=make_image,
    description="thermal free-free emission (Moschou+2018), refractive ray tracing",
    options=("gaunt",)))

__all__ = ["bremsstrahlung_coefficients", "magnetized_ff_coefficients",
           "magnetoionic_params", "wind_optical_depth", "wind_flux_density",
           "make_image", "compare_refraction", "MECHANISM"]
