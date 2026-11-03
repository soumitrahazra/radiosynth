# radiosynth - From MHD to MHz: synthetic radio images of MHD models of the Sun and stars
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
"""Simplistic mechanism (Benkevitch et al. 2010): dimensionless emissivity
peaking near the critical layer.  Uses the refractive ray tracer in common/.

>>> from radiosynth import simplistic
>>> img = simplistic.make_image(grid, "80MHz", observer=(-215, 0, 0))
"""
from ..common.mechanism import Mechanism, register
from .emission import simplistic_coefficients


def make_image(grid, frequencies, **kw):
    """Refractive ray-traced image with the simplistic emissivity
    (same keywords as radiosynth.make_radio_image)."""
    from ..common.image import make_radio_image
    kw.pop("emission", None)
    return make_radio_image(grid, frequencies, emission="simplistic", **kw)


def compare_refraction(grid, frequencies, **kw):
    """Refracted vs straight-line image (see radiosynth.compare_refraction)."""
    from ..common.image import compare_refraction as _cmp
    kw.pop("emission", None)
    return _cmp(grid, frequencies, emission="simplistic", **kw)


MECHANISM = register(Mechanism(
    "simplistic", coefficients=simplistic_coefficients, make_image=make_image,
    description="ad-hoc emissivity near the critical layer (Benkevitch+2010); "
                "dimensionless intensity"))

__all__ = ["simplistic_coefficients", "make_image", "compare_refraction", "MECHANISM"]
