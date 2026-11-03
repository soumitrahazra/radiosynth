# radiosynth - From MHD to MHz: synthetic radio images of MHD models of the Sun and stars
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
"""Registry of radio emission mechanisms.

Each mechanism lives in its own sub-package of radiosynth
(``simplistic/``, ``bremsstrahlung/``, ``gyroresonance/``) and registers
itself here when it is imported.  The shared machinery in ``common/`` (grids,
readers, ray tracer, image driver, figures) only talks to mechanisms through
this registry, so a new mechanism is added by creating a new folder and
calling :func:`register` - nothing in ``common/`` has to change.

A mechanism entry has

name          : the string used in ``emission=...`` and on the command line
coefficients  : callable ``(ne, te, freq, length_unit=..., **options) ->
                (kappa [1/m], S)`` giving the absorption coefficient and the
                source function at sample points, for the refractive ray
                tracer (None if the mechanism needs more than n_e and T_e,
                e.g. the magnetic field for gyroresonance)
make_image    : callable ``(grid, frequencies, **kw) -> RadioImage``
description   : one line shown by ``python -m radiosynth --list-mechanisms``
"""
import importlib

_REGISTRY = {}

# sub-packages that provide mechanisms (imported on demand)
BUILTIN = ("simplistic", "bremsstrahlung", "gyroresonance", "gyrosynchrotron", "combined")


class Mechanism:
    """Description of one emission mechanism (see module docstring)."""

    def __init__(self, name, coefficients=None, make_image=None, description="",
                 options=()):
        self.name = name
        self.coefficients = coefficients
        self.make_image = make_image
        self.description = description
        self.options = tuple(options)

    def __repr__(self):
        return f"Mechanism({self.name!r}: {self.description})"


def register(mech):
    """Add (or replace) a mechanism in the registry."""
    _REGISTRY[mech.name.lower()] = mech
    return mech


def get_mechanism(name):
    """Return the Mechanism called ``name``, importing its sub-package if needed."""
    key = name.lower()
    if key not in _REGISTRY and key in BUILTIN:
        importlib.import_module(f"radiosynth.{key}")
    if key not in _REGISTRY:
        raise ValueError(f"Unknown radio emission mechanism '{name}' "
                         f"(available: {', '.join(available_mechanisms())})")
    return _REGISTRY[key]


def available_mechanisms():
    for b in BUILTIN:
        try:
            importlib.import_module(f"radiosynth.{b}")
        except ImportError:                     # pragma: no cover
            pass
    return sorted(_REGISTRY)
